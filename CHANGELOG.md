# Changelog

## OpenAI LLM Reasoning Integration - 2026-08-28

### What was added

Wired real OpenAI API calls into the correlator and skeptic stages, replacing
the pure rule-based diagnosis path with LLM-backed reasoning when
`OPENAI_API_KEY` is configured.

- **`src/llm.py`** — shared OpenAI client module with lazy initialization,
  structured JSON calls, automatic retry (2 attempts, 30s timeout), and
  cumulative usage/cost tracking.
- **`src/correlator.py`** — when LLM is available, sends all extracted evidence
  (deploys, config changes, error traces, health signals, network alerts,
  webhook status, failure concentration) to `gpt-4.1-mini` and asks for a
  structured root-cause diagnosis. Validates the response against `VALID_CAUSES`,
  enforces the 0.60 confidence gate, and falls back to rule-based scoring on
  any failure.
- **`src/skeptic.py`** — when LLM is available, sends evidence plus the primary
  diagnosis to the model for adversarial review. Individual penalties capped at
  0.15, total at 0.50. The hard invariant `final_confidence <= primary_confidence`
  is enforced programmatically regardless of LLM output.
- Every pipeline record carries a `reasoning_mode` field: `LLM_REASONED`,
  `RULE_BASED`, or `RULE_BASED_FALLBACK`.
- Rule-based scores are always computed and included in the evidence dict,
  regardless of which path produces the final diagnosis.

### Fallback guarantee

A fresh clone with no `OPENAI_API_KEY` produces identical results to the
pre-LLM codebase. If the API key is set but a call fails (timeout, rate limit,
invalid response, network error), the pipeline falls back automatically and
labels the output `RULE_BASED_FALLBACK`.

### Test coverage

`tests/test_llm_integration.py` (12 tests) covers:
- Rule-based labeling when no key is set
- Fallback activation when LLM calls fail
- Rejection of invalid LLM responses (unknown cause)
- Acceptance of valid LLM responses
- Confidence gate enforcement on LLM output
- Skeptic invariant preservation (confidence clamp, penalty caps)

### Hugging Face Inference API support

Added as a second backend when OpenAI credits are unavailable.  `src/llm.py`
checks `OPENAI_API_KEY` first, then `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`.
The HF path uses the OpenAI-compatible `router.huggingface.co/v1/` endpoint
with `Qwen/Qwen2.5-72B-Instruct` as the default model (free tier).

### Live evaluation results (2026-08-28)

61/61 incidents got `LLM_REASONED` via Qwen 2.5 72B on Hugging Face:

| Metric | Rule-based | LLM | Delta |
|---|---:|---:|---:|
| Overall accuracy | 43.1% | 56.9% | +13.8pp |
| config_change | 20% | 60% | +40pp |
| network_issue | 36% | 54.5% | +18.5pp |
| Misdiagnoses | 3 | 19 | +16 |
| Ambiguous honesty | 100% | 0% | -100pp |

LLM reasoning adds genuine diagnostic value (+13.8pp overall), especially on
weak-signal causes.  But it is overconfident: 19 misdiagnoses vs 3, and it
never escalates ambiguous cases.  The rule-based path is the safer autonomous
decision-maker until the LLM's confidence calibration is addressed.

### Configuration

| Env var | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | Enable LLM reasoning (OpenAI) | (empty = check HF) |
| `HF_TOKEN` | Enable LLM reasoning (Hugging Face) | (empty = rule-based) |
| `OPENAI_MODEL` | Model override | `gpt-4.1-mini` / `Qwen/Qwen2.5-72B-Instruct` |

## Label-Leakage Elimination - 2026-08-28

### What was wrong

The 100% root-cause accuracy figure was not real. The synthetic data generator
(`data/simulate.py`) produced every evidence field as a deterministic 1:1
function of the ground-truth cause. Eight independent leak channels were
identified:

1. `error_traces[].error_code` — a dict lookup `{cause: code}` that was the
   exact inverse of the correlator's `ERROR_SIGNATURES`.
2. Deploy logs with matching `affected_method`/`affected_route` — exclusive to
   `bad_deploy`.
3. `config_change` deploy events — exclusive to `config_change`.
4. `route_health:healthcheck_failed` alerts — exclusive to `bank_psp_downtime`.
5. `network_packet_loss` alerts — exclusive to `network_issue`.
6. Webhook `delivery_status="failed"` — exclusive to `gateway_error`.
7. Webhook `delivery_status="timed_out"` — exclusive to `network_issue`.
8. `FAILURE_REASON_BY_CAUSE` — a deterministic dict for payment failure reasons.

Ablation with `ERROR_SIGNATURES={}` (removing only channel 1) showed true
accuracy was 60.8%. The correlator was not diagnosing — it was inverting lookup
tables the same codebase created.

### Why the earlier fix did not survive

The first leakage pass addressed only the `trace_code` dict and added no
regression test. When the correlator was later rebuilt from scratch (replacing
`investigator.py`), the remaining seven channels were never audited and the
problem was re-introduced without detection.

### What was fixed

- Replaced all 8 deterministic signal generators with probability distributions
  per cause, with deliberate cross-cause overlap.
- Every error code now appears in at least 2 different causes' distributions.
- Deploy overlaps, config changes, route-health alerts, network alerts, and
  webhook failures are all probabilistic per cause (e.g. `deploy_overlap` has
  P=0.80 for `bad_deploy` but P=0.12 for every other cause).
- `FAILURE_REASON_BY_CAUSE` replaced with per-cause probability distributions.

### Honest accuracy after the fix

| Root cause | Accuracy | Notes |
|---|---:|---|
| bad_deploy | 50% | Deploy overlap is a strong independent signal |
| bank_psp_downtime | 50% | Route health alerts are informative but shared |
| gateway_error | 60% | Benefits from error code + webhook corroboration |
| config_change | 20% | Hardest — config events are now shared across causes |
| network_issue | 36% | Network alerts are shared with other route-level causes |
| **Overall (clear)** | **43.1%** | |
| Ambiguous honesty | 100% | All ambiguous cases correctly escalated |

### Structural regression test

`tests/test_no_label_leakage.py` contains four tests that structurally prevent
re-introduction of label leakage:

1. **No single field > 85% standalone accuracy** — trains a majority-vote
   classifier per field and asserts it cannot predict cause alone.
2. **No field is bijective with cause** — asserts every field value that appears
   ≥ 5 times maps to at least 2 different causes.
3. **Error code distribution overlap** — asserts every `ERROR_SIGNATURES` code
   is produced by at least 2 causes.
4. **Signal cross-cause overlap** — asserts deploy, config, health, network,
   and webhook signals each appear for at least 2 causes.

If any of these tests fail, the simulator has re-introduced a near-1:1 mapping
and the accuracy figure is unreliable.

## Verified Live Recovery + Test Hermeticity - 2026-08-28

- Carried one incident end to end against Razorpay's real test-mode API: `INC-0001`,
  Payment Link `plink_TV3xDAz77LgtbO`, INR 7,147, captured as `pay_TV4A0vFG7rC8L5`.
  `impact.recovered_amount_inr` for that incident is now a measured 7,147 with
  `recovery_measurement_type: ACTUAL TEST-MODE`; the modeled 37,861 is preserved
  under `modeled_recovered_amount_inr`. **1 of 61 incidents (1.6%) carries real
  captured money** — INR 7,147 of an aggregate INR 2,415,145, or 0.296%. The
  aggregate figure is unchanged and remains modeled.
- Added `src/reconcile_links.py`, a manual reconcile fallback that polls Payment
  Link status over the authenticated test API. The signed-webhook path was **not**
  exercised: no public webhook tunnel was available and `RAZORPAY_WEBHOOK_SECRET`
  was unset. Poll-sourced audit entries are labeled `AUTHENTICATED_TEST_API_POLL`,
  never `VERIFIED_WEBHOOK_SIGNATURE`, so the trail cannot claim an HMAC check that
  never ran.

### Fixed: `pytest` was making live Razorpay API calls

**What happened.** With a real `.env` on disk, running the test suite created 10
real Payment Links against the configured Razorpay account. The registry held 13
links when the demo had deliberately created only 3.

**Why it is dangerous.** The safety gate at `RazorpayTestGateway._credentials`
refuses any key that is not `rzp_test_`-prefixed, so a live key would have been
rejected at the gateway — but only because that one check held. Nothing else in
the test path was asking whether it should be talking to Razorpay at all: the
suite reached the network on its own initiative, at whatever account the ambient
environment pointed it at, with no cap the developer had opted into. The sharpest
symptom is the name of the test that did it:
`test_missing_credentials_falls_back_without_an_api_call` made a real API call.
A test asserting that no call happens was itself the caller.

**How it was found.** Not by a failing test — the suite was green. It was found by
counting rows in `data/created_test_links.json` after a run and finding more links
than the demo had created. Two tests then began failing the moment a real `.env`
existed, which is the tell: the suite had only ever been green because no
credentials were configured. Causation was confirmed by moving `.env` aside (8
passed), and moving it back (2 failed).

**Root cause — three compounding effects.**
1. `load_environment()` calls `load_dotenv()`, which repopulates exactly the
   variables the credential tests delete with `monkeypatch.delenv`.
2. The module-level singleton `_gateway = RazorpayTestGateway()` runs
   `load_environment()` at *import* time, before any fixture can patch the loader.
3. The real-uvicorn tests serve from a daemon thread that outlives the test body.
   `monkeypatch` is function-scoped, so its teardown restored `LIVE_API_MODE=true`
   while that thread was still running the pipeline.

Fixing only (1) left 3 leaked links. Fixing (1) and (2) left 2. All three had to go.

**How the fix is enforced.** `tests/conftest.py` writes `LIVE_API_MODE=false` and
empty credentials directly into `os.environ` at conftest import — before the test
modules import `src`, and `load_dotenv()` does not override variables that already
exist. That gives every thread in the process a credential-free, live-API-off
floor for the whole session, independent of fixture scope and of import order. The
autouse fixture patching `load_environment` sits on top for tests that assert on
`delenv`/`setenv` behaviour. Verified by running the full suite repeatedly with a
real `.env` present and confirming the link registry does not grow.

## Pattern Recall - 2026-08-27

- Added `src/memory.py`: a hand-crafted five-feature vector per incident (method, PSP/bank route, failure reason, failure concentration, deploy overlap) compared by cosine similarity.
- Recall is causally honest: the store only ever contains incidents already processed in the same batch, so the first incident of any batch has zero matches and no incident can see one that follows it.
- Added a running per-root-cause resolution track record ("reroute traffic resolved bank_psp_downtime in 9/9 prior cases") surfaced as supporting evidence.
- Added a "Similar past incidents" panel to the incident detail view, with click-through to the recalled incident.
- Recall never edits `predicted_cause` or `confidence`; the correlator and skeptic remain the only stages that move a diagnosis.

## Real API Integration Milestone - 2026-08-27

- Added an opt-in Razorpay SDK path for capped, throttled test Payment Link creation.
- Added verified, idempotent webhook reconciliation for captured, failed, and paid test events.
- Added persistent test-link audit metadata and a dry-run-first cleanup command.
- Added `.env` loading, strict test-key enforcement, generated-output hygiene, and a demo-day preflight checklist.
- Added clear `SIMULATED`, `LIVE TEST-MODE`, and `ACTUAL TEST-MODE` labeling across API records, audit trails, documentation, and the dashboard.
- Kept the detector, correlator, recovery decision gates, and modeled GMV calculations unchanged.
