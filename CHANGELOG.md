# Changelog

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
