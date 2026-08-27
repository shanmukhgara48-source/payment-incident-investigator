# Changelog

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
