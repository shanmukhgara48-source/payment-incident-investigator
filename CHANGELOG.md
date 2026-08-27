# Changelog

## Real API Integration Milestone - 2026-08-27

- Added an opt-in Razorpay SDK path for capped, throttled test Payment Link creation.
- Added verified, idempotent webhook reconciliation for captured, failed, and paid test events.
- Added persistent test-link audit metadata and a dry-run-first cleanup command.
- Added `.env` loading, strict test-key enforcement, generated-output hygiene, and a demo-day preflight checklist.
- Added clear `SIMULATED`, `LIVE TEST-MODE`, and `ACTUAL TEST-MODE` labeling across API records, audit trails, documentation, and the dashboard.
- Kept the detector, correlator, recovery decision gates, and modeled GMV calculations unchanged.
