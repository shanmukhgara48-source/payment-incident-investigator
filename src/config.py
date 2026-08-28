"""Human-set policy knobs for bounded revenue recovery.

These values are deliberately centralized so reviewers can audit and tune the
autonomous-action boundary without searching through pipeline logic.
"""

import os

# Diagnoses below this score can only be escalated to a human. The correlator
# uses the same boundary before it is willing to name a root cause.
MIN_CONFIDENCE_FOR_AUTO_ACTION = 0.60

# A single payment above this amount is never included in an autonomous retry
# or Payment Link campaign.
MAX_AUTO_RETRY_AMOUNT_INR = 50_000

# Stops repeated recovery attempts from turning into an unbounded retry loop.
MAX_RETRIES_PER_PAYMENT = 2

# Merchant notifications are suppressed below this failed-GMV exposure to
# avoid alert fatigue. The confidence gate still applies above the threshold.
MERCHANT_NOTIFICATION_EXPOSURE_THRESHOLD_INR = 100_000

# MODELING ASSUMPTION for this synthetic demo only. This is not a measured
# Razorpay statistic and must remain labeled anywhere the result is displayed.
ASSUMED_RECOVERY_SUCCESS_RATE = 0.35

# Route failures cannot be retried until an explicit passing health signal is
# present; absence of a failing signal is not considered confirmation.
ROUTE_HEALTH_CONFIRMATION_REQUIRED = True

# Real Razorpay calls are opt-in and test-mode only. The environment flag is
# read at call time so `python -m src.run_demo --live-api` can enable it safely.
LIVE_API_MODE_ENV_VAR = "LIVE_API_MODE"
RAZORPAY_TEST_KEY_PREFIX = "rzp_test_"

# Hard caps for the Razorpay test-mode side effect. The per-run limit protects
# the account-wide test Payment Link quota when the full batch is evaluated.
MAX_REAL_LINKS_PER_INCIDENT = 3
MAX_REAL_LINKS_PER_DEMO_RUN = 3

# Simple in-process rate limit: never send more than one create/cancel request
# per second to Razorpay's test endpoints.
REAL_API_CALL_INTERVAL_SECONDS = 1.0

# Recovery links are intentionally short-lived during a demo.
PAYMENT_LINK_EXPIRY_MINUTES = 30

# Duration (minutes) used to model how much future GMV would continue failing
# on a degraded route if no reroute action were taken. Only applies to incidents
# where the recommended action is "reroute traffic".
PROTECTED_WINDOW_MINUTES = 30


def live_api_mode_enabled() -> bool:
    """Return whether the explicitly gated Razorpay test-mode path is enabled."""

    return os.getenv(LIVE_API_MODE_ENV_VAR, "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
