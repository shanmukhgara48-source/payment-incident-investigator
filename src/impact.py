"""GMV impact calculations over actual current-window payment attempts."""

from __future__ import annotations

try:
    from .recovery import (
        ASSUMED_RECOVERY_SUCCESS_RATE,
        MAX_AUTO_RETRY_AMOUNT_INR,
        MAX_RETRIES_PER_PAYMENT,
    )
except ImportError:  # Supports `python src/evaluate.py`.
    from recovery import (
        ASSUMED_RECOVERY_SUCCESS_RATE,
        MAX_AUTO_RETRY_AMOUNT_INR,
        MAX_RETRIES_PER_PAYMENT,
    )


def calculate_impact(incident: dict) -> dict:
    attempts = [event for event in incident["payment_events"] if event["window"] == "current"]
    failures = [event for event in attempts if event["status"] == "failed"]
    recoverable = [
        event
        for event in failures
        if event["high_intent"]
        and event["amount_inr"] <= MAX_AUTO_RETRY_AMOUNT_INR
        and event["retry_count"] < MAX_RETRIES_PER_PAYMENT
    ]
    attempted_gmv = sum(event["amount_inr"] for event in attempts)
    failed_gmv = sum(event["amount_inr"] for event in failures)
    recoverable_gmv = sum(event["amount_inr"] for event in recoverable)
    recovered_amount = round(recoverable_gmv * ASSUMED_RECOVERY_SUCCESS_RATE)
    above_cap = [event for event in failures if event["amount_inr"] > MAX_AUTO_RETRY_AMOUNT_INR]
    retry_exhausted = [event for event in failures if event["retry_count"] >= MAX_RETRIES_PER_PAYMENT]

    return {
        "attempted_gmv_inr": attempted_gmv,
        "failed_gmv_inr": failed_gmv,
        "recoverable_gmv_inr": recoverable_gmv,
        "recovered_amount_inr": recovered_amount,
        "recovered_amount_basis": (
            f"MODELING ASSUMPTION: {ASSUMED_RECOVERY_SUCCESS_RATE:.0%} success rate applied "
            "to recoverable_gmv_inr; not a measured statistic."
        ),
        "assumed_recovery_success_rate": ASSUMED_RECOVERY_SUCCESS_RATE,
        "attempt_count": len(attempts),
        "failed_count": len(failures),
        "recoverable_payment_count": len(recoverable),
        "above_auto_retry_cap_count": len(above_cap),
        "retry_limit_reached_count": len(retry_exhausted),
        "eligibility_policy": {
            "high_intent_required": True,
            "max_auto_retry_amount_inr": MAX_AUTO_RETRY_AMOUNT_INR,
            "max_retries_per_payment": MAX_RETRIES_PER_PAYMENT,
        },
    }
