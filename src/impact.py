"""GMV impact calculations over actual current-window payment attempts."""

from __future__ import annotations

try:
    from .config import (
        ASSUMED_RECOVERY_SUCCESS_RATE,
        MAX_AUTO_RETRY_AMOUNT_INR,
        MAX_RETRIES_PER_PAYMENT,
        PROTECTED_WINDOW_MINUTES,
    )
except ImportError:  # Supports `python src/evaluate.py`.
    from config import (
        ASSUMED_RECOVERY_SUCCESS_RATE,
        MAX_AUTO_RETRY_AMOUNT_INR,
        MAX_RETRIES_PER_PAYMENT,
        PROTECTED_WINDOW_MINUTES,
    )


def _estimate_gmv_protected(incident: dict, failed_gmv: int) -> dict:
    """Estimate future GMV that would continue failing if a degraded route is not rerouted.

    Uses the incident's per-minute failure curve to extrapolate the observed
    failure rate forward for PROTECTED_WINDOW_MINUTES. If no per-minute data
    exists, falls back to a linear extrapolation from the current window.
    """
    window = incident.get("window", {})
    failure_by_minute = incident.get("failure_by_minute", [])

    if failure_by_minute:
        # Use the last few minutes' average failure GMV per minute
        recent = failure_by_minute[-3:] if len(failure_by_minute) >= 3 else failure_by_minute
        avg_failure_gmv_per_minute = sum(b["failed_gmv_inr"] for b in recent) / len(recent)
    else:
        # Fallback: spread failed_gmv evenly across the observed window
        from datetime import datetime
        start_iso = window.get("current_start", "")
        end_iso = window.get("current_end", "")
        if start_iso and end_iso:
            start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
            window_minutes = max(1, int((end - start).total_seconds() / 60))
        else:
            window_minutes = 15  # safe default
        avg_failure_gmv_per_minute = failed_gmv / window_minutes

    protected_gmv = round(avg_failure_gmv_per_minute * PROTECTED_WINDOW_MINUTES)
    # Cap: cannot exceed the incident's total attempted GMV in the window
    attempts = [e for e in incident.get("payment_events", []) if e["window"] == "current"]
    attempted_gmv = sum(e["amount_inr"] for e in attempts)
    protected_gmv = min(protected_gmv, attempted_gmv)
    protected_gmv = max(0, protected_gmv)

    return {
        "gmv_protected_inr": protected_gmv,
        "gmv_protected_basis": (
            f"MODELING ASSUMPTION: extrapolated observed failure rate forward "
            f"for {PROTECTED_WINDOW_MINUTES} minutes; represents prevented future "
            f"failures, not recovered past payments."
        ),
        "gmv_protected_window_minutes": PROTECTED_WINDOW_MINUTES,
    }


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
    retry_recovered_amount = round(recoverable_gmv * ASSUMED_RECOVERY_SUCCESS_RATE)
    above_cap = [event for event in failures if event["amount_inr"] > MAX_AUTO_RETRY_AMOUNT_INR]
    retry_exhausted = [event for event in failures if event["retry_count"] >= MAX_RETRIES_PER_PAYMENT]

    protection = _estimate_gmv_protected(incident, failed_gmv)

    return {
        "attempted_gmv_inr": attempted_gmv,
        "failed_gmv_inr": failed_gmv,
        "recoverable_gmv_inr": recoverable_gmv,
        "retry_recovered_amount_inr": retry_recovered_amount,
        "recovered_amount_inr": retry_recovered_amount,  # backward compat, used by webhook flow
        "recovered_amount_basis": (
            f"MODELING ASSUMPTION: {ASSUMED_RECOVERY_SUCCESS_RATE:.0%} success rate applied "
            "to recoverable_gmv_inr; not a measured statistic."
        ),
        "assumed_recovery_success_rate": ASSUMED_RECOVERY_SUCCESS_RATE,
        **protection,
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
