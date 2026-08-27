"""Counterfactual cost-of-delay analysis for payment incidents.

Given an incident and a hypothetical detection delay (positive = later,
negative = earlier), recompute failed_gmv and recoverable_gmv using the
incident's actual per-minute failure-rate curve.
"""

from __future__ import annotations

try:
    from .config import (
        ASSUMED_RECOVERY_SUCCESS_RATE,
        MAX_AUTO_RETRY_AMOUNT_INR,
        MAX_RETRIES_PER_PAYMENT,
    )
except ImportError:
    from config import (
        ASSUMED_RECOVERY_SUCCESS_RATE,
        MAX_AUTO_RETRY_AMOUNT_INR,
        MAX_RETRIES_PER_PAYMENT,
    )

MIN_DELAY_MINUTES = -20
MAX_DELAY_MINUTES = 20
DEFAULT_DETECTION_MINUTE_OFFSET = 4


def _build_minute_buckets(incident: dict) -> list[dict]:
    """Return per-minute failure buckets, preferring pre-computed data."""
    if "failure_by_minute" in incident and incident["failure_by_minute"]:
        return incident["failure_by_minute"]
    # Fallback: compute from raw payment events
    current_events = [e for e in incident["payment_events"] if e["window"] == "current"]
    if not current_events:
        return []
    start_iso = incident["window"]["current_start"]
    from datetime import datetime, timedelta, timezone
    start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    window_minutes = 15  # default
    end_iso = incident["window"].get("current_end")
    if end_iso:
        end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
        window_minutes = max(1, int((end - start).total_seconds() / 60))
    buckets = []
    for minute_offset in range(window_minutes):
        bucket_start = start + timedelta(minutes=minute_offset)
        bucket_end = start + timedelta(minutes=minute_offset + 1)
        bs = bucket_start.isoformat().replace("+00:00", "Z")
        be = bucket_end.isoformat().replace("+00:00", "Z")
        evts = [e for e in current_events if bs <= e["timestamp"] < be]
        failed = [e for e in evts if e["status"] == "failed"]
        buckets.append({
            "minute_offset": minute_offset,
            "timestamp": bs,
            "total_count": len(evts),
            "failed_count": len(failed),
            "failed_gmv_inr": sum(e["amount_inr"] for e in failed),
        })
    return buckets


def _recoverable_gmv_for_events(events: list[dict]) -> int:
    """Compute recoverable GMV from raw failed payment events."""
    return sum(
        e["amount_inr"]
        for e in events
        if e["status"] == "failed"
        and e.get("high_intent")
        and e["amount_inr"] <= MAX_AUTO_RETRY_AMOUNT_INR
        and e.get("retry_count", 0) < MAX_RETRIES_PER_PAYMENT
    )


def estimate_gmv_saved(incident: dict, delay_minutes: int) -> dict:
    """Estimate GMV impact under a hypothetical detection delay.

    delay_minutes = 0 reproduces the actual recorded values (no change).
    delay_minutes < 0 means earlier detection; some later failures prevented.
    delay_minutes > 0 means later detection; but can't exceed actual totals.

    Model: the actual window ran its full course. In the counterfactual, the
    detector fires (actual_detection + delay_minutes) minutes into the current
    window and instantly mitigates all failures from that minute onward.
    At delay=0, the counterfactual detection equals the window end (no
    prevention = actual outcome). Negative delays shift the cutoff earlier.
    Positive delays still can't exceed the window end, so they clamp to actual.
    """
    if not MIN_DELAY_MINUTES <= delay_minutes <= MAX_DELAY_MINUTES:
        raise ValueError(
            f"delay_minutes must be between {MIN_DELAY_MINUTES} and {MAX_DELAY_MINUTES}"
        )

    buckets = _build_minute_buckets(incident)
    window_minutes = len(buckets) if buckets else 15
    detection_offset = incident.get("window", {}).get(
        "detection_minute_offset", DEFAULT_DETECTION_MINUTE_OFFSET
    )

    # In the actual scenario, failures ran for the full window despite detection
    # at minute `detection_offset`. The counterfactual imagines perfect instant
    # mitigation at a shifted detection point.
    # Reference: delay=0 means "what actually happened" = full window of failures.
    # We anchor: counterfactual_cutoff = window_minutes + delay_minutes
    # At delay=0: cutoff = window_minutes => all buckets count => matches actual.
    # At delay=-5: cutoff = window_minutes - 5 => last 5 minutes prevented.
    # At delay=+5: cutoff = window_minutes + 5 => clamped to window_minutes => same as actual.
    counterfactual_cutoff = window_minutes + delay_minutes
    # Clamp between 0 and window_minutes
    counterfactual_cutoff = max(0, min(counterfactual_cutoff, window_minutes))

    # Compute actual values (sum of all current-window failures)
    actual_failed_gmv = sum(b["failed_gmv_inr"] for b in buckets)

    # Hypothetical: only failures in minutes 0..(cutoff-1) accumulate
    hypothetical_failed_gmv = 0
    for b in buckets:
        if b["minute_offset"] < counterfactual_cutoff:
            hypothetical_failed_gmv += b["failed_gmv_inr"]
    hypothetical_failed_gmv = min(hypothetical_failed_gmv, actual_failed_gmv)
    hypothetical_failed_gmv = max(0, hypothetical_failed_gmv)

    # Recoverable GMV scales proportionally with failed GMV
    current_events = [e for e in incident.get("payment_events", []) if e["window"] == "current"]
    actual_recoverable = _recoverable_gmv_for_events(current_events)

    if actual_failed_gmv > 0:
        ratio = hypothetical_failed_gmv / actual_failed_gmv
    else:
        ratio = 1.0 if delay_minutes >= 0 else 0.0
    hypothetical_recoverable = round(actual_recoverable * ratio)
    hypothetical_recovered = round(hypothetical_recoverable * ASSUMED_RECOVERY_SUCCESS_RATE)
    actual_recovered = round(actual_recoverable * ASSUMED_RECOVERY_SUCCESS_RATE)
    gmv_saved = actual_failed_gmv - hypothetical_failed_gmv

    # Build cumulative curve for the chart
    cumulative_curve = []
    running_total = 0
    for b in buckets:
        running_total += b["failed_gmv_inr"]
        cumulative_curve.append({
            "minute_offset": b["minute_offset"],
            "timestamp": b["timestamp"],
            "cumulative_failed_gmv_inr": running_total,
            "failed_gmv_inr": b["failed_gmv_inr"],
        })

    return {
        "incident_id": incident.get("incident_id", "unknown"),
        "delay_minutes": delay_minutes,
        "actual_detection_minute": detection_offset,
        "hypothetical_detection_minute": counterfactual_cutoff,
        "actual_failed_gmv_inr": actual_failed_gmv,
        "hypothetical_failed_gmv_inr": hypothetical_failed_gmv,
        "gmv_saved_inr": gmv_saved,
        "actual_recoverable_gmv_inr": actual_recoverable,
        "hypothetical_recoverable_gmv_inr": hypothetical_recoverable,
        "actual_recovered_amount_inr": actual_recovered,
        "hypothetical_recovered_amount_inr": hypothetical_recovered,
        "assumed_recovery_success_rate": ASSUMED_RECOVERY_SUCCESS_RATE,
        "cumulative_failure_curve": cumulative_curve,
        "disclaimer": "Modeled estimate based on failure-rate trend, not a guarantee.",
    }
