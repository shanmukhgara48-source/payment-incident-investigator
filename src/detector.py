"""Pair-level payment degradation detection against each pair's own baseline."""

from __future__ import annotations

from collections import defaultdict


MIN_SAMPLE_SIZE = 20
MIN_SUCCESS_RATE_DROP = 0.05


def _pair_key(event: dict) -> tuple[str, str]:
    return event["sub_type"], event["route"]


def _summarize(events: list[dict]) -> dict[tuple[str, str], dict]:
    summaries = defaultdict(lambda: {"attempts": 0, "successes": 0, "failures": 0, "sample": None})
    for event in events:
        bucket = summaries[_pair_key(event)]
        bucket["attempts"] += 1
        bucket["successes"] += event["status"] == "success"
        bucket["failures"] += event["status"] == "failed"
        bucket["sample"] = event
    return summaries


def detect_degradations(incident: dict) -> dict:
    """Return every degraded method/route pair and the largest primary drop."""
    baseline_events = [event for event in incident["payment_events"] if event["window"] == "baseline"]
    current_events = [event for event in incident["payment_events"] if event["window"] == "current"]
    baseline = _summarize(baseline_events)
    current = _summarize(current_events)
    total_current_failures = sum(event["status"] == "failed" for event in current_events)
    degradations = []

    for key, current_stats in current.items():
        baseline_stats = baseline.get(key)
        if not baseline_stats:
            continue
        baseline_rate = baseline_stats["successes"] / baseline_stats["attempts"]
        current_rate = current_stats["successes"] / current_stats["attempts"]
        success_rate_drop = baseline_rate - current_rate
        if (
            baseline_stats["attempts"] < MIN_SAMPLE_SIZE
            or current_stats["attempts"] < MIN_SAMPLE_SIZE
            or success_rate_drop < MIN_SUCCESS_RATE_DROP
        ):
            continue

        sample = current_stats["sample"]
        degradation = {
            "method": sample["method"],
            "sub_type": sample["sub_type"],
            "method_display": sample["method_display"],
            "route_type": sample["route_type"],
            "route": sample["route"],
            "baseline_attempts": baseline_stats["attempts"],
            "current_attempts": current_stats["attempts"],
            "baseline_success_rate": round(baseline_rate, 4),
            "current_success_rate": round(current_rate, 4),
            "success_rate_drop": round(success_rate_drop, 4),
            "failure_rate_increase_pct": round(success_rate_drop * 100, 1),
            "current_failure_count": current_stats["failures"],
            "failure_concentration_pct": round(
                100 * current_stats["failures"] / max(1, total_current_failures), 1
            ),
            "window_start": incident["window"]["current_start"],
            "window_end": incident["window"]["current_end"],
        }
        degradations.append(degradation)

    degradations.sort(key=lambda item: item["success_rate_drop"], reverse=True)
    return {
        "incident_id": incident["incident_id"],
        "detected": bool(degradations),
        "thresholds": {
            "min_sample_size": MIN_SAMPLE_SIZE,
            "min_success_rate_drop": MIN_SUCCESS_RATE_DROP,
        },
        "primary_degradation": degradations[0] if degradations else None,
        "degradations": degradations,
    }
