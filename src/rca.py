"""Human-readable RCA generation backed only by computed evidence."""

from __future__ import annotations

from datetime import datetime


CAUSE_LABELS = {
    "bad_deploy": "merchant deploy regression",
    "bank_psp_downtime": "external PSP/bank degradation",
    "gateway_error": "gateway degradation",
    "config_change": "Razorpay routing config regression",
    "network_issue": "payment network degradation",
}


def _time_label(timestamp: str) -> str:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime("%H:%M UTC")


def generate_rca(detection: dict, correlation: dict) -> str:
    primary = detection.get("primary_degradation")
    if not primary:
        return "No pair-level degradation crossed the detection threshold. Signals inconclusive - escalating for manual review."

    evidence = correlation["evidence"]
    route_prefix = {
        "psp": "PSP",
        "bank": "bank",
        "gateway": "gateway",
    }.get(primary["route_type"], "route")
    deploy_note = (
        f"Merchant deploy overlap: {evidence['deploy_overlap_detail']}"
        if evidence["deploy_overlap"]
        else "No merchant deploy overlap"
    )
    config_note = (
        f"Razorpay routing config changed: {evidence['config_change_detail']}"
        if evidence["config_change_overlap"]
        else "Razorpay routing config unchanged"
    )
    prefix = (
        f"At {_time_label(primary['window_start'])}, {primary['method_display']} failures rose "
        f"{primary['failure_rate_increase_pct']:.1f}%. "
        f"{primary['failure_concentration_pct']:.1f}% came from {route_prefix} {primary['route']}. "
        f"{deploy_note}. {config_note}."
    )
    if correlation["predicted_cause"] == "unresolved":
        return f"{prefix} Signals inconclusive - escalating for manual review."
    return (
        f"{prefix} Likely {CAUSE_LABELS[correlation['predicted_cause']]} "
        f"(confidence {correlation['confidence']:.2f})."
    )
