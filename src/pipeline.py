"""End-to-end alert -> RCA -> impact -> recovery incident pipeline."""

from __future__ import annotations

from collections import Counter

try:
    from .correlator import correlate
    from .detector import detect_degradations
    from .impact import calculate_impact
    from .rca import generate_rca
    from .recovery import recommend_recovery
except ImportError:  # Supports direct script imports from src/.
    from correlator import correlate
    from detector import detect_degradations
    from impact import calculate_impact
    from rca import generate_rca
    from recovery import recommend_recovery


def _top_failure_reason(incident: dict, detection: dict) -> tuple[str, int]:
    primary = detection.get("primary_degradation")
    if not primary:
        return "none", 0
    reasons = Counter(
        event["failure_reason"]
        for event in incident["payment_events"]
        if event["window"] == "current"
        and event["status"] == "failed"
        and event["sub_type"] == primary["sub_type"]
        and event["route"] == primary["route"]
    )
    return reasons.most_common(1)[0] if reasons else ("none", 0)


def _timeline_markers(
    incident: dict,
    detection: dict,
    correlation: dict,
    impact: dict,
    recovery: dict,
) -> list[dict]:
    primary = detection.get("primary_degradation")
    markers = []

    for event in incident["deploy_logs"]:
        if event.get("affected_method") or event.get("event_type") == "config_change":
            label = (
                f"Routing config {event['version']} changed"
                if event["event_type"] == "config_change"
                else f"Deploy {event['service']} {event['version']} at {event['rollout_pct']}%"
            )
            markers.append({"timestamp": event["timestamp"], "kind": event["event_type"], "label": label})
    for event in incident["alerts"]:
        markers.append(
            {
                "timestamp": event["timestamp"],
                "kind": "alert",
                "label": f"Alert: {event['metric']} ({event['threshold_breached']})",
            }
        )
    for event in incident["error_traces"]:
        markers.append(
            {
                "timestamp": event["timestamp"],
                "kind": "error",
                "label": f"Error spike: {event['error_code']} x{event['count']} on {event['affected_endpoint']}",
            }
        )
    if primary:
        markers.append(
            {
                "timestamp": primary["window_start"],
                "kind": "method",
                "label": f"Affected: {primary['method_display']} via {primary['route']}",
            }
        )

    top_reason, reason_count = _top_failure_reason(incident, detection)
    end_time = incident["window"]["current_end"]
    markers.extend(
        [
            {
                "timestamp": end_time,
                "kind": "impact",
                "label": f"Failed GMV: INR {impact['failed_gmv_inr']:,}",
            },
            {
                "timestamp": end_time,
                "kind": "error",
                "label": f"Top failure reason: {top_reason} ({reason_count} events)",
            },
            {
                "timestamp": end_time,
                "kind": "action",
                "label": f"Action: {recovery['primary_action']}",
            },
            {
                "timestamp": end_time,
                "kind": "recovery",
                "label": (
                    f"Modeled recovered: INR {impact['recovered_amount_inr']:,} "
                    f"({impact['assumed_recovery_success_rate']:.0%} assumption)"
                ),
            },
        ]
    )
    return sorted(markers, key=lambda marker: (marker["timestamp"], marker["kind"]))


def run_incident(incident: dict) -> dict:
    detection = detect_degradations(incident)
    correlation = correlate(incident, detection)
    rca_text = generate_rca(detection, correlation)
    impact = calculate_impact(incident)
    recovery = recommend_recovery(incident, correlation, impact)
    top_reason, top_reason_count = _top_failure_reason(incident, detection)

    return {
        "incident_id": incident["incident_id"],
        "window": incident["window"],
        "ground_truth": incident["ground_truth"],
        "detection": detection,
        "correlation": correlation,
        "rca_text": rca_text,
        "impact": impact,
        "recovery": recovery,
        "top_failure_reason": top_reason,
        "top_failure_reason_count": top_reason_count,
        "timeline": _timeline_markers(incident, detection, correlation, impact, recovery),
        "audit_trail": recovery["audit_trail"],
    }


def run_pipeline(incidents: list[dict]) -> list[dict]:
    return [run_incident(incident) for incident in incidents]
