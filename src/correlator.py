"""Deterministic, evidence-reporting root-cause correlation engine."""

from __future__ import annotations

from datetime import datetime, timedelta

try:
    from .config import MIN_CONFIDENCE_FOR_AUTO_ACTION
except ImportError:  # Supports direct imports from src/.
    from config import MIN_CONFIDENCE_FOR_AUTO_ACTION

MIN_CONFIDENCE_TO_RESOLVE = MIN_CONFIDENCE_FOR_AUTO_ACTION

ERROR_SIGNATURES = {
    "MERCHANT_5XX": ("bad_deploy", "merchant-side"),
    "PSP_UNAVAILABLE": ("bank_psp_downtime", "PSP-side"),
    "BANK_TIMEOUT": ("bank_psp_downtime", "bank-side"),
    "GATEWAY_502": ("gateway_error", "gateway-side"),
    "ROUTE_NOT_FOUND": ("config_change", "routing-config"),
    "NET_CONN_RESET": ("network_issue", "network-side"),
}

VALID_CAUSES = {
    "bad_deploy",
    "bank_psp_downtime",
    "gateway_error",
    "config_change",
    "network_issue",
    "unresolved",
}


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _in_overlap(timestamp: str, start: datetime, end: datetime) -> bool:
    value = _parse_time(timestamp)
    return start - timedelta(minutes=5) <= value <= end


def correlate(incident: dict, detection: dict) -> dict:
    primary = detection.get("primary_degradation")
    if not primary:
        return {
            "incident_id": incident["incident_id"],
            "predicted_cause": "unresolved",
            "confidence": 0.0,
            "supporting_signal_count": 0,
            "evidence": {"reason": "No degradation crossed the detector threshold."},
        }

    start = _parse_time(primary["window_start"])
    end = _parse_time(primary["window_end"])
    method = primary["sub_type"]
    route = primary["route"]

    overlapping_deploys = [
        event
        for event in incident["deploy_logs"]
        if event.get("event_type") == "deploy"
        and event.get("affected_method") == method
        and event.get("affected_route") == route
        and _in_overlap(event["timestamp"], start, end)
    ]
    config_changes = [
        event
        for event in incident["deploy_logs"]
        if event.get("event_type") == "config_change"
        and event.get("affected_method") == method
        and event.get("affected_route") == route
        and _in_overlap(event["timestamp"], start, end)
    ]
    relevant_traces = [
        event
        for event in incident["error_traces"]
        if event.get("affected_method") == method
        and event.get("affected_route") == route
        and _in_overlap(event["timestamp"], start, end)
    ]
    dominant_trace = max(relevant_traces, key=lambda item: item["count"], default=None)
    signature = ERROR_SIGNATURES.get(dominant_trace["error_code"]) if dominant_trace else None
    concentration = primary["failure_concentration_pct"]
    route_health_failed = any(
        event["metric"] == f"route_health:{route}"
        and event["threshold_breached"] == "healthcheck_failed"
        for event in incident["alerts"]
        if _in_overlap(event["timestamp"], start, end)
    )
    route_health_confirmed = any(
        event["metric"] == f"route_health:{route}"
        and event["threshold_breached"] == "healthcheck_passed"
        for event in incident["alerts"]
        if _in_overlap(event["timestamp"], start, end)
    )
    network_alert = any(
        event["metric"].startswith("network_packet_loss")
        for event in incident["alerts"]
        if _in_overlap(event["timestamp"], start, end)
    )
    webhook_failed = any(
        event["delivery_status"] in {"failed", "timed_out"}
        for event in incident["webhook_events"]
        if _in_overlap(event["timestamp"], start, end)
    )

    scores = {cause: 0.0 for cause in VALID_CAUSES if cause != "unresolved"}
    signals = {cause: [] for cause in scores}

    def add(cause: str, weight: float, signal: str) -> None:
        scores[cause] += weight
        signals[cause].append(signal)

    if overlapping_deploys:
        add("bad_deploy", 0.40, "matching merchant deploy overlap")
    if config_changes:
        add("config_change", 0.45, "matching Razorpay routing config change")
    if signature:
        add(signature[0], 0.35, f"{signature[1]} error signature {dominant_trace['error_code']}")
    if route_health_failed:
        add("bank_psp_downtime", 0.30, "route health check failed")
    if network_alert:
        add("network_issue", 0.35, "independent packet-loss alert")
    if webhook_failed:
        if dominant_trace and dominant_trace["error_code"] == "GATEWAY_502":
            add("gateway_error", 0.25, "webhook delivery failures agree with gateway trace")
        elif dominant_trace and dominant_trace["error_code"] == "NET_CONN_RESET":
            add("network_issue", 0.15, "webhook timeouts agree with network trace")
    if concentration >= 50:
        for cause in scores:
            if signals[cause]:
                add(cause, 0.20, f"{concentration:.1f}% failure concentration")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_cause, best_score = ranked[0]
    runner_up_score = ranked[1][1]
    confidence = round(min(0.99, best_score), 2)
    if confidence < MIN_CONFIDENCE_TO_RESOLVE or best_score - runner_up_score < 0.15:
        predicted_cause = "unresolved"
    else:
        predicted_cause = best_cause

    deploy = overlapping_deploys[0] if overlapping_deploys else None
    config = config_changes[0] if config_changes else None
    evidence = {
        "deploy_overlap": bool(deploy),
        "deploy_overlap_detail": (
            f"{deploy['service']} {deploy['version']} at {deploy['timestamp']} ({deploy['rollout_pct']}% rollout)"
            if deploy
            else "No matching merchant deploy in the degradation window."
        ),
        "config_change_overlap": bool(config),
        "config_change_detail": (
            f"{config['service']} {config['version']} at {config['timestamp']}"
            if config
            else "Razorpay routing config unchanged in the degradation window."
        ),
        "dominant_error_code": dominant_trace["error_code"] if dominant_trace else None,
        "dominant_error_count": dominant_trace["count"] if dominant_trace else 0,
        "error_signature_side": signature[1] if signature else "unknown",
        "concentration_pct": concentration,
        "concentration_method": primary["method_display"],
        "concentration_route": route,
        "concentration_route_type": primary["route_type"],
        "route_health_confirmed": route_health_confirmed,
        "route_health_failed": route_health_failed,
        "network_alert": network_alert,
        "webhook_delivery_degraded": webhook_failed,
        "score_by_cause": {key: round(value, 2) for key, value in scores.items()},
        "confidence_rule": (
            f"highest independent-signal score {best_score:.2f}; "
            f"resolve only at >= {MIN_CONFIDENCE_TO_RESOLVE:.2f} with >= 0.15 lead"
        ),
        "supporting_signals": signals[best_cause],
    }
    return {
        "incident_id": incident["incident_id"],
        "predicted_cause": predicted_cause,
        "confidence": confidence,
        "supporting_signal_count": len(signals[best_cause]),
        "evidence": evidence,
    }
