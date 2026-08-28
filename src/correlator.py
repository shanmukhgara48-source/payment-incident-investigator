"""Evidence-reporting root-cause correlation engine.

Two diagnosis paths:
- **LLM_REASONED**: When OPENAI_API_KEY is set, the full evidence is sent to
  an OpenAI model for structured root-cause reasoning.
- **RULE_BASED** / **RULE_BASED_FALLBACK**: Deterministic weighted scoring.
  Used when no key is configured, or as automatic fallback when the LLM call
  fails.

The rule-based scores are always computed and included in the evidence dict
regardless of which path produces the final diagnosis.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

try:
    from .config import MIN_CONFIDENCE_FOR_AUTO_ACTION
    from .float_compare import gte, lt
    from .llm import llm_available, llm_call
except ImportError:  # Supports direct imports from src/.
    from config import MIN_CONFIDENCE_FOR_AUTO_ACTION
    from float_compare import gte, lt
    from llm import llm_available, llm_call

logger = logging.getLogger(__name__)

MIN_CONFIDENCE_TO_RESOLVE = MIN_CONFIDENCE_FOR_AUTO_ACTION

# A resolved diagnosis must also lead the runner-up by this margin.
MIN_MARGIN_OVER_RUNNER_UP = 0.15

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

_CORRELATOR_SYSTEM_PROMPT = """\
You are a payment incident root-cause analyst.  Given structured evidence from
a payment degradation incident, determine the most likely root cause.

Valid root causes (pick exactly one):
- bad_deploy: A merchant-side code deployment caused the failures.
- bank_psp_downtime: The PSP or bank endpoint is degraded or down.
- gateway_error: The payment gateway itself is experiencing errors.
- config_change: A routing configuration change caused misrouting.
- network_issue: Network-level problems (packet loss, connection resets).
- unresolved: Evidence is insufficient or contradictory — escalate.

Respond with valid JSON:
{
  "predicted_cause": "<one of the six values above>",
  "confidence": <float between 0.0 and 0.99>,
  "explanation": "<1-2 sentence explanation of your reasoning>",
  "supporting_signals": ["<signal 1>", "<signal 2>"]
}

Rules:
- If confidence < 0.60 or evidence genuinely points to multiple causes
  equally, set predicted_cause to "unresolved".
- Do not invent evidence.  Base your reasoning only on the signals provided.
- Confidence 0.99 means near-certainty; reserve it for overwhelming evidence.
"""


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _in_overlap(timestamp: str, start: datetime, end: datetime) -> bool:
    value = _parse_time(timestamp)
    return start - timedelta(minutes=5) <= value <= end


def _format_evidence_prompt(
    incident: dict,
    primary: dict,
    overlapping_deploys: list,
    config_changes: list,
    dominant_trace: dict | None,
    signature: tuple | None,
    concentration: float,
    route_health_failed: bool,
    route_health_confirmed: bool,
    network_alert: bool,
    webhook_failed: bool,
    top_failure_reason: str,
    top_failure_count: int,
) -> str:
    """Format all evidence into a structured prompt for the LLM."""
    method = primary["sub_type"]
    route = primary["route"]
    rate_increase = primary.get("rate_increase_pct", "unknown")

    lines = [
        f"Incident: {incident['incident_id']}",
        f"Payment method: {method} via {route}",
        f"Failure rate increase: {rate_increase}%",
        f"Failure concentration on this route: {concentration:.1f}%",
        "",
        "=== Deploy events in window ===",
    ]
    if overlapping_deploys:
        for d in overlapping_deploys:
            lines.append(
                f"- {d['service']} {d['version']} at {d['timestamp']} "
                f"({d['rollout_pct']}% rollout)"
            )
    else:
        lines.append("None")

    lines.append("")
    lines.append("=== Config changes in window ===")
    if config_changes:
        for c in config_changes:
            lines.append(f"- {c['service']} {c['version']} at {c['timestamp']}")
    else:
        lines.append("None")

    lines.append("")
    lines.append("=== Error traces ===")
    if dominant_trace:
        lines.append(
            f"- Dominant error: {dominant_trace['error_code']} "
            f"x{dominant_trace['count']} on {dominant_trace.get('affected_endpoint', 'unknown')}"
        )
        if signature:
            lines.append(
                f"- Error signature interpretation: {signature[1]} "
                f"(typically associated with {signature[0]})"
            )
        else:
            lines.append("- No known error signature match")
    else:
        lines.append("No error traces in window")

    lines.append("")
    lines.append("=== Health signals ===")
    if route_health_failed:
        lines.append(f"- Route health check for {route}: FAILED")
    elif route_health_confirmed:
        lines.append(f"- Route health check for {route}: PASSED")
    else:
        lines.append(f"- Route health check for {route}: no signal")
    lines.append(f"- Network packet-loss alert: {'YES' if network_alert else 'NO'}")
    lines.append(
        f"- Webhook delivery degraded: {'YES' if webhook_failed else 'NO'}"
    )

    lines.append("")
    lines.append("=== Payment failure context ===")
    lines.append(f"- Top failure reason: {top_failure_reason} ({top_failure_count} events)")

    return "\n".join(lines)


def _try_llm_correlate(
    incident: dict,
    primary: dict,
    overlapping_deploys: list,
    config_changes: list,
    dominant_trace: dict | None,
    signature: tuple | None,
    concentration: float,
    route_health_failed: bool,
    route_health_confirmed: bool,
    network_alert: bool,
    webhook_failed: bool,
    top_failure_reason: str,
    top_failure_count: int,
) -> dict | None:
    """Attempt an LLM-based diagnosis.  Returns validated result or None."""
    if not llm_available():
        return None

    user_prompt = _format_evidence_prompt(
        incident,
        primary,
        overlapping_deploys,
        config_changes,
        dominant_trace,
        signature,
        concentration,
        route_health_failed,
        route_health_confirmed,
        network_alert,
        webhook_failed,
        top_failure_reason,
        top_failure_count,
    )

    result = llm_call(_CORRELATOR_SYSTEM_PROMPT, user_prompt)
    if result is None:
        return None

    # Validate the LLM response
    cause = result.get("predicted_cause")
    if cause not in VALID_CAUSES:
        logger.warning(
            "LLM returned invalid cause %r for %s; falling back",
            cause,
            incident["incident_id"],
        )
        return None

    confidence = result.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        logger.warning("LLM returned non-numeric confidence; falling back")
        return None

    confidence = round(min(0.99, max(0.0, float(confidence))), 2)

    # Enforce the same resolve gate the rule-based path uses
    if lt(confidence, MIN_CONFIDENCE_TO_RESOLVE) and cause != "unresolved":
        cause = "unresolved"

    result["predicted_cause"] = cause
    result["confidence"] = confidence
    return result


def correlate(incident: dict, detection: dict) -> dict:
    primary = detection.get("primary_degradation")
    if not primary:
        return {
            "incident_id": incident["incident_id"],
            "predicted_cause": "unresolved",
            "confidence": 0.0,
            "supporting_signal_count": 0,
            "evidence": {"reason": "No degradation crossed the detector threshold."},
            "reasoning_mode": "RULE_BASED",
        }

    start = _parse_time(primary["window_start"])
    end = _parse_time(primary["window_end"])
    method = primary["sub_type"]
    route = primary["route"]

    # ── evidence extraction (unchanged) ──────────────────────────────
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

    # ── rule-based scoring (always computed, included in evidence) ────
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
    if gte(concentration, 50):
        for cause in scores:
            if signals[cause]:
                add(cause, 0.20, f"{concentration:.1f}% failure concentration")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_cause, best_score = ranked[0]
    runner_up_score = ranked[1][1]
    rule_confidence = round(min(0.99, best_score), 2)
    if lt(rule_confidence, MIN_CONFIDENCE_TO_RESOLVE) or lt(
        best_score - runner_up_score, MIN_MARGIN_OVER_RUNNER_UP
    ):
        rule_predicted = "unresolved"
    else:
        rule_predicted = best_cause

    # ── top failure reason (for LLM prompt context) ──────────────────
    from collections import Counter

    failure_reasons = Counter(
        e["failure_reason"]
        for e in incident["payment_events"]
        if e["window"] == "current"
        and e["status"] == "failed"
        and e["sub_type"] == method
        and e["route"] == route
    )
    if failure_reasons:
        top_failure_reason, top_failure_count = failure_reasons.most_common(1)[0]
    else:
        top_failure_reason, top_failure_count = "none", 0

    # ── LLM diagnosis attempt ────────────────────────────────────────
    reasoning_mode = "RULE_BASED"
    llm_meta = None
    llm_explanation = None
    predicted_cause = rule_predicted
    confidence = rule_confidence

    llm_result = _try_llm_correlate(
        incident,
        primary,
        overlapping_deploys,
        config_changes,
        dominant_trace,
        signature,
        concentration,
        route_health_failed,
        route_health_confirmed,
        network_alert,
        webhook_failed,
        top_failure_reason,
        top_failure_count,
    )

    if llm_result is not None:
        predicted_cause = llm_result["predicted_cause"]
        confidence = llm_result["confidence"]
        llm_explanation = llm_result.get("explanation", "")
        llm_meta = llm_result.get("_llm_meta")
        reasoning_mode = "LLM_REASONED"
    elif llm_available():
        reasoning_mode = "RULE_BASED_FALLBACK"

    # ── build result ─────────────────────────────────────────────────
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
            f"resolve only at >= {MIN_CONFIDENCE_TO_RESOLVE:.2f} "
            f"with >= {MIN_MARGIN_OVER_RUNNER_UP:.2f} lead"
        ),
        "supporting_signals": (
            llm_result.get("supporting_signals", signals[best_cause])
            if llm_result is not None
            else signals[best_cause]
        ),
    }
    if llm_explanation:
        evidence["llm_explanation"] = llm_explanation

    result = {
        "incident_id": incident["incident_id"],
        "predicted_cause": predicted_cause,
        "confidence": confidence,
        "supporting_signal_count": (
            len(evidence["supporting_signals"])
            if evidence["supporting_signals"]
            else len(signals[best_cause])
        ),
        "evidence": evidence,
        "reasoning_mode": reasoning_mode,
    }
    if llm_meta:
        result["llm_meta"] = llm_meta
    return result
