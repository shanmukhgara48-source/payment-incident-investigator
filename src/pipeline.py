"""End-to-end alert -> RCA -> impact -> recovery incident pipeline."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable

try:
    from .config import (
        ASSUMED_RECOVERY_SUCCESS_RATE,
        MAX_AUTO_RETRY_AMOUNT_INR,
        MAX_RETRIES_PER_PAYMENT,
        MERCHANT_NOTIFICATION_EXPOSURE_THRESHOLD_INR,
        MIN_CONFIDENCE_FOR_AUTO_ACTION,
        ROUTE_HEALTH_CONFIRMATION_REQUIRED,
    )
    from .correlator import correlate
    from .detector import detect_degradations
    from .impact import calculate_impact
    from .memory import IncidentMemory, empty_recall, extract_features
    from .rca import generate_rca
    from .recovery import recommend_recovery
    from .skeptic import skeptic_review as run_skeptic_review
except ImportError:  # Supports direct script imports from src/.
    from config import (
        ASSUMED_RECOVERY_SUCCESS_RATE,
        MAX_AUTO_RETRY_AMOUNT_INR,
        MAX_RETRIES_PER_PAYMENT,
        MERCHANT_NOTIFICATION_EXPOSURE_THRESHOLD_INR,
        MIN_CONFIDENCE_FOR_AUTO_ACTION,
        ROUTE_HEALTH_CONFIRMATION_REQUIRED,
    )
    from correlator import correlate
    from detector import detect_degradations
    from impact import calculate_impact
    from memory import IncidentMemory, empty_recall, extract_features
    from rca import generate_rca
    from recovery import recommend_recovery
    from skeptic import skeptic_review as run_skeptic_review


logger = logging.getLogger(__name__)


def _safe_stage(
    stage: str,
    incident_id: str,
    operation: Callable[[], dict | str],
    fallback: Callable[[Exception], dict | str],
    stage_errors: list[dict],
) -> dict | str:
    try:
        result = operation()
        logger.debug(
            "pipeline stage completed",
            extra={"incident_id": incident_id, "stage": stage},
        )
        return result
    except Exception as exc:  # A single malformed incident must not stop the batch.
        logger.exception(
            "pipeline stage failed; using safe fallback",
            extra={"incident_id": incident_id, "stage": stage},
        )
        stage_errors.append({"stage": stage, "error": str(exc)})
        return fallback(exc)


def _detection_fallback(incident_id: str, exc: Exception) -> dict:
    return {
        "incident_id": incident_id,
        "detected": False,
        "thresholds": {},
        "primary_degradation": None,
        "degradations": [],
        "stage_error": str(exc),
    }


def _correlation_fallback(incident_id: str, exc: Exception) -> dict:
    return {
        "incident_id": incident_id,
        "predicted_cause": "unresolved",
        "confidence": 0.0,
        "supporting_signal_count": 0,
        "evidence": {
            "reason": f"Correlation unavailable: {exc}",
            "route_health_confirmed": False,
        },
        "stage_error": str(exc),
    }


def _impact_fallback(exc: Exception) -> dict:
    return {
        "attempted_gmv_inr": 0,
        "failed_gmv_inr": 0,
        "recoverable_gmv_inr": 0,
        "recovered_amount_inr": 0,
        "recovered_amount_basis": (
            f"MODELING ASSUMPTION: {ASSUMED_RECOVERY_SUCCESS_RATE:.0%} success rate; "
            "impact calculation unavailable."
        ),
        "assumed_recovery_success_rate": ASSUMED_RECOVERY_SUCCESS_RATE,
        "attempt_count": 0,
        "failed_count": 0,
        "recoverable_payment_count": 0,
        "above_auto_retry_cap_count": 0,
        "retry_limit_reached_count": 0,
        "eligibility_policy": {
            "high_intent_required": True,
            "max_auto_retry_amount_inr": MAX_AUTO_RETRY_AMOUNT_INR,
            "max_retries_per_payment": MAX_RETRIES_PER_PAYMENT,
        },
        "stage_error": str(exc),
    }


def _skeptic_fallback(correlation: dict, exc: Exception) -> dict:
    return {
        "outcome": "confirmed",
        "summary": f"Skeptic review unavailable: {exc}",
        "primary_confidence": correlation.get("confidence", 0.0),
        "total_penalty": 0.0,
        "final_confidence": correlation.get("confidence", 0.0),
        "checks_performed": 0,
        "challenges_raised": 0,
        "checks": [],
        "challenges": [],
        "stage_error": str(exc),
    }


def _recovery_fallback(incident: dict, exc: Exception) -> dict:
    incident_id = incident.get("incident_id", "unknown")
    timestamp = incident.get("window", {}).get("current_end", "unknown")
    audit = {
        "incident_id": incident_id,
        "action": "escalate to human",
        "reason": f"Recovery stage unavailable: {exc}",
        "bounded_by": "PIPELINE_STAGE_FAILURE",
        "timestamp": timestamp,
    }
    return {
        "primary_action": "escalate to human",
        "auto_action_taken": False,
        "merchant_notification_sent": False,
        "modeled_recovered_amount_inr": 0,
        "modeled_recovered_amount_basis": "No recovery modeled because the recovery stage failed.",
        "recovery_mode": "SIMULATED",
        "live_api_requested": False,
        "test_payment_links": [],
        "live_api_fallback_reason": "Recovery stage failed before an API action could be attempted.",
        "audit_trail": [audit],
        "policy": {
            "MIN_CONFIDENCE_FOR_AUTO_ACTION": MIN_CONFIDENCE_FOR_AUTO_ACTION,
            "MAX_AUTO_RETRY_AMOUNT_INR": MAX_AUTO_RETRY_AMOUNT_INR,
            "MAX_RETRIES_PER_PAYMENT": MAX_RETRIES_PER_PAYMENT,
            "ROUTE_HEALTH_CONFIRMATION_REQUIRED": ROUTE_HEALTH_CONFIRMATION_REQUIRED,
            "MERCHANT_NOTIFICATION_EXPOSURE_THRESHOLD_INR": MERCHANT_NOTIFICATION_EXPOSURE_THRESHOLD_INR,
            "ASSUMED_RECOVERY_SUCCESS_RATE": ASSUMED_RECOVERY_SUCCESS_RATE,
            "TEST_MODE_ONLY": True,
        },
        "stage_error": str(exc),
    }


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
                "label": f"Detection: {event['metric']} ({event['threshold_breached']})",
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
                "kind": "rca",
                "label": (
                    f"RCA: {correlation['predicted_cause']} "
                    f"(confidence {correlation['confidence']:.2f})"
                ),
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
                    (
                        f"Actual TEST-MODE recovery: INR {impact['recovered_amount_inr']:,}"
                        if impact.get("recovery_measurement_type") == "ACTUAL TEST-MODE"
                        else f"Modeled recovered: INR {impact['recovered_amount_inr']:,} "
                        f"({impact['assumed_recovery_success_rate']:.0%} assumption)"
                    )
                ),
            },
        ]
    )
    kind_order = {
        "deploy": 0,
        "config_change": 0,
        "method": 1,
        "alert": 2,
        "error": 3,
        "impact": 4,
        "rca": 5,
        "action": 6,
        "recovery": 7,
    }
    return sorted(
        markers,
        key=lambda marker: (marker["timestamp"], kind_order.get(marker["kind"], 99)),
    )


def run_incident(incident: dict, memory: IncidentMemory | None = None) -> dict:
    """Process one incident. `memory` holds only incidents processed before it."""
    incident_id = incident.get("incident_id", "unknown")
    stage_errors: list[dict] = []
    detection = _safe_stage(
        "detector",
        incident_id,
        lambda: detect_degradations(incident),
        lambda exc: _detection_fallback(incident_id, exc),
        stage_errors,
    )
    correlation = _safe_stage(
        "correlator",
        incident_id,
        lambda: correlate(incident, detection),
        lambda exc: _correlation_fallback(incident_id, exc),
        stage_errors,
    )
    skeptic = _safe_stage(
        "skeptic",
        incident_id,
        lambda: run_skeptic_review(incident, detection, correlation),
        lambda exc: _skeptic_fallback(correlation, exc),
        stage_errors,
    )
    # Build an adjusted correlation with final_confidence for downstream stages.
    # The original correlation is preserved as primary_diagnosis in the record.
    adjusted_correlation = {**correlation, "confidence": skeptic["final_confidence"]}
    # If skeptic lowered confidence below the resolve threshold, mark unresolved
    if (
        adjusted_correlation["confidence"] < MIN_CONFIDENCE_FOR_AUTO_ACTION
        and correlation["predicted_cause"] != "unresolved"
        and skeptic["outcome"] == "challenged"
    ):
        adjusted_correlation["predicted_cause"] = "unresolved"

    try:
        top_reason, top_reason_count = _top_failure_reason(incident, detection)
    except Exception as exc:
        logger.exception(
            "failed to summarize failure reason",
            extra={"incident_id": incident_id, "stage": "record_assembly"},
        )
        stage_errors.append({"stage": "record_assembly", "error": str(exc)})
        top_reason, top_reason_count = "unknown", 0

    # Pattern recall happens before the diagnosis is finalized, but the store only
    # ever contains incidents already processed in this batch - never future ones.
    recall_features: dict = {}
    pattern_recall = empty_recall()
    if memory is not None:
        recall_features = _safe_stage(
            "memory_features",
            incident_id,
            lambda: extract_features(detection, adjusted_correlation, top_reason),
            lambda exc: {},
            stage_errors,
        )
        if recall_features:
            pattern_recall = _safe_stage(
                "memory_recall",
                incident_id,
                lambda: memory.recall(recall_features, adjusted_correlation["predicted_cause"]),
                lambda exc: empty_recall(recall_features),
                stage_errors,
            )

    rca_text = _safe_stage(
        "rca",
        incident_id,
        lambda: generate_rca(detection, adjusted_correlation),
        lambda exc: "RCA unavailable. Signals inconclusive - escalating for manual review.",
        stage_errors,
    )
    impact = _safe_stage(
        "impact",
        incident_id,
        lambda: calculate_impact(incident),
        _impact_fallback,
        stage_errors,
    )
    recovery_correlation = adjusted_correlation
    if any(error["stage"] == "impact" for error in stage_errors):
        recovery_correlation = _correlation_fallback(
            incident_id, RuntimeError("impact unavailable; autonomous recovery disabled")
        )
    recovery = _safe_stage(
        "recovery",
        incident_id,
        lambda: recommend_recovery(incident, recovery_correlation, impact),
        lambda exc: _recovery_fallback(incident, exc),
        stage_errors,
    )
    if memory is not None and recall_features:
        # Only after this incident is fully resolved does it become visible to
        # the incidents that follow it.
        _safe_stage(
            "memory_remember",
            incident_id,
            lambda: memory.remember(
                incident_id, recall_features, adjusted_correlation, recovery, impact
            )
            or {},
            lambda exc: {},
            stage_errors,
        )

    try:
        timeline = _timeline_markers(incident, detection, adjusted_correlation, impact, recovery)
    except Exception as exc:
        logger.exception(
            "failed to assemble timeline",
            extra={"incident_id": incident_id, "stage": "timeline"},
        )
        stage_errors.append({"stage": "timeline", "error": str(exc)})
        timeline = []

    return {
        "incident_id": incident_id,
        "window": incident.get("window", {}),
        "ground_truth": incident.get(
            "ground_truth", {"cause": "unresolved", "is_ambiguous": True}
        ),
        "detection": detection,
        "correlation": adjusted_correlation,
        "primary_diagnosis": correlation,
        "skeptic_review": skeptic,
        "pattern_recall": pattern_recall,
        "rca_text": rca_text,
        "impact": impact,
        "recovery": recovery,
        "top_failure_reason": top_reason,
        "top_failure_reason_count": top_reason_count,
        "timeline": timeline,
        "audit_trail": recovery["audit_trail"],
        "stage_errors": stage_errors,
    }


def run_pipeline(incidents: list[dict]) -> list[dict]:
    logger.info(
        "batch pipeline started",
        extra={"incident_id": "batch", "stage": "pipeline"},
    )
    # One store per batch, filled in processing order: incident N can only ever
    # recall incidents 0..N-1.
    memory = IncidentMemory()
    records = [run_incident(incident, memory) for incident in incidents]
    logger.info(
        "batch pipeline completed incidents=%s incidents_with_errors=%s "
        "incidents_with_similar_past=%s",
        len(records),
        sum(bool(record["stage_errors"]) for record in records),
        sum(bool(record["pattern_recall"]["match_count"]) for record in records),
        extra={"incident_id": "batch", "stage": "pipeline"},
    )
    return records
