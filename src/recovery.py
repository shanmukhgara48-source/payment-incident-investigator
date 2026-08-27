"""Confidence-gated, bounded revenue recovery decision engine."""

from __future__ import annotations


MIN_CONFIDENCE_FOR_AUTO_ACTION = 0.60
MAX_AUTO_RETRY_AMOUNT_INR = 50_000
MAX_RETRIES_PER_PAYMENT = 2
MERCHANT_NOTIFICATION_EXPOSURE_THRESHOLD_INR = 100_000
ASSUMED_RECOVERY_SUCCESS_RATE = 0.35

ROUTE_LEVEL_CAUSES = {"bank_psp_downtime", "gateway_error", "network_issue"}
PRIMARY_ACTIONS = {
    "reroute traffic",
    "retry later",
    "create Payment Links for high-intent failures",
    "notify merchant",
    "escalate to human",
}


def recommend_recovery(incident: dict, correlation: dict, impact: dict) -> dict:
    timestamp = incident["window"]["current_end"]
    incident_id = incident["incident_id"]
    cause = correlation["predicted_cause"]
    confidence = correlation["confidence"]
    audit_trail = []

    def log(action: str, reason: str, bounded_by: str) -> None:
        audit_trail.append(
            {
                "incident_id": incident_id,
                "action": action,
                "reason": reason,
                "bounded_by": bounded_by,
                "timestamp": timestamp,
            }
        )

    if cause == "unresolved" or confidence < MIN_CONFIDENCE_FOR_AUTO_ACTION:
        primary_action = "escalate to human"
        log(
            primary_action,
            f"Cause is {cause} at confidence {confidence:.2f}; no autonomous recovery action taken.",
            "MIN_CONFIDENCE_FOR_AUTO_ACTION",
        )
        estimated_recovery_activated = False
    else:
        route_health_confirmed = correlation["evidence"]["route_health_confirmed"]
        if cause in ROUTE_LEVEL_CAUSES and not route_health_confirmed:
            primary_action = "reroute traffic"
            log(
                primary_action,
                f"{cause} is route-level and the affected route health check has not passed.",
                "ROUTE_HEALTH_CONFIRMATION_REQUIRED",
            )
            log(
                "retry blocked",
                "Retrying into a still-degraded route would consume the bounded retry budget.",
                "MAX_RETRIES_PER_PAYMENT",
            )
            estimated_recovery_activated = False
        elif impact["recoverable_payment_count"] > 0:
            primary_action = "create Payment Links for high-intent failures"
            log(
                primary_action,
                f"{impact['recoverable_payment_count']} high-intent failures are inside amount and retry caps.",
                "MAX_AUTO_RETRY_AMOUNT_INR, MAX_RETRIES_PER_PAYMENT",
            )
            estimated_recovery_activated = True
        else:
            primary_action = "retry later"
            log(
                primary_action,
                "No failed payment is currently eligible for an autonomous Payment Link action.",
                "MAX_AUTO_RETRY_AMOUNT_INR, MAX_RETRIES_PER_PAYMENT",
            )
            estimated_recovery_activated = False

        if impact["above_auto_retry_cap_count"]:
            log(
                "high-value payments escalated",
                f"{impact['above_auto_retry_cap_count']} failed payments exceed the per-payment cap.",
                "MAX_AUTO_RETRY_AMOUNT_INR",
            )
        if impact["retry_limit_reached_count"]:
            log(
                "retry-exhausted payments skipped",
                f"{impact['retry_limit_reached_count']} failed payments have reached the retry limit.",
                "MAX_RETRIES_PER_PAYMENT",
            )

    confidence_gate_passed = cause != "unresolved" and confidence >= MIN_CONFIDENCE_FOR_AUTO_ACTION
    notification_sent = (
        confidence_gate_passed
        and impact["failed_gmv_inr"] >= MERCHANT_NOTIFICATION_EXPOSURE_THRESHOLD_INR
    )
    if not confidence_gate_passed:
        log(
            "merchant notification suppressed",
            "Low-confidence incidents are escalation-only; no autonomous merchant communication sent.",
            "MIN_CONFIDENCE_FOR_AUTO_ACTION",
        )
    elif notification_sent:
        log(
            "notify merchant",
            f"Failed GMV exposure INR {impact['failed_gmv_inr']:,} crossed the configured threshold.",
            "MERCHANT_NOTIFICATION_EXPOSURE_THRESHOLD_INR",
        )
    else:
        log(
            "merchant notification suppressed",
            f"Failed GMV exposure INR {impact['failed_gmv_inr']:,} stayed below the configured threshold.",
            "MERCHANT_NOTIFICATION_EXPOSURE_THRESHOLD_INR",
        )

    modeled_recovered = impact["recovered_amount_inr"] if estimated_recovery_activated else 0
    return {
        "primary_action": primary_action,
        "auto_action_taken": primary_action != "escalate to human",
        "merchant_notification_sent": notification_sent,
        "modeled_recovered_amount_inr": modeled_recovered,
        "modeled_recovered_amount_basis": impact["recovered_amount_basis"],
        "audit_trail": audit_trail,
        "policy": {
            "MIN_CONFIDENCE_FOR_AUTO_ACTION": MIN_CONFIDENCE_FOR_AUTO_ACTION,
            "MAX_AUTO_RETRY_AMOUNT_INR": MAX_AUTO_RETRY_AMOUNT_INR,
            "MAX_RETRIES_PER_PAYMENT": MAX_RETRIES_PER_PAYMENT,
            "ROUTE_HEALTH_CONFIRMATION_REQUIRED": True,
            "MERCHANT_NOTIFICATION_EXPOSURE_THRESHOLD_INR": MERCHANT_NOTIFICATION_EXPOSURE_THRESHOLD_INR,
            "ASSUMED_RECOVERY_SUCCESS_RATE": ASSUMED_RECOVERY_SUCCESS_RATE,
        },
    }
