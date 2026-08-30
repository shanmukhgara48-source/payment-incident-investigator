"""Prepare fine-tuning JSONL from the train split.

Usage:
    .venv/bin/python -m data.prepare_finetune

Reads data/finetune_train.json, runs the detector + evidence extraction for each
incident (same code path as the live correlator), and writes chat-formatted JSONL
suitable for HuggingFace SFT training.

Output: data/finetune_train.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from src.correlator import (
    ERROR_SIGNATURES,
    _CORRELATOR_SYSTEM_PROMPT,
    _format_evidence_prompt,
    _in_overlap,
    _parse_time,
)
from src.detector import detect_degradations
from collections import Counter


def _extract_evidence_prompt(incident: dict) -> str | None:
    """Run detector + evidence extraction to produce the same prompt the live
    correlator would send to the LLM.  Returns None if no degradation detected."""
    detection = detect_degradations(incident)
    primary = detection.get("primary_degradation")
    if not primary:
        return None

    start = _parse_time(primary["window_start"])
    end = _parse_time(primary["window_end"])
    method = primary["sub_type"]
    route = primary["route"]

    overlapping_deploys = [
        e for e in incident["deploy_logs"]
        if e.get("event_type") == "deploy"
        and e.get("affected_method") == method
        and e.get("affected_route") == route
        and _in_overlap(e["timestamp"], start, end)
    ]
    config_changes = [
        e for e in incident["deploy_logs"]
        if e.get("event_type") == "config_change"
        and e.get("affected_method") == method
        and e.get("affected_route") == route
        and _in_overlap(e["timestamp"], start, end)
    ]
    relevant_traces = [
        e for e in incident["error_traces"]
        if e.get("affected_method") == method
        and e.get("affected_route") == route
        and _in_overlap(e["timestamp"], start, end)
    ]
    dominant_trace = max(relevant_traces, key=lambda x: x["count"], default=None)
    signature = ERROR_SIGNATURES.get(dominant_trace["error_code"]) if dominant_trace else None
    concentration = primary["failure_concentration_pct"]
    route_health_failed = any(
        e["metric"] == f"route_health:{route}"
        and e["threshold_breached"] == "healthcheck_failed"
        for e in incident["alerts"]
        if _in_overlap(e["timestamp"], start, end)
    )
    route_health_confirmed = any(
        e["metric"] == f"route_health:{route}"
        and e["threshold_breached"] == "healthcheck_passed"
        for e in incident["alerts"]
        if _in_overlap(e["timestamp"], start, end)
    )
    network_alert = any(
        e["metric"].startswith("network_packet_loss")
        for e in incident["alerts"]
        if _in_overlap(e["timestamp"], start, end)
    )
    webhook_failed = any(
        e["delivery_status"] in {"failed", "timed_out"}
        for e in incident["webhook_events"]
        if _in_overlap(e["timestamp"], start, end)
    )

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

    return _format_evidence_prompt(
        incident, primary, overlapping_deploys, config_changes,
        dominant_trace, signature, concentration,
        route_health_failed, route_health_confirmed,
        network_alert, webhook_failed,
        top_failure_reason, top_failure_count,
    )


def _build_target_response(incident: dict) -> str:
    """Build the ground-truth assistant response in the same JSON format
    the correlator prompt expects."""
    cause = incident["ground_truth"]["cause"]
    is_ambiguous = incident["ground_truth"]["is_ambiguous"]

    if is_ambiguous or cause == "unresolved":
        return json.dumps({
            "predicted_cause": "unresolved",
            "confidence": 0.35,
            "explanation": "Evidence is insufficient or contradictory — multiple signals point to different causes with no clear winner. Escalating.",
            "supporting_signals": ["mixed/weak signals"],
        })

    # For clear cases, build a realistic explanation
    explanations = {
        "bad_deploy": "Overlapping merchant deploy correlates with the failure window and error signature.",
        "bank_psp_downtime": "PSP/bank-side errors and route health check failure indicate downstream provider outage.",
        "gateway_error": "Gateway-side errors dominate the trace, with webhook delivery failures corroborating.",
        "config_change": "Routing configuration change in the window correlates with the failure pattern.",
        "network_issue": "Network packet-loss alerts and connection reset errors indicate infrastructure-level problems.",
    }
    signals_map = {
        "bad_deploy": ["deploy overlap", "error signature match"],
        "bank_psp_downtime": ["route health failed", "PSP error codes"],
        "gateway_error": ["GATEWAY_502 errors", "webhook delivery failures"],
        "config_change": ["config change overlap", "routing errors"],
        "network_issue": ["network packet-loss alert", "NET_CONN_RESET errors"],
    }

    return json.dumps({
        "predicted_cause": cause,
        "confidence": 0.78,
        "explanation": explanations.get(cause, "Evidence supports this diagnosis."),
        "supporting_signals": signals_map.get(cause, ["corroborating evidence"]),
    })


# Simplified system prompt for fine-tuning (no few-shot examples — the model
# learns from training data instead)
_FINETUNE_SYSTEM_PROMPT = """\
You are a payment incident root-cause analyst. Given structured evidence from \
a payment degradation incident, determine the most likely root cause.

Valid root causes (pick exactly one):
- bad_deploy: A merchant-side code deployment caused the failures.
- bank_psp_downtime: The PSP or bank endpoint is degraded or down.
- gateway_error: The payment gateway itself is experiencing errors.
- config_change: A routing configuration change caused misrouting.
- network_issue: Network-level problems (packet loss, connection resets).
- unresolved: Evidence is insufficient or contradictory — escalate.

CRITICAL: If the evidence does not clearly support one cause, respond with \
predicted_cause="unresolved". Being wrong with high confidence is WORSE than \
saying "unresolved".

Respond with valid JSON:
{"predicted_cause": "<cause>", "confidence": <0.0-0.99>, "explanation": "<reasoning>", "supporting_signals": ["<signal>"]}"""


def main() -> None:
    data_dir = Path(__file__).parent
    train_path = data_dir / "finetune_train.json"

    if not train_path.exists():
        print("ERROR: Run data.split_dataset first to generate finetune_train.json")
        return

    incidents = json.loads(train_path.read_text())
    print(f"Processing {len(incidents)} training incidents...")

    output_path = data_dir / "finetune_train.jsonl"
    skipped = 0
    written = 0

    with open(output_path, "w") as f:
        for inc in incidents:
            user_prompt = _extract_evidence_prompt(inc)
            if user_prompt is None:
                skipped += 1
                continue

            target = _build_target_response(inc)

            record = {
                "messages": [
                    {"role": "system", "content": _FINETUNE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": target},
                ]
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    print(f"Written: {written}, Skipped (no degradation): {skipped}")
    print(f"Output: {output_path}")

    # Show cause distribution in training data
    cause_counts = Counter(
        inc["ground_truth"]["cause"] for inc in incidents
    )
    print(f"Cause distribution: {dict(cause_counts)}")


if __name__ == "__main__":
    main()
