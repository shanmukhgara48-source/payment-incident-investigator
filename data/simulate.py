"""Deterministic synthetic data for the payment incident pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


SEED = 20260827
DEFAULT_INCIDENT_COUNT = 60
AMBIGUOUS_FRACTION = 0.15
MIN_INCIDENT_COUNT = 10
MAX_INCIDENT_COUNT = 500
MIN_AMBIGUOUS_FRACTION = 0.0
MAX_AMBIGUOUS_FRACTION = 0.5
BASELINE_WINDOW_MINUTES = 30
CURRENT_WINDOW_MINUTES = 15

logger = logging.getLogger(__name__)

ROUTE_PROFILES = [
    {
        "method": "UPI",
        "sub_type": "upi_collect",
        "display_name": "UPI Collect",
        "route_type": "psp",
        "route": "abc@upi",
    },
    {
        "method": "UPI",
        "sub_type": "upi_intent",
        "display_name": "UPI Intent",
        "route_type": "psp",
        "route": "swift@upi",
    },
    {
        "method": "Cards",
        "sub_type": "cards",
        "display_name": "Cards",
        "route_type": "gateway",
        "route": "Razorpay Gateway West",
    },
    {
        "method": "Netbanking",
        "sub_type": "netbanking",
        "display_name": "Netbanking",
        "route_type": "bank",
        "route": "ICICI Bank",
    },
]

CAUSES = [
    "bad_deploy",
    "bank_psp_downtime",
    "gateway_error",
    "config_change",
    "network_issue",
]

# ── Probabilistic signal parameters ────────────────────────────────
# Each cause produces evidence signals with controlled probabilities,
# ensuring deliberate overlap between causes.  No single signal field
# is a 1:1 function of the ground-truth cause.

ERROR_CODE_DISTRIBUTIONS: dict[str, list[tuple[str, float]]] = {
    "bad_deploy": [
        ("MERCHANT_5XX", 0.50),
        ("GATEWAY_502", 0.17),
        ("BANK_TIMEOUT", 0.12),
        ("ROUTE_NOT_FOUND", 0.08),
        ("NET_CONN_RESET", 0.05),
        ("UNKNOWN_UPSTREAM", 0.08),
    ],
    "bank_psp_downtime": [
        ("PSP_UNAVAILABLE", 0.45),
        ("BANK_TIMEOUT", 0.25),
        ("GATEWAY_502", 0.12),
        ("NET_CONN_RESET", 0.08),
        ("ROUTE_NOT_FOUND", 0.05),
        ("UNKNOWN_UPSTREAM", 0.05),
    ],
    "gateway_error": [
        ("GATEWAY_502", 0.42),
        ("BANK_TIMEOUT", 0.17),
        ("PSP_UNAVAILABLE", 0.13),
        ("ROUTE_NOT_FOUND", 0.08),
        ("NET_CONN_RESET", 0.10),
        ("UNKNOWN_UPSTREAM", 0.10),
    ],
    "config_change": [
        ("ROUTE_NOT_FOUND", 0.35),
        ("GATEWAY_502", 0.22),
        ("MERCHANT_5XX", 0.18),
        ("BANK_TIMEOUT", 0.10),
        ("NET_CONN_RESET", 0.05),
        ("UNKNOWN_UPSTREAM", 0.10),
    ],
    "network_issue": [
        ("NET_CONN_RESET", 0.38),
        ("BANK_TIMEOUT", 0.22),
        ("GATEWAY_502", 0.13),
        ("PSP_UNAVAILABLE", 0.08),
        ("ROUTE_NOT_FOUND", 0.07),
        ("UNKNOWN_UPSTREAM", 0.12),
    ],
}

# P(signal present | cause) for each corroborating signal type.
SIGNAL_PROBABILITIES: dict[str, dict[str, float]] = {
    "deploy_overlap": {
        "bad_deploy": 0.80,
        "bank_psp_downtime": 0.12,
        "gateway_error": 0.12,
        "config_change": 0.12,
        "network_issue": 0.12,
    },
    "config_change_overlap": {
        "config_change": 0.75,
        "bad_deploy": 0.10,
        "bank_psp_downtime": 0.10,
        "gateway_error": 0.10,
        "network_issue": 0.10,
    },
    "route_health_failed": {
        "bank_psp_downtime": 0.75,
        "gateway_error": 0.20,
        "network_issue": 0.15,
        "bad_deploy": 0.05,
        "config_change": 0.05,
    },
    "network_alert": {
        "network_issue": 0.75,
        "bank_psp_downtime": 0.15,
        "gateway_error": 0.10,
        "bad_deploy": 0.05,
        "config_change": 0.05,
    },
    "webhook_degraded": {
        "gateway_error": 0.65,
        "network_issue": 0.60,
        "bank_psp_downtime": 0.20,
        "bad_deploy": 0.10,
        "config_change": 0.10,
    },
}

FAILURE_REASON_DISTRIBUTIONS: dict[str, list[tuple[str, float]]] = {
    "bad_deploy": [
        ("gateway_error", 0.50),
        ("bank_timeout", 0.20),
        ("psp_not_available", 0.15),
        ("otp_failed", 0.10),
        ("server_error", 0.05),
    ],
    "bank_psp_downtime": [
        ("psp_not_available", 0.45),
        ("bank_timeout", 0.25),
        ("gateway_error", 0.20),
        ("otp_failed", 0.10),
    ],
    "gateway_error": [
        ("gateway_error", 0.50),
        ("bank_timeout", 0.20),
        ("psp_not_available", 0.15),
        ("server_error", 0.10),
        ("otp_failed", 0.05),
    ],
    "config_change": [
        ("gateway_error", 0.40),
        ("bank_timeout", 0.25),
        ("psp_not_available", 0.20),
        ("otp_failed", 0.15),
    ],
    "network_issue": [
        ("bank_timeout", 0.45),
        ("gateway_error", 0.25),
        ("psp_not_available", 0.15),
        ("otp_failed", 0.10),
        ("server_error", 0.05),
    ],
    "unresolved": [
        ("bank_timeout", 0.30),
        ("gateway_error", 0.25),
        ("psp_not_available", 0.20),
        ("otp_failed", 0.15),
        ("server_error", 0.10),
    ],
}


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def event_time(start: datetime, minutes: int, index: int, total: int) -> datetime:
    seconds = max(1, minutes * 60 - 1)
    return start + timedelta(seconds=(index * 97 + total * 11) % seconds)


def _payment_events(
    rng: random.Random,
    incident_id: str,
    profile: dict,
    window: str,
    start: datetime,
    minutes: int,
    total: int,
    failure_rate: float,
    cause: str,
) -> list[dict]:
    failure_count = round(total * failure_rate)
    outcomes = ["failed"] * failure_count + ["success"] * (total - failure_count)
    rng.shuffle(outcomes)
    events = []

    for index, status in enumerate(outcomes):
        high_value = rng.random() < 0.06
        amount = rng.randint(52_000, 90_000) if high_value else rng.randint(199, 18_000)
        high_intent = status == "failed" and rng.random() < 0.58
        retry_count = rng.choices([0, 1, 2, 3], weights=[60, 24, 12, 4])[0]
        reason = "none"
        if status == "failed":
            if window == "current":
                dist = FAILURE_REASON_DISTRIBUTIONS.get(
                    cause, FAILURE_REASON_DISTRIBUTIONS["unresolved"]
                )
                reasons, weights = zip(*dist)
                reason = rng.choices(reasons, weights=weights, k=1)[0]
            else:
                reason = rng.choice(["otp_failed", "bank_timeout", "gateway_error"])

        events.append(
            {
                "source": "payment_events",
                "payment_id": f"pay_{incident_id.lower().replace('-', '')}_{profile['sub_type']}_{window}_{index:03d}",
                "timestamp": iso(event_time(start, minutes, index, total)),
                "window": window,
                "method": profile["method"],
                "sub_type": profile["sub_type"],
                "method_display": profile["display_name"],
                "route_type": profile["route_type"],
                "route": profile["route"],
                "amount_inr": amount,
                "status": status,
                "failure_reason": reason,
                "high_intent": high_intent,
                "retry_count": retry_count,
            }
        )
    return events


def _signal_streams(
    rng: random.Random,
    cause: str,
    target: dict,
    current_start: datetime,
    alert_time: datetime,
    severity: float,
    ambiguous: bool,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    # Background deploy -- always present, never matches the target pair.
    deploy_logs = [
        {
            "source": "deploy_logs",
            "event_type": "deploy",
            "service": "settlements-ledger",
            "version": f"v4.{rng.randint(1, 9)}.{rng.randint(0, 9)}",
            "rollout_pct": 10,
            "timestamp": iso(current_start - timedelta(minutes=9)),
            "affected_method": None,
            "affected_route": None,
        }
    ]
    # Success-rate alert -- always present.
    alerts = [
        {
            "source": "alerts",
            "metric": f"payment_success_rate:{target['sub_type']}:{target['route']}",
            "threshold_breached": "rolling_baseline_minus_5pp",
            "timestamp": iso(alert_time),
        }
    ]
    # Baseline webhook -- always delivered.
    webhook_events = [
        {
            "source": "webhook_events",
            "type": "payment.failed",
            "delivery_status": "delivered",
            "timestamp": iso(current_start + timedelta(minutes=5)),
        }
    ]
    error_traces: list[dict] = []

    if ambiguous:
        error_traces.extend(
            [
                {
                    "source": "error_traces",
                    "error_code": "UNKNOWN_UPSTREAM",
                    "count": max(2, round(severity * 12)),
                    "timestamp": iso(current_start + timedelta(minutes=3)),
                    "affected_endpoint": "/v1/payments/authorize",
                    "affected_method": target["sub_type"],
                    "affected_route": target["route"],
                },
                {
                    "source": "error_traces",
                    "error_code": "MERCHANT_5XX",
                    "count": max(2, round(severity * 10)),
                    "timestamp": iso(current_start + timedelta(minutes=7)),
                    "affected_endpoint": "/v1/payments/authorize",
                    "affected_method": target["sub_type"],
                    "affected_route": target["route"],
                },
            ]
        )
        return deploy_logs, alerts, webhook_events, error_traces

    # ── Non-ambiguous: probabilistic signal generation ──────────────
    # Each signal type is generated with a cause-dependent probability,
    # with deliberate overlap between causes.

    # Error trace -- sampled from a per-cause distribution.
    dist = ERROR_CODE_DISTRIBUTIONS[cause]
    codes, weights = zip(*dist)
    error_code = rng.choices(codes, weights=weights, k=1)[0]
    error_traces.append(
        {
            "source": "error_traces",
            "error_code": error_code,
            "count": max(8, round(severity * 100)),
            "timestamp": iso(current_start + timedelta(minutes=3)),
            "affected_endpoint": "/v1/payments/authorize",
            "affected_method": target["sub_type"],
            "affected_route": target["route"],
        }
    )

    # Deploy overlap -- probabilistic per cause.
    if rng.random() < SIGNAL_PROBABILITIES["deploy_overlap"][cause]:
        deploy_logs.append(
            {
                "source": "deploy_logs",
                "event_type": "deploy",
                "service": "merchant-checkout",
                "version": f"v8.{rng.randint(10, 19)}.{rng.randint(0, 9)}",
                "rollout_pct": rng.choice([25, 50, 100]),
                "timestamp": iso(current_start - timedelta(minutes=2)),
                "affected_method": target["sub_type"],
                "affected_route": target["route"],
            }
        )

    # Config change overlap -- probabilistic per cause.
    if rng.random() < SIGNAL_PROBABILITIES["config_change_overlap"][cause]:
        deploy_logs.append(
            {
                "source": "deploy_logs",
                "event_type": "config_change",
                "service": "razorpay-routing-config",
                "version": f"ruleset-{rng.randint(120, 999)}",
                "rollout_pct": 100,
                "timestamp": iso(current_start - timedelta(minutes=1)),
                "affected_method": target["sub_type"],
                "affected_route": target["route"],
            }
        )

    # Route health alert -- probabilistic per cause.
    if rng.random() < SIGNAL_PROBABILITIES["route_health_failed"][cause]:
        alerts.append(
            {
                "source": "alerts",
                "metric": f"route_health:{target['route']}",
                "threshold_breached": "healthcheck_failed",
                "timestamp": iso(current_start + timedelta(minutes=2)),
            }
        )

    # Network packet-loss alert -- probabilistic per cause.
    if rng.random() < SIGNAL_PROBABILITIES["network_alert"][cause]:
        alerts.append(
            {
                "source": "alerts",
                "metric": "network_packet_loss:payments-edge",
                "threshold_breached": "packet_loss_above_8pct",
                "timestamp": iso(current_start + timedelta(minutes=1)),
            }
        )

    # Webhook delivery degradation -- probabilistic per cause.
    if rng.random() < SIGNAL_PROBABILITIES["webhook_degraded"][cause]:
        delivery_status = rng.choice(["failed", "timed_out"])
        webhook_events.append(
            {
                "source": "webhook_events",
                "type": "payment.failed",
                "delivery_status": delivery_status,
                "timestamp": iso(current_start + timedelta(minutes=rng.choice([4, 6]))),
            }
        )

    return deploy_logs, alerts, webhook_events, error_traces


def _failure_by_minute(payment_events: list[dict], current_start: datetime) -> list[dict]:
    """Per-minute failure buckets for the current window (used by counterfactual analysis)."""
    buckets: list[dict] = []
    for minute_offset in range(CURRENT_WINDOW_MINUTES):
        bucket_start_iso = iso(current_start + timedelta(minutes=minute_offset))
        bucket_end_iso = iso(current_start + timedelta(minutes=minute_offset + 1))
        bucket_events = [
            e for e in payment_events
            if e["window"] == "current"
            and bucket_start_iso <= e["timestamp"] < bucket_end_iso
        ]
        failed = [e for e in bucket_events if e["status"] == "failed"]
        buckets.append({
            "minute_offset": minute_offset,
            "timestamp": bucket_start_iso,
            "total_count": len(bucket_events),
            "failed_count": len(failed),
            "failed_gmv_inr": sum(e["amount_inr"] for e in failed),
        })
    return buckets


def generate_incident(index: int, ambiguous_indexes: set[int], seed: int = SEED) -> dict:
    rng = random.Random(seed + index * 101)
    incident_id = f"INC-{index + 1:04d}"
    current_start = datetime(2026, 8, 20, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=index * 37)
    alert_time = current_start + timedelta(minutes=4)
    baseline_start = current_start - timedelta(minutes=BASELINE_WINDOW_MINUTES)
    target = ROUTE_PROFILES[index % len(ROUTE_PROFILES)]
    ambiguous = index in ambiguous_indexes
    cause = "unresolved" if ambiguous else CAUSES[index % len(CAUSES)]
    injected_spike = round(rng.uniform(0.05, 0.40), 3)

    payment_events = []
    for profile in ROUTE_PROFILES:
        is_target = profile["sub_type"] == target["sub_type"] and profile["route"] == target["route"]
        total = 100 if is_target else 35
        baseline_failure_rate = rng.uniform(0.015, 0.035)
        current_failure_rate = min(0.70, baseline_failure_rate + injected_spike) if is_target else rng.uniform(0.015, 0.04)
        payment_events.extend(
            _payment_events(
                rng,
                incident_id,
                profile,
                "baseline",
                baseline_start,
                BASELINE_WINDOW_MINUTES,
                total,
                baseline_failure_rate,
                cause,
            )
        )
        payment_events.extend(
            _payment_events(
                rng,
                incident_id,
                profile,
                "current",
                current_start,
                CURRENT_WINDOW_MINUTES,
                total,
                current_failure_rate,
                cause,
            )
        )

    payment_events.sort(key=lambda event: event["timestamp"])
    deploy_logs, alerts, webhook_events, error_traces = _signal_streams(
        rng, cause, target, current_start, alert_time, injected_spike, ambiguous
    )

    current_failures = [
        e for e in payment_events
        if e["window"] == "current" and e["status"] == "failed"
    ]
    if current_failures:
        reason_counts = Counter(e["failure_reason"] for e in current_failures)
        top_reason = reason_counts.most_common(1)[0][0]
    else:
        top_reason = "none"

    failure_by_minute = _failure_by_minute(payment_events, current_start)

    return {
        "incident_id": incident_id,
        "window": {
            "baseline_start": iso(baseline_start),
            "current_start": iso(current_start),
            "current_end": iso(current_start + timedelta(minutes=CURRENT_WINDOW_MINUTES)),
            "detection_minute_offset": 4,
        },
        "failure_by_minute": failure_by_minute,
        "payment_events": payment_events,
        "deploy_logs": sorted(deploy_logs, key=lambda event: event["timestamp"]),
        "alerts": sorted(alerts, key=lambda event: event["timestamp"]),
        "webhook_events": sorted(webhook_events, key=lambda event: event["timestamp"]),
        "error_traces": sorted(error_traces, key=lambda event: event["timestamp"]),
        "ground_truth": {
            "cause": cause,
            "is_ambiguous": ambiguous,
            "affected_method": target["sub_type"],
            "affected_method_display": target["display_name"],
            "affected_route": target["route"],
            "injected_failure_rate_spike": injected_spike,
            "top_failure_reason": top_reason,
        },
    }


# ── deliberately constructed skeptic-gate case ──────────────────────
# Every randomly generated incident lands a primary diagnosis at 0.80-0.99
# confidence, which is too far above MIN_CONFIDENCE_FOR_AUTO_ACTION (0.60) for
# the skeptic's penalty schedule (max ~0.22 for a bad_deploy) to ever reach the
# gate. So the 60-incident batch shows challenges but never a challenge that
# actually blocks auto-action -- a real gap in the demo evidence.
#
# This incident closes that gap. It is fully hardcoded, not random: a
# "bad_deploy" whose entire support is a single overlapping deploy touching
# 5% of traffic, with no error-trace signature to corroborate it, while
# failures are smeared across all four routes.
#
#   correlator : deploy overlap 0.40 + >=50% concentration 0.20 = 0.60
#                (exactly clears the resolve bar, runner-up 0.00)
#   skeptic    : small_blast_radius        -0.100  (5% rollout <= 25%)
#                low_failure_concentration -0.056  (58.8% < 80%)
#   final      : 0.60 - 0.156 = 0.44  -> BELOW the 0.60 gate -> escalate
#
# Keep the numbers below in sync with that arithmetic; tests/test_skeptic.py
# asserts the end state.
SKEPTIC_GATE_ROLLOUT_PCT = 5
SKEPTIC_GATE_TARGET_TOTAL = 100
SKEPTIC_GATE_TARGET_BASELINE_FAILURE_RATE = 0.03   # 3 of 100
SKEPTIC_GATE_TARGET_CURRENT_FAILURE_RATE = 0.30    # 30 of 100
SKEPTIC_GATE_OTHER_TOTAL = 35
SKEPTIC_GATE_OTHER_BASELINE_FAILURE_RATE = 1 / 35  # 1 of 35
SKEPTIC_GATE_OTHER_CURRENT_FAILURE_RATE = 0.20     # 7 of 35


def build_skeptic_gate_incident(index: int, seed: int = SEED) -> dict:
    """Build the constructed incident that drives confidence below the gate.

    `index` is the zero-based position it occupies, so with the default batch
    of 60 it becomes INC-0061.
    """
    rng = random.Random(seed + index * 101)
    incident_id = f"INC-{index + 1:04d}"
    current_start = datetime(2026, 8, 20, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=index * 37)
    alert_time = current_start + timedelta(minutes=4)
    baseline_start = current_start - timedelta(minutes=BASELINE_WINDOW_MINUTES)
    target = ROUTE_PROFILES[0]  # UPI Collect / abc@upi

    # Failure counts are chosen so the target pair holds the largest success-rate
    # drop (it stays the primary degradation) while carrying only 30 of the 51
    # current-window failures -- 58.8% concentration, above the correlator's 50%
    # bonus threshold but below the skeptic's 80% dilution threshold.
    payment_events = []
    for profile in ROUTE_PROFILES:
        is_target = profile["sub_type"] == target["sub_type"] and profile["route"] == target["route"]
        total = SKEPTIC_GATE_TARGET_TOTAL if is_target else SKEPTIC_GATE_OTHER_TOTAL
        baseline_rate = (
            SKEPTIC_GATE_TARGET_BASELINE_FAILURE_RATE
            if is_target
            else SKEPTIC_GATE_OTHER_BASELINE_FAILURE_RATE
        )
        current_rate = (
            SKEPTIC_GATE_TARGET_CURRENT_FAILURE_RATE
            if is_target
            else SKEPTIC_GATE_OTHER_CURRENT_FAILURE_RATE
        )
        for window, start, minutes, rate in (
            ("baseline", baseline_start, BASELINE_WINDOW_MINUTES, baseline_rate),
            ("current", current_start, CURRENT_WINDOW_MINUTES, current_rate),
        ):
            payment_events.extend(
                _payment_events(
                    rng, incident_id, profile, window, start, minutes, total, rate, "bad_deploy"
                )
            )
    payment_events.sort(key=lambda event: event["timestamp"])

    deploy_logs = [
        # Unrelated background deploy: no affected method/route, so it never matches.
        {
            "source": "deploy_logs",
            "event_type": "deploy",
            "service": "settlements-ledger",
            "version": "v4.2.1",
            "rollout_pct": 10,
            "timestamp": iso(current_start - timedelta(minutes=9)),
            "affected_method": None,
            "affected_route": None,
        },
        # The sole piece of evidence behind the bad_deploy diagnosis -- and it
        # only ever reached 5% of traffic.
        {
            "source": "deploy_logs",
            "event_type": "deploy",
            "service": "merchant-checkout",
            "version": "v8.14.0",
            "rollout_pct": SKEPTIC_GATE_ROLLOUT_PCT,
            "timestamp": iso(current_start - timedelta(minutes=2)),
            "affected_method": target["sub_type"],
            "affected_route": target["route"],
        },
    ]
    alerts = [
        {
            "source": "alerts",
            "metric": f"payment_success_rate:{target['sub_type']}:{target['route']}",
            "threshold_breached": "rolling_baseline_minus_5pp",
            "timestamp": iso(alert_time),
        }
    ]
    # Delivered, not failed: no webhook signal for gateway_error or network_issue.
    webhook_events = [
        {
            "source": "webhook_events",
            "type": "payment.failed",
            "delivery_status": "delivered",
            "timestamp": iso(current_start + timedelta(minutes=5)),
        }
    ]
    # Deliberately empty. With no dominant error code there is no signature to
    # corroborate bad_deploy -- which is what holds confidence down at exactly
    # 0.60 instead of the 0.95 a normal bad_deploy incident scores.
    error_traces: list[dict] = []

    return {
        "incident_id": incident_id,
        "window": {
            "baseline_start": iso(baseline_start),
            "current_start": iso(current_start),
            "current_end": iso(current_start + timedelta(minutes=CURRENT_WINDOW_MINUTES)),
            "detection_minute_offset": 4,
        },
        "failure_by_minute": _failure_by_minute(payment_events, current_start),
        "payment_events": payment_events,
        "deploy_logs": sorted(deploy_logs, key=lambda event: event["timestamp"]),
        "alerts": alerts,
        "webhook_events": webhook_events,
        "error_traces": error_traces,
        "ground_truth": {
            # Ground truth is "unresolved" on purpose. The evidence genuinely
            # does not identify a cause: a 5% deploy cannot explain a 27-point
            # drop spread over four routes. Escalating is the correct outcome,
            # so this is scored as an ambiguous case the system is expected to
            # be honest about, not as a clear case it is expected to name.
            "cause": "unresolved",
            "is_ambiguous": True,
            "affected_method": target["sub_type"],
            "affected_method_display": target["display_name"],
            "affected_route": target["route"],
            "injected_failure_rate_spike": round(
                SKEPTIC_GATE_TARGET_CURRENT_FAILURE_RATE
                - SKEPTIC_GATE_TARGET_BASELINE_FAILURE_RATE,
                3,
            ),
            "top_failure_reason": Counter(
                e["failure_reason"]
                for e in payment_events
                if e["window"] == "current" and e["status"] == "failed"
            ).most_common(1)[0][0],
            "constructed_for": "skeptic_confidence_gate",
            "construction_note": (
                "Deliberately constructed to exercise the skeptic gate: a thinly "
                "supported bad_deploy diagnosis (5% rollout, no error signature, "
                "58.8% failure concentration) that the skeptic pushes from 0.60 "
                "to 0.44, below MIN_CONFIDENCE_FOR_AUTO_ACTION."
            ),
        },
    }


def validate_simulation_parameters(count: int, ambiguous_fraction: float) -> None:
    if not MIN_INCIDENT_COUNT <= count <= MAX_INCIDENT_COUNT:
        raise ValueError(
            f"incident count must be between {MIN_INCIDENT_COUNT} and {MAX_INCIDENT_COUNT}"
        )
    if not MIN_AMBIGUOUS_FRACTION <= ambiguous_fraction <= MAX_AMBIGUOUS_FRACTION:
        raise ValueError(
            "ambiguous-case ratio must be between "
            f"{MIN_AMBIGUOUS_FRACTION:.1f} and {MAX_AMBIGUOUS_FRACTION:.1f}"
        )


def generate_dataset(
    count: int = DEFAULT_INCIDENT_COUNT,
    ambiguous_fraction: float = AMBIGUOUS_FRACTION,
    seed: int = SEED,
    include_skeptic_case: bool = True,
) -> dict:
    """Generate `count` random incidents, plus the constructed skeptic-gate case.

    `include_skeptic_case` appends one extra, fully deterministic incident
    (INC-0061 for the default batch of 60) that drives confidence below the
    auto-action gate. Callers that need exactly `count` incidents -- the
    /ws/live stream, whose query parameter is a promise -- pass False.
    """
    validate_simulation_parameters(count, ambiguous_fraction)
    ambiguous_count = round(count * ambiguous_fraction)
    if ambiguous_fraction > 0:
        ambiguous_count = max(1, ambiguous_count)
    ambiguous_indexes = set()
    if ambiguous_count:
        ambiguous_indexes = set(range(5, count, max(1, count // ambiguous_count)))
        if len(ambiguous_indexes) > ambiguous_count:
            ambiguous_indexes = set(sorted(ambiguous_indexes)[:ambiguous_count])
        candidate = count - 1
        while len(ambiguous_indexes) < ambiguous_count:
            ambiguous_indexes.add(candidate)
            candidate -= 1

    incidents = [generate_incident(index, ambiguous_indexes, seed) for index in range(count)]
    constructed_ids: list[str] = []
    if include_skeptic_case:
        skeptic_case = build_skeptic_gate_incident(count, seed)
        incidents.append(skeptic_case)
        constructed_ids.append(skeptic_case["incident_id"])

    ambiguous_total = len(ambiguous_indexes) + len(constructed_ids)
    return {
        "metadata": {
            "seed": seed,
            "incident_count": len(incidents),
            "random_incident_count": count,
            "constructed_incident_ids": constructed_ids,
            "ambiguous_incident_count": ambiguous_total,
            "ambiguous_fraction": ambiguous_total / len(incidents),
            "baseline_window_minutes": BASELINE_WINDOW_MINUTES,
            "current_window_minutes": CURRENT_WINDOW_MINUTES,
            "synthetic_data_notice": "All events and money values are synthetic.",
        },
        "incidents": incidents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_INCIDENT_COUNT)
    parser.add_argument("--ambiguous-ratio", type=float, default=AMBIGUOUS_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("incidents.json"),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s level=%(levelname)s logger=%(name)s "
            "incident_id=%(incident_id)s stage=%(stage)s message=%(message)s"
        ),
    )
    dataset = generate_dataset(args.count, args.ambiguous_ratio, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "generated incidents=%s ambiguous=%s output=%s",
        dataset["metadata"]["incident_count"],
        dataset["metadata"]["ambiguous_incident_count"],
        args.output,
        extra={"incident_id": "batch", "stage": "simulate"},
    )


if __name__ == "__main__":
    main()
