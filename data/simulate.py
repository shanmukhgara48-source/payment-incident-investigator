"""Deterministic synthetic data for the payment incident pipeline."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


SEED = 20260827
DEFAULT_INCIDENT_COUNT = 60
AMBIGUOUS_FRACTION = 0.15
BASELINE_WINDOW_MINUTES = 30
CURRENT_WINDOW_MINUTES = 15

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

FAILURE_REASON_BY_CAUSE = {
    "bad_deploy": "gateway_error",
    "bank_psp_downtime": "psp_not_available",
    "gateway_error": "gateway_error",
    "config_change": "gateway_error",
    "network_issue": "bank_timeout",
    "unresolved": "bank_timeout",
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
                reason = FAILURE_REASON_BY_CAUSE[cause]
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
    alerts = [
        {
            "source": "alerts",
            "metric": f"payment_success_rate:{target['sub_type']}:{target['route']}",
            "threshold_breached": "rolling_baseline_minus_5pp",
            "timestamp": iso(alert_time),
        }
    ]
    webhook_events = [
        {
            "source": "webhook_events",
            "type": "payment.failed",
            "delivery_status": "delivered",
            "timestamp": iso(current_start + timedelta(minutes=5)),
        }
    ]
    error_traces = []

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

    trace_code = {
        "bad_deploy": "MERCHANT_5XX",
        "bank_psp_downtime": "PSP_UNAVAILABLE",
        "gateway_error": "GATEWAY_502",
        "config_change": "ROUTE_NOT_FOUND",
        "network_issue": "NET_CONN_RESET",
    }[cause]
    endpoint = "/v1/payments/authorize"
    error_traces.append(
        {
            "source": "error_traces",
            "error_code": trace_code,
            "count": max(8, round(severity * 100)),
            "timestamp": iso(current_start + timedelta(minutes=3)),
            "affected_endpoint": endpoint,
            "affected_method": target["sub_type"],
            "affected_route": target["route"],
        }
    )

    if cause == "bad_deploy":
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
    elif cause == "config_change":
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
    elif cause == "bank_psp_downtime":
        alerts.append(
            {
                "source": "alerts",
                "metric": f"route_health:{target['route']}",
                "threshold_breached": "healthcheck_failed",
                "timestamp": iso(current_start + timedelta(minutes=2)),
            }
        )
    elif cause == "gateway_error":
        webhook_events.append(
            {
                "source": "webhook_events",
                "type": "payment.failed",
                "delivery_status": "failed",
                "timestamp": iso(current_start + timedelta(minutes=4)),
            }
        )
    elif cause == "network_issue":
        alerts.append(
            {
                "source": "alerts",
                "metric": "network_packet_loss:payments-edge",
                "threshold_breached": "packet_loss_above_8pct",
                "timestamp": iso(current_start + timedelta(minutes=1)),
            }
        )
        webhook_events.append(
            {
                "source": "webhook_events",
                "type": "payment.failed",
                "delivery_status": "timed_out",
                "timestamp": iso(current_start + timedelta(minutes=6)),
            }
        )

    return deploy_logs, alerts, webhook_events, error_traces


def generate_incident(index: int, ambiguous_indexes: set[int]) -> dict:
    rng = random.Random(SEED + index * 101)
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
    top_reason = FAILURE_REASON_BY_CAUSE[cause]

    return {
        "incident_id": incident_id,
        "window": {
            "baseline_start": iso(baseline_start),
            "current_start": iso(current_start),
            "current_end": iso(current_start + timedelta(minutes=CURRENT_WINDOW_MINUTES)),
        },
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


def generate_dataset(count: int = DEFAULT_INCIDENT_COUNT) -> dict:
    if count < 1:
        raise ValueError("count must be positive")
    ambiguous_count = max(1, round(count * AMBIGUOUS_FRACTION))
    ambiguous_indexes = set(range(5, count, max(1, count // ambiguous_count)))
    if len(ambiguous_indexes) > ambiguous_count:
        ambiguous_indexes = set(sorted(ambiguous_indexes)[:ambiguous_count])
    while len(ambiguous_indexes) < ambiguous_count:
        ambiguous_indexes.add(count - len(ambiguous_indexes) - 1)

    return {
        "metadata": {
            "seed": SEED,
            "incident_count": count,
            "ambiguous_incident_count": len(ambiguous_indexes),
            "ambiguous_fraction": len(ambiguous_indexes) / count,
            "baseline_window_minutes": BASELINE_WINDOW_MINUTES,
            "current_window_minutes": CURRENT_WINDOW_MINUTES,
            "synthetic_data_notice": "All events and money values are synthetic.",
        },
        "incidents": [generate_incident(index, ambiguous_indexes) for index in range(count)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_INCIDENT_COUNT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("incidents.json"),
    )
    args = parser.parse_args()
    dataset = generate_dataset(args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    print(
        f"Generated {dataset['metadata']['incident_count']} incidents "
        f"({dataset['metadata']['ambiguous_incident_count']} ambiguous) -> {args.output}"
    )


if __name__ == "__main__":
    main()
