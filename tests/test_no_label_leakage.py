"""Structural regression test: no single evidence field should be a
near-perfect predictor of the ground-truth root cause.

BACKGROUND

The first version of the simulator generated every evidence field as a
deterministic 1:1 function of the true cause — ``trace_code`` was a dict
lookup, deploy overlaps were exclusive to ``bad_deploy``, route-health
alerts were exclusive to ``bank_psp_downtime``, etc. The correlator then
inverted those same lookups to "predict" the cause, achieving 100%
accuracy on synthetic data without learning anything.

This test catches that pattern structurally: for each evidence field the
correlator consumes, it trains a trivial majority-vote classifier and
asserts that standalone accuracy stays below a ceiling. If any single
field crosses the ceiling, the simulator has re-introduced a near-1:1
mapping and the accuracy figure is meaningless.

THRESHOLD RATIONALE

With 5 equally likely causes, a random field gives ~20% accuracy. A
genuinely informative field (e.g. deploy overlap, which is more common
for bad_deploy but also appears for other causes at ~12%) can reach
40-55% standalone accuracy honestly — it's a useful signal, not the
whole answer. The ceiling of 85% is set to catch 1:1 or near-1:1
mappings (which hit ~100%) while leaving headroom for honestly
informative fields. This means: any field that can predict the cause
correctly more than 85% of the time on its own is *too* predictive to
be realistic.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta

from data.simulate import CAUSES, generate_dataset
from src.correlator import ERROR_SIGNATURES
from src.detector import detect_degradations

# Maximum standalone accuracy any single evidence field may achieve
# when used alone to predict root cause. See module docstring.
MAX_SINGLE_FIELD_ACCURACY = 0.85


def _predictive_ceiling(values: list, causes: list) -> tuple[float, dict]:
    """Accuracy of a majority-vote lookup table from field values to causes.

    Returns (accuracy, {value: best_cause}) so failures can be diagnosed.
    """
    value_counts: dict[str, Counter] = defaultdict(Counter)
    for value, cause in zip(values, causes):
        value_counts[str(value)][cause] += 1

    correct = 0
    lookup = {}
    for value_str, counts in value_counts.items():
        best_cause, best_count = counts.most_common(1)[0]
        correct += best_count
        lookup[value_str] = best_cause

    return correct / len(causes), lookup


def _extract_evidence_fields(incidents: list[dict]) -> dict[str, list]:
    """Extract every field the correlator, skeptic, and memory consume."""
    fields: dict[str, list] = {
        "dominant_error_code": [],
        "has_deploy_overlap": [],
        "has_config_change": [],
        "has_route_health_failed": [],
        "has_network_alert": [],
        "has_webhook_degraded": [],
        "top_failure_reason": [],
        "error_signature_cause": [],
    }

    for incident in incidents:
        detection = detect_degradations(incident)
        primary = detection.get("primary_degradation")
        if not primary:
            for key in fields:
                fields[key].append(None)
            continue

        method = primary["sub_type"]
        route = primary["route"]
        start = datetime.fromisoformat(
            primary["window_start"].replace("Z", "+00:00")
        )
        end = datetime.fromisoformat(
            primary["window_end"].replace("Z", "+00:00")
        )

        # Dominant error trace
        relevant_traces = [
            t for t in incident["error_traces"]
            if t.get("affected_method") == method
            and t.get("affected_route") == route
        ]
        dominant = max(relevant_traces, key=lambda t: t["count"], default=None)
        error_code = dominant["error_code"] if dominant else None
        fields["dominant_error_code"].append(error_code)

        # Error signature -> cause mapping
        sig = ERROR_SIGNATURES.get(error_code) if error_code else None
        fields["error_signature_cause"].append(sig[0] if sig else None)

        # Deploy overlap
        has_deploy = any(
            d.get("event_type") == "deploy"
            and d.get("affected_method") == method
            and d.get("affected_route") == route
            for d in incident["deploy_logs"]
        )
        fields["has_deploy_overlap"].append(has_deploy)

        # Config change overlap
        has_config = any(
            d.get("event_type") == "config_change"
            and d.get("affected_method") == method
            and d.get("affected_route") == route
            for d in incident["deploy_logs"]
        )
        fields["has_config_change"].append(has_config)

        # Route health failed
        has_health = any(
            a["metric"] == f"route_health:{route}"
            and a["threshold_breached"] == "healthcheck_failed"
            for a in incident["alerts"]
        )
        fields["has_route_health_failed"].append(has_health)

        # Network alert
        has_network = any(
            a["metric"].startswith("network_packet_loss")
            for a in incident["alerts"]
        )
        fields["has_network_alert"].append(has_network)

        # Webhook degraded
        has_webhook = any(
            w["delivery_status"] in {"failed", "timed_out"}
            for w in incident["webhook_events"]
        )
        fields["has_webhook_degraded"].append(has_webhook)

        # Top failure reason
        current_failures = [
            e for e in incident["payment_events"]
            if e["window"] == "current"
            and e["status"] == "failed"
            and e["sub_type"] == method
            and e["route"] == route
        ]
        if current_failures:
            reason_counts = Counter(e["failure_reason"] for e in current_failures)
            fields["top_failure_reason"].append(reason_counts.most_common(1)[0][0])
        else:
            fields["top_failure_reason"].append(None)

    return fields


def test_no_single_field_is_a_near_perfect_cause_predictor():
    """The core structural guard: no evidence field alone achieves > 85%
    accuracy at predicting root cause across the full synthetic dataset."""
    dataset = generate_dataset(count=60, ambiguous_fraction=0.0, seed=42)
    incidents = dataset["incidents"]
    # Exclude the constructed skeptic case (its ground_truth is unresolved)
    clear = [
        inc for inc in incidents
        if not inc["ground_truth"]["is_ambiguous"]
    ]
    causes = [inc["ground_truth"]["cause"] for inc in clear]

    fields = _extract_evidence_fields(clear)

    violations = []
    for field_name, values in fields.items():
        # Skip if all None
        if all(v is None for v in values):
            continue
        accuracy, lookup = _predictive_ceiling(values, causes)
        if accuracy > MAX_SINGLE_FIELD_ACCURACY:
            violations.append(
                f"{field_name}: standalone accuracy {accuracy:.1%} "
                f"(ceiling {MAX_SINGLE_FIELD_ACCURACY:.0%}); "
                f"lookup={lookup}"
            )

    assert not violations, (
        "Label leakage detected — the following evidence field(s) can predict "
        "root cause with near-perfect accuracy on their own, indicating the "
        "simulator encodes the answer:\n" + "\n".join(violations)
    )


def test_no_field_is_bijective_with_cause():
    """Stronger structural check: no field value should map to exactly one
    cause across the entire dataset. A perfectly bijective mapping is the
    hallmark of a label leak even if overall accuracy is below the ceiling."""
    dataset = generate_dataset(count=100, ambiguous_fraction=0.0, seed=99)
    clear = [
        inc for inc in dataset["incidents"]
        if not inc["ground_truth"]["is_ambiguous"]
    ]
    causes = [inc["ground_truth"]["cause"] for inc in clear]
    fields = _extract_evidence_fields(clear)

    for field_name, values in fields.items():
        if all(v is None for v in values):
            continue
        value_causes: dict[str, set] = defaultdict(set)
        value_counts: dict[str, int] = Counter()
        for value, cause in zip(values, causes):
            key = str(value)
            value_causes[key].add(cause)
            value_counts[key] += 1

        for value_str, cause_set in value_causes.items():
            if value_str in ("None",):
                continue
            count = value_counts[value_str]
            # Allow rare values (< 5 occurrences) to be single-cause
            if count < 5:
                continue
            assert len(cause_set) > 1, (
                f"Label leakage: {field_name}={value_str} maps exclusively to "
                f"cause={cause_set.pop()} across {count} incidents"
            )


def test_error_code_distribution_has_overlap():
    """Direct check that the simulator's error code distribution produces
    each code from multiple causes, not just one."""
    dataset = generate_dataset(count=200, ambiguous_fraction=0.0, seed=123)
    clear = [
        inc for inc in dataset["incidents"]
        if not inc["ground_truth"]["is_ambiguous"]
    ]

    code_to_causes: dict[str, set] = defaultdict(set)
    for inc in clear:
        cause = inc["ground_truth"]["cause"]
        for trace in inc["error_traces"]:
            code_to_causes[trace["error_code"]].add(cause)

    # Every code that appears in ERROR_SIGNATURES (the correlator's lookup)
    # must be produced by at least 2 different causes
    for code in ERROR_SIGNATURES:
        if code not in code_to_causes:
            continue  # Code not generated is fine
        assert len(code_to_causes[code]) >= 2, (
            f"Error code {code} is only produced by cause "
            f"{code_to_causes[code]}; this enables a 1:1 lookup leak"
        )


def test_cause_specific_signals_have_cross_cause_overlap():
    """Deploy overlaps, config changes, route health alerts, and network
    alerts must each appear for at least 2 different causes."""
    dataset = generate_dataset(count=200, ambiguous_fraction=0.0, seed=456)
    clear = [
        inc for inc in dataset["incidents"]
        if not inc["ground_truth"]["is_ambiguous"]
    ]

    signal_causes: dict[str, set] = {
        "deploy_overlap": set(),
        "config_change": set(),
        "route_health_failed": set(),
        "network_alert": set(),
        "webhook_degraded": set(),
    }

    for inc in clear:
        cause = inc["ground_truth"]["cause"]
        detection = detect_degradations(inc)
        primary = detection.get("primary_degradation")
        if not primary:
            continue
        method = primary["sub_type"]
        route = primary["route"]

        if any(
            d.get("event_type") == "deploy"
            and d.get("affected_method") == method
            and d.get("affected_route") == route
            for d in inc["deploy_logs"]
        ):
            signal_causes["deploy_overlap"].add(cause)

        if any(
            d.get("event_type") == "config_change"
            and d.get("affected_method") == method
            and d.get("affected_route") == route
            for d in inc["deploy_logs"]
        ):
            signal_causes["config_change"].add(cause)

        if any(
            a["metric"] == f"route_health:{route}"
            and a["threshold_breached"] == "healthcheck_failed"
            for a in inc["alerts"]
        ):
            signal_causes["route_health_failed"].add(cause)

        if any(
            a["metric"].startswith("network_packet_loss")
            for a in inc["alerts"]
        ):
            signal_causes["network_alert"].add(cause)

        if any(
            w["delivery_status"] in {"failed", "timed_out"}
            for w in inc["webhook_events"]
        ):
            signal_causes["webhook_degraded"].add(cause)

    for signal, causes_seen in signal_causes.items():
        assert len(causes_seen) >= 2, (
            f"Signal '{signal}' only appears for cause(s) {causes_seen}; "
            "it must appear for at least 2 different causes to prevent "
            "a 1:1 lookup leak"
        )
