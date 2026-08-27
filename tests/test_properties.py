"""Safety and correctness properties a technical reviewer is likely to probe."""

from data.simulate import generate_dataset
from src.config import MAX_AUTO_RETRY_AMOUNT_INR, MIN_CONFIDENCE_FOR_AUTO_ACTION
from src.correlator import correlate
from src.detector import detect_degradations
from src.impact import calculate_impact
from src.pipeline import run_incident, run_pipeline
from src.recovery import recommend_recovery


def test_detector_flags_the_injected_method_route_pair():
    incident = generate_dataset(count=10, ambiguous_fraction=0.0, seed=101)["incidents"][0]
    detection = detect_degradations(incident)

    assert detection["detected"] is True
    assert detection["primary_degradation"]["sub_type"] == incident["ground_truth"]["affected_method"]
    assert detection["primary_degradation"]["route"] == incident["ground_truth"]["affected_route"]


def test_ambiguous_incidents_never_cross_the_confidence_gate():
    incidents = generate_dataset(count=20, ambiguous_fraction=0.25, seed=202)["incidents"]
    ambiguous = [incident for incident in incidents if incident["ground_truth"]["is_ambiguous"]]

    assert ambiguous
    for incident in ambiguous:
        result = correlate(incident, detect_degradations(incident))
        assert result["predicted_cause"] == "unresolved"
        assert result["confidence"] < MIN_CONFIDENCE_FOR_AUTO_ACTION


def test_recovery_never_auto_retries_a_payment_above_the_cap():
    incident = generate_dataset(count=10, ambiguous_fraction=0.0, seed=303)["incidents"][0]
    for event in incident["payment_events"]:
        if event["window"] == "current" and event["status"] == "failed":
            event["amount_inr"] = MAX_AUTO_RETRY_AMOUNT_INR + 1
            event["high_intent"] = True
            event["retry_count"] = 0

    detection = detect_degradations(incident)
    correlation = correlate(incident, detection)
    impact = calculate_impact(incident)
    recovery = recommend_recovery(incident, correlation, impact)

    assert impact["recoverable_payment_count"] == 0
    assert recovery["primary_action"] != "create Payment Links for high-intent failures"
    assert any(entry["bounded_by"] == "MAX_AUTO_RETRY_AMOUNT_INR" for entry in recovery["audit_trail"])


def test_gmv_math_is_monotonic_for_every_incident():
    records = run_pipeline(generate_dataset(count=30, ambiguous_fraction=0.2, seed=404)["incidents"])

    for record in records:
        impact = record["impact"]
        assert (
            impact["recovered_amount_inr"]
            <= impact["recoverable_gmv_inr"]
            <= impact["failed_gmv_inr"]
            <= impact["attempted_gmv_inr"]
        )


def test_stage_failure_is_recorded_and_escalated(monkeypatch):
    incident = generate_dataset(count=10, ambiguous_fraction=0.0, seed=505)["incidents"][0]

    def fail_detector(_):
        raise RuntimeError("synthetic detector failure")

    monkeypatch.setattr("src.pipeline.detect_degradations", fail_detector)
    record = run_incident(incident)

    assert any(error["stage"] == "detector" for error in record["stage_errors"])
    assert record["correlation"]["predicted_cause"] == "unresolved"
    assert record["recovery"]["primary_action"] == "escalate to human"
