"""Pattern recall correctness, with causal honesty as the headline property."""

import copy

from data.simulate import generate_dataset
from src.memory import (
    SIMILARITY_THRESHOLD,
    IncidentMemory,
    build_feature_vector,
    cosine_similarity,
    extract_features,
)
from src.pipeline import run_pipeline


def _synthetic_incident(dataset_incident, incident_id):
    """Clone a real incident under a new id so the pipeline still accepts it."""
    clone = copy.deepcopy(dataset_incident)
    clone["incident_id"] = incident_id
    for event in clone["payment_events"]:
        event["payment_id"] = event["payment_id"].replace(
            dataset_incident["incident_id"].lower().replace("-", ""),
            incident_id.lower().replace("-", ""),
        )
    return clone


def test_first_incident_in_a_batch_has_no_similar_matches():
    """The key causal-honesty check: incident 0 cannot see anything."""
    for seed in (11, 202, 3003):
        records = run_pipeline(
            generate_dataset(count=20, ambiguous_fraction=0.15, seed=seed)["incidents"]
        )
        first = records[0]["pattern_recall"]

        assert first["prior_incidents_considered"] == 0
        assert first["match_count"] == 0
        assert first["matches"] == []
        assert first["resolution_track_record"] is None
        assert first["supporting_evidence"] is None


def test_no_incident_can_recall_an_incident_that_comes_after_it():
    """Every match must point strictly backwards in processing order."""
    records = run_pipeline(
        generate_dataset(count=30, ambiguous_fraction=0.15, seed=707)["incidents"]
    )
    order = {record["incident_id"]: index for index, record in enumerate(records)}

    for index, record in enumerate(records):
        recall = record["pattern_recall"]
        assert recall["prior_incidents_considered"] == index
        for match in recall["matches"]:
            assert order[match["incident_id"]] < index


def test_near_identical_second_incident_surfaces_the_first():
    """Same method/route/failure_reason must produce a high-similarity match."""
    dataset = generate_dataset(count=10, ambiguous_fraction=0.0, seed=909)
    source = dataset["incidents"][0]
    first = _synthetic_incident(source, "INC-8001")
    second = _synthetic_incident(source, "INC-8002")

    records = run_pipeline([first, second])
    first_recall = records[0]["pattern_recall"]
    second_recall = records[1]["pattern_recall"]

    assert first_recall["match_count"] == 0

    assert second_recall["match_count"] == 1
    match = second_recall["matches"][0]
    assert match["incident_id"] == "INC-8001"
    assert match["similarity"] >= SIMILARITY_THRESHOLD
    assert {"method", "route", "failure_reason"} <= set(match["matched_on"])
    assert match["root_cause"] == records[0]["correlation"]["predicted_cause"]
    assert match["action_taken"] == records[0]["recovery"]["primary_action"]
    assert match["outcome"]
    assert "INC-8001" in second_recall["supporting_evidence"]


def test_a_different_route_does_not_clear_the_threshold():
    features = {
        "method": "upi_collect",
        "route": "abc@upi",
        "failure_reason": "gateway_error",
        "concentration_pct": 90.0,
        "deploy_overlap": False,
    }
    other_route = {**features, "route": "swift@upi"}
    other_reason = {**features, "failure_reason": "bank_timeout"}

    identical = cosine_similarity(
        build_feature_vector(features), build_feature_vector(features)
    )
    assert round(identical, 6) == 1.0
    assert (
        cosine_similarity(build_feature_vector(features), build_feature_vector(other_route))
        < SIMILARITY_THRESHOLD
    )
    assert (
        cosine_similarity(build_feature_vector(features), build_feature_vector(other_reason))
        < SIMILARITY_THRESHOLD
    )


def test_resolution_track_record_counts_only_prior_cases():
    memory = IncidentMemory()
    resolved_recovery = {
        "primary_action": "reroute traffic",
        "auto_action_taken": True,
        "modeled_recovered_amount_inr": 0,
    }
    escalated_recovery = {
        "primary_action": "escalate to human",
        "auto_action_taken": False,
        "modeled_recovered_amount_inr": 0,
    }
    impact = {"assumed_recovery_success_rate": 0.35}
    features = {
        "method": "netbanking",
        "route": "ICICI Bank",
        "failure_reason": "bank_timeout",
        "concentration_pct": 88.0,
        "deploy_overlap": False,
    }

    for index in range(3):
        memory.remember(
            f"INC-70{index:02d}",
            features,
            {"predicted_cause": "bank_psp_downtime", "confidence": 0.9},
            resolved_recovery,
            impact,
        )
    memory.remember(
        "INC-7099",
        features,
        {"predicted_cause": "bank_psp_downtime", "confidence": 0.7},
        escalated_recovery,
        impact,
    )

    track = memory.recall(features, "bank_psp_downtime")["resolution_track_record"]
    assert track["dominant_action"] == "reroute traffic"
    assert track["resolved_count"] == 3
    assert track["prior_case_count"] == 4
    assert track["statement"] == "reroute traffic resolved bank_psp_downtime in 3/4 prior cases"
    assert "not a measured payment success rate" in track["basis"]

    # A cause with no prior history has no track record at all.
    assert memory.recall(features, "config_change")["resolution_track_record"] is None


def test_recall_never_changes_the_diagnosis():
    """Recall is evidence only: batch diagnoses must match per-incident ones."""
    incidents = generate_dataset(count=15, ambiguous_fraction=0.2, seed=515)["incidents"]
    with_memory = run_pipeline(incidents)

    from src.pipeline import run_incident

    for incident, record in zip(incidents, with_memory):
        isolated = run_incident(incident)  # no memory passed in
        assert isolated["correlation"]["predicted_cause"] == record["correlation"]["predicted_cause"]
        assert isolated["correlation"]["confidence"] == record["correlation"]["confidence"]
        assert isolated["recovery"]["primary_action"] == record["recovery"]["primary_action"]
        assert isolated["pattern_recall"]["match_count"] == 0
        assert record["pattern_recall"]["influences_diagnosis"] is False


def test_extract_features_survives_an_undetected_incident():
    detection = {"primary_degradation": None}
    correlation = {"evidence": {}}
    features = extract_features(detection, correlation, "none")

    assert features == {
        "method": "none",
        "route": "none",
        "failure_reason": "none",
        "concentration_pct": 0.0,
        "deploy_overlap": False,
    }
