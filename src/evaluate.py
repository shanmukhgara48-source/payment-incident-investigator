"""Run and score the full synthetic incident pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .pipeline import run_pipeline
    from .recovery import ASSUMED_RECOVERY_SUCCESS_RATE
except ImportError:  # Supports `python src/evaluate.py`.
    from pipeline import run_pipeline
    from recovery import ASSUMED_RECOVERY_SUCCESS_RATE


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "incidents.json"
DEFAULT_RESULTS = ROOT / "results.json"


def evaluate(data_path: Path = DEFAULT_DATA, output_path: Path = DEFAULT_RESULTS) -> dict:
    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    records = run_pipeline(dataset["incidents"])
    clear = [record for record in records if not record["ground_truth"]["is_ambiguous"]]
    ambiguous = [record for record in records if record["ground_truth"]["is_ambiguous"]]

    detection_matches = [
        record
        for record in records
        if record["detection"]["detected"]
        and record["detection"]["primary_degradation"]["sub_type"]
        == record["ground_truth"]["affected_method"]
        and record["detection"]["primary_degradation"]["route"]
        == record["ground_truth"]["affected_route"]
    ]
    correct_diagnoses = [
        record
        for record in clear
        if record["correlation"]["predicted_cause"] == record["ground_truth"]["cause"]
    ]
    honest_ambiguous = [
        record for record in ambiguous if record["correlation"]["predicted_cause"] == "unresolved"
    ]
    misdiagnoses = [
        {
            "incident_id": record["incident_id"],
            "exception_type": "misdiagnosis",
            "expected_cause": record["ground_truth"]["cause"],
            "predicted_cause": record["correlation"]["predicted_cause"],
            "confidence": record["correlation"]["confidence"],
        }
        for record in clear
        if record["correlation"]["predicted_cause"] != record["ground_truth"]["cause"]
    ]
    detection_exceptions = [
        {
            "incident_id": record["incident_id"],
            "exception_type": "detection_miss_or_wrong_pair",
            "expected_method": record["ground_truth"]["affected_method"],
            "expected_route": record["ground_truth"]["affected_route"],
            "detected_pair": record["detection"].get("primary_degradation"),
        }
        for record in records
        if record not in detection_matches
    ]
    false_confident_ambiguous = [
        {
            "incident_id": record["incident_id"],
            "exception_type": "false_confident_on_ambiguous",
            "predicted_cause": record["correlation"]["predicted_cause"],
            "confidence": record["correlation"]["confidence"],
        }
        for record in ambiguous
        if record["correlation"]["predicted_cause"] != "unresolved"
    ]

    aggregate = {
        "incident_count": len(records),
        "clear_incident_count": len(clear),
        "ambiguous_incident_count": len(ambiguous),
        "detection_accuracy": len(detection_matches) / len(records),
        "root_cause_accuracy_clear_cases": len(correct_diagnoses) / max(1, len(clear)),
        "honesty_rate_on_ambiguous_cases": len(honest_ambiguous) / max(1, len(ambiguous)),
        "total_attempted_gmv_inr": sum(record["impact"]["attempted_gmv_inr"] for record in records),
        "total_failed_gmv_inr": sum(record["impact"]["failed_gmv_inr"] for record in records),
        "total_recoverable_gmv_inr": sum(record["impact"]["recoverable_gmv_inr"] for record in records),
        "total_recovered_amount_inr": sum(record["impact"]["recovered_amount_inr"] for record in records),
        "recovered_amount_basis": (
            f"MODELING ASSUMPTION: {ASSUMED_RECOVERY_SUCCESS_RATE:.0%} success rate applied to "
            "recoverable_gmv_inr; not a measured statistic."
        ),
        "escalation_count": sum(
            record["recovery"]["primary_action"] == "escalate to human" for record in records
        ),
        "high_value_payment_escalation_count": sum(
            record["impact"]["above_auto_retry_cap_count"] for record in records
        ),
        "merchant_notification_count": sum(
            record["recovery"]["merchant_notification_sent"] for record in records
        ),
        "misdiagnosis_count": len(misdiagnoses),
        "detection_exception_count": len(detection_exceptions),
        "false_confident_ambiguous_count": len(false_confident_ambiguous),
    }
    all_exceptions = detection_exceptions + misdiagnoses + false_confident_ambiguous
    results = {
        "metadata": {
            **dataset["metadata"],
            "track": "AI Revenue Recovery",
            "pipeline": "alert -> root cause -> business impact -> safe recovery action -> measured result",
            "full_dataset_evaluated": True,
        },
        "assumptions": {
            "recovery_success_rate": ASSUMED_RECOVERY_SUCCESS_RATE,
            "recovery_success_rate_label": "Modeling assumption only; not a real measured Razorpay statistic.",
        },
        "aggregate_metrics": aggregate,
        "example_rca_outputs": [record["rca_text"] for record in records[:3]],
        "exceptions": all_exceptions,
        "incidents": records,
    }
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def _money(value: int) -> str:
    return f"INR {value:,}"


def print_report(results: dict) -> None:
    metrics = results["aggregate_metrics"]
    print("PAYMENT INCIDENT INVESTIGATOR + REVENUE RECOVERY COMMANDER")
    print(f"Incidents evaluated: {metrics['incident_count']} (full dataset)")
    print(f"Detection accuracy: {metrics['detection_accuracy']:.1%}")
    print(f"Root-cause accuracy, clear cases: {metrics['root_cause_accuracy_clear_cases']:.1%}")
    print(f"Honesty rate, ambiguous cases: {metrics['honesty_rate_on_ambiguous_cases']:.1%}")
    print(f"Attempted GMV: {_money(metrics['total_attempted_gmv_inr'])}")
    print(f"Failed GMV: {_money(metrics['total_failed_gmv_inr'])}")
    print(f"Recoverable GMV: {_money(metrics['total_recoverable_gmv_inr'])}")
    print(
        f"Modeled recovered amount: {_money(metrics['total_recovered_amount_inr'])} "
        f"(ASSUMPTION: {results['assumptions']['recovery_success_rate']:.0%} success rate; not measured)"
    )
    print(f"Incident escalations: {metrics['escalation_count']}")
    print(f"Misdiagnoses: {metrics['misdiagnosis_count']} (full list in results.json)")
    print("\nSample RCA outputs:")
    for text in results["example_rca_outputs"]:
        print(f"- {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    results = evaluate(args.data, args.output)
    print_report(results)
    print(f"\nWrote {len(results['incidents'])} complete incident records to {args.output}")


if __name__ == "__main__":
    main()
