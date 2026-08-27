import unittest

from data.simulate import DEFAULT_INCIDENT_COUNT, generate_dataset
from src.pipeline import run_pipeline
from src.recovery import ASSUMED_RECOVERY_SUCCESS_RATE, MIN_CONFIDENCE_FOR_AUTO_ACTION


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)
        cls.output = run_pipeline(cls.dataset["incidents"])

    def test_dataset_shape_and_sources(self):
        self.assertEqual(len(self.dataset["incidents"]), 60)
        self.assertEqual(self.dataset["metadata"]["ambiguous_incident_count"], 9)
        for incident in self.dataset["incidents"]:
            self.assertTrue(incident["payment_events"])
            self.assertTrue(incident["deploy_logs"])
            self.assertTrue(incident["alerts"])
            self.assertTrue(incident["webhook_events"])
            self.assertTrue(incident["error_traces"])

    def test_detector_quality_floor(self):
        matches = 0
        for record in self.output:
            primary = record["detection"]["primary_degradation"]
            truth = record["ground_truth"]
            if primary and (primary["sub_type"], primary["route"]) == (
                truth["affected_method"],
                truth["affected_route"],
            ):
                matches += 1
        self.assertGreaterEqual(matches / len(self.output), 0.95)

    def test_ambiguous_cases_are_honest_and_gated(self):
        ambiguous = [record for record in self.output if record["ground_truth"]["is_ambiguous"]]
        self.assertTrue(ambiguous)
        for record in ambiguous:
            self.assertEqual(record["correlation"]["predicted_cause"], "unresolved")
            self.assertLess(record["correlation"]["confidence"], MIN_CONFIDENCE_FOR_AUTO_ACTION)
            self.assertEqual(record["recovery"]["primary_action"], "escalate to human")
            self.assertFalse(record["recovery"]["merchant_notification_sent"])

    def test_impact_math_and_audit_schema(self):
        for record in self.output:
            impact = record["impact"]
            self.assertEqual(
                impact["recovered_amount_inr"],
                round(impact["recoverable_gmv_inr"] * ASSUMED_RECOVERY_SUCCESS_RATE),
            )
            self.assertIn("MODELING ASSUMPTION", impact["recovered_amount_basis"])
            self.assertTrue(record["audit_trail"])
            for entry in record["audit_trail"]:
                self.assertLessEqual(
                    {"incident_id", "action", "reason", "bounded_by", "timestamp"}, set(entry)
                )


if __name__ == "__main__":
    unittest.main()
