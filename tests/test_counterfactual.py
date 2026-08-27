import unittest

from data.simulate import DEFAULT_INCIDENT_COUNT, generate_dataset
from src.counterfactual import estimate_gmv_saved, MIN_DELAY_MINUTES, MAX_DELAY_MINUTES


class CounterfactualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)

    def test_delay_zero_matches_actual(self):
        """delay=0 must return the actual recorded failed/recoverable values."""
        for incident in self.dataset["incidents"][:10]:
            result = estimate_gmv_saved(incident, 0)
            self.assertEqual(result["delay_minutes"], 0)
            self.assertEqual(
                result["actual_failed_gmv_inr"],
                result["hypothetical_failed_gmv_inr"],
                f"delay=0 should match actual for {incident['incident_id']}",
            )
            self.assertEqual(result["gmv_saved_inr"], 0)
            self.assertEqual(
                result["actual_recoverable_gmv_inr"],
                result["hypothetical_recoverable_gmv_inr"],
            )
            self.assertEqual(
                result["actual_recovered_amount_inr"],
                result["hypothetical_recovered_amount_inr"],
            )

    def test_gmv_saved_never_exceeds_total_failed(self):
        """GMV saved can never exceed the incident's total failed_gmv."""
        for incident in self.dataset["incidents"][:10]:
            for delay in [MIN_DELAY_MINUTES, -10, -5, 0, 5, 10, MAX_DELAY_MINUTES]:
                result = estimate_gmv_saved(incident, delay)
                self.assertGreaterEqual(result["gmv_saved_inr"], 0,
                    f"gmv_saved should be >= 0 for delay={delay}")
                self.assertLessEqual(result["gmv_saved_inr"], result["actual_failed_gmv_inr"],
                    f"gmv_saved should not exceed actual_failed_gmv for delay={delay}")
                self.assertLessEqual(
                    result["hypothetical_failed_gmv_inr"],
                    result["actual_failed_gmv_inr"],
                    f"hypothetical failed should not exceed actual for delay={delay}",
                )
                self.assertGreaterEqual(result["hypothetical_failed_gmv_inr"], 0)

    def test_earlier_detection_saves_more(self):
        """Earlier detection (negative delay) should save at least as much as later."""
        for incident in self.dataset["incidents"][:10]:
            early = estimate_gmv_saved(incident, -10)
            late = estimate_gmv_saved(incident, 10)
            self.assertGreaterEqual(
                early["gmv_saved_inr"],
                late["gmv_saved_inr"],
                f"earlier detection should save more for {incident['incident_id']}",
            )

    def test_out_of_range_raises(self):
        incident = self.dataset["incidents"][0]
        with self.assertRaises(ValueError):
            estimate_gmv_saved(incident, -21)
        with self.assertRaises(ValueError):
            estimate_gmv_saved(incident, 21)

    def test_cumulative_curve_present_and_monotonic(self):
        incident = self.dataset["incidents"][0]
        result = estimate_gmv_saved(incident, 0)
        curve = result["cumulative_failure_curve"]
        self.assertTrue(len(curve) > 0)
        for i in range(1, len(curve)):
            self.assertGreaterEqual(
                curve[i]["cumulative_failed_gmv_inr"],
                curve[i - 1]["cumulative_failed_gmv_inr"],
            )

    def test_disclaimer_present(self):
        incident = self.dataset["incidents"][0]
        result = estimate_gmv_saved(incident, 0)
        self.assertIn("not a guarantee", result["disclaimer"])


if __name__ == "__main__":
    unittest.main()
