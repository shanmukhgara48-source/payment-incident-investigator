"""Tests for the skeptic agent — adversarial second-pass review."""

import unittest

from data.simulate import DEFAULT_INCIDENT_COUNT, generate_dataset
from src.detector import detect_degradations
from src.correlator import correlate
from src.pipeline import run_pipeline
from src.skeptic import skeptic_review


class SkepticInvariantTest(unittest.TestCase):
    """Hard invariant: final_confidence <= primary confidence for EVERY incident."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)
        cls.records = run_pipeline(cls.dataset["incidents"])

    def test_final_confidence_never_exceeds_primary(self):
        for record in self.records:
            primary_conf = record["primary_diagnosis"]["confidence"]
            final_conf = record["skeptic_review"]["final_confidence"]
            self.assertLessEqual(
                final_conf,
                primary_conf,
                f"{record['incident_id']}: final_confidence {final_conf} > "
                f"primary confidence {primary_conf}",
            )

    def test_final_confidence_matches_adjusted_correlation(self):
        """The adjusted correlation stored in record['correlation'] must use final_confidence."""
        for record in self.records:
            self.assertEqual(
                record["correlation"]["confidence"],
                record["skeptic_review"]["final_confidence"],
                f"{record['incident_id']}: correlation confidence does not match "
                f"skeptic final_confidence",
            )

    def test_primary_diagnosis_preserved_separately(self):
        """primary_diagnosis must be present and distinct from the adjusted correlation."""
        for record in self.records:
            self.assertIn("primary_diagnosis", record)
            self.assertIn("skeptic_review", record)
            self.assertIn("confidence", record["primary_diagnosis"])
            self.assertIn("predicted_cause", record["primary_diagnosis"])

    def test_skeptic_review_has_required_fields(self):
        for record in self.records:
            sr = record["skeptic_review"]
            for field in [
                "outcome",
                "summary",
                "primary_confidence",
                "total_penalty",
                "final_confidence",
                "checks_performed",
                "challenges_raised",
                "checks",
                "challenges",
            ]:
                self.assertIn(field, sr, f"{record['incident_id']} missing skeptic field: {field}")
            self.assertIn(sr["outcome"], {"confirmed", "challenged"})
            self.assertGreaterEqual(sr["checks_performed"], 5, "Should run at least 5 rules")

    def test_ambiguous_cases_unaffected_by_skeptic(self):
        """Ambiguous (unresolved) cases should pass through with zero penalty."""
        ambiguous = [r for r in self.records if r["ground_truth"]["is_ambiguous"]]
        self.assertTrue(ambiguous, "Dataset should have ambiguous cases")
        for record in ambiguous:
            sr = record["skeptic_review"]
            self.assertEqual(sr["outcome"], "confirmed")
            self.assertEqual(sr["total_penalty"], 0.0)
            self.assertEqual(sr["final_confidence"], sr["primary_confidence"])


class SkepticChallengeTest(unittest.TestCase):
    """Test that the skeptic actually catches weak diagnoses — not just that it runs."""

    def test_small_rollout_deploy_is_challenged(self):
        """A bad_deploy with rollout_pct <= 25% should be challenged by the
        small_blast_radius rule."""
        dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)
        # INC-0011 is a bad_deploy with 25% rollout in the default dataset
        inc_0011 = dataset["incidents"][10]  # index 10 = INC-0011
        self.assertEqual(inc_0011["incident_id"], "INC-0011")
        self.assertEqual(inc_0011["ground_truth"]["cause"], "bad_deploy")

        detection = detect_degradations(inc_0011)
        correlation = correlate(inc_0011, detection)
        self.assertEqual(correlation["predicted_cause"], "bad_deploy")
        primary_conf = correlation["confidence"]

        review = skeptic_review(inc_0011, detection, correlation)

        self.assertEqual(review["outcome"], "challenged")
        self.assertGreater(review["total_penalty"], 0)
        self.assertLess(review["final_confidence"], primary_conf)

        rule_names = [c["rule"] for c in review["challenges"]]
        self.assertIn("small_blast_radius", rule_names)

    def test_low_concentration_is_challenged(self):
        """An incident with failure concentration < 80% should be challenged."""
        dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)
        # INC-0026 has concentration 75.0%
        inc_0026 = dataset["incidents"][25]  # index 25 = INC-0026
        self.assertEqual(inc_0026["incident_id"], "INC-0026")

        detection = detect_degradations(inc_0026)
        correlation = correlate(inc_0026, detection)
        primary_conf = correlation["confidence"]

        review = skeptic_review(inc_0026, detection, correlation)

        self.assertEqual(review["outcome"], "challenged")
        self.assertLess(review["final_confidence"], primary_conf)

        rule_names = [c["rule"] for c in review["challenges"]]
        self.assertIn("low_failure_concentration", rule_names)

    def test_stacked_penalties_on_weak_deploy(self):
        """INC-0031 has both small rollout AND low concentration — both rules should fire."""
        dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)
        inc_0031 = dataset["incidents"][30]  # index 30 = INC-0031
        self.assertEqual(inc_0031["incident_id"], "INC-0031")

        detection = detect_degradations(inc_0031)
        correlation = correlate(inc_0031, detection)
        review = skeptic_review(inc_0031, detection, correlation)

        self.assertEqual(review["outcome"], "challenged")
        self.assertGreaterEqual(review["challenges_raised"], 2)
        rule_names = [c["rule"] for c in review["challenges"]]
        self.assertIn("small_blast_radius", rule_names)
        self.assertIn("low_failure_concentration", rule_names)
        # Stacked penalty should be larger than either individual rule
        self.assertGreater(review["total_penalty"], 0.10)

    def test_strong_incident_is_confirmed(self):
        """INC-0001 has 100% rollout and 89% concentration — should be confirmed."""
        dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)
        inc_0001 = dataset["incidents"][0]
        self.assertEqual(inc_0001["incident_id"], "INC-0001")

        detection = detect_degradations(inc_0001)
        correlation = correlate(inc_0001, detection)
        review = skeptic_review(inc_0001, detection, correlation)

        self.assertEqual(review["outcome"], "confirmed")
        self.assertEqual(review["total_penalty"], 0.0)
        self.assertEqual(review["final_confidence"], correlation["confidence"])


if __name__ == "__main__":
    unittest.main()
