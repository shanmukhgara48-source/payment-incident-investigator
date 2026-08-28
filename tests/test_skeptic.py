"""Tests for the skeptic agent — adversarial second-pass review."""

import unittest

from data.simulate import (
    DEFAULT_INCIDENT_COUNT,
    build_skeptic_gate_incident,
    generate_dataset,
)
from src.config import MIN_CONFIDENCE_FOR_AUTO_ACTION
from src.detector import detect_degradations
from src.correlator import correlate
from src.pipeline import run_incident, run_pipeline
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

    def test_generated_ambiguous_cases_pass_through_skeptic_untouched(self):
        """The generator's ambiguous incidents are already unresolved at the
        correlator, so every skeptic rule early-returns on them."""
        generated = [
            r for r in self.records
            if r["ground_truth"]["is_ambiguous"]
            and r["ground_truth"].get("constructed_for") is None
        ]
        self.assertTrue(generated, "Dataset should have generated ambiguous cases")
        for record in generated:
            sr = record["skeptic_review"]
            self.assertEqual(sr["outcome"], "confirmed")
            self.assertEqual(sr["total_penalty"], 0.0)
            self.assertEqual(sr["final_confidence"], sr["primary_confidence"])

    def test_no_ambiguous_case_is_ever_auto_actioned(self):
        """The invariant that actually matters. Unlike the test above, this one
        holds for the constructed skeptic-gate case too -- there the correlator
        IS fooled, and the skeptic is the only thing standing between a 0.60
        bad_deploy diagnosis and an automated recovery action."""
        ambiguous = [r for r in self.records if r["ground_truth"]["is_ambiguous"]]
        self.assertTrue(ambiguous, "Dataset should have ambiguous cases")
        for record in ambiguous:
            self.assertEqual(
                record["correlation"]["predicted_cause"],
                "unresolved",
                f"{record['incident_id']} named a cause on ambiguous evidence",
            )
            self.assertLess(
                record["skeptic_review"]["final_confidence"],
                MIN_CONFIDENCE_FOR_AUTO_ACTION,
                f"{record['incident_id']} finished at or above the auto-action gate",
            )
            self.assertEqual(record["recovery"]["primary_action"], "escalate to human")

    def test_batch_contains_a_diagnosis_the_skeptic_actually_blocked(self):
        """Demo-evidence guard for the gap this case was built to close.

        Before INC-0061 the batch was 54 confirmed / 6 challenged / 0 pushed
        below the gate: the skeptic never once changed an outcome. At least one
        incident must clear the gate on primary confidence and then be driven
        under it by the skeptic, or that claim is unsupported again."""
        blocked = [
            r for r in self.records
            if r["skeptic_review"]["primary_confidence"] >= MIN_CONFIDENCE_FOR_AUTO_ACTION
            and r["skeptic_review"]["final_confidence"] < MIN_CONFIDENCE_FOR_AUTO_ACTION
        ]
        self.assertTrue(
            blocked,
            "No incident is blocked by the skeptic; the confidence gate is "
            "untested by the dataset.",
        )
        for record in blocked:
            self.assertEqual(record["skeptic_review"]["outcome"], "challenged")
            self.assertEqual(record["recovery"]["primary_action"], "escalate to human")


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


class SkepticGateCaseTest(unittest.TestCase):
    """INC-0061: the constructed incident that exercises the confidence gate.

    Pins the exact arithmetic documented in data/simulate.py. If a rule
    penalty or a correlator weight changes, this fails loudly rather than
    silently ceasing to test the gate.
    """

    @classmethod
    def setUpClass(cls):
        cls.incident = build_skeptic_gate_incident(DEFAULT_INCIDENT_COUNT)
        cls.record = run_incident(cls.incident)

    def test_it_is_the_sixty_first_incident_of_the_default_dataset(self):
        dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)
        last = dataset["incidents"][-1]
        self.assertEqual(last["incident_id"], "INC-0061")
        self.assertEqual(
            last["ground_truth"]["constructed_for"], "skeptic_confidence_gate"
        )

    def test_correlator_is_fooled_into_a_thin_bad_deploy_diagnosis(self):
        primary = self.record["primary_diagnosis"]
        self.assertEqual(primary["predicted_cause"], "bad_deploy")
        self.assertEqual(primary["confidence"], 0.60)
        # It clears the resolve bar on deploy overlap + concentration alone,
        # with no error signature behind it.
        self.assertIsNone(primary["evidence"]["dominant_error_code"])

    def test_skeptic_fires_the_two_intended_rules(self):
        sr = self.record["skeptic_review"]
        self.assertEqual(sr["outcome"], "challenged")
        fired = {c["rule"] for c in sr["challenges"]}
        self.assertEqual(fired, {"small_blast_radius", "low_failure_concentration"})
        self.assertEqual(sr["total_penalty"], 0.156)

    def test_final_confidence_lands_below_the_auto_action_gate(self):
        sr = self.record["skeptic_review"]
        self.assertEqual(sr["primary_confidence"], 0.60)
        self.assertEqual(sr["final_confidence"], 0.44)
        self.assertLess(sr["final_confidence"], MIN_CONFIDENCE_FOR_AUTO_ACTION)

    def test_it_escalates_instead_of_auto_actioning(self):
        self.assertEqual(self.record["correlation"]["predicted_cause"], "unresolved")
        self.assertEqual(
            self.record["recovery"]["primary_action"], "escalate to human"
        )
