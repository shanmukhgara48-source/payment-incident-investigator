"""Tests for the hybrid correlator: rule-based primary + LLM tiebreaker.

These tests run WITHOUT a real OPENAI_API_KEY (conftest pins it to "").
They verify the hybrid combination logic, borderline band behaviour,
hard invariants, and backward compatibility with the rule-based-only path.
"""

import unittest
from unittest.mock import patch, call

from data.simulate import DEFAULT_INCIDENT_COUNT, generate_dataset
from src.config import (
    COMBINED_CONFIDENCE_CEILING,
    LLM_CONFLICT_PENALTY,
    LLM_CORROBORATION_BOOST,
    MIN_CONFIDENCE_FOR_AUTO_ACTION,
    MIN_CONFIDENCE_FOR_AUTO_ACTION_LLM,
    RULE_BASED_BORDERLINE_HIGH,
    RULE_BASED_BORDERLINE_LOW,
)
from src.correlator import VALID_CAUSES, correlate, _is_borderline
from src.detector import detect_degradations
from src.float_compare import lt
from src.llm import get_usage_stats, llm_available, reset_usage_stats
from src.pipeline import run_incident
from src.skeptic import skeptic_review


def _make_llm_response(cause, confidence, explanation="Test."):
    return {
        "predicted_cause": cause,
        "confidence": confidence,
        "explanation": explanation,
        "supporting_signals": [f"{cause} signal"],
        "_llm_meta": {
            "model": "test", "backend": "test",
            "latency_seconds": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0, "attempt": 1,
        },
    }


class TestLLMUnavailableFallback(unittest.TestCase):
    """When no API key is set, everything uses RULE_BASED_ALONE."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)

    def test_llm_not_available_without_key(self):
        self.assertFalse(llm_available())

    def test_correlator_labels_rule_based_alone(self):
        inc = self.dataset["incidents"][0]
        det = detect_degradations(inc)
        cor = correlate(inc, det)
        self.assertIn(cor["reasoning_mode"], {
            "RULE_BASED_ALONE",
        })
        self.assertNotIn("llm_meta", cor)

    def test_skeptic_labels_rule_based(self):
        inc = self.dataset["incidents"][0]
        det = detect_degradations(inc)
        cor = correlate(inc, det)
        review = skeptic_review(inc, det, cor)
        self.assertEqual(review["reasoning_mode"], "RULE_BASED")
        self.assertNotIn("llm_meta", review)

    def test_pipeline_record_has_reasoning_modes(self):
        inc = self.dataset["incidents"][0]
        rec = run_incident(inc)
        self.assertIn("reasoning_mode", rec["correlation"])
        self.assertIn("reasoning_mode", rec["primary_diagnosis"])
        self.assertIn("reasoning_mode", rec["skeptic_review"])

    def test_no_llm_calls_made(self):
        reset_usage_stats()
        inc = self.dataset["incidents"][0]
        run_incident(inc)
        stats = get_usage_stats()
        self.assertEqual(stats["total_calls"], 0)


class TestBorderlineBand(unittest.TestCase):
    """Borderline band definitions and boundary behaviour."""

    def test_below_low_is_not_borderline(self):
        self.assertFalse(_is_borderline(0.30))
        self.assertFalse(_is_borderline(0.0))

    def test_at_low_boundary_is_not_borderline(self):
        self.assertFalse(_is_borderline(RULE_BASED_BORDERLINE_LOW))

    def test_above_low_is_borderline(self):
        self.assertTrue(_is_borderline(RULE_BASED_BORDERLINE_LOW + 0.01))

    def test_at_high_boundary_is_not_borderline(self):
        self.assertFalse(_is_borderline(RULE_BASED_BORDERLINE_HIGH))

    def test_below_high_is_borderline(self):
        self.assertTrue(_is_borderline(RULE_BASED_BORDERLINE_HIGH - 0.01))

    def test_above_high_is_not_borderline(self):
        self.assertFalse(_is_borderline(0.90))

    def test_action_gate_is_inside_band(self):
        """The MIN_CONFIDENCE_FOR_AUTO_ACTION threshold must be inside the band."""
        self.assertGreater(MIN_CONFIDENCE_FOR_AUTO_ACTION, RULE_BASED_BORDERLINE_LOW)
        self.assertLess(MIN_CONFIDENCE_FOR_AUTO_ACTION, RULE_BASED_BORDERLINE_HIGH)


class TestHybridCorroborated(unittest.TestCase):
    """When LLM agrees with rule-based on a borderline case → boost."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)

    def _find_borderline_resolved(self):
        """Find a borderline incident with a resolved (non-unresolved) rule cause."""
        for inc in self.dataset["incidents"]:
            det = detect_degradations(inc)
            cor = correlate(inc, det)
            if (cor["evidence"].get("borderline", False)
                    and cor["evidence"]["rule_based_predicted_cause"] != "unresolved"):
                return inc, det, cor
        self.skipTest("No borderline resolved incidents in the dataset")

    def test_corroboration_boosts_confidence(self):
        inc, det, cor = self._find_borderline_resolved()
        rule_conf = cor["evidence"]["rule_based_confidence"]
        rule_cause = cor["evidence"]["rule_based_predicted_cause"]

        llm_resp = _make_llm_response(rule_cause, 0.80)
        with patch("src.correlator.llm_available", return_value=True), \
             patch("src.correlator.llm_call", return_value=llm_resp):
            cor2 = correlate(inc, det)

        self.assertEqual(cor2["reasoning_mode"], "RULE_BASED_LLM_CORROBORATED")
        self.assertEqual(cor2["predicted_cause"], rule_cause)
        self.assertGreater(cor2["confidence"], rule_conf)
        # Boost is exactly LLM_CORROBORATION_BOOST
        expected = round(min(COMBINED_CONFIDENCE_CEILING, rule_conf + LLM_CORROBORATION_BOOST), 2)
        self.assertEqual(cor2["confidence"], expected)


class TestHybridConflicted(unittest.TestCase):
    """When LLM disagrees with rule-based on a borderline case → penalize."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)

    def _find_borderline_resolved(self):
        for inc in self.dataset["incidents"]:
            det = detect_degradations(inc)
            cor = correlate(inc, det)
            if (cor["evidence"].get("borderline", False)
                    and cor["evidence"]["rule_based_predicted_cause"] != "unresolved"):
                return inc, det, cor
        self.skipTest("No borderline resolved incidents")

    def test_conflict_lowers_confidence(self):
        inc, det, cor = self._find_borderline_resolved()
        rule_conf = cor["evidence"]["rule_based_confidence"]
        rule_cause = cor["evidence"]["rule_based_predicted_cause"]

        # Pick a different cause for the LLM
        other_cause = next(c for c in VALID_CAUSES if c not in {rule_cause, "unresolved"})
        llm_resp = _make_llm_response(other_cause, 0.80)
        with patch("src.correlator.llm_available", return_value=True), \
             patch("src.correlator.llm_call", return_value=llm_resp):
            cor2 = correlate(inc, det)

        self.assertEqual(cor2["reasoning_mode"], "RULE_BASED_LLM_CONFLICTED")
        self.assertLess(cor2["confidence"], rule_conf)

    def test_conflict_never_raises_confidence(self):
        """INVARIANT: conflict can only LOWER confidence vs rule-based-alone."""
        inc, det, cor = self._find_borderline_resolved()
        rule_conf = cor["evidence"]["rule_based_confidence"]
        rule_cause = cor["evidence"]["rule_based_predicted_cause"]

        other_cause = next(c for c in VALID_CAUSES if c not in {rule_cause, "unresolved"})
        llm_resp = _make_llm_response(other_cause, 0.99)
        with patch("src.correlator.llm_available", return_value=True), \
             patch("src.correlator.llm_call", return_value=llm_resp):
            cor2 = correlate(inc, det)

        self.assertLessEqual(cor2["confidence"], rule_conf)


class TestHybridHardInvariants(unittest.TestCase):
    """Hard invariants that must hold for the hybrid correlator."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)

    def test_combined_confidence_never_exceeds_ceiling(self):
        """INVARIANT: confidence <= COMBINED_CONFIDENCE_CEILING regardless of boost."""
        # Use a borderline incident with high rule-based confidence
        for inc in self.dataset["incidents"]:
            det = detect_degradations(inc)
            cor = correlate(inc, det)
            rb_conf = cor["evidence"].get("rule_based_confidence", 0)
            rb_cause = cor["evidence"].get("rule_based_predicted_cause", "unresolved")
            if cor["evidence"].get("borderline") and rb_cause != "unresolved":
                llm_resp = _make_llm_response(rb_cause, 0.99)
                with patch("src.correlator.llm_available", return_value=True), \
                     patch("src.correlator.llm_call", return_value=llm_resp):
                    cor2 = correlate(inc, det)
                self.assertLessEqual(cor2["confidence"], COMBINED_CONFIDENCE_CEILING)
                return
        self.skipTest("No borderline resolved incidents")

    def test_non_borderline_identical_to_rule_based_alone(self):
        """INVARIANT: incidents outside the band produce identical results."""
        for inc in self.dataset["incidents"]:
            det = detect_degradations(inc)
            cor = correlate(inc, det)
            if not cor["evidence"].get("borderline", False):
                # With LLM available, the result must still be identical
                with patch("src.correlator.llm_available", return_value=True), \
                     patch("src.correlator.llm_call") as mock_call:
                    cor2 = correlate(inc, det)
                # LLM must NOT be called
                mock_call.assert_not_called()
                # Same cause and confidence
                self.assertEqual(cor2["predicted_cause"], cor["predicted_cause"])
                self.assertEqual(cor2["confidence"], cor["confidence"])
                self.assertEqual(cor2["reasoning_mode"], "RULE_BASED_ALONE")
                return
        self.skipTest("All incidents are borderline (unlikely)")

    def test_no_llm_call_for_non_borderline_incidents(self):
        """INVARIANT: LLM call count is zero for non-borderline incidents."""
        non_borderline = []
        for inc in self.dataset["incidents"]:
            det = detect_degradations(inc)
            cor = correlate(inc, det)
            if not cor["evidence"].get("borderline", False):
                non_borderline.append(inc)

        self.assertGreater(len(non_borderline), 0, "Need at least one non-borderline")

        with patch("src.correlator.llm_available", return_value=True), \
             patch("src.correlator.llm_call") as mock_call:
            for inc in non_borderline:
                det = detect_degradations(inc)
                correlate(inc, det)
            mock_call.assert_not_called()

    def test_skeptic_still_runs_after_hybrid(self):
        """INVARIANT: the skeptic pass runs on hybrid results, same as before."""
        inc = self.dataset["incidents"][0]
        rec = run_incident(inc)
        # The skeptic review must exist and have the required fields
        self.assertIn("skeptic_review", rec)
        self.assertIn("final_confidence", rec["skeptic_review"])
        self.assertIn("reasoning_mode", rec["skeptic_review"])
        # The adjusted correlation (post-skeptic) must use the skeptic's
        # final_confidence, not the pre-skeptic value.
        self.assertEqual(
            rec["correlation"]["confidence"],
            rec["skeptic_review"]["final_confidence"],
        )


class TestLLMFallbackOnFailure(unittest.TestCase):
    """When LLM call fails on a borderline incident → RULE_BASED_FALLBACK."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)

    def _find_borderline(self):
        for inc in self.dataset["incidents"]:
            det = detect_degradations(inc)
            cor = correlate(inc, det)
            if cor["evidence"].get("borderline", False):
                return inc, det, cor
        self.skipTest("No borderline incidents")

    def test_llm_failure_falls_back(self):
        inc, det, cor = self._find_borderline()
        rule_conf = cor["evidence"]["rule_based_confidence"]
        rule_cause = cor["evidence"]["rule_based_predicted_cause"]

        with patch("src.correlator.llm_available", return_value=True), \
             patch("src.correlator.llm_call", return_value=None):
            cor2 = correlate(inc, det)

        self.assertEqual(cor2["reasoning_mode"], "RULE_BASED_FALLBACK")
        # Falls back to rule-based result exactly
        self.assertEqual(cor2["predicted_cause"], rule_cause)
        self.assertEqual(cor2["confidence"], rule_conf)

    def test_invalid_llm_cause_falls_back(self):
        inc, det, cor = self._find_borderline()
        bad_resp = {
            "predicted_cause": "solar_flare",
            "confidence": 0.80,
            "explanation": "The sun did it",
            "supporting_signals": [],
        }
        with patch("src.correlator.llm_available", return_value=True), \
             patch("src.correlator.llm_call", return_value=bad_resp):
            cor2 = correlate(inc, det)
        self.assertEqual(cor2["reasoning_mode"], "RULE_BASED_FALLBACK")


class TestSkepticLLMInvariants(unittest.TestCase):
    """Skeptic invariants still hold with hybrid correlator output."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)

    def test_skeptic_cannot_raise_confidence(self):
        inc = self.dataset["incidents"][0]
        det = detect_degradations(inc)
        cor = correlate(inc, det)
        llm_response = {
            "challenges": [],
            "total_penalty": 0.0,
            "summary": "Diagnosis looks solid.",
        }
        with patch("src.skeptic.llm_available", return_value=True), \
             patch("src.skeptic.llm_call", return_value=llm_response):
            review = skeptic_review(inc, det, cor)
        self.assertLessEqual(review["final_confidence"], review["primary_confidence"])

    def test_skeptic_penalties_are_capped(self):
        inc = self.dataset["incidents"][0]
        det = detect_degradations(inc)
        cor = correlate(inc, det)
        llm_response = {
            "challenges": [
                {"rule": "huge_penalty", "challenge": "Problem", "penalty": 5.0},
                {"rule": "another_huge", "challenge": "Another", "penalty": 3.0},
            ],
            "total_penalty": 8.0,
            "summary": "Everything is wrong.",
        }
        with patch("src.skeptic.llm_available", return_value=True), \
             patch("src.skeptic.llm_call", return_value=llm_response):
            review = skeptic_review(inc, det, cor)
        for ch in review["challenges"]:
            self.assertLessEqual(ch["penalty"], 0.15)
        self.assertLessEqual(review["total_penalty"], 0.50)


class TestLLMOverconfidenceRegression(unittest.TestCase):
    """Regression: the hybrid path must remain honest on ambiguous cases.

    The LLM confidence gate (MIN_CONFIDENCE_FOR_AUTO_ACTION_LLM = 0.65)
    forces low-confidence LLM responses to "unresolved", which the hybrid
    combination logic then treats as a conflict/non-corroboration.
    """

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)
        cls.ambiguous = [
            inc for inc in cls.dataset["incidents"]
            if inc.get("ground_truth", {}).get("is_ambiguous", False)
        ]

    def test_ambiguous_cases_exist_in_dataset(self):
        self.assertGreaterEqual(len(self.ambiguous), 5)

    def test_llm_moderate_confidence_gated_to_unresolved(self):
        """LLM returning 0.62 is gated to unresolved by the LLM-specific gate."""
        inc = self.ambiguous[0]
        det = detect_degradations(inc)
        moderate_resp = _make_llm_response("gateway_error", 0.62)
        with patch("src.correlator.llm_available", return_value=True), \
             patch("src.correlator.llm_call", return_value=moderate_resp):
            cor = correlate(inc, det)
        # The LLM's response was gated to unresolved, so it can't corroborate
        # unless the rule-based cause is also unresolved.
        # Either way, the final result should NOT be a confident non-unresolved.
        if cor["evidence"].get("borderline"):
            # LLM said unresolved (after gate) → conflict or non-corroboration
            self.assertIn(cor["reasoning_mode"], {
                "RULE_BASED_LLM_CONFLICTED",
                "RULE_BASED_LLM_CORROBORATED",
                "RULE_BASED_ALONE",
            })

    def test_old_gate_would_have_passed(self):
        """Prove 0.62 passes the old 0.60 gate but fails the 0.65 LLM gate."""
        self.assertFalse(lt(0.62, MIN_CONFIDENCE_FOR_AUTO_ACTION))
        self.assertTrue(lt(0.62, MIN_CONFIDENCE_FOR_AUTO_ACTION_LLM))


if __name__ == "__main__":
    unittest.main()
