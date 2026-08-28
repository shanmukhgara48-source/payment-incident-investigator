"""Tests for the LLM integration: fallback behaviour, labeling, and invariants.

These tests run WITHOUT a real OPENAI_API_KEY (conftest pins it to "").
They verify that the pipeline works identically in rule-based mode and that
the fallback labeling is correct when an LLM call would fail.
"""

import unittest
from unittest.mock import patch

from data.simulate import DEFAULT_INCIDENT_COUNT, generate_dataset
from src.correlator import VALID_CAUSES, correlate
from src.detector import detect_degradations
from src.llm import get_usage_stats, llm_available, reset_usage_stats
from src.pipeline import run_incident
from src.skeptic import skeptic_review


class TestLLMUnavailableFallback(unittest.TestCase):
    """When OPENAI_API_KEY is empty, everything uses RULE_BASED."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)

    def test_llm_not_available_without_key(self):
        self.assertFalse(llm_available())

    def test_correlator_labels_rule_based(self):
        inc = self.dataset["incidents"][0]
        det = detect_degradations(inc)
        cor = correlate(inc, det)
        self.assertEqual(cor["reasoning_mode"], "RULE_BASED")
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


class TestLLMFallbackOnFailure(unittest.TestCase):
    """When OPENAI_API_KEY is set but the call fails, label is RULE_BASED_FALLBACK."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)

    def test_correlator_falls_back_on_llm_failure(self):
        inc = self.dataset["incidents"][0]
        det = detect_degradations(inc)
        with patch("src.correlator.llm_available", return_value=True), \
             patch("src.correlator.llm_call", return_value=None):
            cor = correlate(inc, det)
        self.assertEqual(cor["reasoning_mode"], "RULE_BASED_FALLBACK")
        self.assertIn(cor["predicted_cause"], VALID_CAUSES)

    def test_skeptic_falls_back_on_llm_failure(self):
        inc = self.dataset["incidents"][0]
        det = detect_degradations(inc)
        cor = correlate(inc, det)
        with patch("src.skeptic.llm_available", return_value=True), \
             patch("src.skeptic.llm_call", return_value=None):
            review = skeptic_review(inc, det, cor)
        self.assertEqual(review["reasoning_mode"], "RULE_BASED_FALLBACK")
        # Skeptic invariant still holds
        self.assertLessEqual(review["final_confidence"], review["primary_confidence"])

    def test_correlator_rejects_invalid_llm_cause(self):
        """If LLM returns an invalid cause, fall back to rule-based."""
        inc = self.dataset["incidents"][0]
        det = detect_degradations(inc)
        bad_response = {
            "predicted_cause": "solar_flare",
            "confidence": 0.80,
            "explanation": "The sun did it",
            "supporting_signals": [],
        }
        with patch("src.correlator.llm_available", return_value=True), \
             patch("src.correlator.llm_call", return_value=bad_response):
            cor = correlate(inc, det)
        self.assertEqual(cor["reasoning_mode"], "RULE_BASED_FALLBACK")
        self.assertIn(cor["predicted_cause"], VALID_CAUSES)

    def test_correlator_accepts_valid_llm_response(self):
        """A well-formed LLM response is used."""
        inc = self.dataset["incidents"][0]
        det = detect_degradations(inc)
        good_response = {
            "predicted_cause": "bad_deploy",
            "confidence": 0.85,
            "explanation": "Deploy overlap with matching error signature.",
            "supporting_signals": ["deploy overlap", "error trace"],
            "_llm_meta": {
                "model": "gpt-4.1-mini",
                "latency_seconds": 1.2,
                "prompt_tokens": 500,
                "completion_tokens": 80,
                "total_tokens": 580,
                "attempt": 1,
            },
        }
        with patch("src.correlator.llm_available", return_value=True), \
             patch("src.correlator.llm_call", return_value=good_response):
            cor = correlate(inc, det)
        self.assertEqual(cor["reasoning_mode"], "LLM_REASONED")
        self.assertEqual(cor["predicted_cause"], "bad_deploy")
        self.assertEqual(cor["confidence"], 0.85)
        self.assertIn("llm_meta", cor)
        self.assertEqual(cor["evidence"]["llm_explanation"], "Deploy overlap with matching error signature.")

    def test_correlator_enforces_confidence_gate_on_llm(self):
        """Even when LLM returns a cause with low confidence, the gate applies."""
        inc = self.dataset["incidents"][0]
        det = detect_degradations(inc)
        low_conf_response = {
            "predicted_cause": "gateway_error",
            "confidence": 0.45,
            "explanation": "Weak gateway signal.",
            "supporting_signals": [],
        }
        with patch("src.correlator.llm_available", return_value=True), \
             patch("src.correlator.llm_call", return_value=low_conf_response):
            cor = correlate(inc, det)
        self.assertEqual(cor["reasoning_mode"], "LLM_REASONED")
        self.assertEqual(cor["predicted_cause"], "unresolved")
        self.assertEqual(cor["confidence"], 0.45)


class TestSkepticLLMInvariants(unittest.TestCase):
    """LLM skeptic must preserve all hard invariants."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)

    def test_llm_skeptic_cannot_raise_confidence(self):
        """Even if LLM tries to raise confidence, the clamp prevents it."""
        inc = self.dataset["incidents"][0]
        det = detect_degradations(inc)
        cor = correlate(inc, det)
        # Simulate LLM returning negative penalty (trying to raise confidence)
        llm_response = {
            "challenges": [],
            "total_penalty": 0.0,
            "summary": "Diagnosis looks solid.",
        }
        with patch("src.skeptic.llm_available", return_value=True), \
             patch("src.skeptic.llm_call", return_value=llm_response):
            review = skeptic_review(inc, det, cor)
        self.assertLessEqual(review["final_confidence"], review["primary_confidence"])

    def test_llm_skeptic_penalties_are_capped(self):
        """Individual penalties capped at 0.15, total at 0.50."""
        inc = self.dataset["incidents"][0]
        det = detect_degradations(inc)
        cor = correlate(inc, det)
        llm_response = {
            "challenges": [
                {"rule": "huge_penalty", "challenge": "Massive problem", "penalty": 5.0},
                {"rule": "another_huge", "challenge": "Another one", "penalty": 3.0},
            ],
            "total_penalty": 8.0,
            "summary": "Everything is wrong.",
        }
        with patch("src.skeptic.llm_available", return_value=True), \
             patch("src.skeptic.llm_call", return_value=llm_response):
            review = skeptic_review(inc, det, cor)
        # Each individual penalty capped at 0.15, total at 0.50
        for ch in review["challenges"]:
            self.assertLessEqual(ch["penalty"], 0.15)
        self.assertLessEqual(review["total_penalty"], 0.50)
        self.assertLessEqual(review["final_confidence"], review["primary_confidence"])


if __name__ == "__main__":
    unittest.main()
