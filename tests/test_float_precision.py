"""Threshold comparisons must answer the decimal question, not the binary one.

The bug these cover: `0.97 - 0.92` evaluates to `0.04999999999999993`, so a
degradation whose true drop is exactly the `MIN_SUCCESS_RATE_DROP = 0.05` gate
was rejected by the detector (INC-0045, baseline 97/100 -> current 92/100).
"""

from decimal import Decimal
from fractions import Fraction

import pytest

from src.correlator import MIN_MARGIN_OVER_RUNNER_UP, correlate
from src.detector import MIN_SUCCESS_RATE_DROP, detect_degradations
from src.float_compare import close, gt, gte, lt, lte
from src.skeptic import skeptic_review

from data.simulate import generate_dataset


# ── the exact failing values from INC-0045 ──────────────────────────

def test_inc_0045_raw_arithmetic_still_loses_the_bit():
    """Pin the underlying float behaviour so the regression stays legible."""
    assert 0.97 - 0.92 == pytest.approx(0.05)
    assert (0.97 - 0.92) != 0.05
    assert repr(0.97 - 0.92) == "0.04999999999999993"
    # The naive comparison is what shipped, and it is wrong in decimal terms.
    # (Parenthesized: `a < b is True` would chain into `(a < b) and (b is True)`.)
    assert ((0.97 - 0.92) < 0.05) is True


def test_inc_0045_drop_now_meets_the_detector_threshold():
    """97/100 -> 92/100 is a drop of exactly 0.05 and must not be rejected."""
    baseline_rate = 97 / 100
    current_rate = 92 / 100
    drop = baseline_rate - current_rate

    # Exact arithmetic says this drop is precisely the threshold.
    assert Fraction(97, 100) - Fraction(92, 100) == Fraction(5, 100)
    # The naive comparison rejects it...
    assert drop < MIN_SUCCESS_RATE_DROP
    # ...and the tolerance-based one correctly does not.
    assert not lt(drop, MIN_SUCCESS_RATE_DROP)
    assert gte(drop, MIN_SUCCESS_RATE_DROP)


def test_detector_detects_a_pair_whose_drop_is_exactly_the_threshold():
    """End-to-end: the INC-0045 shape reaches the detector and is detected."""
    incident = _incident_with_rates(successes_baseline=97, successes_current=92)
    detection = detect_degradations(incident)

    assert detection["detected"] is True
    primary = detection["primary_degradation"]
    assert primary["success_rate_drop"] == pytest.approx(0.05)
    assert primary["baseline_success_rate"] == pytest.approx(0.97)
    assert primary["current_success_rate"] == pytest.approx(0.92)


def test_detector_still_rejects_a_drop_genuinely_below_the_threshold():
    """The fix must not turn the gate into a rubber stamp."""
    # 97/100 -> 93/100 is a drop of 0.04, genuinely under the gate.
    incident = _incident_with_rates(successes_baseline=97, successes_current=93)
    detection = detect_degradations(incident)

    assert detection["detected"] is False
    assert detection["primary_degradation"] is None


# ── general property: correctness at threshold boundaries ───────────

# Thresholds actually used as decision gates in this project.
PROJECT_THRESHOLDS = [0.05, 0.15, 0.30, 0.60, 0.75, 0.80]


def _decimal_pairs_summing_to(threshold: float):
    """Float pairs (a, b) whose exact decimal difference is `threshold`."""
    step = Decimal("0.01")
    target = Decimal(str(threshold))
    pairs = []
    value = target
    while value <= Decimal("1.00"):
        pairs.append((float(value), float(value - target)))
        value += step
    return pairs


@pytest.mark.parametrize("threshold", PROJECT_THRESHOLDS)
def test_difference_equal_to_threshold_is_never_treated_as_below(threshold):
    """For every a-b whose true difference IS the threshold, `lt` says False.

    This is the general form of the INC-0045 bug: any decimal pair can land a
    few ULPs low after subtraction.
    """
    pairs = _decimal_pairs_summing_to(threshold)
    assert pairs, "expected at least one boundary pair to test"

    for high, low in pairs:
        difference = high - low
        exact = Decimal(str(high)) - Decimal(str(low))
        assert exact == Decimal(str(threshold))

        # The mathematically correct answers, at the boundary:
        assert not lt(difference, threshold), f"{high} - {low} wrongly below {threshold}"
        assert gte(difference, threshold), f"{high} - {low} wrongly fails >= {threshold}"


def test_the_boundary_suite_actually_exercises_the_bug():
    """Guard the test above: prove the naive comparison fails on this corpus.

    Without this, `test_difference_equal_to_threshold_is_never_treated_as_below`
    could pass with the fix reverted and prove nothing. Note that not every
    threshold triggers it -- 0.75 is exactly representable in binary, so no pair
    at that threshold misbehaves. The property must hold everywhere; the bug
    only needs to appear somewhere.
    """
    naive_failures = [
        (threshold, high, low)
        for threshold in PROJECT_THRESHOLDS
        for high, low in _decimal_pairs_summing_to(threshold)
        if (high - low) < threshold
    ]

    assert naive_failures, "corpus contains no float error; it cannot catch a regression"
    # The specific pair from INC-0045 is in there.
    assert (0.05, 0.97, 0.92) in naive_failures


@pytest.mark.parametrize("threshold", PROJECT_THRESHOLDS)
def test_values_meaningfully_below_threshold_still_compare_as_below(threshold):
    """The tolerance must not swallow a real difference."""
    for delta in (0.01, 0.001, 0.0001):
        value = threshold - delta
        assert lt(value, threshold), f"{value} should be below {threshold}"
        assert not gte(value, threshold)


# ── helper semantics ────────────────────────────────────────────────

def test_lt_and_gte_are_exact_complements():
    for value in (0.0, 0.05, 0.1499999, 0.15, 0.6, 0.75, 1.0, 0.97 - 0.92):
        for threshold in PROJECT_THRESHOLDS:
            assert lt(value, threshold) is not gte(value, threshold)


def test_gt_and_lte_are_exact_complements():
    for value in (0.0, 0.05, 0.1499999, 0.15, 0.6, 0.75, 1.0, 0.97 - 0.92):
        for threshold in PROJECT_THRESHOLDS:
            assert gt(value, threshold) is not lte(value, threshold)


def test_close_absorbs_representation_error_but_not_real_differences():
    assert close(0.97 - 0.92, 0.05)
    assert close(0.1 + 0.2, 0.3)
    assert not close(0.0501, 0.05)
    assert not close(0.6, 0.61)


def test_tolerance_handles_a_zero_threshold():
    """A relative tolerance alone degenerates at 0.0; abs_tol must cover it."""
    assert close(0.0, 0.0)
    assert gte(0.0, 0.0)
    assert not lt(0.0, 0.0)


# ── the deliberate exception: the skeptic's clamp stays strict ───────

def test_skeptic_confidence_clamp_remains_strict():
    """The skeptic may only hold or lower confidence, with no tolerance slack."""
    incidents = generate_dataset(count=15, ambiguous_fraction=0.2, seed=707)["incidents"]

    for incident in incidents:
        detection = detect_degradations(incident)
        correlation = correlate(incident, detection)
        review = skeptic_review(incident, detection, correlation)
        assert review["final_confidence"] <= review["primary_confidence"]


# ── correlator margin gate ──────────────────────────────────────────

def test_correlator_margin_gate_uses_the_shared_threshold():
    assert MIN_MARGIN_OVER_RUNNER_UP == 0.15
    # A margin of exactly 0.15 must count as meeting the gate.
    assert gte(0.75 - 0.60, MIN_MARGIN_OVER_RUNNER_UP)
    assert gte(0.55 - 0.40, MIN_MARGIN_OVER_RUNNER_UP)
    assert not lt(0.95 - 0.80, MIN_MARGIN_OVER_RUNNER_UP)


# ── helpers ─────────────────────────────────────────────────────────

def _incident_with_rates(*, successes_baseline: int, successes_current: int) -> dict:
    """Build a one-pair incident with exact 100-attempt success rates."""
    # The simulator enforces a minimum batch of 10; we only need the first.
    incident = generate_dataset(count=10, ambiguous_fraction=0.0, seed=20260827)["incidents"][0]
    template = incident["payment_events"][0]

    def events(window: str, successes: int) -> list[dict]:
        built = []
        for index in range(100):
            event = dict(template)
            event["window"] = window
            event["status"] = "success" if index < successes else "failed"
            event["payment_id"] = f"pay_test_{window}_{index:03d}"
            event["failure_reason"] = "none" if index < successes else "bank_timeout"
            built.append(event)
        return built

    incident["payment_events"] = (
        events("baseline", successes_baseline) + events("current", successes_current)
    )
    return incident
