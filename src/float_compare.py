"""Tolerance-based comparisons for human-meaningful decision thresholds.

Binary floating point cannot represent most decimal values exactly, so a
comparison that is correct in decimal arithmetic can be wrong in `float`:

    >>> 0.97 - 0.92
    0.04999999999999993
    >>> (0.97 - 0.92) < 0.05      # decimal says 0.05 < 0.05 is False
    True

That single bit of representation error was enough to make the detector reject
a real degradation (INC-0045: baseline 97/100, current 92/100, a drop of
exactly 0.05 against a `MIN_SUCCESS_RATE_DROP = 0.05` gate).

Every threshold in this project is a human-set decimal policy value -- 0.05,
0.15, 0.60, 0.75, 50, 80. When code asks "is this value at least the
threshold?", it means the decimal question, not the binary one. These helpers
answer the decimal question.

TOLERANCE
    `rel_tol=1e-9` sits roughly eight orders of magnitude above the
    representation error being corrected (~1e-16) and many orders below the
    smallest distinction this domain cares about: confidences carry 2 decimal
    places, success-rate drops 4, cosine similarity 3. So the tolerance
    absorbs binary noise and cannot absorb a real difference.

    `abs_tol=1e-12` keeps the helpers well-behaved when a threshold is 0.0,
    where a relative tolerance alone degenerates.

USE THESE FOR
    Comparing a computed float against a policy threshold.

DO NOT USE THESE FOR
    Invariant guards that must stay strict (see `skeptic.py`, where a clamp
    fires when final confidence exceeds primary confidence -- a tolerant
    comparison would let a tiny excess escape the clamp), integer comparisons,
    or guards against division by zero.
"""

from __future__ import annotations

import math

# See the TOLERANCE note above before changing either value.
DEFAULT_REL_TOL = 1e-9
DEFAULT_ABS_TOL = 1e-12


def close(left: float, right: float, *, rel_tol: float = DEFAULT_REL_TOL,
          abs_tol: float = DEFAULT_ABS_TOL) -> bool:
    """Return whether two floats are equal once representation error is ignored."""
    return math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol)


def gte(value: float, threshold: float, *, rel_tol: float = DEFAULT_REL_TOL,
        abs_tol: float = DEFAULT_ABS_TOL) -> bool:
    """`value >= threshold`, treating near-equal values as equal.

    A value that lands a few ULPs below the threshold because of binary
    representation still counts as meeting it.
    """
    return value >= threshold or close(value, threshold, rel_tol=rel_tol, abs_tol=abs_tol)


def lt(value: float, threshold: float, *, rel_tol: float = DEFAULT_REL_TOL,
       abs_tol: float = DEFAULT_ABS_TOL) -> bool:
    """`value < threshold`, treating near-equal values as equal (so: not below).

    Exact logical complement of `gte`, so a gate written as `lt(...)` and one
    written as `not gte(...)` can never disagree.
    """
    return not gte(value, threshold, rel_tol=rel_tol, abs_tol=abs_tol)


def lte(value: float, threshold: float, *, rel_tol: float = DEFAULT_REL_TOL,
        abs_tol: float = DEFAULT_ABS_TOL) -> bool:
    """`value <= threshold`, treating near-equal values as equal."""
    return value <= threshold or close(value, threshold, rel_tol=rel_tol, abs_tol=abs_tol)


def gt(value: float, threshold: float, *, rel_tol: float = DEFAULT_REL_TOL,
       abs_tol: float = DEFAULT_ABS_TOL) -> bool:
    """`value > threshold`, treating near-equal values as equal (so: not above).

    Exact logical complement of `lte`.
    """
    return not lte(value, threshold, rel_tol=rel_tol, abs_tol=abs_tol)
