"""Skeptic agent: adversarial second-pass review of the primary correlator diagnosis.

The skeptic examines the same evidence the correlator saw and looks for
counter-signals or inconsistencies.  It can only hold or LOWER confidence —
never raise it.  The output records every check performed and whether it
challenged the primary or confirmed it.
"""

from __future__ import annotations

try:
    from .correlator import ERROR_SIGNATURES
except ImportError:
    from correlator import ERROR_SIGNATURES

# ── penalty schedule ────────────────────────────────────────────────
# Each rule that fires returns a tuple (challenge_text, penalty).
# Penalties are subtracted from primary confidence to yield the
# final_confidence.  Multiple rules can fire and their penalties stack,
# but the result is floored at 0.0.


def _rule_small_blast_radius(
    predicted_cause: str,
    incident: dict,
    evidence: dict,
    detection: dict,
) -> tuple[str, float] | None:
    """If diagnosis is bad_deploy but rollout_pct is small, challenge."""
    if predicted_cause != "bad_deploy":
        return None
    primary = detection.get("primary_degradation")
    if not primary:
        return None
    method = primary["sub_type"]
    route = primary["route"]
    deploys = [
        d for d in incident["deploy_logs"]
        if d.get("event_type") == "deploy"
        and d.get("affected_method") == method
        and d.get("affected_route") == route
    ]
    if not deploys:
        return None
    rollout = max(d["rollout_pct"] for d in deploys)
    if rollout <= 25:
        return (
            f"Deploy rollout is only {rollout}% — too small a blast radius "
            "to explain the observed failure concentration.",
            0.10,
        )
    return None


def _rule_error_signature_mismatch(
    predicted_cause: str,
    _incident: dict,
    evidence: dict,
    _detection: dict,
) -> tuple[str, float] | None:
    """If the dominant error code maps to a *different* cause, challenge."""
    error_code = evidence.get("dominant_error_code")
    if not error_code or error_code not in ERROR_SIGNATURES:
        return None
    expected_cause = ERROR_SIGNATURES[error_code][0]
    if expected_cause != predicted_cause:
        side = ERROR_SIGNATURES[error_code][1]
        return (
            f"Dominant error code {error_code} is a {side} signature "
            f"(typical of {expected_cause}), not {predicted_cause}.",
            0.12,
        )
    return None


def _rule_missing_corroborating_health_signal(
    predicted_cause: str,
    _incident: dict,
    evidence: dict,
    _detection: dict,
) -> tuple[str, float] | None:
    """Route-level causes without a matching health check failure are weaker."""
    if predicted_cause == "bank_psp_downtime" and not evidence.get("route_health_failed"):
        return (
            "Diagnosed bank/PSP downtime but no route health check failure was observed.",
            0.08,
        )
    if predicted_cause == "network_issue" and not evidence.get("network_alert"):
        return (
            "Diagnosed network issue but no independent packet-loss alert was observed.",
            0.08,
        )
    return None


def _rule_low_failure_concentration(
    predicted_cause: str,
    _incident: dict,
    evidence: dict,
    _detection: dict,
) -> tuple[str, float] | None:
    """If failure concentration is below 80%, the signal is diluted."""
    if predicted_cause == "unresolved":
        return None
    concentration = evidence.get("concentration_pct", 100)
    if concentration < 80:
        penalty = round(0.04 + 0.06 * ((80 - concentration) / 80), 3)
        return (
            f"Failure concentration is only {concentration:.1f}% — "
            "failures are spread across multiple routes, weakening the "
            "single-cause hypothesis.",
            min(penalty, 0.10),
        )
    return None


def _rule_thin_margin_over_runner_up(
    predicted_cause: str,
    _incident: dict,
    evidence: dict,
    _detection: dict,
) -> tuple[str, float] | None:
    """If the top score barely beats the runner-up, the diagnosis is fragile."""
    if predicted_cause == "unresolved":
        return None
    scores = evidence.get("score_by_cause", {})
    if not scores:
        return None
    sorted_vals = sorted(scores.values(), reverse=True)
    if len(sorted_vals) < 2:
        return None
    margin = sorted_vals[0] - sorted_vals[1]
    if 0.15 <= margin < 0.30:
        return (
            f"Margin over the next-best explanation is only {margin:.2f} — "
            "an alternative cause is almost as well-supported.",
            0.06,
        )
    return None


# All skeptic rules in execution order
SKEPTIC_RULES = [
    ("small_blast_radius", _rule_small_blast_radius),
    ("error_signature_mismatch", _rule_error_signature_mismatch),
    ("missing_corroborating_health_signal", _rule_missing_corroborating_health_signal),
    ("low_failure_concentration", _rule_low_failure_concentration),
    ("thin_margin_over_runner_up", _rule_thin_margin_over_runner_up),
]


def skeptic_review(
    incident: dict,
    detection: dict,
    correlation: dict,
) -> dict:
    """Run all skeptic rules against the primary diagnosis.

    Returns a dict containing:
    - ``outcome``: ``"confirmed"`` or ``"challenged"``
    - ``checks``: list of every rule that was evaluated
    - ``challenges``: list of rules that fired (subset of checks)
    - ``total_penalty``: sum of penalties from fired rules
    - ``final_confidence``: primary confidence minus total_penalty (floored at 0.0)
    """
    predicted_cause = correlation.get("predicted_cause", "unresolved")
    primary_confidence = correlation.get("confidence", 0.0)
    evidence = correlation.get("evidence", {})

    checks: list[dict] = []
    challenges: list[dict] = []
    total_penalty = 0.0

    for rule_name, rule_fn in SKEPTIC_RULES:
        result = rule_fn(predicted_cause, incident, evidence, detection)
        if result is not None:
            challenge_text, penalty = result
            entry = {
                "rule": rule_name,
                "fired": True,
                "challenge": challenge_text,
                "penalty": round(penalty, 3),
            }
            checks.append(entry)
            challenges.append(entry)
            total_penalty += penalty
        else:
            checks.append({
                "rule": rule_name,
                "fired": False,
                "challenge": None,
                "penalty": 0.0,
            })

    total_penalty = round(total_penalty, 3)
    final_confidence = round(max(0.0, primary_confidence - total_penalty), 2)

    # Hard invariant: skeptic can only hold or lower confidence
    if final_confidence > primary_confidence:
        final_confidence = primary_confidence

    outcome = "challenged" if challenges else "confirmed"
    summary = (
        "No strong counter-evidence found; primary diagnosis stands."
        if outcome == "confirmed"
        else "; ".join(c["challenge"] for c in challenges)
    )

    return {
        "outcome": outcome,
        "summary": summary,
        "primary_confidence": primary_confidence,
        "total_penalty": total_penalty,
        "final_confidence": final_confidence,
        "checks_performed": len(checks),
        "challenges_raised": len(challenges),
        "checks": checks,
        "challenges": challenges,
    }
