"""Skeptic agent: adversarial second-pass review of the primary correlator diagnosis.

Two review paths:
- **LLM_REASONED**: When OPENAI_API_KEY is set, the evidence and primary
  diagnosis are sent to an OpenAI model to find counter-arguments.
- **RULE_BASED** / **RULE_BASED_FALLBACK**: Five deterministic rules check
  for blast-radius, signature mismatch, missing corroboration, low
  concentration, and thin margin.

The skeptic can only hold or LOWER confidence — never raise it.  The output
records every check performed and whether it challenged the primary or
confirmed it.
"""

from __future__ import annotations

import logging

try:
    from .correlator import ERROR_SIGNATURES
    from .float_compare import gte, lt
    from .llm import llm_available, llm_call
except ImportError:
    from correlator import ERROR_SIGNATURES
    from float_compare import gte, lt
    from llm import llm_available, llm_call

logger = logging.getLogger(__name__)

# ── penalty schedule ────────────────────────────────────────────────
# Each rule that fires returns a tuple (challenge_text, penalty).
# Penalties are subtracted from primary confidence to yield the
# final_confidence.  Multiple rules can fire and their penalties stack,
# but the result is floored at 0.0.

_SKEPTIC_SYSTEM_PROMPT = """\
You are an adversarial reviewer of payment incident diagnoses.  Your job is to
find weaknesses, contradictions, and counter-evidence in the primary diagnosis.
You can only LOWER confidence, never raise it.

Respond with valid JSON:
{
  "challenges": [
    {
      "rule": "short_descriptive_name",
      "challenge": "1-sentence explanation of the weakness",
      "penalty": 0.XX
    }
  ],
  "total_penalty": 0.XX,
  "summary": "1-2 sentence overall assessment"
}

Rules for your response:
- Each individual penalty must be between 0.01 and 0.15.
- total_penalty must equal the sum of individual penalties (max 0.50).
- If you find NO genuine weaknesses, return {"challenges": [], "total_penalty": 0.0, "summary": "..."}.
- Do NOT manufacture problems — only flag real inconsistencies visible in the evidence.
- Common things to check:
  * Small deploy rollout cannot explain widespread failures.
  * Error signature points to a different cause than diagnosed.
  * Missing corroborating health/network signal for the diagnosed cause.
  * Low failure concentration weakens a single-cause hypothesis.
  * Close margin between top two candidate causes.
"""


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
    if lt(concentration, 80):
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
    # `margin` is a difference of two accumulated float scores, so both ends of
    # this band need decimal-sense comparisons or a margin of exactly 0.15 or
    # 0.30 can fall on the wrong side of the rule.
    if gte(margin, 0.15) and lt(margin, 0.30):
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


def _format_skeptic_prompt(
    incident: dict,
    detection: dict,
    correlation: dict,
) -> str:
    """Format evidence and primary diagnosis for the LLM skeptic."""
    evidence = correlation.get("evidence", {})
    primary = detection.get("primary_degradation", {})

    lines = [
        f"Incident: {correlation.get('incident_id', 'unknown')}",
        f"Primary diagnosis: {correlation.get('predicted_cause', 'unresolved')}",
        f"Primary confidence: {correlation.get('confidence', 0.0):.2f}",
        "",
        "=== Evidence summary ===",
        f"Payment method: {primary.get('sub_type', '?')} via {primary.get('route', '?')}",
        f"Failure concentration: {evidence.get('concentration_pct', '?')}%",
        f"Deploy overlap: {evidence.get('deploy_overlap_detail', 'none')}",
        f"Config change: {evidence.get('config_change_detail', 'none')}",
        f"Dominant error code: {evidence.get('dominant_error_code', 'none')}",
        f"Error signature side: {evidence.get('error_signature_side', 'unknown')}",
        f"Route health failed: {evidence.get('route_health_failed', False)}",
        f"Route health confirmed: {evidence.get('route_health_confirmed', False)}",
        f"Network alert: {evidence.get('network_alert', False)}",
        f"Webhook degraded: {evidence.get('webhook_delivery_degraded', False)}",
    ]

    scores = evidence.get("score_by_cause", {})
    if scores:
        lines.append("")
        lines.append("=== Rule-based scores by cause ===")
        for cause, score in sorted(scores.items(), key=lambda x: -x[1]):
            lines.append(f"  {cause}: {score:.2f}")

    supporting = evidence.get("supporting_signals", [])
    if supporting:
        lines.append("")
        lines.append("=== Supporting signals for primary diagnosis ===")
        for sig in supporting:
            lines.append(f"  - {sig}")

    llm_expl = evidence.get("llm_explanation")
    if llm_expl:
        lines.append("")
        lines.append(f"=== Correlator reasoning ===\n{llm_expl}")

    return "\n".join(lines)


def _try_llm_skeptic(
    incident: dict,
    detection: dict,
    correlation: dict,
    primary_confidence: float,
) -> dict | None:
    """Attempt an LLM-based adversarial review.  Returns validated result or None."""
    if not llm_available():
        return None

    predicted_cause = correlation.get("predicted_cause", "unresolved")
    if predicted_cause == "unresolved":
        # Nothing to challenge — LLM adds no value here
        return None

    user_prompt = _format_skeptic_prompt(incident, detection, correlation)
    result = llm_call(_SKEPTIC_SYSTEM_PROMPT, user_prompt)
    if result is None:
        return None

    # Validate structure
    challenges = result.get("challenges")
    if not isinstance(challenges, list):
        logger.warning("LLM skeptic returned non-list challenges; falling back")
        return None

    validated_challenges = []
    total_penalty = 0.0
    for ch in challenges:
        if not isinstance(ch, dict):
            continue
        penalty = ch.get("penalty", 0)
        if not isinstance(penalty, (int, float)) or penalty < 0:
            continue
        penalty = min(float(penalty), 0.15)  # cap individual penalties
        validated_challenges.append({
            "rule": str(ch.get("rule", "llm_challenge")),
            "fired": True,
            "challenge": str(ch.get("challenge", "")),
            "penalty": round(penalty, 3),
        })
        total_penalty += penalty

    total_penalty = round(min(total_penalty, 0.50), 3)  # cap total
    final_confidence = round(max(0.0, primary_confidence - total_penalty), 2)

    # Hard invariant: skeptic can only hold or lower confidence.
    if final_confidence > primary_confidence:
        final_confidence = primary_confidence

    return {
        "challenges": validated_challenges,
        "total_penalty": total_penalty,
        "final_confidence": final_confidence,
        "summary": result.get("summary", ""),
        "_llm_meta": result.get("_llm_meta"),
    }


def _run_rule_based_skeptic(
    predicted_cause: str,
    primary_confidence: float,
    incident: dict,
    evidence: dict,
    detection: dict,
) -> dict:
    """Run the five deterministic skeptic rules."""
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

    # Hard invariant
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


def skeptic_review(
    incident: dict,
    detection: dict,
    correlation: dict,
) -> dict:
    """Run adversarial review against the primary diagnosis.

    Returns a dict containing:
    - ``outcome``: ``"confirmed"`` or ``"challenged"``
    - ``checks``: list of every rule that was evaluated
    - ``challenges``: list of rules that fired (subset of checks)
    - ``total_penalty``: sum of penalties from fired rules
    - ``final_confidence``: primary confidence minus total_penalty (floored at 0.0)
    - ``reasoning_mode``: ``LLM_REASONED``, ``RULE_BASED``, or ``RULE_BASED_FALLBACK``
    """
    predicted_cause = correlation.get("predicted_cause", "unresolved")
    primary_confidence = correlation.get("confidence", 0.0)
    evidence = correlation.get("evidence", {})

    # Try LLM path first
    llm_result = _try_llm_skeptic(incident, detection, correlation, primary_confidence)

    if llm_result is not None:
        challenges = llm_result["challenges"]
        outcome = "challenged" if challenges else "confirmed"

        review = {
            "outcome": outcome,
            "summary": llm_result["summary"],
            "primary_confidence": primary_confidence,
            "total_penalty": llm_result["total_penalty"],
            "final_confidence": llm_result["final_confidence"],
            "checks_performed": len(challenges),
            "challenges_raised": len(challenges),
            "checks": challenges,
            "challenges": challenges,
            "reasoning_mode": "LLM_REASONED",
        }
        if llm_result.get("_llm_meta"):
            review["llm_meta"] = llm_result["_llm_meta"]
        return review

    # Rule-based path
    review = _run_rule_based_skeptic(
        predicted_cause, primary_confidence, incident, evidence, detection,
    )
    review["reasoning_mode"] = (
        "RULE_BASED_FALLBACK" if llm_available() else "RULE_BASED"
    )
    return review
