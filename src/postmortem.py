"""Generate a clean markdown postmortem document for a single incident."""

from __future__ import annotations

from datetime import datetime


def _time(value: str | None) -> str:
    if not value:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return str(value)


def _money(value: int | float | None) -> str:
    if value is None:
        return "INR 0"
    return f"INR {int(value):,}"


def _cause_label(value: str | None) -> str:
    return (value or "unresolved").replace("_", " ").title()


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def generate_postmortem(record: dict) -> str:
    """Return a complete markdown postmortem for one pipeline record."""

    incident_id = record.get("incident_id", "Unknown")
    window = record.get("window", {})
    correlation = record.get("correlation", {})
    primary_diag = record.get("primary_diagnosis")
    skeptic = record.get("skeptic_review")
    pattern_recall = record.get("pattern_recall")
    impact = record.get("impact", {})
    recovery = record.get("recovery", {})
    detection = record.get("detection", {})
    rca_text = record.get("rca_text", "")
    timeline = record.get("timeline", [])
    audit_trail = record.get("audit_trail", [])

    # Header
    primary = detection.get("primary_degradation")
    method_display = primary["method_display"] if primary else "Unknown method"
    cause = correlation.get("predicted_cause", "unresolved")
    confidence = correlation.get("confidence", 0)
    status = "Escalated" if recovery.get("primary_action") == "escalate to human" else "Resolved"

    lines: list[str] = []
    lines.append(f"# Postmortem: {incident_id}")
    lines.append("")
    lines.append(f"**Status:** {status}  ")
    lines.append(f"**Method:** {method_display}  ")
    lines.append(f"**Root cause:** {_cause_label(cause)}  ")
    lines.append(f"**Confidence:** {confidence:.2f}  ")
    lines.append(f"**Window:** {_time(window.get('current_start'))} — {_time(window.get('current_end'))}  ")
    lines.append(f"**Recovery mode:** {recovery.get('recovery_mode', 'SIMULATED')}  ")
    lines.append("")

    # Summary / RCA
    lines.append("## Root cause analysis")
    lines.append("")
    if rca_text:
        lines.append(f"> {rca_text}")
    else:
        lines.append("> No RCA text available.")
    lines.append("")

    # Detection
    if primary:
        lines.append("## Detection")
        lines.append("")
        lines.append(f"- **Success rate drop:** {_pct(primary.get('success_rate_drop'))}")
        lines.append(f"- **Failure concentration:** {primary.get('failure_concentration_pct', 0):.1f}%")
        lines.append(f"- **Affected route:** {primary.get('route', 'Unknown')} ({primary.get('route_type', 'unknown')})")
        lines.append(f"- **Baseline success rate:** {_pct(primary.get('baseline_success_rate'))}")
        lines.append(f"- **Current success rate:** {_pct(primary.get('current_success_rate'))}")
        lines.append("")

    # Timeline
    if timeline:
        lines.append("## Timeline")
        lines.append("")
        lines.append("| Time | Event | Detail |")
        lines.append("|------|-------|--------|")
        for marker in timeline:
            t = _time(marker.get("timestamp"))
            kind = (marker.get("kind", "")).replace("_", " ").title()
            label = marker.get("label", "")
            lines.append(f"| {t} | {kind} | {label} |")
        lines.append("")

    # Business impact
    lines.append("## Business impact")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Attempted GMV | {_money(impact.get('attempted_gmv_inr'))} |")
    lines.append(f"| Failed GMV | {_money(impact.get('failed_gmv_inr'))} |")
    lines.append(f"| Recoverable GMV | {_money(impact.get('recoverable_gmv_inr'))} |")
    lines.append(f"| Recovered amount | {_money(impact.get('recovered_amount_inr'))} |")
    lines.append(f"| Failed payment count | {impact.get('failed_count', 0)} |")
    lines.append(f"| Recoverable payment count | {impact.get('recoverable_payment_count', 0)} |")
    basis = impact.get("recovered_amount_basis", "")
    if basis:
        lines.append("")
        lines.append(f"*{basis}*")
    lines.append("")

    # Action taken
    lines.append("## Action taken")
    lines.append("")
    lines.append(f"**Primary action:** {recovery.get('primary_action', 'Unknown')}")
    lines.append("")
    if recovery.get("merchant_notification_sent"):
        lines.append("Merchant notification was sent.")
        lines.append("")

    # Skeptic review — only if the feature exists in this record
    if skeptic and isinstance(skeptic, dict) and "outcome" in skeptic:
        lines.append("## Skeptic review")
        lines.append("")
        outcome = skeptic["outcome"]
        lines.append(f"**Outcome:** {outcome.title()}")
        if primary_diag and isinstance(primary_diag, dict):
            lines.append(f"  ")
            lines.append(f"**Primary confidence:** {primary_diag.get('confidence', 0):.2f}  ")
            lines.append(f"**Final confidence:** {skeptic.get('final_confidence', 0):.2f}  ")
            penalty = skeptic.get("total_penalty", 0)
            if penalty:
                lines.append(f"**Total penalty:** -{penalty:.3f}  ")
        lines.append("")
        summary = skeptic.get("summary", "")
        if summary:
            lines.append(f"> {summary}")
            lines.append("")
        checks = skeptic.get("checks", [])
        if checks:
            for check in checks:
                rule = (check.get("rule", "")).replace("_", " ").title()
                if check.get("fired"):
                    challenge = check.get("challenge", "")
                    p = check.get("penalty", 0)
                    lines.append(f"- **{rule}** — {challenge} (penalty: -{p:.3f})")
                else:
                    lines.append(f"- {rule} — passed")
            lines.append("")

    # Similar past incidents — only if the feature exists in this record
    if pattern_recall and isinstance(pattern_recall, dict):
        matches = pattern_recall.get("matches", [])
        prior = pattern_recall.get("prior_incidents_considered", 0)
        if prior > 0:
            lines.append("## Similar past incidents")
            lines.append("")
            if matches:
                lines.append(f"{len(matches)} similar incident(s) found out of {prior} compared:")
                lines.append("")
                for m in matches:
                    mid = m.get("incident_id", "Unknown")
                    sim = m.get("similarity", 0)
                    rc = _cause_label(m.get("root_cause"))
                    action = m.get("action_taken", "Unknown")
                    outcome = m.get("outcome", "Unknown")
                    lines.append(f"- **{mid}** (similarity {sim:.2f}) — {rc}, action: {action}, outcome: {outcome}")
                lines.append("")
            else:
                lines.append(f"No similar incidents found among {prior} compared.")
                lines.append("")
            track = pattern_recall.get("resolution_track_record")
            if track and isinstance(track, dict):
                statement = track.get("statement", "")
                if statement:
                    lines.append(f"*{statement}*")
                    lines.append("")

    # Audit trail
    if audit_trail:
        lines.append("## Audit trail")
        lines.append("")
        lines.append("| Time | Action | Reason | Bounded by |")
        lines.append("|------|--------|--------|------------|")
        for entry in audit_trail:
            t = _time(entry.get("timestamp"))
            action = entry.get("action", "")
            reason = entry.get("reason", "")
            bounded = entry.get("bounded_by", "")
            lines.append(f"| {t} | {action} | {reason} | {bounded} |")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*This postmortem was generated from synthetic incident data. "
                 "All events, money values, and recovery outcomes are simulated. "
                 "TEST MODE ONLY — no production money was moved.*")
    lines.append("")

    return "\n".join(lines)
