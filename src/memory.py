"""Pattern recall across incidents already processed in the current batch.

Each incident is reduced to a small hand-crafted feature vector (no embedding
model) and compared against previously processed incidents with cosine
similarity.  Two guarantees matter here:

1. **Causal honesty.** ``IncidentMemory`` only ever holds incidents that were
   already fully processed.  The first incident of a batch therefore has zero
   possible matches, and no incident can ever see one that comes after it.
   ``recall()`` is always called before ``remember()`` for the same incident.
2. **Recall is evidence, not a verdict.** The recalled history is attached to
   the record as supporting context.  It never edits ``predicted_cause`` or
   ``confidence`` — the correlator and skeptic remain the only things that can
   move a diagnosis.
"""

from __future__ import annotations

from collections import Counter
from math import sqrt

try:
    from .float_compare import lt
except ImportError:  # Supports direct imports from src/.
    from float_compare import lt

# Cosine floor for calling two incidents "similar". With the weights below this
# effectively requires method, route, and failure reason to all agree; a single
# categorical mismatch lands around 0.69.
SIMILARITY_THRESHOLD = 0.75

# The UI panel shows the top 1-2; keeping the API in step avoids a long tail.
MAX_SIMILAR_MATCHES = 2

# Categorical identity dominates; the two numeric signals are deliberately
# secondary tie-breakers rather than drivers.
FEATURE_WEIGHTS = {
    "method": 1.0,
    "route": 1.0,
    "failure_reason": 1.0,
    "concentration": 0.5,
    "deploy_overlap": 0.5,
}

CATEGORICAL_FEATURES = ("method", "route", "failure_reason")

RESOLUTION_BASIS = (
    "'Resolved' means the bounded pipeline took an autonomous recovery action "
    "instead of escalating to a human. It is not a measured payment success rate."
)


def extract_features(detection: dict, correlation: dict, top_failure_reason: str) -> dict:
    """Pull the five recall features out of already-computed pipeline stages."""
    primary = detection.get("primary_degradation") or {}
    evidence = correlation.get("evidence") or {}
    concentration = primary.get("failure_concentration_pct")
    if concentration is None:
        concentration = evidence.get("concentration_pct", 0.0)
    return {
        "method": primary.get("sub_type") or "none",
        "route": primary.get("route") or "none",
        "failure_reason": top_failure_reason or "none",
        "concentration_pct": round(float(concentration or 0.0), 1),
        "deploy_overlap": bool(evidence.get("deploy_overlap", False)),
    }


def build_feature_vector(features: dict) -> dict[str, float]:
    """Return a sparse weighted vector; zero dimensions are dropped as no-ops."""
    concentration = min(1.0, max(0.0, float(features["concentration_pct"]) / 100))
    vector = {
        f"method={features['method']}": FEATURE_WEIGHTS["method"],
        f"route={features['route']}": FEATURE_WEIGHTS["route"],
        f"failure_reason={features['failure_reason']}": FEATURE_WEIGHTS["failure_reason"],
        "concentration": FEATURE_WEIGHTS["concentration"] * concentration,
        "deploy_overlap": FEATURE_WEIGHTS["deploy_overlap"] if features["deploy_overlap"] else 0.0,
    }
    return {key: value for key, value in vector.items() if value}


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right[key] for key, value in left.items() if key in right)
    norm_left = sqrt(sum(value * value for value in left.values()))
    norm_right = sqrt(sum(value * value for value in right.values()))
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def _shared_features(left: dict, right: dict) -> list[str]:
    shared = [name for name in CATEGORICAL_FEATURES if left[name] == right[name]]
    if left["deploy_overlap"] == right["deploy_overlap"]:
        shared.append("deploy_overlap")
    return shared


def _outcome_label(recovery: dict, impact: dict) -> str:
    action = recovery.get("primary_action", "unknown action")
    recovered = recovery.get("modeled_recovered_amount_inr", 0) or 0
    if not recovery.get("auto_action_taken"):
        return f"{action} - no autonomous recovery action taken"
    if recovered:
        measurement = impact.get("recovery_measurement_type")
        if measurement == "ACTUAL TEST-MODE":
            return f"{action} - actual TEST-MODE recovery INR {recovered:,}"
        rate = impact.get("assumed_recovery_success_rate", 0)
        return f"{action} - modeled recovery INR {recovered:,} ({rate:.0%} assumption)"
    return f"{action} - bounded action taken, no recovery amount modeled"


class IncidentMemory:
    """Append-only store of processed incidents, queried in processing order."""

    def __init__(
        self,
        threshold: float = SIMILARITY_THRESHOLD,
        max_matches: int = MAX_SIMILAR_MATCHES,
    ) -> None:
        self.threshold = threshold
        self.max_matches = max_matches
        self._entries: list[dict] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    def recall(self, features: dict, predicted_cause: str) -> dict:
        """Match `features` against prior incidents only. Never mutates state."""
        vector = build_feature_vector(features)
        scored = []
        # Reversed + a stable sort means ties resolve to the most recent precedent.
        for entry in reversed(self._entries):
            similarity = round(cosine_similarity(vector, entry["vector"]), 3)
            if lt(similarity, self.threshold):
                continue
            scored.append(
                {
                    "incident_id": entry["incident_id"],
                    "similarity": similarity,
                    "matched_on": _shared_features(features, entry["features"]),
                    "root_cause": entry["root_cause"],
                    "confidence": entry["confidence"],
                    "action_taken": entry["action_taken"],
                    "outcome": entry["outcome"],
                    "resolved": entry["resolved"],
                    "features": entry["features"],
                }
            )
        scored.sort(key=lambda match: match["similarity"], reverse=True)
        matches = scored[: self.max_matches]
        track_record = self._track_record(predicted_cause)
        return {
            "prior_incidents_considered": len(self._entries),
            "similarity_threshold": self.threshold,
            "features": features,
            "match_count": len(matches),
            "total_above_threshold": len(scored),
            "matches": matches,
            "resolution_track_record": track_record,
            "supporting_evidence": self._supporting_evidence(matches, track_record),
            "influences_diagnosis": False,
        }

    def remember(
        self,
        incident_id: str,
        features: dict,
        correlation: dict,
        recovery: dict,
        impact: dict,
    ) -> None:
        """Record a finished incident so later incidents in the batch can see it."""
        self._entries.append(
            {
                "incident_id": incident_id,
                "features": dict(features),
                "vector": build_feature_vector(features),
                "root_cause": correlation.get("predicted_cause", "unresolved"),
                "confidence": correlation.get("confidence", 0.0),
                "action_taken": recovery.get("primary_action", "unknown action"),
                "outcome": _outcome_label(recovery, impact),
                "resolved": bool(recovery.get("auto_action_taken")),
            }
        )

    def _track_record(self, cause: str) -> dict | None:
        """Running resolution success rate for `cause` across prior incidents."""
        prior = [entry for entry in self._entries if entry["root_cause"] == cause]
        if not prior:
            return None
        by_action: dict[str, dict] = {}
        for entry in prior:
            bucket = by_action.setdefault(entry["action_taken"], {"attempts": 0, "resolved": 0})
            bucket["attempts"] += 1
            bucket["resolved"] += entry["resolved"]
        dominant_action = Counter(entry["action_taken"] for entry in prior).most_common(1)[0][0]
        resolved_by_dominant = by_action[dominant_action]["resolved"]
        total = len(prior)
        if resolved_by_dominant:
            statement = f"{dominant_action} resolved {cause} in {resolved_by_dominant}/{total} prior cases"
        else:
            escalated = sum(1 for entry in prior if not entry["resolved"])
            statement = (
                f"{cause} was escalated to a human in {escalated}/{total} prior cases; "
                "no autonomous action has resolved it yet"
            )
        return {
            "root_cause": cause,
            "dominant_action": dominant_action,
            "resolved_count": resolved_by_dominant,
            "prior_case_count": total,
            "resolution_rate": round(resolved_by_dominant / total, 3),
            "statement": statement,
            "by_action": by_action,
            "basis": RESOLUTION_BASIS,
        }

    @staticmethod
    def _supporting_evidence(matches: list[dict], track_record: dict | None) -> str | None:
        parts = []
        if matches:
            best = matches[0]
            parts.append(
                f"Resembles {best['incident_id']} (similarity {best['similarity']:.2f}), "
                f"diagnosed {best['root_cause']}; outcome: {best['outcome']}."
            )
        if track_record:
            parts.append(f"{track_record['statement']}.")
        return " ".join(parts) if parts else None


def empty_recall(features: dict | None = None) -> dict:
    """Recall shape used when the memory stage is unavailable for an incident."""
    return {
        "prior_incidents_considered": 0,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "features": features or {},
        "match_count": 0,
        "total_above_threshold": 0,
        "matches": [],
        "resolution_track_record": None,
        "supporting_evidence": None,
        "influences_diagnosis": False,
    }
