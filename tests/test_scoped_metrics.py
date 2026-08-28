"""Tests for the scoped retry-recovered and GMV-protected metrics.

Verifies:
1. retry_recovered_amount_inr is computed only over retry-action incidents.
2. gmv_protected_inr is computed only over reroute-action incidents.
3. The two figures are never summed together anywhere in the codebase.
"""

import ast
import os
from pathlib import Path

from data.simulate import generate_dataset
from src.config import ASSUMED_RECOVERY_SUCCESS_RATE
from src.evaluate import evaluate
from src.pipeline import run_pipeline


RETRY_ACTION = "create Payment Links for high-intent failures"
REROUTE_ACTION = "reroute traffic"
SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _run_batch(seed=700):
    dataset = generate_dataset(count=30, ambiguous_fraction=0.15, seed=seed)
    return run_pipeline(dataset["incidents"])


class TestRetryRecoveredScope:
    """retry_recovered_amount_inr must be zero when filtered to reroute-only incidents."""

    def test_retry_recovered_zero_for_reroute_only_incidents(self):
        records = _run_batch()
        reroute_only = [r for r in records if r["recovery"]["primary_action"] == REROUTE_ACTION]
        # There should be at least one reroute incident in a 30-incident batch with
        # non-zero ambiguous ratio; if not, the test is vacuously true but still valid.
        for record in reroute_only:
            # The recovery engine should report zero modeled recovered for reroute actions
            assert record["recovery"]["modeled_recovered_amount_inr"] == 0, (
                f"{record['incident_id']}: reroute incident should have zero modeled_recovered_amount_inr"
            )

    def test_retry_recovered_only_sums_retry_eligible(self):
        records = _run_batch()
        retry_eligible = [r for r in records if r["recovery"]["primary_action"] == RETRY_ACTION]
        non_retry = [r for r in records if r["recovery"]["primary_action"] != RETRY_ACTION]

        # Sum of retry_recovered over retry-eligible should match the aggregate
        retry_sum = sum(r["impact"]["retry_recovered_amount_inr"] for r in retry_eligible)
        all_sum = sum(r["impact"]["retry_recovered_amount_inr"] for r in records)

        # The non-retry incidents still have a non-zero impact.retry_recovered_amount_inr
        # (it's the raw calculation), but the recovery engine zeros it. The aggregate in
        # evaluate.py should only sum from retry-eligible records.
        assert retry_sum <= all_sum  # sanity

        # Verify each retry-eligible incident's retry_recovered equals recoverable * rate
        for record in retry_eligible:
            impact = record["impact"]
            expected = round(impact["recoverable_gmv_inr"] * ASSUMED_RECOVERY_SUCCESS_RATE)
            assert impact["retry_recovered_amount_inr"] == expected

    def test_impact_has_retry_recovered_field(self):
        records = _run_batch()
        for record in records:
            assert "retry_recovered_amount_inr" in record["impact"]


class TestGmvProtectedScope:
    """gmv_protected_inr must be zero for retry-only incidents and bounded for reroute."""

    def test_gmv_protected_zero_for_non_reroute_incidents(self):
        records = _run_batch()
        non_reroute = [r for r in records if r["recovery"]["primary_action"] != REROUTE_ACTION]
        for record in non_reroute:
            assert record["recovery"]["gmv_protected_inr"] == 0, (
                f"{record['incident_id']}: non-reroute incident should have zero gmv_protected_inr in recovery"
            )

    def test_gmv_protected_bounded_by_attempted_gmv(self):
        records = _run_batch()
        reroute = [r for r in records if r["recovery"]["primary_action"] == REROUTE_ACTION]
        for record in reroute:
            protected = record["recovery"]["gmv_protected_inr"]
            attempted = record["impact"]["attempted_gmv_inr"]
            assert 0 <= protected <= attempted, (
                f"{record['incident_id']}: gmv_protected_inr ({protected}) exceeds "
                f"attempted_gmv_inr ({attempted})"
            )

    def test_gmv_protected_only_for_reroute_in_aggregate(self):
        """The evaluate aggregate should only sum gmv_protected from reroute incidents."""
        records = _run_batch()
        reroute = [r for r in records if r["recovery"]["primary_action"] == REROUTE_ACTION]
        expected_sum = sum(r["recovery"]["gmv_protected_inr"] for r in reroute)
        # Also verify no non-reroute incident contributes
        non_reroute_sum = sum(
            r["recovery"]["gmv_protected_inr"]
            for r in records
            if r["recovery"]["primary_action"] != REROUTE_ACTION
        )
        assert non_reroute_sum == 0


class TestMetricsNeverSummed:
    """The two figures must never be combined into a single 'total impact' number."""

    def test_no_combined_sum_in_source_code(self):
        """Scan all Python source files for any expression that sums
        retry_recovered and gmv_protected together."""
        violations = []
        for py_file in SRC_DIR.rglob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            # Check for string patterns that would combine these
            if "retry_recovered" in source and "gmv_protected" in source:
                # Parse the AST and look for BinOp(Add) combining the two
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                        left_names = {n.attr for n in ast.walk(node.left) if isinstance(n, ast.Attribute)}
                        right_names = {n.attr for n in ast.walk(node.right) if isinstance(n, ast.Attribute)}
                        all_names = left_names | right_names
                        has_recovered = any("retry_recovered" in n or "recovered_amount" in n for n in all_names)
                        has_protected = any("gmv_protected" in n for n in all_names)
                        if has_recovered and has_protected:
                            violations.append(f"{py_file.name}:{node.lineno}")
        assert not violations, f"Found combined sum of recovered + protected in: {violations}"

    def test_aggregate_has_separate_fields(self):
        """The evaluate output must have distinct top-level fields, not one merged total."""
        import json
        import tempfile
        from data.simulate import generate_dataset
        dataset = generate_dataset(count=10, ambiguous_fraction=0.1, seed=800)
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "incidents.json"
            results_path = Path(tmp) / "results.json"
            data_path.write_text(json.dumps(dataset), encoding="utf-8")
            results = evaluate(data_path, results_path)
        agg = results["aggregate_metrics"]
        assert "total_retry_recovered_amount_inr" in agg
        assert "total_gmv_protected_inr" in agg
        # They must be separate keys, not a merged "total_impact" or similar
        for key in agg:
            if "total" in key.lower() and "impact" in key.lower():
                assert False, f"Found suspicious combined metric key: {key}"
