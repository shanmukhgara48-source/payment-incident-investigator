"""Live LLM evaluation: run the full pipeline and report honest accuracy.

Usage:
    .venv/bin/python -m src.evaluate_llm

Requires OPENAI_API_KEY set in .env or environment.
Reports per-cause accuracy, fallback counts, cost, and latency.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

# Load .env before importing src modules
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from data.simulate import DEFAULT_INCIDENT_COUNT, generate_dataset
from src.llm import get_backend, get_model, get_usage_stats, llm_available, reset_client, reset_usage_stats
from src.pipeline import run_pipeline


def main() -> None:
    reset_client()
    reset_usage_stats()

    if not llm_available():
        print("ERROR: No LLM API key set (OPENAI_API_KEY or HF_TOKEN).")
        print("Add one to .env or export it in your shell.")
        sys.exit(1)

    # Force client init so backend is resolved
    from src.llm import get_client
    get_client()
    backend = get_backend()
    model = get_model()
    print(f"Backend: {backend}")
    print(f"Model: {model}")
    print()

    # Generate the deterministic dataset
    print("Generating dataset...")
    dataset = generate_dataset(DEFAULT_INCIDENT_COUNT)
    incidents = dataset["incidents"]
    print(f"Dataset: {len(incidents)} incidents")
    print()

    # Run the full pipeline with LLM
    print("Running pipeline with LLM reasoning...")
    t0 = time.monotonic()
    records = run_pipeline(incidents)
    wall_time = time.monotonic() - t0
    print(f"Pipeline completed in {wall_time:.1f}s")
    print()

    # === Reasoning mode breakdown ===
    correlator_modes = Counter(r["correlation"]["reasoning_mode"] for r in records)
    skeptic_modes = Counter(r["skeptic_review"]["reasoning_mode"] for r in records)
    print("=== REASONING MODE BREAKDOWN ===")
    print(f"Correlator: {dict(correlator_modes)}")
    print(f"Skeptic:    {dict(skeptic_modes)}")
    print()

    # === Accuracy evaluation ===
    # Separate clear vs ambiguous
    clear = [r for r in records if not r["ground_truth"]["is_ambiguous"]]
    ambiguous = [r for r in records if r["ground_truth"]["is_ambiguous"]]

    # Overall accuracy on clear cases
    correct = sum(
        1 for r in clear
        if r["correlation"]["predicted_cause"] == r["ground_truth"]["cause"]
    )
    unresolved = sum(
        1 for r in clear
        if r["correlation"]["predicted_cause"] == "unresolved"
    )
    misdiagnosed = sum(
        1 for r in clear
        if r["correlation"]["predicted_cause"] != r["ground_truth"]["cause"]
        and r["correlation"]["predicted_cause"] != "unresolved"
    )

    print("=== OVERALL ACCURACY (clear cases) ===")
    print(f"Total clear cases: {len(clear)}")
    print(f"Correct: {correct} ({correct/len(clear)*100:.1f}%)")
    print(f"Unresolved (escalated): {unresolved}")
    print(f"Misdiagnosed: {misdiagnosed}")
    print()

    # Per-cause accuracy
    causes = ["bad_deploy", "bank_psp_downtime", "gateway_error", "config_change", "network_issue"]
    print("=== PER-CAUSE ACCURACY ===")
    print(f"{'Root cause':<22} {'Correct':>8} {'Unresolved':>10} {'Misdiag':>8} {'Accuracy':>9}")
    print("-" * 60)

    per_cause_accuracy = {}
    for cause in causes:
        cause_records = [r for r in clear if r["ground_truth"]["cause"] == cause]
        if not cause_records:
            continue
        c = sum(1 for r in cause_records if r["correlation"]["predicted_cause"] == cause)
        u = sum(1 for r in cause_records if r["correlation"]["predicted_cause"] == "unresolved")
        m = len(cause_records) - c - u
        acc = c / len(cause_records) * 100
        per_cause_accuracy[cause] = acc
        print(f"{cause:<22} {f'{c}/{len(cause_records)}':>8} {u:>10} {m:>8} {acc:>8.1f}%")

    print(f"{'Overall (clear)':<22} {f'{correct}/{len(clear)}':>8} {unresolved:>10} {misdiagnosed:>8} {correct/len(clear)*100:>8.1f}%")
    print()

    # Ambiguous case honesty
    ambiguous_correct = sum(
        1 for r in ambiguous
        if r["correlation"]["predicted_cause"] == "unresolved"
    )
    print(f"Ambiguous honesty: {ambiguous_correct}/{len(ambiguous)} ({ambiguous_correct/len(ambiguous)*100:.0f}%)")
    print()

    # === Comparison vs rule-based baseline ===
    baseline = {
        "bad_deploy": 50.0,
        "bank_psp_downtime": 50.0,
        "gateway_error": 60.0,
        "config_change": 20.0,
        "network_issue": 36.0,
    }
    baseline_overall = 43.1

    print("=== LLM vs RULE-BASED COMPARISON ===")
    print(f"{'Root cause':<22} {'Rule-based':>10} {'LLM':>10} {'Delta':>10}")
    print("-" * 55)
    for cause in causes:
        if cause in per_cause_accuracy:
            rb = baseline[cause]
            llm = per_cause_accuracy[cause]
            delta = llm - rb
            sign = "+" if delta > 0 else ""
            print(f"{cause:<22} {rb:>9.1f}% {llm:>9.1f}% {sign}{delta:>8.1f}pp")

    overall_llm = correct / len(clear) * 100
    overall_delta = overall_llm - baseline_overall
    sign = "+" if overall_delta > 0 else ""
    print(f"{'Overall':<22} {baseline_overall:>9.1f}% {overall_llm:>9.1f}% {sign}{overall_delta:>8.1f}pp")
    print()

    # === Misdiagnosis detail ===
    if misdiagnosed > 0:
        print("=== MISDIAGNOSES ===")
        for r in clear:
            pred = r["correlation"]["predicted_cause"]
            truth = r["ground_truth"]["cause"]
            if pred != truth and pred != "unresolved":
                mode = r["correlation"]["reasoning_mode"]
                print(f"  {r['incident_id']}: predicted={pred}, truth={truth}, mode={mode}")
        print()

    # === Cost and latency ===
    stats = get_usage_stats()
    print("=== COST AND LATENCY ===")
    print(f"Model: {stats['model']}")
    print(f"Total LLM calls: {stats['total_calls']}")
    print(f"Total prompt tokens: {stats['total_prompt_tokens']:,}")
    print(f"Total completion tokens: {stats['total_completion_tokens']:,}")
    print(f"Total latency: {stats['total_latency_seconds']:.1f}s")
    print(f"Estimated cost: ${stats['estimated_cost_usd']:.6f}")
    print()

    llm_reasoned_count = correlator_modes.get("LLM_REASONED", 0) + skeptic_modes.get("LLM_REASONED", 0)
    if llm_reasoned_count > 0:
        avg_latency = stats["total_latency_seconds"] / stats["total_calls"]
        print(f"Average latency per LLM call: {avg_latency:.2f}s")

        # Separate correlator vs skeptic latency from records
        cor_latencies = [
            r["correlation"]["llm_meta"]["latency_seconds"]
            for r in records
            if r["correlation"].get("llm_meta")
        ]
        skep_latencies = [
            r["skeptic_review"]["llm_meta"]["latency_seconds"]
            for r in records
            if r["skeptic_review"].get("llm_meta")
        ]
        if cor_latencies:
            print(f"  Correlator avg: {sum(cor_latencies)/len(cor_latencies):.2f}s ({len(cor_latencies)} calls)")
        if skep_latencies:
            print(f"  Skeptic avg: {sum(skep_latencies)/len(skep_latencies):.2f}s ({len(skep_latencies)} calls)")
        print()

        # Demo extrapolation
        print("=== DEMO EXTRAPOLATION ===")
        print(f"Wall-clock time for {len(incidents)} incidents: {wall_time:.1f}s")
        print(f"  Of which LLM latency: {stats['total_latency_seconds']:.1f}s")
        print(f"  Pipeline overhead: {wall_time - stats['total_latency_seconds']:.1f}s")
        print(f"Cost per full run: ${stats['estimated_cost_usd']:.4f}")
        print(f"Cost for 10 demo runs: ${stats['estimated_cost_usd'] * 10:.4f}")
    else:
        print("No LLM calls were made (all rule-based).")
        print(f"Wall-clock time: {wall_time:.1f}s")

    print()
    print("=== VERDICT ===")
    if llm_reasoned_count > 0:
        if overall_delta > 5:
            print(f"LLM reasoning improved overall accuracy by {sign}{overall_delta:.1f}pp.")
        elif overall_delta > 0:
            print(f"LLM reasoning showed modest improvement: {sign}{overall_delta:.1f}pp overall.")
        elif overall_delta == 0:
            print("LLM reasoning produced identical accuracy to rule-based.")
        else:
            print(f"LLM reasoning performed worse than rule-based: {overall_delta:.1f}pp overall.")

        # Check weak causes specifically
        for cause in ["config_change", "network_issue"]:
            if cause in per_cause_accuracy:
                d = per_cause_accuracy[cause] - baseline[cause]
                s = "+" if d > 0 else ""
                print(f"  {cause}: {s}{d:.1f}pp {'(improved)' if d > 0 else '(no improvement)' if d == 0 else '(regressed)'}")
    else:
        print("No LLM calls were made. Set OPENAI_API_KEY to enable LLM evaluation.")

    # Save results for README update
    output = {
        "overall_accuracy_pct": round(correct / len(clear) * 100, 1),
        "per_cause": per_cause_accuracy,
        "misdiagnosed": misdiagnosed,
        "unresolved": unresolved,
        "ambiguous_honesty_pct": round(ambiguous_correct / len(ambiguous) * 100, 1),
        "correlator_modes": dict(correlator_modes),
        "skeptic_modes": dict(skeptic_modes),
        "cost": stats,
        "wall_time_seconds": round(wall_time, 1),
    }
    with open("data/llm_evaluation.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to data/llm_evaluation.json")


if __name__ == "__main__":
    main()
