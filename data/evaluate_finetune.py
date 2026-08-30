"""Evaluate the fine-tuned model on the held-out test split.

Usage:
    .venv/bin/python -m data.evaluate_finetune

Runs the fine-tuned model, rule-based baseline, and (if available) the hybrid
system on the SAME test split for a fair four-way comparison.

Output: data/finetune_eval.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prepare_finetune import _extract_evidence_prompt, _FINETUNE_SYSTEM_PROMPT
from src.correlator import correlate, VALID_CAUSES
from src.detector import detect_degradations


MODEL_DIR = Path(__file__).parent / "finetune_model"


def _run_finetuned_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    incidents: list[dict],
    device: str,
) -> list[dict]:
    """Run the fine-tuned model on each incident and return predictions."""
    results = []

    for inc in incidents:
        user_prompt = _extract_evidence_prompt(inc)
        if user_prompt is None:
            results.append({
                "incident_id": inc["incident_id"],
                "predicted_cause": "unresolved",
                "confidence": 0.0,
            })
            continue

        messages = [
            {"role": "system", "content": _FINETUNE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Parse JSON response
        try:
            # Try to extract JSON from the response
            if "{" in response_text:
                json_start = response_text.index("{")
                json_end = response_text.rindex("}") + 1
                parsed = json.loads(response_text[json_start:json_end])
            else:
                parsed = json.loads(response_text)

            cause = parsed.get("predicted_cause", "unresolved")
            if cause not in VALID_CAUSES:
                cause = "unresolved"
            confidence = float(parsed.get("confidence", 0.0))
            confidence = min(0.99, max(0.0, confidence))
        except (json.JSONDecodeError, ValueError, KeyError):
            cause = "unresolved"
            confidence = 0.0

        results.append({
            "incident_id": inc["incident_id"],
            "predicted_cause": cause,
            "confidence": round(confidence, 2),
            "raw_response": response_text[:500],
        })

    return results


def _run_rule_based(incidents: list[dict]) -> list[dict]:
    """Run the rule-based correlator on each incident."""
    results = []
    for inc in incidents:
        detection = detect_degradations(inc)
        corr = correlate(inc, detection)
        results.append({
            "incident_id": inc["incident_id"],
            "predicted_cause": corr["predicted_cause"],
            "confidence": corr["confidence"],
            "reasoning_mode": corr["reasoning_mode"],
        })
    return results


def _evaluate(
    predictions: list[dict],
    incidents: list[dict],
    label: str,
) -> dict:
    """Compute accuracy metrics for a set of predictions."""
    clear = [i for i in incidents if not i["ground_truth"]["is_ambiguous"]]
    ambiguous = [i for i in incidents if i["ground_truth"]["is_ambiguous"]]

    pred_map = {p["incident_id"]: p for p in predictions}

    correct = 0
    unresolved = 0
    misdiagnosed = 0
    per_cause_correct = Counter()
    per_cause_total = Counter()

    for inc in clear:
        pred = pred_map[inc["incident_id"]]["predicted_cause"]
        truth = inc["ground_truth"]["cause"]
        per_cause_total[truth] += 1
        if pred == truth:
            correct += 1
            per_cause_correct[truth] += 1
        elif pred == "unresolved":
            unresolved += 1
        else:
            misdiagnosed += 1

    # Ambiguous honesty: should predict "unresolved"
    ambiguous_correct = sum(
        1 for inc in ambiguous
        if pred_map[inc["incident_id"]]["predicted_cause"] == "unresolved"
    )

    per_cause_acc = {}
    for cause in sorted(per_cause_total):
        total = per_cause_total[cause]
        cor = per_cause_correct[cause]
        per_cause_acc[cause] = round(cor / total * 100, 1) if total > 0 else 0.0

    return {
        "label": label,
        "total_clear": len(clear),
        "total_ambiguous": len(ambiguous),
        "correct": correct,
        "accuracy_pct": round(correct / len(clear) * 100, 1) if clear else 0.0,
        "unresolved": unresolved,
        "misdiagnosed": misdiagnosed,
        "ambiguous_honesty_pct": round(ambiguous_correct / len(ambiguous) * 100, 1) if ambiguous else 0.0,
        "per_cause_accuracy": per_cause_acc,
    }


def main() -> None:
    if not MODEL_DIR.exists():
        print("ERROR: Fine-tuned model not found. Run data.finetune_lora first.")
        sys.exit(1)

    # Load test split
    test_path = Path(__file__).parent / "finetune_test.json"
    if not test_path.exists():
        print("ERROR: Test split not found. Run data.split_dataset first.")
        sys.exit(1)

    test_incidents = json.loads(test_path.read_text())
    print(f"Test incidents: {len(test_incidents)}")

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # === 1. Rule-based baseline ===
    print("\n=== Running rule-based baseline ===")
    rule_preds = _run_rule_based(test_incidents)
    rule_eval = _evaluate(rule_preds, test_incidents, "Rule-based")
    print(f"  Accuracy: {rule_eval['accuracy_pct']}%")
    print(f"  Honesty: {rule_eval['ambiguous_honesty_pct']}%")
    print(f"  Misdiagnosed: {rule_eval['misdiagnosed']}")

    # === 2. Fine-tuned model ===
    print("\n=== Loading fine-tuned model ===")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    print("=== Running fine-tuned model on test split ===")
    t0 = time.monotonic()
    ft_preds = _run_finetuned_model(model, tokenizer, test_incidents, device)
    ft_time = time.monotonic() - t0
    ft_eval = _evaluate(ft_preds, test_incidents, "Fine-tuned")
    print(f"  Accuracy: {ft_eval['accuracy_pct']}%")
    print(f"  Honesty: {ft_eval['ambiguous_honesty_pct']}%")
    print(f"  Misdiagnosed: {ft_eval['misdiagnosed']}")
    print(f"  Inference time: {ft_time:.1f}s")

    # === Print comparison table ===
    print("\n" + "=" * 65)
    print("HELD-OUT TEST SPLIT COMPARISON")
    print("=" * 65)
    print(f"{'Metric':<25} {'Rule-based':>12} {'Fine-tuned':>12}")
    print("-" * 65)
    print(f"{'Overall accuracy':.<25} {rule_eval['accuracy_pct']:>11.1f}% {ft_eval['accuracy_pct']:>11.1f}%")
    print(f"{'Ambiguous honesty':.<25} {rule_eval['ambiguous_honesty_pct']:>11.1f}% {ft_eval['ambiguous_honesty_pct']:>11.1f}%")
    print(f"{'Misdiagnosed':.<25} {rule_eval['misdiagnosed']:>12} {ft_eval['misdiagnosed']:>12}")
    print(f"{'Unresolved':.<25} {rule_eval['unresolved']:>12} {ft_eval['unresolved']:>12}")

    all_causes = sorted(set(list(rule_eval["per_cause_accuracy"]) + list(ft_eval["per_cause_accuracy"])))
    for cause in all_causes:
        r = rule_eval["per_cause_accuracy"].get(cause, 0.0)
        f = ft_eval["per_cause_accuracy"].get(cause, 0.0)
        print(f"  {cause:.<23} {r:>11.1f}% {f:>11.1f}%")

    # === Save results ===
    output = {
        "test_split_size": len(test_incidents),
        "rule_based": rule_eval,
        "finetuned": ft_eval,
        "finetuned_inference_seconds": round(ft_time, 1),
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
    }
    out_path = Path(__file__).parent / "finetune_eval.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nResults saved to {out_path}")

    # === Misdiagnosis detail ===
    ft_misdiag = [
        (p, inc) for p, inc in zip(ft_preds, test_incidents)
        if not inc["ground_truth"]["is_ambiguous"]
        and p["predicted_cause"] != inc["ground_truth"]["cause"]
        and p["predicted_cause"] != "unresolved"
    ]
    if ft_misdiag:
        print(f"\n=== FINE-TUNED MISDIAGNOSES ({len(ft_misdiag)}) ===")
        for p, inc in ft_misdiag:
            print(f"  {p['incident_id']}: predicted={p['predicted_cause']}, truth={inc['ground_truth']['cause']}")


if __name__ == "__main__":
    main()
