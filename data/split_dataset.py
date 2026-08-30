"""Generate an expanded dataset and split into train/val/test for fine-tuning.

Usage:
    .venv/bin/python -m data.split_dataset [--count 300] [--seed 20260827]

Produces:
    data/finetune_train.json
    data/finetune_val.json
    data/finetune_test.json
    data/finetune_meta.json   (split sizes, per-cause counts, overlap check)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from data.simulate import AMBIGUOUS_FRACTION, SEED, generate_dataset


def stratified_split(
    incidents: list[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split incidents into train/val/test, stratified by ground-truth cause."""
    rng = random.Random(seed + 9999)

    # Group by cause (including "unresolved" for ambiguous)
    by_cause: dict[str, list[dict]] = defaultdict(list)
    for inc in incidents:
        cause = inc["ground_truth"]["cause"]
        by_cause[cause].append(inc)

    train, val, test = [], [], []

    for cause, group in sorted(by_cause.items()):
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_frac))
        n_val = max(1, round(n * val_frac))
        n_test = n - n_train - n_val
        if n_test < 1:
            # Steal from train if needed
            n_test = 1
            n_train = n - n_val - n_test

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    # Shuffle within each split (so they're not grouped by cause)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def count_causes(incidents: list[dict]) -> dict[str, int]:
    return dict(Counter(inc["ground_truth"]["cause"] for inc in incidents))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print(f"Generating {args.count} incidents (+ skeptic gate case)...")
    dataset = generate_dataset(
        count=args.count,
        ambiguous_fraction=AMBIGUOUS_FRACTION,
        seed=args.seed,
        include_skeptic_case=True,
    )
    incidents = dataset["incidents"]
    print(f"Total incidents: {len(incidents)}")

    train, val, test = stratified_split(incidents, seed=args.seed)

    # Verify no overlap
    train_ids = {i["incident_id"] for i in train}
    val_ids = {i["incident_id"] for i in val}
    test_ids = {i["incident_id"] for i in test}

    assert len(train_ids & val_ids) == 0, "Train/val overlap!"
    assert len(train_ids & test_ids) == 0, "Train/test overlap!"
    assert len(val_ids & test_ids) == 0, "Val/test overlap!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(incidents), "Missing incidents!"

    out_dir = Path(__file__).parent
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"finetune_{name}.json"
        path.write_text(json.dumps(split, indent=2) + "\n")
        print(f"  {name}: {len(split)} incidents -> {path}")

    meta = {
        "total": len(incidents),
        "train": {"count": len(train), "causes": count_causes(train)},
        "val": {"count": len(val), "causes": count_causes(val)},
        "test": {"count": len(test), "causes": count_causes(test)},
        "overlap_check": "PASS",
        "seed": args.seed,
        "generated_count": args.count,
    }
    meta_path = out_dir / "finetune_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nMeta saved to {meta_path}")

    # Print summary table
    print(f"\n{'Split':<8} {'Total':>6}  ", end="")
    all_causes = sorted(set(c for split in [train, val, test] for c in count_causes(split)))
    for c in all_causes:
        print(f"{c:>20}", end="")
    print()
    print("-" * (16 + 20 * len(all_causes)))
    for name, split in [("train", train), ("val", val), ("test", test)]:
        counts = count_causes(split)
        print(f"{name:<8} {len(split):>6}  ", end="")
        for c in all_causes:
            print(f"{counts.get(c, 0):>20}", end="")
        print()


if __name__ == "__main__":
    main()
