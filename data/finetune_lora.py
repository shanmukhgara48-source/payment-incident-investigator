"""LoRA fine-tune Qwen2.5-0.5B-Instruct on payment incident diagnosis data.

Usage:
    .venv/bin/python -m data.finetune_lora

Runs on Apple MPS (M-series GPU). With 8GB RAM and a 0.5B model + LoRA,
this should complete in ~15-30 minutes for 3 epochs over 212 examples.

Output: data/finetune_model/ (merged LoRA weights)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path(__file__).parent / "finetune_output"
MERGED_DIR = Path(__file__).parent / "finetune_model"


def load_training_data() -> Dataset:
    """Load the JSONL training data as a HuggingFace Dataset."""
    data_path = Path(__file__).parent / "finetune_train.jsonl"
    if not data_path.exists():
        print("ERROR: Run data.prepare_finetune first")
        sys.exit(1)

    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))

    # Convert to the format trl expects
    texts = []
    for rec in records:
        messages = rec["messages"]
        texts.append({"messages": messages})

    return Dataset.from_list(texts)


def main() -> None:
    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Using Apple MPS (M-series GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA GPU")
    else:
        device = "cpu"
        print(f"WARNING: No GPU available, using CPU (will be slow)")

    print(f"Base model: {BASE_MODEL}")
    print()

    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,  # MPS needs float32
        trust_remote_code=True,
    )

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Load dataset
    print("Loading training data...")
    train_dataset = load_training_data()
    print(f"Training examples: {len(train_dataset)}")

    # Training arguments
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=5,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,  # MPS doesn't support fp16 training well
        bf16=False,
        max_length=512,
        packing=True,  # Pack short sequences together for speed
        dataloader_pin_memory=False,  # Required for MPS
        report_to="none",
    )

    # Create trainer
    print("Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Train
    print("\n=== Starting fine-tuning ===")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  LoRA rank: {lora_config.r}")
    print()

    result = trainer.train()

    print(f"\n=== Training complete ===")
    print(f"  Total steps: {result.global_step}")
    print(f"  Training loss: {result.training_loss:.4f}")
    print(f"  Runtime: {result.metrics.get('train_runtime', 0):.1f}s")

    # Save LoRA adapter
    print("\nSaving LoRA adapter...")
    trainer.save_model(str(OUTPUT_DIR / "final"))

    # Merge LoRA weights into base model and save
    print("Merging LoRA weights into base model...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"Merged model saved to {MERGED_DIR}")

    # Save training metrics
    metrics = {
        "base_model": BASE_MODEL,
        "training_examples": len(train_dataset),
        "epochs": int(training_args.num_train_epochs),
        "total_steps": result.global_step,
        "final_loss": round(result.training_loss, 4),
        "runtime_seconds": round(result.metrics.get("train_runtime", 0), 1),
        "lora_rank": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "learning_rate": training_args.learning_rate,
        "device": device,
    }
    metrics_path = Path(__file__).parent / "finetune_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
