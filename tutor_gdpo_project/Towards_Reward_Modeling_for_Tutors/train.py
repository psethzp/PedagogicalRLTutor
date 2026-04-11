import argparse
import random
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback
from trl import RewardConfig, RewardTrainer


def load_and_convert_dataset(csv_path: str, seed: int = 42, shuffle: bool = False) -> Dataset:
    """
    Convert pairwise CSV to TRL RewardTrainer format.

    Expected columns:
      - conversation_history
      - gold_solution
      - response_a
      - response_b
      - label (1 => A preferred, 0 => B preferred)
    """
    df = pd.read_csv(csv_path)
    required_cols = {"conversation_history", "gold_solution", "response_a", "response_b", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    examples = []
    for _, row in df.iterrows():
        user_content = f"{row['conversation_history']}\n\nGold Solution: {row['gold_solution']}"

        if int(row["label"]) == 1:
            chosen_response = row["response_a"]
            rejected_response = row["response_b"]
        else:
            chosen_response = row["response_b"]
            rejected_response = row["response_a"]

        examples.append(
            {
                "chosen": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": chosen_response},
                ],
                "rejected": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": rejected_response},
                ],
            }
        )

    if shuffle:
        random.seed(seed)
        random.shuffle(examples)

    return Dataset.from_list(examples)


def build_reward_config(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        optim="adamw_torch",
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        max_length=args.max_length,
        dataset_num_proc=args.dataset_num_proc,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=args.report_to,
        logging_first_step=True,
        seed=args.seed,
        remove_unused_columns=False,
        disable_dropout=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reward model with TRL RewardTrainer.")
    parser.add_argument("--train-csv", required=True, help="Path to training CSV.")
    parser.add_argument("--eval-csv", required=True, help="Path to eval CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-0.5B-Instruct", help="HF model name/path.")

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=int, default=5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none", help='e.g. "none", "wandb".')
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--shuffle-train", action="store_true", help="Shuffle converted training examples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading and converting datasets...")
    train_dataset = load_and_convert_dataset(args.train_csv, seed=args.seed, shuffle=args.shuffle_train)
    eval_dataset = load_and_convert_dataset(args.eval_csv, seed=args.seed, shuffle=False)
    print(f"Train size: {len(train_dataset)}")
    print(f"Eval size: {len(eval_dataset)}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    print("Loading model...")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        torch_dtype=dtype,
    )

    config = build_reward_config(args)
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=0.0,
    )

    print("Initializing RewardTrainer...")
    trainer = RewardTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[early_stopping_callback],
    )

    print("Starting training...")
    trainer.train()

    final_model_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    print(f"\nModel saved to: {final_model_path}")
    print(f"Total parameters: {sum(p.numel() for p in trainer.model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad) / 1e6:.2f}M")


if __name__ == "__main__":
    main()
