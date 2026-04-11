import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_and_convert_dataset(csv_path: str) -> Dataset:
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

    return Dataset.from_list(examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for reward model and save predictions/metrics.")
    parser.add_argument("--model-path", default=None, help="Path to trained reward model directory.")
    parser.add_argument(
        "--download-from-hf",
        action="store_true",
        help="If set, download model files from Hugging Face before inference (instead of --model-path).",
    )
    parser.add_argument(
        "--hf-repo-id",
        default="kpetyxova/towards-reward-modeling-tutors",
        help="Hugging Face repo id (default: kpetyxova/towards-reward-modeling-tutors).",
    )
    parser.add_argument("--test-csv", required=True, help="Path to CSV used for inference.")
    parser.add_argument("--output-dir", required=True, help="Directory to store predictions and metrics.")
    parser.add_argument("--max-length", type=int, default=1024, help="Truncation length for chat template encoding.")
    parser.add_argument("--predictions-file", default="test_inference_predictions.csv")
    parser.add_argument("--metrics-file", default="test_inference_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.download_from_hf and args.model_path:
        raise ValueError("Choose one model source: either --model-path OR --download-from-hf (not both).")

    use_hf = args.download_from_hf or args.model_path is None

    if use_hf:
        local_model_dir = snapshot_download(
            repo_id=args.hf_repo_id,
        )
        model_source = local_model_dir
        print(f"Model source: Hugging Face ({args.hf_repo_id})")
        print(f"Downloaded model to: {local_model_dir}")
    else:
        if not os.path.isdir(args.model_path):
            raise ValueError(
                f"--model-path does not exist or is not a directory: {args.model_path}. "
                "Provide a valid local model folder or omit --model-path to use the default HF repo."
            )
        model_source = args.model_path
        print(f"Model source: local folder ({model_source})")

    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    model.to(device)
    model.eval()

    print(f"Tokenizer padding side: {tokenizer.padding_side}")
    print(f"Tokenizer truncation side: {tokenizer.truncation_side}")

    test_dataset = load_and_convert_dataset(args.test_csv)
    test_df_original = pd.read_csv(args.test_csv)
    assert len(test_dataset) == len(test_df_original), "Mismatch between converted dataset and input CSV."
    print(f"Loaded {len(test_dataset)} test examples")

    def to_device(batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for key, value in batch.items():
            out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return out

    def coerce_encoding(enc: Any) -> Dict[str, torch.Tensor]:
        if hasattr(enc, "data") and isinstance(getattr(enc, "data", None), dict):
            enc = dict(enc)

        if isinstance(enc, dict):
            if "input_ids" in enc and isinstance(enc["input_ids"], torch.Tensor):
                if "attention_mask" not in enc or not isinstance(enc["attention_mask"], torch.Tensor):
                    enc["attention_mask"] = torch.ones_like(enc["input_ids"])
                return enc
            if "input_ids" in enc:
                input_ids = enc["input_ids"]
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids, dtype=torch.long)
                enc["input_ids"] = input_ids
                enc["attention_mask"] = torch.ones_like(input_ids)
                return enc

        if isinstance(enc, torch.Tensor):
            if enc.ndim == 1:
                enc = enc.unsqueeze(0)
            return {"input_ids": enc, "attention_mask": torch.ones_like(enc)}

        if isinstance(enc, (list, tuple)):
            enc = torch.tensor(enc, dtype=torch.long)
            if enc.ndim == 1:
                enc = enc.unsqueeze(0)
            return {"input_ids": enc, "attention_mask": torch.ones_like(enc)}

        raise TypeError(f"Unexpected tokenizer output type: {type(enc)}")

    def ensure_batched(enc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key in ("input_ids", "attention_mask"):
            if key in enc and isinstance(enc[key], torch.Tensor) and enc[key].ndim == 1:
                enc[key] = enc[key].unsqueeze(0)
        return enc

    def encode_messages(messages: Any, max_length: int) -> Dict[str, torch.Tensor]:
        enc = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        return ensure_batched(coerce_encoding(enc))

    def tensors_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
        if a.ndim == 1:
            a = a.unsqueeze(0)
        if b.ndim == 1:
            b = b.unsqueeze(0)
        if a.shape != b.shape:
            return False
        return bool(torch.equal(a, b))

    def score_encoded(enc: Dict[str, torch.Tensor]) -> float:
        with torch.no_grad():
            out = model(**to_device(enc))
        return float(out.logits.squeeze())

    rows = []
    correct_non_tie = 0
    non_tie_total = 0
    correct_all = 0
    ties_count = 0
    identical_count = 0

    for idx in tqdm(range(len(test_dataset)), desc="Scoring responses"):
        ex = test_dataset[idx]
        orig = test_df_original.iloc[idx]

        enc_ch = encode_messages(ex["chosen"], max_length=args.max_length)
        enc_rj = encode_messages(ex["rejected"], max_length=args.max_length)

        ids_ch, ids_rj = enc_ch["input_ids"], enc_rj["input_ids"]
        is_identical = tensors_identical(ids_ch, ids_rj)
        if is_identical:
            identical_count += 1

        score_ch = score_encoded(enc_ch)
        score_rj = score_encoded(enc_rj)

        is_tie = abs(score_ch - score_rj) == 0.0
        if is_tie:
            ties_count += 1

        metric_included = (not is_tie) and (not is_identical)
        if metric_included:
            non_tie_total += 1
            if score_ch > score_rj:
                correct_non_tie += 1

        if score_ch > score_rj:
            correct_all += 1

        gold_label = orig.get("label", np.nan)
        if gold_label in [0, 1]:
            if gold_label == 1:
                score_a, score_b = score_ch, score_rj
            else:
                score_a, score_b = score_rj, score_ch
            pred_label = 1 if score_a > score_b else (0 if score_a < score_b else np.nan)
        else:
            score_a = np.nan
            score_b = np.nan
            pred_label = np.nan

        rows.append(
            {
                "idx": idx,
                "conversation_id": orig.get("conversation_id", ""),
                "model_a": orig.get("model_a", ""),
                "model_b": orig.get("model_b", ""),
                "conversation_history": orig.get("conversation_history", ""),
                "gold_solution": orig.get("gold_solution", ""),
                "response_a": orig.get("response_a", ""),
                "response_b": orig.get("response_b", ""),
                "len_tokens_chosen": int(enc_ch["input_ids"].shape[-1]),
                "len_tokens_rejected": int(enc_rj["input_ids"].shape[-1]),
                "is_identical_after_truncation": bool(is_identical),
                "score_chosen": score_ch,
                "score_rejected": score_rj,
                "is_tie": bool(is_tie),
                "metric_included": bool(metric_included),
                "pair_correct_chosen_over_rejected": bool(score_ch > score_rj),
                "gold_label": gold_label,
                "score_a": score_a,
                "score_b": score_b,
                "pred_label": pred_label,
                "pred_correct_vs_label": (
                    pred_label == gold_label
                    if (gold_label in [0, 1] and not np.isnan(pred_label))
                    else False
                ),
            }
        )

    pred_df = pd.DataFrame(rows)
    predictions_path = Path(args.output_dir) / args.predictions_file
    metrics_json_path = Path(args.output_dir) / args.metrics_file
    metrics_history_csv = Path(args.output_dir) / "metrics_history.csv"

    pred_df.to_csv(predictions_path, index=False)
    print(f"\nSaved predictions to: {predictions_path}")

    total = len(test_dataset)
    acc_excl_ties = correct_non_tie / max(1, non_tie_total)
    acc_incl_ties_as_wrong = correct_all / max(1, total)

    mask_eval = pred_df["metric_included"] == True
    label_acc_excl_ties = (
        (pred_df.loc[mask_eval, "pred_label"] == pred_df.loc[mask_eval, "gold_label"]).mean()
        if mask_eval.any()
        else float("nan")
    )
    label_acc_incl_ties = (pred_df["pred_label"] == pred_df["gold_label"]).sum() / max(1, len(pred_df))

    score_a_mean = float(np.nanmean(pred_df["score_a"]))
    score_b_mean = float(np.nanmean(pred_df["score_b"]))
    score_a_std = float(np.nanstd(pred_df["score_a"]))
    score_b_std = float(np.nanstd(pred_df["score_b"]))
    abs_margin_mean = float(np.nanmean(np.abs(pred_df["score_a"] - pred_df["score_b"])))

    run_ts = datetime.now(timezone.utc).isoformat()
    model_name = getattr(getattr(model, "config", None), "_name_or_path", "unknown_model")
    metrics = {
        "run_timestamp_utc": run_ts,
        "model_name": model_name,
        "dataset_size": total,
        "max_length": args.max_length,
        "tokenizer_padding_side": tokenizer.padding_side,
        "tokenizer_truncation_side": tokenizer.truncation_side,
        "non_tie_pairs": non_tie_total,
        "ties": ties_count,
        "identical_after_truncation": identical_count,
        "accuracy_excluding_ties": round(acc_excl_ties, 6),
        "accuracy_including_ties_as_wrong": round(acc_incl_ties_as_wrong, 6),
        "label_accuracy_excluding_ties": round(float(label_acc_excl_ties), 6)
        if not np.isnan(label_acc_excl_ties)
        else None,
        "label_accuracy_including_ties_as_wrong": round(float(label_acc_incl_ties), 6)
        if not np.isnan(label_acc_incl_ties)
        else None,
        "score_a_mean": round(score_a_mean, 6),
        "score_a_std": round(score_a_std, 6),
        "score_b_mean": round(score_b_mean, 6),
        "score_b_std": round(score_b_std, 6),
        "abs_margin_mean": round(abs_margin_mean, 6),
    }

    with metrics_json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)
    print(f"Saved metrics to: {metrics_json_path}")

    metrics_row = {
        "run_timestamp_utc": run_ts,
        "model_name": model_name,
        "dataset_size": total,
        "max_length": args.max_length,
        "non_tie_pairs": non_tie_total,
        "ties": ties_count,
        "identical_after_truncation": identical_count,
        "acc_excl_ties": metrics["accuracy_excluding_ties"],
        "acc_incl_ties_as_wrong": metrics["accuracy_including_ties_as_wrong"],
        "label_acc_excl_ties": metrics["label_accuracy_excluding_ties"],
        "label_acc_incl_ties_as_wrong": metrics["label_accuracy_including_ties_as_wrong"],
        "score_a_mean": metrics["score_a_mean"],
        "score_b_mean": metrics["score_b_mean"],
        "abs_margin_mean": metrics["abs_margin_mean"],
        "tokenizer_padding_side": metrics["tokenizer_padding_side"],
        "tokenizer_truncation_side": metrics["tokenizer_truncation_side"],
    }
    hist_df = pd.DataFrame([metrics_row])
    if metrics_history_csv.exists():
        hist_df.to_csv(metrics_history_csv, mode="a", header=False, index=False)
    else:
        hist_df.to_csv(metrics_history_csv, index=False)
    print(f"Appended metrics history to: {metrics_history_csv}")

    print("\n=== Inference Summary ===")
    print(f"Total examples: {total}")
    print(f"Ties: {ties_count}")
    print(f"Identical after truncation: {identical_count}")
    print(f"Non-tie pairs: {non_tie_total}")
    print(f"Accuracy excluding ties: {acc_excl_ties:.4f}")
    print(f"Accuracy including ties as wrong: {acc_incl_ties_as_wrong:.4f}")


if __name__ == "__main__":
    main()
