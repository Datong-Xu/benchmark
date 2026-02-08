#!/usr/bin/env python3
"""
HellaSwag Evaluation Script for GatedDeltaNet (and other fla models).

Usage:
    python eval_hellaswag.py \
        --model_path /path/to/model \
        --data_dir /path/to/hellaswag \
        --batch_size 8

HellaSwag 是一个常识推理评测：给定一段上下文，从 4 个候选续写中选出最合理的。
评测方式：计算每个候选续写在上下文条件下的 log-likelihood，选最高的作为预测。
使用 validation 集 (10,042 题)，因为 test 集没有公开标签。
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pyarrow as pa
import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# ---------------------------------------------------------------------------
# Register fla model classes
# ---------------------------------------------------------------------------
import fla  # noqa: F401

try:
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    AutoConfig.register("gated_deltanet", GatedDeltaNetConfig)
    AutoModelForCausalLM.register(GatedDeltaNetConfig, GatedDeltaNetForCausalLM)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_arrow_dataset(path: str) -> Dataset:
    """Load a HuggingFace Dataset from an Arrow IPC stream file."""
    reader = pa.ipc.open_stream(path)
    table = reader.read_all()
    return Dataset(table)


def preprocess_text(text: str) -> str:
    """Clean HellaSwag text: fix brackets, strip extra whitespace."""
    text = text.strip()
    # HellaSwag has [header] markers and some formatting artifacts
    text = re.sub(r"\[header\]\s*", "", text)
    text = re.sub(r"\[.*?\]\s*", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    contexts: list[str],
    completions_list: list[list[str]],
    device: torch.device,
) -> list[int]:
    """Score multiple (context, completion) pairs and return predicted indices.

    For each example, compute the average log-likelihood of each completion
    given the context, then pick the completion with the highest score.

    Args:
        contexts: List of context strings (one per example).
        completions_list: List of lists of completion strings
                          (4 completions per example).

    Returns:
        List of predicted indices (0-3), one per example.
    """
    predictions = []

    for ctx, completions in zip(contexts, completions_list):
        best_score = float("-inf")
        best_idx = 0

        for idx, completion in enumerate(completions):
            # Tokenize context and full sequence separately
            ctx_ids = tokenizer.encode(ctx, add_special_tokens=True)
            full_text = ctx + " " + completion
            full_ids = tokenizer.encode(full_text, add_special_tokens=True)

            # The completion tokens are the ones after the context
            ctx_len = len(ctx_ids)

            # Truncate if too long
            if len(full_ids) > 2048:
                full_ids = full_ids[:2048]
                if ctx_len >= len(full_ids):
                    # Context alone exceeds limit, skip
                    continue

            input_ids = torch.tensor([full_ids], device=device)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # Compute log-likelihood of completion tokens
            # For token at position t, the model predicts it using logits[t-1]
            # Completion tokens are at positions ctx_len .. len(full_ids)-1
            # So we use logits at positions (ctx_len-1) .. (len(full_ids)-2)
            completion_len = len(full_ids) - ctx_len
            if completion_len <= 0:
                continue

            shift_logits = logits[0, ctx_len - 1 : len(full_ids) - 1, :]  # (comp_len, vocab)
            shift_labels = input_ids[0, ctx_len:]  # (comp_len,)

            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

            # Average log-likelihood (length-normalized)
            avg_log_prob = token_log_probs.mean().item()

            if avg_log_prob > best_score:
                best_score = avg_log_prob
                best_idx = idx

        predictions.append(best_idx)

    return predictions


def evaluate_hellaswag(
    model_path: str,
    data_dir: str,
    batch_size: int = 8,
    device: str = "cuda",
    output_path: str | None = None,
) -> dict:
    """Run the full HellaSwag evaluation.

    Uses the validation set (10,042 examples) since test labels are not public.
    """
    print(f"{'='*60}")
    print(f" HellaSwag Evaluation")
    print(f"{'='*60}")
    print(f"  Model:       {model_path}")
    print(f"  Data dir:    {data_dir}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Device:      {device}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load model & tokenizer
    # ------------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # ------------------------------------------------------------------
    # 2. Load dataset (validation split)
    # ------------------------------------------------------------------
    print("Loading dataset...")
    val_path = os.path.join(data_dir, "hellaswag-validation.arrow")
    dataset = load_arrow_dataset(val_path)
    print(f"Loaded {len(dataset)} validation examples.\n")

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    correct = 0
    total = len(dataset)
    start_time = time.time()

    num_batches = (total + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="HellaSwag", unit="batch"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)

        contexts = []
        completions_list = []
        gold_labels = []

        for i in range(start_idx, end_idx):
            example = dataset[i]
            ctx = preprocess_text(example["ctx"])
            endings = [preprocess_text(e) for e in example["endings"]]
            label = int(example["label"]) if example["label"] != "" else -1

            contexts.append(ctx)
            completions_list.append(endings)
            gold_labels.append(label)

        predictions = score_completions(
            model, tokenizer, contexts, completions_list, device
        )

        for pred, gold in zip(predictions, gold_labels):
            if gold >= 0 and pred == gold:
                correct += 1

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # 4. Print results
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" HellaSwag Results  (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}\n")
    print(f"  Correct:   {correct} / {total}")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Time:      {elapsed:.1f}s  ({elapsed/total*1000:.1f} ms/example)")
    print()

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    results = {
        "model_path": model_path,
        "split": "validation",
        "elapsed_seconds": round(elapsed, 2),
        "overall_accuracy": round(accuracy, 6),
        "correct": correct,
        "total": total,
        "ms_per_example": round(elapsed / total * 1000, 1),
    }

    if output_path is None:
        model_name = Path(model_path).name
        output_path = os.path.join(data_dir, f"results_{model_name}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a language model on HellaSwag.")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the HuggingFace-format model checkpoint.",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory containing HellaSwag arrow files.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for evaluation (default: 8).",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_hellaswag(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output,
    )
