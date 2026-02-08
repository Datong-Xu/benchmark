#!/usr/bin/env python3
"""
CLUE-AFQMC Evaluation Script.

CLUE-AFQMC 是中文语义相似度判断任务 (蚂蚁金融语义相似度)。
给定两个句子，判断是否语义相似 (0=不相似, 1=相似)。
评测方式：few-shot，比较 "是"/"否" 的 log-likelihood。
使用 validation 集评测 (4,316 题)。
"""

import argparse
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

import fla  # noqa: F401

try:
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    AutoConfig.register("gated_deltanet", GatedDeltaNetConfig)
    AutoModelForCausalLM.register(GatedDeltaNetConfig, GatedDeltaNetForCausalLM)
except Exception:
    pass

LABEL_MAP = {0: "否", 1: "是"}


def load_arrow_dataset(path: str) -> Dataset:
    return Dataset(pa.ipc.open_stream(path).read_all())


def format_example(example: dict, include_answer: bool = True) -> str:
    prompt = (
        f"句子一：{example['sentence1']}\n"
        f"句子二：{example['sentence2']}\n"
        f"这两个句子语义相似吗？"
    )
    if include_answer:
        prompt += LABEL_MAP[example["label"]]
    return prompt


def build_prompt(few_shot: list[dict], test_example: dict) -> str:
    header = '判断以下两个句子是否语义相似，回答"是"或"否"。\n\n'
    parts = [format_example(ex, include_answer=True) for ex in few_shot]
    parts.append(format_example(test_example, include_answer=False))
    return header + "\n\n".join(parts)


@torch.no_grad()
def evaluate_example(model, tokenizer, prompt, device) -> int:
    """Return 0 or 1 by comparing logits of '是' vs '否'."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, -2048:]
    outputs = model(input_ids=input_ids)
    last_logits = outputs.logits[0, -1, :]

    yes_id = tokenizer.encode("是", add_special_tokens=False)[-1]
    no_id = tokenizer.encode("否", add_special_tokens=False)[-1]

    return 1 if last_logits[yes_id].item() > last_logits[no_id].item() else 0


def evaluate_clue(
    model_path: str,
    data_dir: str,
    num_few_shot: int = 8,
    device: str = "cuda",
    output_path: str | None = None,
) -> dict:
    print(f"{'='*60}")
    print(f" CLUE-AFQMC Evaluation")
    print(f"{'='*60}")
    print(f"  Model:     {model_path}")
    print(f"  Few-shot:  {num_few_shot}")
    print(f"  Device:    {device}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    train_ds = load_arrow_dataset(os.path.join(data_dir, "clue-train.arrow"))
    val_ds = load_arrow_dataset(os.path.join(data_dir, "clue-validation.arrow"))

    # Balanced few-shot: pick equal number of positive and negative examples
    pos = [train_ds[i] for i in range(len(train_ds)) if train_ds[i]["label"] == 1]
    neg = [train_ds[i] for i in range(len(train_ds)) if train_ds[i]["label"] == 0]
    half = num_few_shot // 2
    few_shot = neg[:half] + pos[:half]
    # interleave
    few_shot_interleaved = []
    for i in range(half):
        if i < len(neg[:half]):
            few_shot_interleaved.append(neg[i])
        if i < len(pos[:half]):
            few_shot_interleaved.append(pos[i])
    few_shot = few_shot_interleaved[:num_few_shot]

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Few-shot: {len(few_shot)}\n")

    correct = 0
    total = len(val_ds)
    tp = fp = tn = fn = 0
    start_time = time.time()

    for i in tqdm(range(total), desc="CLUE-AFQMC", unit="pair"):
        example = val_ds[i]
        prompt = build_prompt(few_shot, example)
        pred = evaluate_example(model, tokenizer, prompt, device)
        gold = example["label"]

        if pred == gold:
            correct += 1
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 0:
            tn += 1
        else:
            fn += 1

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f" CLUE-AFQMC Results  (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Accuracy:   {accuracy:.2%}  ({correct}/{total})")
    print(f"  Precision:  {precision:.2%}")
    print(f"  Recall:     {recall:.2%}")
    print(f"  F1:         {f1:.2%}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}\n")

    results = {
        "model_path": model_path,
        "benchmark": "CLUE-AFQMC",
        "num_few_shot": num_few_shot,
        "elapsed_seconds": round(elapsed, 2),
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "correct": correct,
        "total": total,
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }

    if output_path is None:
        output_path = os.path.join(data_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on CLUE-AFQMC.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument("--num_few_shot", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_clue(
        model_path=args.model_path,
        data_dir=args.data_dir,
        num_few_shot=args.num_few_shot,
        device=args.device,
        output_path=args.output,
    )
