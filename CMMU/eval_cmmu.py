#!/usr/bin/env python3
"""
CMMU Evaluation Script (text-only fallback).

CMMU 是中文多模态多学科评测，原始数据包含图片。
本脚本仅使用文本部分进行评测 (忽略图片信息)，
只评测选择题 (multiple-choice / multiple-response)，跳过填空题。
评测方式：0-shot，计算各选项的 log-likelihood，选最高的作为预测。
"""

import argparse
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import torch
import torch.nn.functional as F
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

OPTION_LABELS = ["A", "B", "C", "D", "E", "F"]


def load_arrow_dataset(path: str) -> Dataset:
    return Dataset(pa.ipc.open_stream(path).read_all())


def build_prompt(question_info: str, options: list[str]) -> str:
    """Build prompt for a CMMU multiple-choice question."""
    prompt = f"题目：{question_info}\n"
    for i, opt in enumerate(options):
        if i < len(OPTION_LABELS):
            prompt += f"{OPTION_LABELS[i]}. {opt}\n"
    prompt += "答案："
    return prompt


@torch.no_grad()
def evaluate_mc(model, tokenizer, prompt, num_options, device) -> str:
    """Predict the best option label by comparing logits."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, -2048:]
    outputs = model(input_ids=input_ids)
    last_logits = outputs.logits[0, -1, :]

    best_label = "A"
    best_logit = float("-inf")
    for i in range(min(num_options, len(OPTION_LABELS))):
        label = OPTION_LABELS[i]
        tid = tokenizer.encode(label, add_special_tokens=False)[-1]
        if last_logits[tid].item() > best_logit:
            best_logit = last_logits[tid].item()
            best_label = label
    return best_label


def evaluate_cmmu(
    model_path: str,
    data_dir: str,
    device: str = "cuda",
    output_path: str | None = None,
) -> dict:
    print(f"{'='*60}")
    print(f" CMMU Evaluation (text-only, MC only)")
    print(f"{'='*60}")
    print(f"  Model:   {model_path}")
    print(f"  Device:  {device}")
    print(f"  Note:    图片信息被忽略 (纯文本模型)")
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

    ds = load_arrow_dataset(os.path.join(data_dir, "cmmu-val.arrow"))
    print(f"Loaded {len(ds)} examples.\n")

    # Filter to MC questions with options
    mc_indices = []
    for i in range(len(ds)):
        ex = ds[i]
        if ex["options"] is not None and len(ex["options"]) >= 2:
            mc_indices.append(i)
    print(f"MC questions with options: {len(mc_indices)} / {len(ds)}")

    skipped_fill = len(ds) - len(mc_indices)
    if skipped_fill > 0:
        print(f"Skipped (fill-in-the-blank / no options): {skipped_fill}\n")

    correct = 0
    total = len(mc_indices)
    subject_results = {}
    start_time = time.time()

    for idx in tqdm(mc_indices, desc="CMMU", unit="q"):
        ex = ds[idx]
        question_info = ex["question_info"] or ""
        options = ex["options"]
        gold_answers = ex["answer"]  # list of strings like ["B"] or ["BC"]
        subject = ex.get("subject", "unknown")

        prompt = build_prompt(question_info, options)
        pred = evaluate_mc(model, tokenizer, prompt, len(options), device)

        # Check if prediction is in any of the gold answers
        is_correct = False
        for ga in gold_answers:
            if pred in ga:  # e.g. pred="B" in gold="BC"
                is_correct = True
                break

        if is_correct:
            correct += 1

        if subject not in subject_results:
            subject_results[subject] = {"correct": 0, "total": 0}
        subject_results[subject]["total"] += 1
        if is_correct:
            subject_results[subject]["correct"] += 1

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0

    # Compute per-subject accuracy
    for subj in subject_results:
        r = subject_results[subj]
        r["accuracy"] = round(r["correct"] / r["total"], 6) if r["total"] > 0 else 0.0

    print(f"\n{'='*60}")
    print(f" CMMU Results  (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  MC Correct: {correct} / {total}")
    print(f"  Accuracy:   {accuracy:.2%}")
    print(f"  (Note: 纯文本模型, 图片信息缺失, 结果仅供参考)\n")
    for subj in sorted(subject_results):
        r = subject_results[subj]
        print(f"  {subj:<15} {r['correct']:>4}/{r['total']:<4}  {r['accuracy']:.2%}")
    print()

    results = {
        "model_path": model_path,
        "benchmark": "CMMU",
        "note": "text-only evaluation, images ignored",
        "elapsed_seconds": round(elapsed, 2),
        "overall_accuracy": round(accuracy, 6),
        "correct": correct,
        "total_mc": total,
        "total_all": len(ds),
        "skipped_non_mc": skipped_fill,
        "subject_results": subject_results,
    }

    if output_path is None:
        output_path = os.path.join(data_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on CMMU (text-only).")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_cmmu(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device,
        output_path=args.output,
    )
