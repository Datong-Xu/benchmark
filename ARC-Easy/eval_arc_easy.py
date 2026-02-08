#!/usr/bin/env python3
"""
ARC-Easy Evaluation Script.

ARC-Easy 是科学推理选择题 (简单子集)，共 2,376 道测试题。
评测方式：25-shot, 计算各选项的 log-likelihood，选最高的作为预测。
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


def load_arrow_dataset(path: str) -> Dataset:
    return Dataset(pa.ipc.open_stream(path).read_all())


def format_example(example: dict, include_answer: bool = True) -> str:
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    prompt = f"Question: {example['question']}\n"
    for label, text in zip(labels, texts):
        prompt += f"{label}. {text}\n"
    prompt += "Answer:"
    if include_answer:
        prompt += f" {example['answerKey']}"
    return prompt


def build_prompt(few_shot_examples: list[dict], test_example: dict) -> str:
    header = "The following are multiple choice questions about science.\n\n"
    few_shot_str = ""
    for ex in few_shot_examples:
        few_shot_str += format_example(ex, include_answer=True) + "\n\n"
    test_str = format_example(test_example, include_answer=False)
    return header + few_shot_str + test_str


@torch.no_grad()
def evaluate_example(
    model, tokenizer, prompt: str,
    answer_labels: list[str], device: str,
) -> int:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, -2048:]
    outputs = model(input_ids=input_ids)
    last_logits = outputs.logits[0, -1, :]

    best_idx = 0
    best_logit = float("-inf")
    for i, label in enumerate(answer_labels):
        tid = tokenizer.encode(label, add_special_tokens=False)[-1]
        if last_logits[tid].item() > best_logit:
            best_logit = last_logits[tid].item()
            best_idx = i
    return best_idx


def evaluate_arc_easy(
    model_path: str,
    data_dir: str,
    num_few_shot: int = 25,
    batch_size: int = 8,
    device: str = "cuda",
    output_path: str | None = None,
) -> dict:
    print(f"{'='*60}")
    print(f" ARC-Easy Evaluation")
    print(f"{'='*60}")
    print(f"  Model:     {model_path}")
    print(f"  Few-shot:  {num_few_shot}")
    print(f"  Device:    {device}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    train_ds = load_arrow_dataset(os.path.join(data_dir, "ai2_arc-train.arrow"))
    test_ds = load_arrow_dataset(os.path.join(data_dir, "ai2_arc-test.arrow"))
    few_shot = [train_ds[i] for i in range(min(num_few_shot, len(train_ds)))]
    print(f"Test: {len(test_ds)} examples, Few-shot: {len(few_shot)}\n")

    correct = 0
    total = len(test_ds)
    start_time = time.time()

    for i in tqdm(range(total), desc="ARC-Easy", unit="q"):
        example = test_ds[i]
        prompt = build_prompt(few_shot, example)
        labels = example["choices"]["label"]
        pred_idx = evaluate_example(model, tokenizer, prompt, labels, device)
        if labels[pred_idx] == example["answerKey"]:
            correct += 1

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f" ARC-Easy Results  (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Correct:  {correct} / {total}")
    print(f"  Accuracy: {accuracy:.2%}\n")

    results = {
        "model_path": model_path,
        "benchmark": "ARC-Easy",
        "num_few_shot": num_few_shot,
        "elapsed_seconds": round(elapsed, 2),
        "overall_accuracy": round(accuracy, 6),
        "correct": correct,
        "total": total,
    }

    if output_path is None:
        output_path = os.path.join(data_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on ARC-Easy.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument("--num_few_shot", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_arc_easy(
        model_path=args.model_path,
        data_dir=args.data_dir,
        num_few_shot=args.num_few_shot,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output,
    )
