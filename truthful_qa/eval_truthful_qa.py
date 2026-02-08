#!/usr/bin/env python3
"""
TruthfulQA Evaluation Script (MC1).

TruthfulQA 测试模型抵抗常见误解/幻觉的能力，共 817 题。
评测方式 (MC1)：对每道题，将 Best Answer (正确) 和每个 Incorrect Answer
拼成候选，计算各候选在问题条件下的 log-likelihood，
如果正确答案得分最高则判定正确。

参考: https://arxiv.org/abs/2109.07958
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


def parse_answers(answer_str: str) -> list[str]:
    """Parse semicolon-separated answer strings."""
    return [a.strip() for a in answer_str.split(";") if a.strip()]


@torch.no_grad()
def score_completion(
    model, tokenizer, question: str, answer: str, device: str,
) -> float:
    """Compute average log-likelihood of the answer given the question."""
    prompt = f"Q: {question}\nA:"
    full_text = f"{prompt} {answer}"

    ctx_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_ids = tokenizer.encode(full_text, add_special_tokens=True)

    ctx_len = len(ctx_ids)
    if len(full_ids) > 2048:
        full_ids = full_ids[:2048]
    if ctx_len >= len(full_ids):
        return float("-inf")

    input_ids = torch.tensor([full_ids], device=device)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    shift_logits = logits[0, ctx_len - 1 : len(full_ids) - 1, :]
    shift_labels = input_ids[0, ctx_len:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
    return token_log_probs.mean().item()


def evaluate_truthful_qa(
    model_path: str,
    data_dir: str,
    device: str = "cuda",
    output_path: str | None = None,
) -> dict:
    print(f"{'='*60}")
    print(f" TruthfulQA Evaluation (MC1)")
    print(f"{'='*60}")
    print(f"  Model:   {model_path}")
    print(f"  Device:  {device}")
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

    ds = load_arrow_dataset(os.path.join(data_dir, "truthful_qa-train.arrow"))
    print(f"Loaded {len(ds)} questions.\n")

    mc1_correct = 0
    total = len(ds)
    start_time = time.time()

    for i in tqdm(range(total), desc="TruthfulQA", unit="q"):
        example = ds[i]
        question = example["Question"]
        best_answer = example["Best Answer"]
        incorrect_answers = parse_answers(example["Incorrect Answers"])

        # MC1: best_answer vs all incorrect answers
        all_answers = [best_answer] + incorrect_answers
        scores = [
            score_completion(model, tokenizer, question, ans, device)
            for ans in all_answers
        ]

        # Correct if best_answer (index 0) has the highest score
        if scores.index(max(scores)) == 0:
            mc1_correct += 1

    elapsed = time.time() - start_time
    mc1_accuracy = mc1_correct / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f" TruthfulQA Results  (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  MC1 Correct:  {mc1_correct} / {total}")
    print(f"  MC1 Accuracy: {mc1_accuracy:.2%}\n")

    results = {
        "model_path": model_path,
        "benchmark": "TruthfulQA",
        "metric": "MC1",
        "elapsed_seconds": round(elapsed, 2),
        "mc1_accuracy": round(mc1_accuracy, 6),
        "mc1_correct": mc1_correct,
        "total": total,
    }

    if output_path is None:
        output_path = os.path.join(data_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on TruthfulQA (MC1).")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_truthful_qa(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device,
        output_path=args.output,
    )
