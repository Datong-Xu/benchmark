#!/usr/bin/env python3
"""
Big-Bench Hard (BBH) Evaluation Script.

BBH 是高难度推理任务集合。每个子任务有独立的 arrow 文件。
评测方式：3-shot, 让模型生成答案，与 target 做精确匹配。

当前已下载子任务会被自动检测并评测。
"""

import argparse
import json
import os
import re
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


def load_arrow_dataset(path: str) -> Dataset:
    return Dataset(pa.ipc.open_stream(path).read_all())


def discover_tasks(data_dir: str) -> dict[str, str]:
    """Discover all BBH subtask arrow files in the data directory.

    Returns: {task_name: arrow_file_path}
    """
    tasks = {}
    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith("big_bench_hard-") and fname.endswith(".arrow"):
            task_name = fname.replace("big_bench_hard-", "").replace(".arrow", "")
            tasks[task_name] = os.path.join(data_dir, fname)
    return tasks


def build_prompt(few_shot_examples: list[dict], test_question: str) -> str:
    """Build a few-shot prompt for a BBH task."""
    parts = []
    for ex in few_shot_examples:
        parts.append(f"Q: {ex['question']}\nA: {ex['target']}")
    parts.append(f"Q: {test_question}\nA:")
    return "\n\n".join(parts)


def extract_answer(generated: str) -> str:
    """Extract answer from generated text — take the first line/sentence."""
    # Take text before first newline
    answer = generated.split("\n")[0].strip()
    # Remove trailing punctuation artifacts
    answer = answer.rstrip(".")
    return answer


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    return text.strip().lower()


@torch.no_grad()
def generate_answer(
    model, tokenizer, prompt: str, device: str, max_new_tokens: int = 64,
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if input_ids.shape[1] > 2048 - max_new_tokens:
        input_ids = input_ids[:, -(2048 - max_new_tokens):]
    prompt_len = input_ids.shape[1]

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = outputs[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def evaluate_bbh(
    model_path: str,
    data_dir: str,
    num_few_shot: int = 3,
    max_new_tokens: int = 64,
    device: str = "cuda",
    output_path: str | None = None,
) -> dict:
    print(f"{'='*60}")
    print(f" Big-Bench Hard Evaluation")
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

    # Discover subtasks
    tasks = discover_tasks(data_dir)
    if not tasks:
        print("No BBH subtask files found!")
        return {}
    print(f"Found {len(tasks)} subtask(s): {', '.join(tasks.keys())}\n")

    task_results = {}
    total_correct = 0
    total_count = 0
    start_time = time.time()

    for task_name, arrow_path in tasks.items():
        ds = load_arrow_dataset(arrow_path)
        # Use first N examples as few-shot, evaluate the rest
        few_shot = [ds[i] for i in range(min(num_few_shot, len(ds)))]
        eval_start = num_few_shot if len(ds) > num_few_shot else 0
        eval_ds = [ds[i] for i in range(eval_start, len(ds))]

        if not eval_ds:
            print(f"  [{task_name}] Not enough examples, skipped.")
            continue

        correct = 0
        for ex in tqdm(eval_ds, desc=f"  {task_name}", unit="q"):
            prompt = build_prompt(few_shot, ex["question"])
            generated = generate_answer(model, tokenizer, prompt, device, max_new_tokens)
            pred = extract_answer(generated)

            if normalize(pred) == normalize(ex["target"]):
                correct += 1

        accuracy = correct / len(eval_ds)
        task_results[task_name] = {
            "correct": correct,
            "total": len(eval_ds),
            "accuracy": round(accuracy, 6),
        }
        total_correct += correct
        total_count += len(eval_ds)
        print(f"  [{task_name}] {correct}/{len(eval_ds)} = {accuracy:.2%}")

    elapsed = time.time() - start_time
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

    print(f"\n{'='*60}")
    print(f" BBH Results  (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Overall: {total_correct}/{total_count} = {overall_accuracy:.2%}\n")

    results = {
        "model_path": model_path,
        "benchmark": "Big-Bench Hard",
        "num_few_shot": num_few_shot,
        "elapsed_seconds": round(elapsed, 2),
        "overall_accuracy": round(overall_accuracy, 6),
        "total_correct": total_correct,
        "total_count": total_count,
        "task_results": task_results,
    }

    if output_path is None:
        output_path = os.path.join(data_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on Big-Bench Hard.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument("--num_few_shot", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_bbh(
        model_path=args.model_path,
        data_dir=args.data_dir,
        num_few_shot=args.num_few_shot,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        output_path=args.output,
    )
