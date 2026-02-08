#!/usr/bin/env python3
"""
C-Eval Evaluation Script.

C-Eval 是中文多学科考试评测，涵盖多个学科领域。
评测方式：5-shot (用 dev 集), 比较 A/B/C/D 选项的 logit，选最高的作为预测。
在 val 集上评测 (test 集标签可能不完整)。
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

ANSWER_LABELS = ["A", "B", "C", "D"]


def load_arrow_dataset(path: str) -> Dataset:
    return Dataset(pa.ipc.open_stream(path).read_all())


def format_example(example: dict, include_answer: bool = True) -> str:
    prompt = f"题目：{example['question']}\n"
    prompt += f"A. {example['A']}\n"
    prompt += f"B. {example['B']}\n"
    prompt += f"C. {example['C']}\n"
    prompt += f"D. {example['D']}\n"
    prompt += "答案："
    if include_answer:
        prompt += example["answer"]
    return prompt


def build_prompt(few_shot: list[dict], test_example: dict) -> str:
    header = "以下是中国考试的单项选择题，请选出正确答案。\n\n"
    parts = [format_example(ex, include_answer=True) for ex in few_shot]
    parts.append(format_example(test_example, include_answer=False))
    return header + "\n\n".join(parts)


@torch.no_grad()
def evaluate_example(model, tokenizer, prompt, device) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, -2048:]
    outputs = model(input_ids=input_ids)
    last_logits = outputs.logits[0, -1, :]

    best_label = "A"
    best_logit = float("-inf")
    for label in ANSWER_LABELS:
        tid = tokenizer.encode(label, add_special_tokens=False)[-1]
        if last_logits[tid].item() > best_logit:
            best_logit = last_logits[tid].item()
            best_label = label
    return best_label


def evaluate_ceval(
    model_path: str,
    data_dir: str,
    num_few_shot: int = 5,
    device: str = "cuda",
    output_path: str | None = None,
) -> dict:
    print(f"{'='*60}")
    print(f" C-Eval Evaluation")
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

    # Use dev for few-shot, val for evaluation
    dev_ds = load_arrow_dataset(os.path.join(data_dir, "ceval-exam-dev.arrow"))
    val_ds = load_arrow_dataset(os.path.join(data_dir, "ceval-exam-val.arrow"))
    few_shot = [dev_ds[i] for i in range(min(num_few_shot, len(dev_ds)))]

    # Also evaluate on test set if it has answers
    test_ds = load_arrow_dataset(os.path.join(data_dir, "ceval-exam-test.arrow"))
    test_has_answers = any(test_ds[i]["answer"].strip() != "" for i in range(min(5, len(test_ds))))

    eval_sets = {"val": val_ds}
    if test_has_answers:
        eval_sets["test"] = test_ds

    print(f"Dev (few-shot): {len(dev_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Evaluating on: {', '.join(eval_sets.keys())}\n")

    all_results = {}
    start_time = time.time()

    for split_name, split_ds in eval_sets.items():
        correct = 0
        total = len(split_ds)

        for i in tqdm(range(total), desc=f"C-Eval ({split_name})", unit="q"):
            example = split_ds[i]
            if not example["answer"].strip():
                total -= 1
                continue
            prompt = build_prompt(few_shot, example)
            pred = evaluate_example(model, tokenizer, prompt, device)
            if pred == example["answer"]:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        all_results[split_name] = {
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 6),
        }
        print(f"  {split_name}: {correct}/{total} = {accuracy:.2%}")

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f" C-Eval Results  (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")
    for split_name, r in all_results.items():
        print(f"  {split_name}: {r['correct']}/{r['total']} = {r['accuracy']:.2%}")
    print()

    results = {
        "model_path": model_path,
        "benchmark": "C-Eval",
        "num_few_shot": num_few_shot,
        "elapsed_seconds": round(elapsed, 2),
        **all_results,
    }

    if output_path is None:
        output_path = os.path.join(data_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on C-Eval.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument("--num_few_shot", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_ceval(
        model_path=args.model_path,
        data_dir=args.data_dir,
        num_few_shot=args.num_few_shot,
        device=args.device,
        output_path=args.output,
    )
