#!/usr/bin/env python3
"""
MMLU Evaluation Script for GatedDeltaNet (and other fla models).

Usage:
    python eval_mmlu.py \
        --model_path /wekafs/datongxu/flame/exp/gdn-340M-10B-test/batch32.seqlen2048.warmup256.update1.steps256.lr3e-4 \
        --data_dir /wekafs/datongxu/benchmark/MMLU \
        --num_few_shot 5 \
        --batch_size 8

The script evaluates a causal language model on the MMLU benchmark by computing
the log-likelihood of each answer choice (A/B/C/D) and selecting the most likely one.
Results are reported per-subject and overall, and saved to a JSON file.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# ---------------------------------------------------------------------------
# Register fla model classes so that AutoModel can load GatedDeltaNet, etc.
# ---------------------------------------------------------------------------
import fla  # noqa: F401 — triggers model registration in some versions

# Explicit registration as a fallback
try:
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    AutoConfig.register("gated_deltanet", GatedDeltaNetConfig)
    AutoModelForCausalLM.register(GatedDeltaNetConfig, GatedDeltaNetForCausalLM)
except Exception:
    pass  # already registered

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANSWER_LABELS = ["A", "B", "C", "D"]

# MMLU subject categories
CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history",
        "high_school_us_history", "high_school_world_history",
        "international_law", "jurisprudence", "logical_fallacies",
        "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management",
        "marketing", "medical_genetics", "miscellaneous",
        "nutrition", "professional_accounting", "professional_medicine",
        "virology",
    ],
}

# Build reverse mapping: subject -> category
SUBJECT_TO_CATEGORY = {}
for cat, subjects in CATEGORIES.items():
    for subj in subjects:
        SUBJECT_TO_CATEGORY[subj] = cat


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_arrow_dataset(path: str) -> Dataset:
    """Load a HuggingFace Dataset from an Arrow IPC stream file."""
    reader = pa.ipc.open_stream(path)
    table = reader.read_all()
    return Dataset(table)


def format_subject(subject: str) -> str:
    """Convert subject slug to readable name, e.g. 'abstract_algebra' -> 'abstract algebra'."""
    return subject.replace("_", " ")


def format_example(example: dict, include_answer: bool = True) -> str:
    """Format a single MMLU example as a string.

    Example output:
        Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.
        A. 0
        B. 1
        C. 2
        D. 3
        Answer: B
    """
    prompt = example["question"] + "\n"
    for i, choice in enumerate(example["choices"]):
        prompt += f"{ANSWER_LABELS[i]}. {choice}\n"
    prompt += "Answer:"
    if include_answer:
        prompt += " " + ANSWER_LABELS[example["answer"]]
    return prompt


def build_prompt(subject: str, few_shot_examples: list[dict], test_example: dict) -> str:
    """Build the full prompt with few-shot examples for a test question."""
    header = (
        f"The following are multiple choice questions (with answers) "
        f"about {format_subject(subject)}.\n\n"
    )
    few_shot_str = ""
    for ex in few_shot_examples:
        few_shot_str += format_example(ex, include_answer=True) + "\n\n"

    test_str = format_example(test_example, include_answer=False)
    return header + few_shot_str + test_str


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    answer_token_ids: list[int],
    device: torch.device,
) -> list[int]:
    """Evaluate a batch of prompts and return predicted answer indices (0-3).

    For each prompt, we run the model and look at the logits at the last
    token position, then pick the answer (A/B/C/D) with the highest logit.
    """
    # Tokenize with left-padding for batched generation
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # logits shape: (batch_size, seq_len, vocab_size)
    logits = outputs.logits

    # Get logits at the last real token position for each example
    # With left-padding, the last token is always at position -1
    last_logits = logits[:, -1, :]  # (batch_size, vocab_size)

    # Extract logits for answer tokens only
    answer_logits = last_logits[:, answer_token_ids]  # (batch_size, 4)

    # Pick the answer with the highest logit
    predictions = answer_logits.argmax(dim=-1).cpu().tolist()
    return predictions


def get_few_shot_examples(dev_dataset: Dataset, subject: str, num_few_shot: int) -> list[dict]:
    """Get few-shot examples for a subject from the dev set."""
    subject_examples = [ex for ex in dev_dataset if ex["subject"] == subject]
    return subject_examples[:num_few_shot]


def evaluate_mmlu(
    model_path: str,
    data_dir: str,
    num_few_shot: int = 5,
    batch_size: int = 8,
    device: str = "cuda",
    output_path: str | None = None,
) -> dict:
    """Run the full MMLU evaluation.

    Returns a dict with per-subject accuracy, per-category accuracy, and overall accuracy.
    """
    print(f"{'='*60}")
    print(f" MMLU Evaluation")
    print(f"{'='*60}")
    print(f"  Model:       {model_path}")
    print(f"  Data dir:    {data_dir}")
    print(f"  Few-shot:    {num_few_shot}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Device:      {device}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load model & tokenizer
    # ------------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Use left-side padding for batched evaluation
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

    # Get token IDs for answer labels A, B, C, D
    answer_token_ids = []
    for label in ANSWER_LABELS:
        # Encode just the label letter; take the last token to avoid BOS
        ids = tokenizer.encode(label, add_special_tokens=False)
        answer_token_ids.append(ids[-1])
    print(f"Answer token IDs: {dict(zip(ANSWER_LABELS, answer_token_ids))}\n")

    # ------------------------------------------------------------------
    # 2. Load datasets
    # ------------------------------------------------------------------
    print("Loading datasets...")
    dev_path = os.path.join(data_dir, "mmlu-dev.arrow")
    test_path = os.path.join(data_dir, "mmlu-test.arrow")

    dev_dataset = load_arrow_dataset(dev_path)
    test_dataset = load_arrow_dataset(test_path)

    # Group test examples by subject
    subject_examples: dict[str, list[dict]] = defaultdict(list)
    for example in test_dataset:
        subject_examples[example["subject"]].append(example)

    subjects = sorted(subject_examples.keys())
    print(f"Loaded {len(test_dataset)} test examples across {len(subjects)} subjects.\n")

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    subject_results: dict[str, dict] = {}
    total_correct = 0
    total_count = 0
    start_time = time.time()

    for subject in tqdm(subjects, desc="Subjects", unit="subj"):
        examples = subject_examples[subject]
        few_shot = get_few_shot_examples(dev_dataset, subject, num_few_shot)

        # Build all prompts for this subject
        prompts = [
            build_prompt(subject, few_shot, ex) for ex in examples
        ]
        gold_answers = [ex["answer"] for ex in examples]

        # Run in batches
        predictions = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_preds = evaluate_batch(
                model, tokenizer, batch_prompts, answer_token_ids, device
            )
            predictions.extend(batch_preds)

        # Compute accuracy for this subject
        correct = sum(p == g for p, g in zip(predictions, gold_answers))
        accuracy = correct / len(gold_answers)
        subject_results[subject] = {
            "correct": correct,
            "total": len(gold_answers),
            "accuracy": accuracy,
        }
        total_correct += correct
        total_count += len(gold_answers)

    elapsed = time.time() - start_time
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

    # ------------------------------------------------------------------
    # 4. Aggregate per-category results
    # ------------------------------------------------------------------
    category_results: dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_correct = 0
        cat_total = 0
        for subj in CATEGORIES[cat]:
            if subj in subject_results:
                cat_correct += subject_results[subj]["correct"]
                cat_total += subject_results[subj]["total"]
        if cat_total > 0:
            category_results[cat] = {
                "correct": cat_correct,
                "total": cat_total,
                "accuracy": cat_correct / cat_total,
            }

    # ------------------------------------------------------------------
    # 5. Print results
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" Results  (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}\n")

    # Per-category
    print(f"{'Category':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)
    for cat in ["STEM", "Humanities", "Social Sciences", "Other"]:
        if cat in category_results:
            r = category_results[cat]
            print(f"{cat:<20} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>10.2%}")
    print("-" * 50)
    print(f"{'Overall':<20} {total_correct:>8} {total_count:>8} {overall_accuracy:>10.2%}")
    print()

    # Per-subject (sorted by accuracy)
    print(f"\n{'Subject':<45} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 75)
    for subj in sorted(subject_results, key=lambda s: subject_results[s]["accuracy"]):
        r = subject_results[subj]
        print(f"{format_subject(subj):<45} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>10.2%}")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    results = {
        "model_path": model_path,
        "num_few_shot": num_few_shot,
        "elapsed_seconds": elapsed,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_count": total_count,
        "category_results": category_results,
        "subject_results": subject_results,
    }

    if output_path is None:
        model_name = Path(model_path).parent.name
        output_path = os.path.join(data_dir, f"results_{model_name}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a language model on MMLU.")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the HuggingFace-format model checkpoint.",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory containing MMLU arrow files.",
    )
    parser.add_argument(
        "--num_few_shot", type=int, default=5,
        help="Number of few-shot examples per subject (default: 5).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for evaluation (default: 8).",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (default: cuda if available).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results (default: auto-generated in data_dir).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_mmlu(
        model_path=args.model_path,
        data_dir=args.data_dir,
        num_few_shot=args.num_few_shot,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output,
    )
