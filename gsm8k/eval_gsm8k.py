#!/usr/bin/env python3
"""
GSM8K Evaluation Script for GatedDeltaNet (and other fla models).

Usage:
    python eval_gsm8k.py \
        --model_path /path/to/model \
        --data_dir /path/to/gsm8k \
        --num_few_shot 8 \
        --batch_size 4 \
        --max_new_tokens 512

The script evaluates a causal language model on GSM8K by generating step-by-step
solutions and extracting the final numeric answer. The gold answer is parsed from
the "#### <number>" pattern in the dataset.
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

# ---------------------------------------------------------------------------
# Register fla model classes so that AutoModel can load GatedDeltaNet, etc.
# ---------------------------------------------------------------------------
import fla  # noqa: F401

try:
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    AutoConfig.register("gated_deltanet", GatedDeltaNetConfig)
    AutoModelForCausalLM.register(GatedDeltaNetConfig, GatedDeltaNetForCausalLM)
except Exception:
    pass  # already registered


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_arrow_dataset(path: str) -> Dataset:
    """Load a HuggingFace Dataset from an Arrow IPC stream file."""
    reader = pa.ipc.open_stream(path)
    table = reader.read_all()
    return Dataset(table)


def extract_gold_answer(answer_text: str) -> str:
    """Extract the numeric answer after '####' from a GSM8K answer string.

    Example input:  "Janet sells 16 - 3 - 4 = 9 duck eggs...\\n#### 18"
    Example output: "18"
    """
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""


def extract_predicted_answer(generated_text: str) -> str:
    """Extract the predicted numeric answer from model-generated text.

    Strategy (in order of priority):
    1. Look for "#### <number>" pattern (model learned the format)
    2. Look for "the answer is <number>" pattern
    3. Fall back to the last number in the text
    """
    # Strategy 1: #### pattern
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", generated_text)
    if match:
        return match.group(1).strip().replace(",", "")

    # Strategy 2: "the answer is" pattern
    match = re.search(r"the answer is\s*:?\s*\$?\s*(-?[\d,]+\.?\d*)", generated_text, re.IGNORECASE)
    if match:
        return match.group(1).strip().replace(",", "")

    # Strategy 3: last number in text
    numbers = re.findall(r"-?[\d,]+\.?\d*", generated_text)
    if numbers:
        return numbers[-1].strip().replace(",", "")

    return ""


def normalize_answer(answer: str) -> float | None:
    """Normalize an answer string to a float for comparison."""
    try:
        return float(answer.replace(",", ""))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
# Fixed 8-shot examples selected from the GSM8K train set (standard practice)
DEFAULT_FEW_SHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees that were planted.\n#### 6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops.\n#### 8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. He got 2 toys each from his mom and dad. So he got 2 * 2 = 4 more toys. 5 + 4 = 9.\n#### 9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29.\n#### 29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each cost 5 * 3 = 15 dollars. So she has 23 - 15 = 8 dollars left.\n#### 8"
    },
]


def format_example(example: dict, include_answer: bool = True) -> str:
    """Format a single GSM8K example."""
    text = f"Question: {example['question']}\nAnswer:"
    if include_answer:
        text += f" {example['answer']}"
    return text


def build_prompt(few_shot_examples: list[dict], test_example: dict) -> str:
    """Build the full prompt with few-shot examples for a test question."""
    parts = []
    for ex in few_shot_examples:
        parts.append(format_example(ex, include_answer=True))
    parts.append(format_example(test_example, include_answer=False))
    return "\n\n".join(parts)


def get_few_shot_from_train(train_dataset: Dataset, num_few_shot: int) -> list[dict]:
    """Get few-shot examples from the train set.

    Uses the default curated examples first, then fills from train set if needed.
    """
    if num_few_shot <= len(DEFAULT_FEW_SHOT_EXAMPLES):
        return DEFAULT_FEW_SHOT_EXAMPLES[:num_few_shot]
    # Use all defaults + supplement from train set
    extra_needed = num_few_shot - len(DEFAULT_FEW_SHOT_EXAMPLES)
    extra = [train_dataset[i] for i in range(min(extra_needed, len(train_dataset)))]
    return DEFAULT_FEW_SHOT_EXAMPLES + extra


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 512,
) -> list[str]:
    """Generate completions for a batch of prompts.

    Returns the generated text (completion only, prompt stripped).
    """
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048 - max_new_tokens,  # leave room for generation
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    prompt_length = input_ids.shape[1]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # greedy decoding
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the generated part
    generated_texts = []
    for output_ids in outputs:
        gen_ids = output_ids[prompt_length:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        generated_texts.append(text)

    return generated_texts


def evaluate_gsm8k(
    model_path: str,
    data_dir: str,
    num_few_shot: int = 8,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    device: str = "cuda",
    output_path: str | None = None,
) -> dict:
    """Run the full GSM8K evaluation.

    Returns a dict with overall accuracy and per-example details.
    """
    print(f"{'='*60}")
    print(f" GSM8K Evaluation")
    print(f"{'='*60}")
    print(f"  Model:          {model_path}")
    print(f"  Data dir:       {data_dir}")
    print(f"  Few-shot:       {num_few_shot}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Device:         {device}")
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
    # 2. Load datasets
    # ------------------------------------------------------------------
    print("Loading datasets...")
    train_path = os.path.join(data_dir, "gsm8k-train.arrow")
    test_path = os.path.join(data_dir, "gsm8k-test.arrow")

    train_dataset = load_arrow_dataset(train_path)
    test_dataset = load_arrow_dataset(test_path)
    print(f"Loaded {len(test_dataset)} test examples, {len(train_dataset)} train examples.\n")

    # Prepare few-shot examples
    few_shot_examples = get_few_shot_from_train(train_dataset, num_few_shot)
    print(f"Using {len(few_shot_examples)} few-shot examples.\n")

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    correct = 0
    total = len(test_dataset)
    details = []
    start_time = time.time()

    # Build all prompts
    all_prompts = [
        build_prompt(few_shot_examples, test_dataset[i])
        for i in range(total)
    ]
    all_gold = [
        extract_gold_answer(test_dataset[i]["answer"])
        for i in range(total)
    ]

    num_batches = (total + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="GSM8K", unit="batch"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_prompts = all_prompts[start_idx:end_idx]
        batch_gold = all_gold[start_idx:end_idx]

        # Generate
        generated_texts = generate_batch(
            model, tokenizer, batch_prompts, device, max_new_tokens
        )

        for i, (gen_text, gold) in enumerate(zip(generated_texts, batch_gold)):
            pred = extract_predicted_answer(gen_text)
            pred_norm = normalize_answer(pred)
            gold_norm = normalize_answer(gold)
            is_correct = (
                pred_norm is not None
                and gold_norm is not None
                and abs(pred_norm - gold_norm) < 1e-3
            )
            if is_correct:
                correct += 1

            details.append({
                "index": start_idx + i,
                "question": test_dataset[start_idx + i]["question"],
                "gold_answer": gold,
                "predicted_answer": pred,
                "is_correct": is_correct,
                "generated_text": gen_text[:500],  # truncate for storage
            })

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # 4. Print results
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" GSM8K Results  (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}\n")
    print(f"  Correct:   {correct} / {total}")
    print(f"  Accuracy:  {accuracy:.2%}")
    print()

    # Show some examples
    print("Sample predictions (first 5 incorrect):")
    print("-" * 60)
    wrong = [d for d in details if not d["is_correct"]]
    for d in wrong[:5]:
        print(f"  Q: {d['question'][:80]}...")
        print(f"  Gold: {d['gold_answer']}  |  Pred: {d['predicted_answer']}")
        print(f"  Generated: {d['generated_text'][:120]}...")
        print()

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    results = {
        "model_path": model_path,
        "num_few_shot": num_few_shot,
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": elapsed,
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

    # Save summary
    if output_path is None:
        model_name = Path(model_path).name
        output_path = os.path.join(data_dir, f"results_{model_name}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")

    # Save detailed per-example results alongside
    detail_path = output_path.replace(".json", "_details.json")
    with open(detail_path, "w") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"Details saved to: {detail_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a language model on GSM8K.")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the HuggingFace-format model checkpoint.",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory containing GSM8K arrow files.",
    )
    parser.add_argument(
        "--num_few_shot", type=int, default=8,
        help="Number of few-shot examples (default: 8).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for generation (default: 4).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Max tokens to generate per example (default: 512).",
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
    evaluate_gsm8k(
        model_path=args.model_path,
        data_dir=args.data_dir,
        num_few_shot=args.num_few_shot,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        output_path=args.output,
    )
