#!/usr/bin/env python3
"""
HumanEval Evaluation Script (pass@1).

HumanEval 测试代码生成能力，共 164 道 Python 编程题。
评测方式：模型根据函数签名和 docstring 生成代码补全，
然后在沙盒中执行测试用例，报告 pass@1。
"""

import argparse
import json
import multiprocessing
import os
import signal
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


# ---------------------------------------------------------------------------
# Safe code execution
# ---------------------------------------------------------------------------
def _run_code(code: str, timeout: int = 5) -> bool:
    """Execute code in a subprocess and return True if all tests pass."""
    def _exec(code_str, result_queue):
        try:
            exec_globals = {}
            exec(code_str, exec_globals)
            result_queue.put(True)
        except Exception:
            result_queue.put(False)

    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_exec, args=(code, queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return False

    try:
        return queue.get_nowait()
    except Exception:
        return False


def check_solution(prompt: str, completion: str, test: str, entry_point: str) -> bool:
    """Combine prompt + completion + test and execute."""
    # Clean completion: stop at common end markers
    stop_sequences = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]
    for stop in stop_sequences:
        idx = completion.find(stop)
        if idx != -1:
            completion = completion[:idx]

    full_code = prompt + completion + "\n" + test + f"\ncheck({entry_point})\n"
    return _run_code(full_code, timeout=10)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_completion(
    model, tokenizer, prompt: str, device: str, max_new_tokens: int = 512,
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if input_ids.shape[1] > 2048 - max_new_tokens:
        input_ids = input_ids[:, -(2048 - max_new_tokens):]
    prompt_len = input_ids.shape[1]

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = outputs[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def evaluate_humaneval(
    model_path: str,
    data_dir: str,
    max_new_tokens: int = 512,
    device: str = "cuda",
    output_path: str | None = None,
) -> dict:
    print(f"{'='*60}")
    print(f" HumanEval Evaluation (pass@1)")
    print(f"{'='*60}")
    print(f"  Model:          {model_path}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Device:         {device}")
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

    ds = load_arrow_dataset(os.path.join(data_dir, "openai_humaneval-test.arrow"))
    print(f"Loaded {len(ds)} problems.\n")

    correct = 0
    total = len(ds)
    details = []
    start_time = time.time()

    for i in tqdm(range(total), desc="HumanEval", unit="prob"):
        example = ds[i]
        prompt = example["prompt"]
        test = example["test"]
        entry_point = example["entry_point"]

        completion = generate_completion(model, tokenizer, prompt, device, max_new_tokens)
        passed = check_solution(prompt, completion, test, entry_point)

        if passed:
            correct += 1
        details.append({
            "task_id": example["task_id"],
            "passed": passed,
            "completion": completion[:500],
        })

    elapsed = time.time() - start_time
    pass_at_1 = correct / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f" HumanEval Results  (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Passed:  {correct} / {total}")
    print(f"  pass@1:  {pass_at_1:.2%}\n")

    results = {
        "model_path": model_path,
        "benchmark": "HumanEval",
        "metric": "pass@1",
        "elapsed_seconds": round(elapsed, 2),
        "pass_at_1": round(pass_at_1, 6),
        "correct": correct,
        "total": total,
    }

    if output_path is None:
        output_path = os.path.join(data_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")

    detail_path = output_path.replace(".json", "_details.json")
    with open(detail_path, "w") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"Details saved to: {detail_path}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on HumanEval (pass@1).")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_humaneval(
        model_path=args.model_path,
        data_dir=args.data_dir,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        output_path=args.output,
    )
