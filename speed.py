#!/usr/bin/env python3
"""
Speed Benchmark — 测试模型前向推理和反向传播速度。

Usage:
    python speed.py \
        --model_path /path/to/model \
        --batch_size 8 \
        --seq_len 2048 \
        --warmup_steps 5 \
        --measure_steps 20 \
        --dtype bfloat16

测试内容:
  1. Forward       — 纯前向推理 (torch.no_grad, eval mode)
  2. Backward      — 仅反向传播 (前向不计时, 只计 backward)
  3. Forward+Backward — 完整训练步 (前向 + 反向, 一起计时)

输出指标:
  - 平均耗时 (ms/step)
  - 吞吐量 (tokens/sec)
  - 显存峰值 (peak GPU memory)
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoConfig

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
# Dtype mapping
# ---------------------------------------------------------------------------
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def get_gpu_memory_mb() -> float:
    """Return peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _make_result(elapsed: float, measure_steps: int, tokens_per_step: int) -> dict:
    """Build a result dict from timing data."""
    avg_ms = (elapsed / measure_steps) * 1000
    tokens_per_sec = tokens_per_step / (elapsed / measure_steps)
    peak_mem_mb = get_gpu_memory_mb()
    return {
        "avg_ms_per_step": round(avg_ms, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_memory_mb": round(peak_mem_mb, 1),
        "total_steps": measure_steps,
        "total_elapsed_sec": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Benchmark: Forward only
# ---------------------------------------------------------------------------
def benchmark_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int,
    measure_steps: int,
) -> dict:
    """Benchmark forward-only speed (inference mode, no grad)."""
    model.eval()
    tokens_per_step = input_ids.shape[0] * input_ids.shape[1]

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(input_ids=input_ids)
    torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(measure_steps):
            _ = model(input_ids=input_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return _make_result(elapsed, measure_steps, tokens_per_step)


# ---------------------------------------------------------------------------
# Benchmark: Backward only
# ---------------------------------------------------------------------------
def benchmark_backward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int,
    measure_steps: int,
) -> dict:
    """Benchmark backward-only speed.

    For each step:
      1. Run forward (NOT timed) to build the computation graph
      2. Synchronize
      3. Run backward (TIMED)
      4. Clear gradients
    """
    model.train()
    tokens_per_step = input_ids.shape[0] * input_ids.shape[1]

    # Warmup
    for _ in range(warmup_steps):
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        torch.cuda.synchronize()
        loss.backward()
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)

    # Measure — only time the backward pass
    torch.cuda.reset_peak_memory_stats()
    total_bwd_time = 0.0
    for _ in range(measure_steps):
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        torch.cuda.synchronize()  # ensure forward is done

        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()  # ensure backward is done
        total_bwd_time += time.perf_counter() - start

        model.zero_grad(set_to_none=True)

    return _make_result(total_bwd_time, measure_steps, tokens_per_step)


# ---------------------------------------------------------------------------
# Benchmark: Forward + Backward
# ---------------------------------------------------------------------------
def benchmark_forward_backward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int,
    measure_steps: int,
) -> dict:
    """Benchmark full training step: forward + backward together."""
    model.train()
    tokens_per_step = input_ids.shape[0] * input_ids.shape[1]

    # Warmup
    for _ in range(warmup_steps):
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(measure_steps):
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return _make_result(elapsed, measure_steps, tokens_per_step)


# ---------------------------------------------------------------------------
# Benchmark one seq_len
# ---------------------------------------------------------------------------
def _bench_one_seq_len(
    model: torch.nn.Module,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    warmup_steps: int,
    measure_steps: int,
    device: str,
) -> dict:
    """Run 3 benchmarks for a single seq_len and return a result dict."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    tokens_per_step = batch_size * seq_len

    print(f"\n  seq_len={seq_len}  ({tokens_per_step:,} tokens/step)")
    print(f"  {'─'*56}")

    fwd = benchmark_forward(model, input_ids, warmup_steps, measure_steps)
    print(f"    Forward          {fwd['avg_ms_per_step']:>10.2f}ms"
          f"   {fwd['tokens_per_sec']:>13,.0f} tok/s"
          f"   {fwd['peak_memory_mb']:>8.0f} MB")

    bwd = benchmark_backward(model, input_ids, warmup_steps, measure_steps)
    print(f"    Backward         {bwd['avg_ms_per_step']:>10.2f}ms"
          f"   {bwd['tokens_per_sec']:>13,.0f} tok/s"
          f"   {bwd['peak_memory_mb']:>8.0f} MB")

    fwd_bwd = benchmark_forward_backward(model, input_ids, warmup_steps, measure_steps)
    print(f"    Forward+Backward {fwd_bwd['avg_ms_per_step']:>10.2f}ms"
          f"   {fwd_bwd['tokens_per_sec']:>13,.0f} tok/s"
          f"   {fwd_bwd['peak_memory_mb']:>8.0f} MB")

    bwd_fwd_ratio = (bwd['avg_ms_per_step'] / fwd['avg_ms_per_step']
                     if fwd['avg_ms_per_step'] > 0 else 0)

    # Free the input tensor to reclaim memory before next seq_len
    del input_ids
    torch.cuda.empty_cache()

    return {
        "seq_len": seq_len,
        "tokens_per_step": tokens_per_step,
        "forward": fwd,
        "backward": bwd,
        "forward_backward": fwd_bwd,
        "backward_to_forward_ratio": round(bwd_fwd_ratio, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_speed_benchmark(
    model_path: str,
    batch_size: int = 8,
    seq_lens: list[int] | None = None,
    warmup_steps: int = 5,
    measure_steps: int = 20,
    dtype: str = "bfloat16",
    device: str = "cuda",
    output_path: str | None = None,
) -> dict:
    """Run speed benchmarks for one or more seq_lens (model loaded once)."""
    if seq_lens is None:
        seq_lens = [2048]
    torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)

    print(f"{'='*60}")
    print(f" Speed Benchmark")
    print(f"{'='*60}")
    print(f"  Model:         {model_path}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Seq lengths:   {seq_lens}")
    print(f"  Warmup steps:  {warmup_steps}")
    print(f"  Measure steps: {measure_steps}")
    print(f"  Dtype:         {dtype}")
    print(f"  Device:        {device}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load model (once)
    # ------------------------------------------------------------------
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {num_params:,}")

    gpu_name = "N/A"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    config = model.config
    vocab_size = getattr(config, "vocab_size", 32000)

    # ------------------------------------------------------------------
    # 2. Benchmark each seq_len
    # ------------------------------------------------------------------
    per_seq_results = []
    for sl in seq_lens:
        r = _bench_one_seq_len(
            model, vocab_size, batch_size, sl,
            warmup_steps, measure_steps, device,
        )
        per_seq_results.append(r)

    # ------------------------------------------------------------------
    # 3. Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" Speed Benchmark Summary")
    print(f"{'='*60}")
    print(f"  GPU:          {gpu_name}")
    print(f"  Parameters:   {num_params:,}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Dtype:        {dtype}")

    # Forward table
    print(f"\n  Forward (ms/step):")
    print(f"  {'seq_len':>10} {'ms/step':>12} {'tokens/sec':>15} {'peak mem':>12}")
    for r in per_seq_results:
        f = r["forward"]
        print(f"  {r['seq_len']:>10} {f['avg_ms_per_step']:>10.2f}ms"
              f"   {f['tokens_per_sec']:>13,.0f}"
              f"   {f['peak_memory_mb']:>9.0f} MB")

    # Forward+Backward table
    print(f"\n  Forward+Backward (ms/step):")
    print(f"  {'seq_len':>10} {'ms/step':>12} {'tokens/sec':>15} {'peak mem':>12}")
    for r in per_seq_results:
        fb = r["forward_backward"]
        print(f"  {r['seq_len']:>10} {fb['avg_ms_per_step']:>10.2f}ms"
              f"   {fb['tokens_per_sec']:>13,.0f}"
              f"   {fb['peak_memory_mb']:>9.0f} MB")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    results = {
        "model_path": model_path,
        "num_parameters": num_params,
        "batch_size": batch_size,
        "seq_lens": seq_lens,
        "dtype": dtype,
        "device": device,
        "gpu_name": gpu_name,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "benchmarks": per_seq_results,
    }

    if output_path is None:
        model_name = Path(model_path).name
        output_path = os.path.join(
            os.path.dirname(__file__), f"speed_results_{model_name}.json"
        )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_seq_lens(value: str) -> list[int]:
    """Parse comma-separated seq_len list, e.g. '512,1024,2048'."""
    return [int(x.strip()) for x in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model forward/backward speed.")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the HuggingFace-format model checkpoint.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for speed test (default: 8).",
    )
    parser.add_argument(
        "--seq_len", type=str, default="2048",
        help="Sequence length(s), comma-separated (e.g. '512,1024,2048,4096').",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=5,
        help="Number of warmup steps (default: 5).",
    )
    parser.add_argument(
        "--measure_steps", type=int, default=20,
        help="Number of measurement steps (default: 20).",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model (default: bfloat16).",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run benchmark on.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_speed_benchmark(
        model_path=args.model_path,
        batch_size=args.batch_size,
        seq_lens=parse_seq_lens(args.seq_len),
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        dtype=args.dtype,
        device=args.device,
        output_path=args.output,
    )
