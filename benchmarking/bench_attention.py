"""
Benchmark Flash Attention implementations (Part 3.B).

Compares:
- `pytorch_sdpa`: torch.nn.functional.scaled_dot_product_attention
- `flash_triton`: FlashAttentionTriton.apply

Run:
    python -m benchmarking.bench_attention
    python -m benchmarking.bench_attention --impl flash_triton
    python -m benchmarking.bench_attention --impl pytorch_sdpa
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List

import pandas as pd
import torch
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flash_attention.flash_attention import FlashAttentionTriton

# ---------------------------------------------------------------------------
# Required config (from homework spec)
# ---------------------------------------------------------------------------
device = "cuda"
dtype = torch.float32
batch_size = 8
nb_warmup = 10
nb_forward_passes = 100
nb_backward_passes = 100

d_models = [64]
context_lengths = [256, 1024, 4096, 8192, 16384]

OUTPUT_CSV = os.path.join(_PROJECT_ROOT, "outputs", "csv", "attention_benchmark.csv")


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "No GPU"


def _maybe_compile(fn: Callable, name: str) -> Callable:
    if not hasattr(torch, "compile"):
        return fn
    try:
        return torch.compile(fn, fullgraph=False)
    except Exception as exc:
        print(f"  warning: could not compile {name} ({exc}); using eager mode.")
        return fn


def _time_fn(fn: Callable[[], None], nb_iters: int) -> Dict[str, float]:
    times_ms: List[float] = []
    peaks_mib: List[float] = []

    for _ in range(nb_iters):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()

        times_ms.append(start.elapsed_time(end))
        peaks_mib.append(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))

    t = torch.tensor(times_ms, device="cpu")
    return {
        "time_ms": float(t.mean().item()),
        "peak_mib": float(max(peaks_mib) if peaks_mib else 0.0),
    }


def _estimate_saved_activations_mib(output: torch.Tensor) -> float:
    """Estimate saved activations by traversing autograd graph saved_tensors."""
    grad_fn = output.grad_fn
    if grad_fn is None:
        return 0.0

    visited_nodes = set()
    seen_tensors = set()
    stack = [grad_fn]
    total_bytes = 0

    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in visited_nodes:
            continue
        visited_nodes.add(node_id)

        saved = getattr(node, "saved_tensors", ())
        for tensor in saved:
            if not isinstance(tensor, torch.Tensor):
                continue
            key = (tensor.data_ptr(), tensor.numel(), tensor.element_size())
            if key in seen_tensors:
                continue
            seen_tensors.add(key)
            total_bytes += tensor.numel() * tensor.element_size()

        next_fns = getattr(node, "next_functions", ())
        for next_fn, _ in next_fns:
            if next_fn is not None:
                stack.append(next_fn)

    return total_bytes / (1024.0 * 1024.0)


def _build_inputs(seq_len: int, d_model: int, requires_grad: bool):
    q = torch.randn(
        batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=requires_grad
    )
    k = torch.randn(
        batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=requires_grad
    )
    v = torch.randn(
        batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=requires_grad
    )
    return q, k, v


def _bench_one(
    impl_name: str,
    impl_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor],
    d_model: int,
    seq_len: int,
    gpu_name: str,
) -> Dict[str, object]:
    row = {
        "implementation": impl_name,
        "d_model": d_model,
        "seq_len": seq_len,
        "forward_ms": None,
        "forward_peak_MiB": None,
        "backward_ms": None,
        "backward_peak_MiB": None,
        "saved_activations_MiB": None,
        "status": "ok",
        "gpu": gpu_name,
    }

    # ------------------------- Forward benchmark -------------------------
    try:
        q, k, v = _build_inputs(seq_len, d_model, requires_grad=True)

        def fwd():
            _ = impl_fn(q, k, v, True)

        for _ in range(nb_warmup):
            fwd()
        torch.cuda.synchronize()

        fwd_stats = _time_fn(fwd, nb_forward_passes)
        row["forward_ms"] = fwd_stats["time_ms"]
        row["forward_peak_MiB"] = fwd_stats["peak_mib"]

        torch.cuda.reset_peak_memory_stats()
        out = impl_fn(q, k, v, True)
        row["saved_activations_MiB"] = _estimate_saved_activations_mib(out)
        del out
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if _is_oom_error(exc):
            row["status"] = "OOM"
            torch.cuda.empty_cache()
            return row
        row["status"] = f"error: {type(exc).__name__}"
        return row

    # ------------------------- Backward benchmark ------------------------
    try:
        q, k, v = _build_inputs(seq_len, d_model, requires_grad=True)

        def bwd():
            if q.grad is not None:
                q.grad = None
            if k.grad is not None:
                k.grad = None
            if v.grad is not None:
                v.grad = None
            out = impl_fn(q, k, v, True)
            loss = out.sum()
            loss.backward()

        for _ in range(nb_warmup):
            bwd()
        torch.cuda.synchronize()

        bwd_stats = _time_fn(bwd, nb_backward_passes)
        row["backward_ms"] = bwd_stats["time_ms"]
        row["backward_peak_MiB"] = bwd_stats["peak_mib"]
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if _is_oom_error(exc):
            row["status"] = "OOM(backward)"
            torch.cuda.empty_cache()
            return row
        row["status"] = f"error: {type(exc).__name__}"
        return row

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impl",
        default="all",
        choices=["all", "pytorch_sdpa", "flash_triton"],
        help="Implementation(s) to benchmark.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available - skipping benchmark.")
        return

    torch.manual_seed(0)
    gpu_name = _gpu_name()
    print(f"GPU: {gpu_name}")

    pytorch_sdpa = _maybe_compile(
        lambda q, k, v, is_causal: F.scaled_dot_product_attention(q, k, v, is_causal=is_causal),
        "pytorch_sdpa",
    )
    flash_triton = _maybe_compile(FlashAttentionTriton.apply, "flash_triton")

    implementations = []
    if args.impl in ("all", "pytorch_sdpa"):
        implementations.append(("pytorch_sdpa", pytorch_sdpa))
    if args.impl in ("all", "flash_triton"):
        implementations.append(("flash_triton", flash_triton))

    results = []
    print("=" * 72)
    for impl_name, impl_fn in implementations:
        print(f"Benchmarking {impl_name}")
        for d_model in d_models:
            for seq_len in context_lengths:
                tag = f"d_model={d_model}, seq_len={seq_len}"
                print(f"  {tag} ...", end=" ", flush=True)
                row = _bench_one(impl_name, impl_fn, d_model, seq_len, gpu_name)
                results.append(row)
                if row["status"] == "ok":
                    print(
                        f"fwd={row['forward_ms']:.3f} ms, "
                        f"bwd={row['backward_ms']:.3f} ms, "
                        f"saved={row['saved_activations_MiB']:.2f} MiB"
                    )
                else:
                    print(row["status"])

    df = pd.DataFrame(results)
    # Ensure required output column order.
    ordered_cols = [
        "implementation",
        "d_model",
        "seq_len",
        "forward_ms",
        "forward_peak_MiB",
        "backward_ms",
        "backward_peak_MiB",
        "saved_activations_MiB",
        "status",
        "gpu",
    ]
    df = df[ordered_cols]

    print("\n--- Summary ---")
    print(df.to_string(index=False))

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
