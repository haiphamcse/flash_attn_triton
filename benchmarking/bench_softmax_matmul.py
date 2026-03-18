"""
Benchmark: Fused Softmax-Matmul (Part 1.B)

Compares Triton fused_softmax vs PyTorch softmax_mult across
varying sequence lengths (d2) and block sizes.

Run: python -m benchmarking.bench_softmax_matmul
"""
import os
import sys

import pandas as pd
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from softmax_matmul.softmax_matmul import fused_softmax, softmax_mult

try:
    from triton.compiler.errors import CompileTimeAssertionFailure
except ImportError:
    CompileTimeAssertionFailure = type("CompileTimeAssertionFailure", (Exception,), {})

try:
    from triton.runtime.errors import OutOfResources
except ImportError:
    OutOfResources = type("OutOfResources", (Exception,), {})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
device = "cuda"
dtype = torch.float32
batch_size = 16
nb_warmup = 10
nb_passes = 100
d1 = 2048
d2_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
d3 = 512
block_sizes = [16, 32, 64]
OUTPUT_CSV = os.path.join(_PROJECT_ROOT, "outputs", "softmax_matmul_benchmark.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _failed_row(d2_val, is_triton, block=pd.NA):
    return {
        "batch_size": batch_size,
        "d1": d1,
        "d2": d2_val,
        "d3": d3,
        "triton": is_triton,
        "BLOCK": block,
        "forward_ms_mean": None,
        "forward_ms_std": None,
        "forward_peak_MiB": None,
    }


def _time_fn(fn, nb_iters):
    """Return list of (elapsed_ms, peak_mib) for *nb_iters* calls."""
    records = []
    for _ in range(nb_iters):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        peak_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        records.append((elapsed_ms, peak_mib))
    return records


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------
def bench_pytorch(d2_val):
    x = torch.randn(batch_size, d1, d2_val, device=device, dtype=dtype)
    V = torch.randn(batch_size, d2_val, d3, device=device, dtype=dtype)

    fn = lambda: softmax_mult(x, V)

    for _ in range(nb_warmup):
        fn()
    torch.cuda.synchronize()

    records = _time_fn(fn, nb_passes)
    times = [r[0] for r in records]
    peaks = [r[1] for r in records]

    t = torch.tensor(times)
    return {
        "batch_size": batch_size,
        "d1": d1,
        "d2": d2_val,
        "d3": d3,
        "triton": False,
        "BLOCK": pd.NA,
        "forward_ms_mean": t.mean().item(),
        "forward_ms_std": t.std().item() if len(times) > 1 else 0.0,
        "forward_peak_MiB": max(peaks),
    }


def bench_triton(d2_val, BLOCK):
    x = torch.randn(batch_size, d1, d2_val, device=device, dtype=dtype)
    V = torch.randn(batch_size, d2_val, d3, device=device, dtype=dtype)

    fn = lambda: fused_softmax(x, V, BLOCK_1=BLOCK, BLOCK_2=BLOCK)

    for _ in range(nb_warmup):
        fn()
    torch.cuda.synchronize()

    records = _time_fn(fn, nb_passes)
    times = [r[0] for r in records]
    peaks = [r[1] for r in records]

    t = torch.tensor(times)
    return {
        "batch_size": batch_size,
        "d1": d1,
        "d2": d2_val,
        "d3": d3,
        "triton": True,
        "BLOCK": BLOCK,
        "forward_ms_mean": t.mean().item(),
        "forward_ms_std": t.std().item() if len(times) > 1 else 0.0,
        "forward_peak_MiB": max(peaks),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("CUDA not available – skipping benchmark.")
        return

    results = []

    # --- PyTorch baseline ---
    print("=== PyTorch softmax_mult ===")
    for d2_val in d2_values:
        print(f"  d2={d2_val} ...", end=" ", flush=True)
        try:
            r = bench_pytorch(d2_val)
            results.append(r)
            print(f"{r['forward_ms_mean']:.3f} ms")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM")
                torch.cuda.empty_cache()
                results.append(_failed_row(d2_val, is_triton=False))
            else:
                raise

    # --- Triton fused_softmax ---
    print("\n=== Triton fused_softmax ===")
    for d2_val in d2_values:
        for BLOCK in block_sizes:
            tag = f"d2={d2_val}, BLOCK={BLOCK}"
            print(f"  {tag} ...", end=" ", flush=True)

            try:
                r = bench_triton(d2_val, BLOCK)
                results.append(r)
                print(f"{r['forward_ms_mean']:.3f} ms")

            except CompileTimeAssertionFailure:
                print("skipped (incompatible)")
                results.append(_failed_row(d2_val, is_triton=True, block=BLOCK))

            except (RuntimeError, OutOfResources) as e:
                err = str(e).lower()
                if "out of memory" in err:
                    print("OOM")
                    torch.cuda.empty_cache()
                else:
                    print(f"error – {e}")
                results.append(_failed_row(d2_val, is_triton=True, block=BLOCK))

    # --- Collect & save ---
    df = pd.DataFrame(results)
    df["BLOCK"] = df["BLOCK"].astype("Int64")

    print("\n--- Summary ---")
    print(df.to_string())

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
