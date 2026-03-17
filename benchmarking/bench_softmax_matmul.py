"""
Benchmark: Fused Softmax-Matmul (Section 1.5 / Part 1.B)

Compares Triton fused_softmax vs PyTorch softmax_mult.
Run: python -m benchmarking.bench_softmax_matmul
"""
import os
import sys

import pandas as pd
import torch

# Add project root so we can import softmax_matmul
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from softmax_matmul.softmax_matmul import fused_softmax, softmax_mult

try:
    from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure
except ImportError:
    CompilationError = Exception
    CompileTimeAssertionFailure = Exception

try:
    from triton.runtime.errors import OutOfResources
except ImportError:
    OutOfResources = type("OutOfResources", (Exception,), {})

# -----------------------------------------------------------------------------
# Config (from homework 1.5.1)
# -----------------------------------------------------------------------------
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


def run_one_forward(fn, x, V, **kwargs):
    """Run a single forward pass and return (elapsed_ms, peak_mib)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    fn(x, V, **kwargs)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_mib = peak_bytes / (1024.0 * 1024.0)
    return elapsed_ms, peak_mib


def run_benchmark_config_triton(d2_val, BLOCK):
    """Benchmark fused_softmax for one (d2, BLOCK). Returns dict or None on failure."""
    if d1 % BLOCK != 0 or d2_val % BLOCK != 0:
        return None
    x = torch.randn(batch_size, d1, d2_val, device=device, dtype=dtype)
    V = torch.randn(batch_size, d2_val, d3, device=device, dtype=dtype)

    def fn():
        return fused_softmax(x, V, BLOCK_1=BLOCK, BLOCK_2=BLOCK)

    # Warmup
    for _ in range(nb_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    peaks = []
    for _ in range(nb_passes):
        t, m = run_one_forward(fused_softmax, x, V, BLOCK_1=BLOCK, BLOCK_2=BLOCK)
        times.append(t)
        peaks.append(m)

    return {
        "batch_size": batch_size,
        "d1": d1,
        "d2": d2_val,
        "d3": d3,
        "triton": True,
        "BLOCK": BLOCK,
        "forward_ms_mean": float(torch.tensor(times).mean().item()),
        "forward_ms_std": float(torch.tensor(times).std().item()) if len(times) > 1 else 0.0,
        "forward_peak_MiB": float(max(peaks)),
    }


def run_benchmark_config_pytorch(d2_val):
    """Benchmark softmax_mult for one d2. No BLOCK."""
    x = torch.randn(batch_size, d1, d2_val, device=device, dtype=dtype)
    V = torch.randn(batch_size, d2_val, d3, device=device, dtype=dtype)

    def fn():
        return softmax_mult(x, V)

    for _ in range(nb_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    peaks = []
    for _ in range(nb_passes):
        t, m = run_one_forward(softmax_mult, x, V)
        times.append(t)
        peaks.append(m)

    return {
        "batch_size": batch_size,
        "d1": d1,
        "d2": d2_val,
        "d3": d3,
        "triton": False,
        "BLOCK": pd.NA,
        "forward_ms_mean": float(torch.tensor(times).mean().item()),
        "forward_ms_std": float(torch.tensor(times).std().item()) if len(times) > 1 else 0.0,
        "forward_peak_MiB": float(max(peaks)),
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    results = []

    # PyTorch baseline (one row per d2)
    print("Benchmarking PyTorch softmax_mult...")
    for d2_val in d2_values:
        try:
            r = run_benchmark_config_pytorch(d2_val)
            results.append(r)
        except (RuntimeError, OutOfMemoryError) as e:
            err_str = str(e).lower()
            if "out of memory" in err_str:
                print(f"  OOM for d2={d2_val}")
                torch.cuda.empty_cache()
                results.append({
                    "batch_size": batch_size,
                    "d1": d1,
                    "d2": d2_val,
                    "d3": d3,
                    "triton": False,
                    "BLOCK": pd.NA,
                    "forward_ms_mean": None,
                    "forward_ms_std": None,
                    "forward_peak_MiB": None,
                })
            else:
                raise

    # Triton fused_softmax (per d2 and BLOCK)
    print("Benchmarking Triton fused_softmax...")
    for d2_val in d2_values:
        for BLOCK in block_sizes:
            if d1 % BLOCK != 0 or d2_val % BLOCK != 0:
                print(f"  Skipping d2={d2_val}, BLOCK={BLOCK} (incompatible)")
                results.append({
                    "batch_size": batch_size,
                    "d1": d1,
                    "d2": d2_val,
                    "d3": d3,
                    "triton": True,
                    "BLOCK": BLOCK,
                    "forward_ms_mean": None,
                    "forward_ms_std": None,
                    "forward_peak_MiB": None,
                })
                continue
            try:
                r = run_benchmark_config_triton(d2_val, BLOCK)
                if r is not None:
                    results.append(r)
            except (RuntimeError, OutOfMemoryError, CompilationError, CompileTimeAssertionFailure, OutOfResources) as e:
                err_str = str(e).lower()
                if "out of memory" in err_str:
                    print(f"  OOM for d2={d2_val}, BLOCK={BLOCK}")
                    torch.cuda.empty_cache()
                else:
                    print(f"  Error for d2={d2_val}, BLOCK={BLOCK}: {e}")
                results.append({
                    "batch_size": batch_size,
                    "d1": d1,
                    "d2": d2_val,
                    "d3": d3,
                    "triton": True,
                    "BLOCK": BLOCK,
                    "forward_ms_mean": None,
                    "forward_ms_std": None,
                    "forward_peak_MiB": None,
                })

    df = pd.DataFrame(results)
    df["BLOCK"] = df["BLOCK"].astype("Int64")

    print("\n--- Summary ---")
    print(df.to_string())

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
