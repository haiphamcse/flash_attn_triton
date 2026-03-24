#!/usr/bin/env python3
"""
Check your grade for the Flash Attention homework.

Usage — run from the root of your repository:

    python check_grade.py              # uses existing test_results.xml
    python check_grade.py --run-tests  # runs pytest first, then grades
"""

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Grading rubric ──────────────────────────────────────────────────────────

# Part 1: Fused Softmax-Matmul (8 pts)
#   - Implementation tests: 4 pts (proportional to tests passed)
#   - Benchmark script + CSV output: 4 pts
#
# Part 2: Flash Attention PyTorch (8 pts)
#   - All tests: 8 pts (proportional)
#
# Part 3: Flash Attention Triton (8 pts)
#   - Implementation tests: 6 pts (proportional)
#   - Benchmark script + CSV output: 2 pts

PART1_IMPL_PTS = 4
PART1_BENCH_PTS = 4
PART2_PTS = 8
PART3_IMPL_PTS = 6
PART3_BENCH_PTS = 2

PART1_PATTERNS = [re.compile(r"test_softmax_matmul\.")]
PART2_PATTERNS = [
    re.compile(r"test_flash_attention\..*\[.*pytorch\]"),
    re.compile(r"test_attention\.test_flash_forward_pass_pytorch"),
    re.compile(r"test_attention\.test_flash_backward_pytorch"),
    re.compile(r"test_flash_memory\."),
]
PART3_PATTERNS = [
    re.compile(r"test_flash_attention\..*\[.*triton\]"),
    re.compile(r"test_attention\.test_flash_forward_pass_triton"),
    re.compile(r"test_attention\.test_flash_backward_triton"),
    re.compile(r"test_triton_usage\."),
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def classify_test(full_name):
    for pat in PART1_PATTERNS:
        if pat.search(full_name):
            return "part1"
    for pat in PART2_PATTERNS:
        if pat.search(full_name):
            return "part2"
    for pat in PART3_PATTERNS:
        if pat.search(full_name):
            return "part3"
    return None


def parse_test_results(xml_path):
    counts = {p: {"passed": 0, "total": 0, "details": []} for p in ("part1", "part2", "part3")}
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        print(f"  ERROR: could not parse {xml_path}")
        return counts

    for tc in tree.findall(".//testcase"):
        classname = tc.get("classname", "")
        name = tc.get("name", "")
        full_name = f"{classname}.{name}"

        part = classify_test(full_name)
        if part is None:
            continue

        counts[part]["total"] += 1
        has_failure = tc.find("failure") is not None
        has_error = tc.find("error") is not None
        has_skipped = tc.find("skipped") is not None
        passed = not has_failure and not has_error and not has_skipped

        if passed:
            counts[part]["passed"] += 1
        else:
            status = "FAIL" if has_failure else ("ERROR" if has_error else "SKIP")
            counts[part]["details"].append((name, status))

    return counts


def check_benchmark(root, bench_name, csv_name):
    bench_path = root / "benchmarking" / bench_name
    csv_path = root / "outputs" / csv_name
    has_bench = bench_path.exists() and bench_path.stat().st_size > 200
    has_csv = csv_path.exists() and csv_path.stat().st_size > 50
    return has_bench, has_csv


def score_bar(score, max_score, width=20):
    filled = round(score / max_score * width) if max_score > 0 else 0
    return "[" + "#" * filled + "." * (width - filled) + "]"


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Check your Flash Attention homework grade.")
    parser.add_argument("--run-tests", action="store_true",
                        help="Run pytest before grading (requires CUDA GPU)")
    args = parser.parse_args()

    root = Path.cwd()
    xml_path = root / "test_results.xml"

    # Optionally run tests
    if args.run_tests:
        print("Running pytest...\n")
        subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "./tests", f"--junitxml={xml_path}"],
            cwd=root,
        )
        print()

    # ── Parse test results ──────────────────────────────────────────────
    print("=" * 60)
    print("  FLASH ATTENTION HOMEWORK — GRADE REPORT")
    print("=" * 60)

    if not xml_path.exists():
        print(f"\n  WARNING: {xml_path.name} not found.")
        print("  Run with --run-tests or run ./test_and_submit.sh first.\n")
        counts = {p: {"passed": 0, "total": 0, "details": []} for p in ("part1", "part2", "part3")}
    else:
        counts = parse_test_results(xml_path)

    # ── Part 1 ──────────────────────────────────────────────────────────
    p1 = counts["part1"]
    p1_impl = (p1["passed"] / p1["total"] * PART1_IMPL_PTS) if p1["total"] > 0 else 0

    p1_has_bench, p1_has_csv = check_benchmark(root, "bench_softmax_matmul.py", "softmax_matmul_benchmark.csv")
    p1_bench = PART1_BENCH_PTS if (p1_has_bench and p1_has_csv) else 0
    p1_score = p1_impl + p1_bench

    print(f"\n  Part 1: Fused Softmax-Matmul           {p1_score:.1f} / {PART1_IMPL_PTS + PART1_BENCH_PTS}")
    print(f"  {score_bar(p1_score, PART1_IMPL_PTS + PART1_BENCH_PTS)}")
    print(f"    Tests:     {p1['passed']}/{p1['total']} passed  =>  {p1_impl:.1f} / {PART1_IMPL_PTS} pts")
    print(f"    Benchmark: script={'OK' if p1_has_bench else 'MISSING'}  csv={'OK' if p1_has_csv else 'MISSING'}  =>  {p1_bench} / {PART1_BENCH_PTS} pts")
    for name, status in p1["details"]:
        print(f"      {status}: {name}")

    # ── Part 2 ──────────────────────────────────────────────────────────
    p2 = counts["part2"]
    p2_score = (p2["passed"] / p2["total"] * PART2_PTS) if p2["total"] > 0 else 0

    print(f"\n  Part 2: Flash Attention PyTorch         {p2_score:.1f} / {PART2_PTS}")
    print(f"  {score_bar(p2_score, PART2_PTS)}")
    print(f"    Tests:     {p2['passed']}/{p2['total']} passed  =>  {p2_score:.1f} / {PART2_PTS} pts")
    for name, status in p2["details"]:
        print(f"      {status}: {name}")

    # ── Part 3 ──────────────────────────────────────────────────────────
    p3 = counts["part3"]
    p3_impl = (p3["passed"] / p3["total"] * PART3_IMPL_PTS) if p3["total"] > 0 else 0

    p3_has_bench, p3_has_csv = check_benchmark(root, "bench_attention.py", "attention_benchmark.csv")
    p3_bench = PART3_BENCH_PTS if (p3_has_bench and p3_has_csv) else 0
    p3_score = p3_impl + p3_bench

    print(f"\n  Part 3: Flash Attention Triton          {p3_score:.1f} / {PART3_IMPL_PTS + PART3_BENCH_PTS}")
    print(f"  {score_bar(p3_score, PART3_IMPL_PTS + PART3_BENCH_PTS)}")
    print(f"    Tests:     {p3['passed']}/{p3['total']} passed  =>  {p3_impl:.1f} / {PART3_IMPL_PTS} pts")
    print(f"    Benchmark: script={'OK' if p3_has_bench else 'MISSING'}  csv={'OK' if p3_has_csv else 'MISSING'}  =>  {p3_bench} / {PART3_BENCH_PTS} pts")
    for name, status in p3["details"]:
        print(f"      {status}: {name}")

    # ── Total ───────────────────────────────────────────────────────────
    total = p1_score + p2_score + p3_score
    print()
    print("=" * 60)
    print(f"  TOTAL                                   {total:.1f} / 24")
    print(f"  {score_bar(total, 24, width=40)}")
    print("=" * 60)

    if total < 24:
        print("\n  Tips to improve your grade:")
        if p1_impl < PART1_IMPL_PTS:
            print("    - Fix failing tests in softmax_matmul/softmax_matmul.py")
        if not p1_has_bench or not p1_has_csv:
            print("    - Complete benchmarking/bench_softmax_matmul.py and run it to produce outputs/softmax_matmul_benchmark.csv")
        if p2_score < PART2_PTS:
            print("    - Fix failing tests in flash_attention/flash_attention.py (PyTorch)")
        if p3_impl < PART3_IMPL_PTS:
            print("    - Fix failing tests in flash_attention/flash_attention.py (Triton)")
        if not p3_has_bench or not p3_has_csv:
            print("    - Complete benchmarking/bench_attention.py and run it to produce outputs/attention_benchmark.csv")
        if not xml_path.exists():
            print("    - Run ./test_and_submit.sh or python check_grade.py --run-tests")
    print()


if __name__ == "__main__":
    main()
