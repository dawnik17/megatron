"""Show where a training step's GPU time goes, from a profile of rank 0: a PyTorch profiler trace (launch.py PROFILE=1,
read by torch.py) or an Nsight Systems SQLite export (PROFILE=nsys, read by nsys.py).

    python3 bridge/profile/breakdown runs/train/gemma4-mfu/<run>/torch_profile/rank-0.json.gz
    python3 bridge/profile/breakdown runs/train/gemma4-mfu/<run>/nsys/trace.sqlite

Each kernel is attributed to the PyTorch operation that launched it; the category comes from that operation or,
failing that, the kernel name. Category times can add up to more than the busy time when kernels overlap on different
streams (communication). With launch.py's mfu_report.txt next to the profile, it also prints the step time without the
profiler and how fast the GEMMs run.
"""

import bisect
import re
import sys
from collections import defaultdict
from pathlib import Path

from . import nsys, torch

# A kernel gets the first category whose test matches (launching operation, kernel name).
CATEGORIES = [
    ("attention, sliding layers (FA2)", lambda op, k: "flash" in k),
    ("attention bmm, global layers (TE)", lambda op, k: op in ("aten::bmm", "aten::baddbmm")),
    (
        "attention softmax, global layers (TE)",
        lambda op, k: "softmax" in op.lower() or "softmax" in k.lower(),
    ),
    (
        "GEMM (linear layers)",
        lambda op, k: re.search(r"nvjet|gemm|Gemm|cutlass|xmma|cublas|wgrad", k) is not None,
    ),
    ("NCCL communication", lambda op, k: "nccl" in k.lower()),
    (
        "Inductor fused (norms, GeLU, softcap)",
        lambda op, k: re.match(r"triton_(poi|red|per)_", k) is not None,
    ),
    ("optimizer", lambda op, k: re.search(r"multi_tensor|adam", k, re.I) is not None),
    ("memcpy / memset", lambda op, k: k.startswith(("Memcpy", "Memset"))),
    (
        "copy / cast / cat",
        lambda op, k: re.search(r"copy|_to_copy|cat|contiguous|clone", op) is not None,
    ),
    ("other ATen kernels", lambda op, k: True),
]
GEMM = "GEMM (linear layers)"


def category(op, kernel):
    return next(label for label, test in CATEGORIES if test(op, kernel))


def merge(intervals):
    """(start, end) intervals merged where they overlap, sorted."""
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def exposed_communication(kernels):
    """{collective: us} of NCCL kernel time during which no other kernel runs: communication nothing hides."""
    compute = merge((ts, ts + dur) for ts, dur, name, _ in kernels if "nccl" not in name.lower())
    starts = [s for s, _ in compute]
    exposed = defaultdict(float)
    for ts, dur, name, _ in kernels:
        if "nccl" in name.lower():
            hidden, i = 0.0, max(bisect.bisect_right(starts, ts) - 1, 0)
            while i < len(compute) and compute[i][0] < ts + dur:
                hidden += max(0.0, min(ts + dur, compute[i][1]) - max(ts, compute[i][0]))
                i += 1
            kind = re.search(r"AllReduce|ReduceScatter|AllGather|Broadcast|SendRecv", name)
            exposed[kind[0] if kind else "other"] += dur - hidden
    return exposed


def main(path):
    kernels, steps = (nsys if path.endswith(".sqlite") else torch).load(path)
    if not kernels:
        sys.exit(f"No kernels in {path}")
    if not steps:  # older traces: one window from the first to the last kernel
        steps = [(min(k[0] for k in kernels), max(k[0] + k[1] for k in kernels))]

    starts = [s for s, _ in steps]
    by_cat, by_op, by_kernel = (
        defaultdict(float),
        defaultdict(lambda: [0.0, 0]),
        defaultdict(lambda: [0.0, 0]),
    )
    per_step, inside = defaultdict(list), []
    for ts, dur, name, op in kernels:
        i = bisect.bisect_right(starts, ts) - 1
        if i < 0 or ts >= steps[i][1]:
            continue
        cat = category(op, name)
        by_cat[cat] += dur
        by_op[(cat, op)][0] += dur
        by_op[(cat, op)][1] += 1
        by_kernel[name][0] += dur
        by_kernel[name][1] += 1
        per_step[i].append((ts, ts + dur))
        inside.append((ts, dur, name, op))

    n = len(steps)
    wall = sum(e - s for s, e in steps) / n / 1e3
    busy = sum(e - s for i in range(n) for s, e in merge(per_step[i])) / n / 1e3
    count = sum(len(per_step[i]) for i in range(n)) / n
    print(
        f"{n} profiled steps: step time {wall:.1f} ms, GPU busy {busy:.1f} ms ({busy / wall:.0%}), "
        f"idle {wall - busy:.1f} ms, {count:,.0f} kernels per step"
    )

    exposed = {k: v / n / 1e3 for k, v in exposed_communication(inside).items()}
    by_size = sorted(exposed.items(), key=lambda x: -x[1])
    print(
        f"Communication with no compute alongside: {sum(exposed.values()):.1f} ms per step ("
        + ", ".join(f"{k} {v:.1f}" for k, v in by_size)
        + ")"
    )
    report = Path(path).resolve().parents[1] / "mfu_report.txt"
    if report.exists():
        text = report.read_text()
        step, tflops, peak = (
            float(re.search(rf"{label}\s*:\s*([\d.]+)", text)[1])
            for label in (r"Step time \(median\)", "Model TFLOP/s per GPU", "Peak BF16 TFLOP/s/GPU")
        )
        flop = tflops * step / 1e3  # TFLOP per GPU per step
        gemm = by_cat[GEMM] / n / 1e6  # seconds
        print(
            f"Without the profiler (mfu_report.txt): {step:.1f} ms per step, so the GPU idles about "
            f"{step - busy:.0f} ms ({(step - busy) / step:.0%})"
        )
        print(
            f"GEMMs: at most {flop / gemm:.0f} TFLOP/s per GPU, {flop / gemm / peak:.0%} of the {peak:g} peak; "
            f"at peak, the whole step would take {flop / peak * 1e3:.0f} ms"
        )

    print("\nGPU time per step by category (ms, share of step time):")
    for cat, t in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat:36s} {t / n / 1e3:8.1f}  {t / n / 1e3 / wall:6.1%}")

    print("\nTop launching operations (ms per step, kernels per step):")
    for (cat, op), (t, c) in sorted(by_op.items(), key=lambda x: -x[1][0])[:25]:
        print(f"  {t / n / 1e3:8.1f}  {c / n:7.0f}  {cat:36s} {op[:70]}")

    print("\nTop kernels (ms per step, launches per step, mean us):")
    for name, (t, c) in sorted(by_kernel.items(), key=lambda x: -x[1][0])[:20]:
        print(f"  {t / n / 1e3:8.1f}  {c / n:7.0f}  {t / c:8.1f}  {name[:110]}")
