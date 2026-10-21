#!/usr/bin/env python3
"""A run folder written by launch.py: its train.log, step by step, and its Model FLOPs Utilization, from config.yaml
and train.log.

    python3 bridge/models/mfu.py runs/train/<model_family>/<run name>      # PEAK_TFLOPS=... to use another peak

  training FLOPs = 3 x forward FLOPs (no activation recompute); forward FLOPs per token come from
                   bridge/models/<model_family>/flops.py, given the model's HF config
  MFU            = training FLOPs per step / (step time x GPUs x peak FLOP/s)

Step time is the median "elapsed time per iteration (ms)" after the first 20 iterations.
"""

import importlib.util
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO = Path(__file__).resolve().parents[2]
SKIP = 20  # compilation, allocator and NCCL warm-up
TIME, TFLOPS = "elapsed time per iteration (ms)", "throughput per GPU (TFLOP/s/GPU)"


def parse(run):
    """A run's train.log: Megatron's fields at each step ({step: {field: text}}), peak memory and allocator retries."""
    text = (Path(run) / "train.log").read_text(errors="replace")
    steps = {}
    for m in re.finditer(r"iteration\s+(\d+)/\s*\d+\s*\|(.*)", text):
        steps[int(m[1])] = {
            k.strip(): v.strip() for k, v in (f.split(":", 1) for f in m[2].split("|") if ":" in f)
        }
    peak = [float(x) for x in re.findall(r"mem-max-reserved-gigabytes: ([\d.]+)", text)]
    retries = [int(x) for x in re.findall(r"mem-alloc-retires: (\d+)", text)]
    return dict(steps=steps, peak_gb=max(peak, default=math.nan), retries=max(retries, default=0))


def values(log, field, first=1, last=None):
    return [
        float(s[field])
        for i, s in sorted(log["steps"].items())
        if first <= i <= (last or i) and field in s
    ]


def steady(log, field=TIME):
    """A field after the first SKIP steps; in a shorter run, at every step."""
    return values(log, field, first=SKIP + 1) or values(log, field)


def step_ms(log):
    return statistics.median(steady(log))


def hf_config(model):
    """The (text) config of an HF checkpoint in .cache/hf."""
    snapshots = REPO / ".cache/hf/hub" / f"models--{model.replace('/', '--')}" / "snapshots"
    config = json.loads(next(snapshots.glob("*/config.json")).read_text())
    return config.get("text_config", config)


def forward_flops_per_token(family, model, seq_len):
    spec = importlib.util.spec_from_file_location(
        "flops", REPO / "bridge/models" / family / "flops.py"
    )
    flops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flops)
    return flops.forward_flops_per_token(hf_config(model), seq_len)


def measure(run):
    """Step time, throughput and MFU of a run folder."""
    cfg = yaml.safe_load((Path(run) / "config.yaml").read_text())
    log = parse(run)
    if not (times := steady(log)):
        sys.exit(f"no 'iteration ... elapsed time per iteration' lines in {run}/train.log")
    m = SimpleNamespace(
        steps=len(times), gbs=cfg["gbs"], seq=cfg["seq_len"], gpus=len(str(cfg["gpus"]).split(","))
    )
    m.step_ms = statistics.median(times)
    m.train_flops = 3 * forward_flops_per_token(cfg["model_family"], cfg["model"], m.seq)
    m.tokens_per_s = m.gbs * m.seq / (m.step_ms / 1000)
    m.tflops = m.train_flops * m.tokens_per_s / m.gpus / 1e12
    m.peak = float(os.environ.get("PEAK_TFLOPS") or cfg["peak_tflops"])
    m.mfu = m.tflops / m.peak * 100
    m.bridge = statistics.median(bridge) if (bridge := steady(log, TFLOPS)) else None
    return m


def report(run):
    """The MFU report of a run folder, as text."""
    m = measure(run)
    lines = [
        f"Iterations used        : {m.steps} (skipped first {SKIP})",
        f"Tokens per step        : {m.gbs * m.seq:,} (GBS={m.gbs} x seq={m.seq})",
        f"Training FLOPs / token : {m.train_flops / 1e9:.2f} GFLOP",
        f"Step time (median)     : {m.step_ms:.1f} ms",
        f"Throughput             : {m.tokens_per_s:,.0f} tokens/s | {m.tokens_per_s / m.gpus:,.0f} tokens/s/GPU",
        f"Model TFLOP/s per GPU  : {m.tflops:.1f}",
        f"Peak BF16 TFLOP/s/GPU  : {m.peak:.0f}",
        f"MFU                    : {m.mfu:.2f} %",
    ]
    if m.bridge is not None:
        lines.append(f"(Bridge-reported       : {m.bridge:.1f} TFLOP/s/GPU, generic GPT formula)")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(report(sys.argv[1]), end="")
