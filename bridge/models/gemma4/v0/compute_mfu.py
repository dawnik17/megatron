#!/usr/bin/env python3
"""Compute Model FLOPs Utilization (MFU) for a Gemma 4 E4B Megatron-Bridge training log.

Forward FLOPs per token are counted from the Gemma 4 E4B text architecture:
  per layer   Q / K / V / O projections (K/V skipped on the 18 KV-shared layers),
              causal attention (sliding layers see <= 512 keys, global layers use 512-dim heads),
              gated GeLU MLP, Per-Layer-Embedding gate + projection
  per model   PLE model projection (hidden -> 42 * 256) and LM head (hidden -> 262144)

  training FLOPs = 3 x forward FLOPs
  MFU            = training FLOPs per step / (step time * num GPUs * peak FLOP/s)

Step time is the median "elapsed time per iteration (ms)" after the first --skip-iters iterations.

Usage:
  python3 compute_mfu.py --log train.log --seq-len 4096 --gbs 64 --num-gpus 8 --peak-tflops 2250
"""

import argparse
import re
import statistics
import sys

# google/gemma-4-E4B-it text_config
HIDDEN = 2560
FFN = 10240
LAYERS = 42
HEADS = 8
KV_HEADS = 2
HEAD_DIM = 256  # sliding-window layers
GLOBAL_HEAD_DIM = 512  # full-attention layers
SLIDING_WINDOW = 512
KV_SHARED_LAYERS = 18  # last 18 layers reuse K/V of earlier layers
PLE_DIM = 256
VOCAB = 262144
LAYER_TYPES = (["sliding"] * 5 + ["full"]) * 7

LOG_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+\s*\|.*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    r"(?:.*?throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+))?"
)


def avg_context(seq_len, window=None):
    """Average number of keys a query attends to under a causal (optionally windowed) mask."""
    if window is None or window >= seq_len:
        return (seq_len + 1) / 2
    return (window * (window + 1) / 2 + (seq_len - window) * window) / seq_len


def forward_flops_per_token(seq_len):
    flops = 0.0
    for i, layer_type in enumerate(LAYER_TYPES):
        is_global = layer_type == "full"
        hd = GLOBAL_HEAD_DIM if is_global else HEAD_DIM
        flops += 2 * 2 * HIDDEN * HEADS * hd  # Q, O
        if i < LAYERS - KV_SHARED_LAYERS:
            flops += 2 * 2 * HIDDEN * KV_HEADS * hd  # K, V
        flops += (
            2
            * 2
            * HEADS
            * hd
            * avg_context(seq_len, None if is_global else SLIDING_WINDOW)
        )  # QK^T, PV
        flops += 3 * 2 * HIDDEN * FFN  # gate, up, down
        flops += 2 * 2 * HIDDEN * PLE_DIM  # PLE gate + projection
    flops += 2 * HIDDEN * LAYERS * PLE_DIM  # PLE model projection
    flops += 2 * HIDDEN * VOCAB  # LM head
    return flops


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--log", required=True, help="Training log (stdout of run_recipe.py)."
    )
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument(
        "--gbs", type=int, required=True, help="Global batch size (sequences per step)."
    )
    p.add_argument("--num-gpus", type=int, required=True)
    p.add_argument(
        "--peak-tflops",
        type=float,
        default=2250.0,
        help="Dense BF16 peak per GPU (TFLOP/s).",
    )
    p.add_argument(
        "--skip-iters", type=int, default=20, help="Warm-up iterations to exclude."
    )
    args = p.parse_args()

    steps = {}
    with open(args.log, errors="replace") as f:
        for line in f:
            m = LOG_RE.search(line)
            if m:
                steps[int(m.group(1))] = (
                    float(m.group(2)),
                    float(m.group(3)) if m.group(3) else None,
                )
    if not steps:
        sys.exit(
            f"ERROR: no 'iteration ... elapsed time per iteration' lines found in {args.log}"
        )
    used = {k: v for k, v in steps.items() if k > args.skip_iters} or steps

    step_ms = statistics.median(v[0] for v in used.values())
    bridge = [v[1] for v in used.values() if v[1] is not None]

    tokens_per_step = args.gbs * args.seq_len
    train_flops_per_token = 3 * forward_flops_per_token(args.seq_len)
    tflops_per_gpu = (
        train_flops_per_token
        * tokens_per_step
        / (step_ms / 1000)
        / args.num_gpus
        / 1e12
    )
    tokens_per_s = tokens_per_step / (step_ms / 1000)

    print(f"Iterations used        : {len(used)} (skipped first {args.skip_iters})")
    print(
        f"Tokens per step        : {tokens_per_step:,} (GBS={args.gbs} x seq={args.seq_len})"
    )
    print(f"Training FLOPs / token : {train_flops_per_token / 1e9:.2f} GFLOP")
    print(f"Step time (median)     : {step_ms:.1f} ms")
    print(
        f"Throughput             : {tokens_per_s:,.0f} tokens/s | {tokens_per_s / args.num_gpus:,.0f} tokens/s/GPU"
    )
    print(f"Model TFLOP/s per GPU  : {tflops_per_gpu:.1f}")
    print(f"Peak BF16 TFLOP/s/GPU  : {args.peak_tflops:.0f}")
    print(f"MFU                    : {tflops_per_gpu / args.peak_tflops * 100:.2f} %")
    if bridge:
        print(
            f"(Bridge-reported       : {statistics.median(bridge):.1f} TFLOP/s/GPU, generic GPT formula)"
        )


if __name__ == "__main__":
    main()
