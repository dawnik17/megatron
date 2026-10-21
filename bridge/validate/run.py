#!/usr/bin/env python3
"""Validate our changes to Megatron-Bridge's Gemma 4 E4B recipe against the recipe as shipped (OOTB).

  python3 bridge/validate/run.py --gpus 2,3        # about 20 minutes; exit status 0 when every check passes

Functional: the same model, trained the same way.
  kernels   every kernel our changes swap in matches Megatron's own against float64, forward and backward, and
            only the parameters train.py freezes go without a gradient (kernels/, on the first GPU)
  training  OOTB and ours train 100 steps on FineWeb-Edu from the same weights and batches: the first steps agree to
            rounding, and after the early loss spikes both end at about the same loss
Non-functional: faster, fits in memory, and stays faster at scale.
  speed and memory of those two runs (two GPUs, NVLink)
  scaling   both again with NCCL kept off NVLink, through host memory: two ranks then spend as long communicating
            per step as 64 data-parallel ranks on nodes that share one 400 Gb/s NIC, a worst case for multi-node

Training goes through bridge/models/launch.py with a config from bridge/models/gemma4/config/ (--config, default v1):
ours is the config as it is, OOTB the same config with ootb=true. GEMMA4_* variables set in front reach both. Runs and
report.md land in runs/validate/<date-time>/.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "bridge/models/gemma4/config"
sys.path.insert(0, str(REPO / "bridge/models"))
import launch  # noqa: E402
from mfu import measure, parse, step_ms, values  # noqa: E402

# Limits; the README has the measurements behind them. The functional ones are about three times the largest
# difference measured between equivalent runs (OOTB, OOTB at MBS 2, ours, ours again).
EARLY_STEPS = 5  # same weights at step 1, then small warm-up updates: the runs stay within rounding of each other
EARLY_LOSS = 2.5e-3  # relative loss difference at each of those steps (largest measured: 7.6e-4)
EARLY_GRAD_NORM = (
    0.06  # relative gradient-norm difference at each of those steps (largest measured: 1.9%)
)
FINAL_STEPS = 20
FINAL_LOSS = 0.10  # relative difference of the mean loss over the last FINAL_STEPS steps: early loss spikes, in
# every run, decide where a 100-step run ends up (two runs of the same code: 3.2% apart)
MEMORY = 0.95  # peak reserved memory, share of the GPU's
SPEEDUP_KEPT = 0.80  # share of the NVLink speedup over OOTB that must remain over the slow link


def wait_until_free(gpus, timeout=120):
    """The previous run's processes can take a few seconds to release their memory."""
    deadline = time.time() + timeout
    while any(g.used >= launch.BUSY for g in launch.gpu_info(gpus)) and time.time() < deadline:
        time.sleep(5)


def kernels(out, gpu, cfg):
    """The kernel check in the container on one GPU: its results, or None if it failed."""
    cmd = [
        *launch.container(gpu, cfg, out),
        "python",
        launch.IN_REPO / "bridge/validate/kernels",
        "--out",
        launch.IN_OUT / "kernels.json",
    ]
    print("  kernels ...", end=" ", flush=True)
    with open(out / "kernels.log", "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    print("done" if code == 0 else f"failed, see {out}/kernels.log", flush=True)
    return json.loads((out / "kernels.json").read_text()) if code == 0 else None


def plot(out, cfg):
    """plot.py in the container, which has matplotlib: whether it wrote training.png."""
    plot_py = launch.IN_REPO / "bridge/validate/plot.py"
    cmd = [*launch.container("void", cfg, out), "python", plot_py, launch.IN_OUT]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def train(out, config, name, gpus, steps, *overrides, **env):
    """One launch.py run of the config: its parsed log and MFU, or None if it failed."""
    wait_until_free(gpus)
    print(f"  {name}: {steps} steps ...", end=" ", flush=True)
    start = time.time()
    code = subprocess.run(
        # the suite sets the steps and the batch (8 sequences per GPU) its limits were measured at
        [
            sys.executable,
            launch.__file__,
            config,
            name,
            f"train_iters={steps}",
            "gbs=null",
            *overrides,
        ],
        stdout=subprocess.DEVNULL,
        env={**os.environ, "GPU_IDS": gpus, "RUN_DIR": str(out / name), **env},
    ).returncode
    print(
        (
            f"{(time.time() - start) / 60:.1f} min"
            if code == 0
            else f"failed, see {out / name}/train.log"
        ),
        flush=True,
    )
    return dict(parse(out / name), mfu=measure(out / name).mfu) if code == 0 else None


def relative(ours, ootb):
    return (ours - ootb) / ootb


def checks(k, ootb, ours, ootb_slow, ours_slow, steps, gpu_gb):
    """Rows of (section, check, measured, limit, passed); passed is None for a row that only informs."""
    rows = []
    add = lambda *row: rows.append(row)  # noqa: E731
    if k is None:
        add("functional", "kernels", "kernels check failed", "", False)
    else:
        g = k["gradients"]
        add(
            "functional",
            "parameters without a gradient",
            f"OOTB {len(g['OOTB'])}, ours {len(g['ours'])}",
            f"exactly the {len(g['frozen'])} frozen",
            g["passed"],
        )
        for r in k["kernels"]:
            worst = max(r["outputs"], key=lambda o: o["ours"] / o["ootb"])
            add(
                "functional",
                f"kernel: {r['kernel']}",
                f"error {worst['ours'] / worst['ootb']:.2f}x OOTB's ({worst['output']})"
                + (", identical" if r["identical"] else ""),
                f"<= {k['margin']}x, all outputs",
                r["passed"],
            )
    if ootb is None or ours is None:
        add("functional", "training", "a training run failed", "", False)
        return rows

    same = sum(
        all(
            ootb["steps"].get(i, {}).get(f) == ours["steps"].get(i, {}).get(f)
            for f in ("consumed samples", "learning rate")
        )
        for i in range(1, steps + 1)
    )
    add(
        "functional",
        "same batches and learning rates",
        f"{same} of {steps} steps",
        "all",
        same == steps,
    )
    for name, run in (("OOTB", ootb), ("ours", ours)):
        bad = [
            i
            for i, s in run["steps"].items()
            if s.get("number of nan iterations", "0") != "0"
            or s.get("number of skipped iterations", "0") != "0"
            or not math.isfinite(float(s.get("lm loss", "nan")))
        ]
        add(
            "functional",
            f"{name}: no NaN or skipped steps",
            f"{len(run['steps'])} steps, {len(bad)} bad",
            f"{steps} steps, 0 bad",
            len(run["steps"]) == steps and not bad,
        )
    for field, limit in (("lm loss", EARLY_LOSS), ("grad norm", EARLY_GRAD_NORM)):
        diff = {
            i: relative(*(float(r["steps"][i][field]) for r in (ours, ootb)))
            for i in range(1, EARLY_STEPS + 1)
        }
        worst = max(diff, key=lambda i: abs(diff[i]))
        add(
            "functional",
            f"{field}, steps 1-{EARLY_STEPS} (same start)",
            f"largest difference {diff[worst]:+.2%} (step {worst})",
            f"within {limit * 100:g}% at each",
            all(abs(d) <= limit for d in diff.values()),
        )
    a, b = (
        statistics.mean(values(r, "lm loss", first=steps - FINAL_STEPS + 1)) for r in (ootb, ours)
    )
    add(
        "functional",
        f"mean lm loss, last {FINAL_STEPS} steps",
        f"{a:.4f} -> {b:.4f} ({relative(b, a):+.2%})",
        f"within {FINAL_LOSS:.0%}",
        abs(relative(b, a)) <= FINAL_LOSS,
    )

    fast = step_ms(ootb) / step_ms(ours)
    add(
        "non-functional",
        "step time",
        f"{step_ms(ootb):.0f} -> {step_ms(ours):.0f} ms ({fast:.2f}x), MFU {ootb['mfu']:.1f}% -> {ours['mfu']:.1f}%",
        "faster",
        fast > 1,
    )
    add(
        "non-functional",
        "peak memory per GPU",
        f"{ootb['peak_gb']:.0f} -> {ours['peak_gb']:.0f} GB of {gpu_gb:.0f} ({ours['peak_gb'] / gpu_gb:.0%})",
        f"<= {MEMORY:.0%}, no allocator retries",
        ours["peak_gb"] <= MEMORY * gpu_gb and ours["retries"] == 0,
    )
    if ootb_slow is None or ours_slow is None:
        add("non-functional", "slow link", "a slow-link run failed", "", False)
        return rows
    slow = step_ms(ootb_slow) / step_ms(ours_slow)
    add(
        "non-functional",
        "slow link: communication per step",
        f"OOTB +{step_ms(ootb_slow) - step_ms(ootb):.0f} ms, ours +{step_ms(ours_slow) - step_ms(ours):.0f} ms",
        "",
        None,
    )
    add(
        "non-functional",
        "slow link: speedup kept",
        f"{step_ms(ootb_slow):.0f} -> {step_ms(ours_slow):.0f} ms ({slow:.2f}x, {slow / fast:.0%} of {fast:.2f}x)",
        f">= {SPEEDUP_KEPT:.0%}",
        slow / fast >= SPEEDUP_KEPT,
    )
    return rows


def passed(rows):
    return all(r[4] is not False for r in rows)


def report(rows, ootb, ours):
    width = max(len(r[1]) for r in rows) + 2
    status = {True: "PASS", False: "FAIL", None: "    "}
    lines = []
    for section in ("functional", "non-functional"):
        lines.append(section.upper())
        lines += [
            f"  {status[ok]}  {check:{width}s}{measured:50s} {limit}"
            for s, check, measured, limit, ok in rows
            if s == section
        ]
    lines.append(f"VERDICT: {'PASS' if passed(rows) else 'FAIL'}")
    if ootb and ours:
        lines += ["", "step   loss OOTB   loss ours   grad norm OOTB   grad norm ours"]
        lines += [
            f"{i:4d} {float(ootb['steps'][i]['lm loss']):11.4f} {float(ours['steps'][i]['lm loss']):11.4f} "
            f"{float(ootb['steps'][i]['grad norm']):16.4g} {float(ours['steps'][i]['grad norm']):16.4g}"
            for i in sorted(set(ootb["steps"]) & set(ours["steps"]))
            if i <= EARLY_STEPS or i % 10 == 0
        ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gpus", default="2,3", help="two or more free GPUs (default 2,3)")
    p.add_argument(
        "--config", default="v1", help="config in bridge/models/gemma4/config/ (default v1)"
    )
    p.add_argument(
        "--steps", type=int, default=100, help="training steps of the functional runs (default 100)"
    )
    p.add_argument(
        "--slow-steps", type=int, default=40, help="steps of the slow-link timing runs (default 40)"
    )
    args = p.parse_args()
    if args.steps < 2 * FINAL_STEPS or args.slow_steps < 30:
        p.error(
            f"--steps must be at least {2 * FINAL_STEPS} and --slow-steps at least 30 (timing skips 20 steps)"
        )
    gpus = launch.gpu_info(args.gpus)
    if len(gpus) < 2 or any(g.used >= launch.BUSY for g in gpus):
        sys.exit(f"need two or more free GPUs; {args.gpus} hold {[g.used for g in gpus]} MiB")

    out = REPO / "runs" / "validate" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    config = CONFIGS / f"{args.config}.yaml"
    cfg = launch.load(config)
    print(f"validating {config.name} against OOTB on GPUs {args.gpus} -> {out}", flush=True)
    k = kernels(out, args.gpus.split(",")[0], cfg)
    slow = dict(NCCL_P2P_DISABLE="1")
    ootb = train(out, config, "ootb", args.gpus, args.steps, "ootb=true")
    ours = train(out, config, "ours", args.gpus, args.steps)
    ootb_slow = train(
        out, config, "ootb-slow-link", args.gpus, args.slow_steps, "ootb=true", **slow
    )
    ours_slow = train(out, config, "ours-slow-link", args.gpus, args.slow_steps, **slow)
    rows = checks(k, ootb, ours, ootb_slow, ours_slow, args.steps, gpus[0].total * 2**20 / 1e9)
    text = report(rows, ootb, ours)
    figure = "\n![Training](training.png)\n" if plot(out, cfg) else ""
    (out / "report.md").write_text(f"```\n{text}\n```\n{figure}")
    print("\n" + text)
    sys.exit(0 if passed(rows) else 1)


if __name__ == "__main__":
    main()
