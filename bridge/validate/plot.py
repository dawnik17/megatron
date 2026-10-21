#!/usr/bin/env python3
"""Plot every training run of a validation suite on shared axes, step by step, from the runs' train.log.

  python bridge/validate/plot.py [runs/validate/<date-time>]     # default: the newest; writes training.png there

Six panels: loss, loss relative to OOTB, gradient norm, step time, throughput and peak memory (Megatron logs memory
only for the first steps, so that one is a bar per run). A run's colour is its family (ootb, ours, ...) and its
slow-link twin is dashed. Needs matplotlib, which the host python3 may lack: use the container's python (run.py does)
or any python with matplotlib.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from run import EARLY_LOSS, EARLY_STEPS, REPO  # noqa: E402
from mfu import TFLOPS, TIME, parse, step_ms  # noqa: E402  (run puts bridge/models on the path)

PALETTE = [
    "#3B6FB6",
    "#E07B39",
    "#3A9E6F",
    "#B5485D",
    "#7E62B3",
    "#C9A227",
    "#4AA5B8",
    "#8C6D5A",
    "#D16BA5",
    "#5F6B7A",
]

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFBFD",
        "axes.edgecolor": "#C9CDD6",
        "axes.grid": True,
        "grid.color": "#E4E7EE",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelcolor": "#3A3F4B",
        "xtick.color": "#5A606C",
        "ytick.color": "#5A606C",
        "legend.frameon": False,
    }
)


def xy(run, field):
    """(steps, values) of one logged field."""
    pts = [(i, float(s[field])) for i, s in sorted(run["steps"].items()) if field in s]
    return np.array(pts).T if pts else np.empty((2, 0))


def family(name):
    return re.sub(r"-slow(-link)?", "", name)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dir", nargs="?", type=Path)
    out = p.parse_args().run_dir or max((REPO / "runs/validate").glob("2*"))
    runs = {d.name: parse(d) for d in out.iterdir() if (d / "train.log").exists()}
    runs = {name: run for name, run in runs.items() if run["steps"]}
    if not runs:
        sys.exit(f"no training steps logged under {out}")

    order = sorted(runs, key=lambda n: (family(n) != "ootb", family(n) != "ours", family(n), n))
    families = list(dict.fromkeys(map(family, order)))
    style = {
        n: dict(
            color=PALETTE[families.index(family(n)) % len(PALETTE)],
            ls="--" if "slow" in n else "-",
            lw=1.7,
        )
        for n in order
    }

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), layout="constrained")
    (loss, diff, grad), (time, tflops, mem) = axes
    for n in order:
        run, kw = runs[n], style[n]
        loss.plot(*xy(run, "lm loss"), label=n, **kw)
        grad.plot(*xy(run, "grad norm"), **kw)
        time.plot(*xy(run, TIME), **kw)
        tflops.plot(*xy(run, TFLOPS), **kw)
        median = step_ms(run)  # run.py's step time
        time.axhline(median, color=kw["color"], ls=":", lw=1, alpha=0.8)
        time.annotate(
            f"{median:.0f}",
            (1, median),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=kw["color"],
        )

    for n in order:
        base = n.replace("ours", "ootb", 1)
        if n.startswith("ours") and base in runs:
            a, b = (dict(zip(*xy(runs[r], "lm loss"))) for r in (n, base))
            steps = sorted(a.keys() & b.keys())
            diff.plot(steps, [100 * (a[s] / b[s] - 1) for s in steps], **style[n])
    diff.set_yscale("symlog", linthresh=1)
    diff.axhline(0, color="#7A808C", lw=1)
    diff.axvspan(0.5, EARLY_STEPS + 0.5, color="#3A9E6F", alpha=0.10, lw=0)
    diff.text(
        EARLY_STEPS + 1,
        0.97,
        f"steps 1-{EARLY_STEPS}: within ±{100 * EARLY_LOSS:g}%",
        transform=diff.get_xaxis_transform(),
        va="top",
        fontsize=8,
        color="#2F7D58",
    )

    steady = np.concatenate([xy(runs[n], TIME)[1][2:] for n in order])
    if steady.size:  # step 1 compiles and warms up: keep it from flattening the rest
        time.set_ylim(0, 1.2 * np.percentile(steady, 98))
    grad.set_yscale("log")
    titles = [
        ("LM loss", ""),
        ("Loss relative to OOTB", "%, log beyond ±1%"),
        ("Gradient norm", "log scale"),
        ("Step time", "ms, dotted: median after step 20"),
        ("Throughput", "TFLOP/s per GPU"),
    ]
    for ax, (title, unit) in zip(axes.flat, titles):
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit, fontsize=9)
        ax.set_xlabel("step", fontsize=9)
        ax.set_xlim(left=0)

    for i, n in enumerate(order):
        bar = mem.barh(
            i,
            runs[n]["peak_gb"],
            color=style[n]["color"],
            hatch="//" if "slow" in n else "",
            edgecolor="white",
            height=0.62,
        )
        mem.bar_label(bar, fmt="%.0f GB", padding=4, fontsize=9, color="#3A3F4B")
    mem.set_yticks(range(len(order)), order)
    mem.invert_yaxis()
    mem.grid(axis="y", visible=False)
    mem.margins(x=0.15)
    mem.set_title("Peak reserved memory", loc="left")
    mem.set_xlabel("GB per GPU, rank 0", fontsize=9)

    fig.suptitle(
        f"Training, OOTB vs ours  ·  {out.name}", fontsize=15, fontweight="bold", x=0.01, ha="left"
    )
    fig.legend(
        *loss.get_legend_handles_labels(), loc="outside lower center", ncol=min(len(order), 6)
    )
    fig.savefig(out / "training.png", dpi=160)
    print(out / "training.png")


if __name__ == "__main__":
    main()
