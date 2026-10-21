#!/usr/bin/env python3
"""Train a Megatron-Bridge model from a YAML config in the NeMo container, then report MFU.

    GPU_IDS=2,3,4,5,6,7 python3 bridge/models/launch.py bridge/models/gemma4/config/v1.yaml [run name] [KEY=VALUE ...]

KEY=VALUE overrides a config key (mbs=2, ootb=true) or, with a dot in KEY, any field of Megatron-Bridge's
ConfigContainer (ddp.overlap_grad_reduce=false). A config's shared fields reach OOTB and ours, its overrides only ours,
its ootb_overrides only OOTB. A config can start from another with base: <name>. The run lands in
runs/train/<model_family>/<run name>/ (or RUN_DIR): config.yaml as run, train.log and mfu_report.txt.
PROFILE=1 (PyTorch profiler) or PROFILE=nsys, set in front or in the config's env, records rank 0's steps 30-31 and
writes profile_breakdown.txt.
The config's env reaches the container, and so do NCCL_*, PYTORCH_* and <MODEL_FAMILY>_* variables set in front, which
win over it. The container sees the repo at /repo and the run's folder at /out, wherever they are on the host. On
another machine, set IMAGE and SQSH (the enroot container and the .sqsh it is created from) and, if needed,
TORCH_CACHE (the compile cache: /raid/$USER/torch-cache if /raid/$USER exists, else .cache/torch).
"""

import getpass
import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
import yaml
import mfu

REPO = Path(__file__).resolve().parents[2]
# where the container sees the repo, the run's folder and the compile cache, whatever their host paths
IN_REPO, IN_OUT, IN_CACHE = Path("/repo"), Path("/out"), Path("/cache")
IMAGE = os.environ.get("IMAGE", "pyxis_nemo-26-08")
SQSH = os.environ.get("SQSH", "/sadata/nikkothari/Flipkart_Nemotron_35_lightning/nemo-26-08.sqsh")
RUN_RECIPE = "/opt/Megatron-Bridge/scripts/training/run_recipe.py"
PEAK = {"A100": 312, "H100": 989, "H200": 989, "B200": 2250, "B300": 2250}  # BF16 TFLOP/s
GPU = namedtuple("GPU", "index name used total")  # memory in MiB
BUSY = 1024  # MiB used from which a GPU counts as in use

DEFAULTS = dict(
    data="fineweb-edu",
    tp=1,
    pp=1,
    mbs=None,
    gbs=None,
    seq_len=4096,
    train_iters=100,
    warmup_iters=10,
    ootb=False,
    shared={},
    overrides={},
    ootb_overrides={},
    env={},
)

# config key -> run_recipe.py option
FLAGS = dict(
    tp="-tp", pp="-pp", seq_len="-sl", gbs="-gb", train_iters="-ms", warmup_iters="--warmup_iters"
)

PLUMBING = [
    "checkpoint.save=null",
    "checkpoint.load=null",
    "checkpoint.pretrained_checkpoint=null",
    "logger.log_interval=1",
    "logger.log_throughput=true",
    "validation.eval_iters=0",
]

# nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite --force-overwrite=true
NSYS = [
    "nsys",
    "profile",
    "-t",
    "cuda,nvtx,osrt",
    "--capture-range=cudaProfilerApi",
    "--capture-range-end=stop",
    "--export=sqlite",
    "--force-overwrite=true",
]

# profiling.profile_step_start=30 profiling.profile_step_end=32
STEPS = ["profiling.profile_step_start=30", "profiling.profile_step_end=32"]


def load(path):
    """A config file on top of its base config; dict-valued keys merge one level deep."""
    cfg = yaml.safe_load(Path(path).read_text())
    if "base" not in cfg:
        return {**DEFAULTS, **cfg}
    base = load(Path(path).with_name(f"{cfg.pop('base')}.yaml"))
    merged = {k: {**base[k], **v} for k, v in cfg.items() if isinstance(v, dict)}
    return {**base, **cfg, **merged}


def config(path, args):
    """The config with KEY=VALUE args applied, and the run name: the first other argument, else the config's."""
    cfg, name = load(path), Path(path).stem
    cfg["extra"] = {}  # dotted overrides from the command line, for OOTB and ours alike
    for arg in args:
        key, eq, value = arg.partition("=")
        if not eq:
            name = arg
        else:
            (cfg["extra"] if "." in key else cfg)[key] = yaml.safe_load(value)
    return cfg, name


def value(v):
    return "null" if v is None else str(v).lower() if isinstance(v, bool) else str(v)


def recipe_args(cfg, ngpu):
    """run_recipe.py's arguments: the run settings, ours' changes or OOTB's, and the run's plumbing."""
    # default batch: 8 sequences per data-parallel rank
    cfg["gbs"] = cfg["gbs"] or 8 * ngpu // (cfg["tp"] * cfg["pp"])
    model, data = cfg["model"], cfg["data"]

    # Set Dataset
    if data == "mock":
        # the model takes the tokenizer's vocabulary: the mock one's 32,000 would shrink it
        vocab = mfu.hf_config(model)["vocab_size"]
        dataset = ["--dataset", "mock", f"tokenizer.vocab_size={vocab}"]
    else:
        # tokenized once by bridge/data/prepare.py
        name = model.replace("/", "--")
        prefix = REPO / ".cache/data" / data / name / f"{data}_text_document"

        if not Path(f"{prefix}.idx").exists():
            sys.exit(f"{prefix}.idx not found; see bridge/data/prepare.py")
        prefix = IN_REPO / prefix.relative_to(REPO)

        dataset = (
            f"--dataset megatron-indexed dataset.data_path={prefix} dataset.path_to_cache={prefix.parent}/index "
            f"tokenizer.tokenizer_type=HuggingFaceTokenizer tokenizer.tokenizer_model={model}"
        ).split()

    recipe = ["--recipe", cfg["recipe"], "--mode", "pretrain"]
    mbs = [] if cfg["ootb"] or not cfg["mbs"] else ["-mb", cfg["mbs"]]
    flags = [a for key, flag in FLAGS.items() for a in (flag, cfg[key])]
    plumbing = [*PLUMBING, f"logger.tensorboard_dir={IN_OUT}/tb_logs"]
    ours_or_ootb = cfg["ootb_overrides"] if cfg["ootb"] else cfg["overrides"]
    changes = {**cfg["shared"], **ours_or_ootb, **cfg["extra"]}
    overrides = [f"{k}={value(v)}" for k, v in changes.items()]
    return [*recipe, *dataset, *mbs, *flags, *plumbing, *overrides]


def profiler(run, cfg):
    """Recipe arguments, a command prefix and the trace to break down, for PROFILE=1 or nsys, set in front or in
    the config's env (in front wins)."""
    kind = str(os.environ.get("PROFILE") or cfg["env"].get("PROFILE", "0"))

    if kind == "1":
        trace = run / "torch_profile/rank-0.json.gz"
        return ["profiling.use_pytorch_profiler=true", *STEPS], [], trace

    if kind == "nsys":
        (run / "nsys").mkdir(exist_ok=True)
        trace = run / "nsys/trace.sqlite"
        nsys = [*NSYS, "-o", f"{IN_OUT}/nsys/trace"]
        return ["profiling.use_nsys_profiler=true", *STEPS], nsys, trace

    return [], [], None


def container(gpus, cfg, out):
    """
    enroot start for the NeMo container, created on first use:
    GPUs, the repo at /repo, the folder out at /out, the compile cache at /cache, and the variables that reach it.
    Variables set in front win over the config's env.
    """

    if (
        IMAGE
        not in subprocess.run(["enroot", "list"], capture_output=True, text=True).stdout.split()
    ):
        if not Path(SQSH).exists():
            sys.exit(f"no enroot container {IMAGE} and no {SQSH}: set SQSH or IMAGE")
        subprocess.run(["enroot", "create", "-n", IMAGE, SQSH], check=True)

    # local disk if there is one (DGX's /raid), so torch.compile and Triton reuse their kernels across runs
    raid = Path("/raid") / getpass.getuser()
    local = raid / "torch-cache" if raid.is_dir() else REPO / ".cache/torch"
    cache = os.environ.get("TORCH_CACHE") or local
    Path(cache).mkdir(parents=True, exist_ok=True)
    prefixes = ("NCCL_", "PYTORCH_", f"{cfg['model_family'].upper()}_")

    env = dict(
        NVIDIA_VISIBLE_DEVICES=gpus,
        HOME="/tmp",
        HF_HOME=f"{IN_REPO}/.cache/hf",
        HF_HUB_OFFLINE=1,
        PYTHONPATH="/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM",
        CUDA_DEVICE_MAX_CONNECTIONS=1,
        TORCHINDUCTOR_CACHE_DIR=f"{IN_CACHE}/inductor",
        TRITON_CACHE_DIR=f"{IN_CACHE}/triton",
    )
    env |= cfg["env"] | {k: v for k, v in os.environ.items() if k.startswith(prefixes)}

    variables = [a for k, v in env.items() for a in ("-e", f"{k}={v}")]
    places = ((REPO, IN_REPO), (out, IN_OUT), (cache, IN_CACHE))
    mounts = [a for host, inside in places for a in ("-m", f"{host}:{inside}")]
    return ["enroot", "start", *variables, *mounts, IMAGE]


def gpu_info(ids):
    """The GPUs in ids (e.g. "2,3"), as nvidia-smi reports them."""
    query = "--query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits".split()
    out = subprocess.run(
        ["nvidia-smi", "-i", ids, *query], capture_output=True, text=True, check=True
    ).stdout
    return [
        GPU(i, name, int(used), int(total))
        for i, name, used, total in (r.split(", ") for r in out.splitlines())
    ]


def peak_tflops(gpus):
    """The GPUs' dense BF16 peak. Exits if one holds memory: a shared GPU makes the step time meaningless."""
    info = gpu_info(gpus)
    if busy := [g for g in info if g.used >= BUSY]:
        sys.exit(f"GPUs already in use: {busy}")
    peak = os.environ.get("PEAK_TFLOPS") or next(
        (v for k, v in PEAK.items() if k in info[0].name), None
    )
    return float(peak) if peak else sys.exit(f"unknown GPU {info[0].name}: set PEAK_TFLOPS")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    cfg, name = config(sys.argv[1], sys.argv[2:])
    gpus = os.environ.get("GPU_IDS") or sys.exit("set GPU_IDS, e.g. GPU_IDS=4,5,6,7")
    ngpu, first = len(gpus.split(",")), int(gpus.split(",")[0])

    run = Path(
        os.environ.get("RUN_DIR") or REPO / "runs/train" / cfg["model_family"] / name
    ).resolve()
    run.mkdir(parents=True, exist_ok=True)
    peak = peak_tflops(gpus)

    args = recipe_args(cfg, ngpu)
    profile, prefix, trace = profiler(run, cfg)
    train_py = IN_REPO / "bridge/models" / cfg["model_family"] / "train.py"
    entry = RUN_RECIPE if cfg["ootb"] else train_py
    port = os.environ.get("MASTER_PORT", 29500 + first)
    torchrun = (
        f"python -m torch.distributed.run --nproc_per_node {ngpu} --master_port {port}".split()
    )
    cmd = [*container(gpus, cfg, run), *prefix, *torchrun, entry, *args, *profile]
    saved = {**cfg, "gpus": gpus, "peak_tflops": peak}
    (run / "config.yaml").write_text(yaml.safe_dump(saved, sort_keys=False))

    with open(run / "train.log", "w", buffering=1) as log:

        def tee(line):
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)

        what = "OOTB recipe" if cfg["ootb"] else "our changes"
        tee(f"{name}: {what}, {sys.argv[1]}, GPUs {gpus}, GBS {cfg['gbs']} -> {run}\n")
        cmd = [str(a) for a in cmd]
        tee(
            f"Running: {" ".join([c + "\n" if (i+1) % 2 == 0 else c for i, c in enumerate(cmd)])}\n"
        )

        pipe = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        proc = subprocess.Popen(cmd, **pipe)
        for line in proc.stdout:
            tee(line)
    if proc.wait():
        sys.exit(proc.returncode)

    (run / "mfu_report.txt").write_text(report := mfu.report(run))
    print(report, end="")
    if trace:
        breakdown = [sys.executable, REPO / "bridge/profile/breakdown", trace]
        out = subprocess.run(breakdown, capture_output=True, text=True).stdout
        (run / "profile_breakdown.txt").write_text(out)
        print(out, end="")


if __name__ == "__main__":
    main()
