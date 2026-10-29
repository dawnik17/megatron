# models

Training runs. Each model family has a folder with its patches, its FLOP count and its configs; `launch.py` turns any
config into a Megatron-Bridge run in the NeMo container.

```
models/
├── launch.py              config -> run_recipe.py arguments -> container, profiler, train.log, MFU, profile breakdown
├── mfu.py                 train.log, step time and MFU of a run folder: python3 bridge/models/mfu.py <run dir>
└── gemma4/
    ├── config/v1.yaml     the best B300 configuration (40.2% MFU), every value written out
    ├── config/v2.yaml     v1 on 80 GB A100s: base: v1, plus what differs
    ├── config/12b.yaml    Gemma 4 12B: base: v1, with our 12B recipe, on 4x 80 GB A100s
    ├── 12B/               the 12B text tower: bridge.py (registers its HF architecture), recipe.py (our recipe)
    ├── train.py           our runtime patches and our recipes; then it runs Megatron-Bridge's run_recipe.py unchanged
    ├── flops.py           forward FLOPs per token for MFU, from the model's HF config
    ├── run.sh             the standalone launcher: settings from environment variables, no config
    └── v0/                the reference run the optimizations started from
```

## Run

From the repo root, on free GPUs. Both ways start the same container and `train.py`.

**1.** `launch.py`**, from a config** (what `bridge/validate` uses):

```bash
GPU_IDS=4,5,6,7 python3 bridge/models/launch.py bridge/models/gemma4/config/v1.yaml
GPU_IDS=2,3,4,5,6,7 python3 bridge/models/launch.py bridge/models/gemma4/config/v2.yaml my-run mbs=1 train_iters=50
GPU_IDS=4,5,6,7 python3 bridge/models/launch.py bridge/models/gemma4/config/v1.yaml ootb ootb=true   # recipe as shipped
GPU_IDS=4,5,6,7 PROFILE=nsys python3 bridge/models/launch.py bridge/models/gemma4/config/v1.yaml v1-nsys
```

After the config come a run name and `KEY=VALUE` overrides: a key without a dot is a config key, one with a dot any
field of Megatron-Bridge's ConfigContainer (`ddp.bucket_size=40000000`). A run lands in
`runs/train/<model_family>/<run name>/` (or `RUN_DIR`): `config.yaml` as it ran, `train.log`, `mfu_report.txt`, and
with `PROFILE=1` (PyTorch profiler) or `PROFILE=nsys`, set in front or in the config's `env` (in front wins), the
trace of steps 30-31 and `profile_breakdown.txt`. The GPU's
peak for MFU is detected (`PEAK_TFLOPS` overrides it). `NCCL_*`, `PYTORCH_*` and `<MODEL_FAMILY>_*` variables set in
front reach the container.

The container sees the repo at `/repo` and the run's folder at `/out`, so the repo works from any folder. On another
machine, set `SQSH` to the NeMo container's `.sqsh` (and `IMAGE` to name it). The compile cache is
`/raid/$USER/torch-cache` where that folder exists (local disk on DGX machines), else `.cache/torch`; `TORCH_CACHE`
overrides it. Both ways read these the same.

**2.** `gemma4/run.sh`**, from environment variables** (standalone, no config):

```bash
GPU_IDS=4,5,6,7 bridge/models/gemma4/run.sh best
GPU_IDS=2,3,4,5,6,7 SEQ_LEN=512 MBS=2 PEAK_TFLOPS=312 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bridge/models/gemma4/run.sh a100
GPU_IDS=4,5,6,7 OOTB=1 bridge/models/gemma4/run.sh ootb                               # recipe as shipped
```

Its settings are `TP`, `MBS` (4), `GBS` (8 per GPU), `SEQ_LEN` (4096), `TRAIN_ITERS` (100), `DATA` (`fineweb-edu` or
`mock`), `OOTB`, `PROFILE` and `PEAK_TFLOPS` (2250, a B300; 312 on A100s). Arguments after the run name go to
`train.py` as Megatron overrides. Runs land in `runs/train/gemma4-mfu/<run name>/`.

## Configs

| Key | Meaning |
|---|---|
| `base` | another config in the same folder to start from; dict-valued keys merge |
| `model_family`, `model`, `recipe` | the folder here, the HF checkpoint, Megatron-Bridge's recipe function |
| `data` | `fineweb-edu` (tokenized by `bridge/data/prepare.py`) or `mock` |
| `tp`, `pp`, `mbs`, `gbs`, `seq_len`, `train_iters`, `warmup_iters` | run settings; `gbs: null` is 8 sequences per GPU, and OOTB keeps the recipe's `mbs` |
| `shared` | ConfigContainer fields for OOTB and ours alike |
| `overrides` | our changes: ours only |
| `ootb_overrides` | `ootb=true` runs `run_recipe.py` instead of `train.py`, with only these |
| `env` | variables for the container, such as `train.py`'s `GEMMA4_*` switches |

## A new model family

A folder `models/<family>/` with a `train.py` that patches functions and then runs `run_recipe.py` (as
`gemma4/train.py` does), a `flops.py` with `forward_flops_per_token(hf_text_config, seq_len)`, and its `config/`.

## A recipe of our own

`run_recipe.py --recipe NAME` only looks in Megatron-Bridge's `megatron.bridge.recipes`. `gemma4/train.py` adds every
function in the `__all__` of `gemma4/<size>/recipe.py` there before it starts `run_recipe.py`, so a config's
`recipe:` can name one (`12B/recipe.py`'s `gemma4_12b_pretrain_h200_bf16_config`). `ootb=true` runs `run_recipe.py`
alone and can't find it. To check a recipe without training, run `train.py` with `--dry-run`: it builds the recipe,
applies the overrides and writes the resulting ConfigContainer:

```bash
enroot start -e NVIDIA_VISIBLE_DEVICES=7 -e HOME=/tmp -e HF_HOME=/repo/.cache/hf -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM \
  -m $PWD:/repo -m /tmp:/out pyxis_nemo-26-08 python /repo/bridge/models/gemma4/train.py \
  --recipe gemma4_12b_pretrain_h200_bf16_config --mode pretrain --dataset mock tokenizer.vocab_size=262144 -tp 4 \
  --dry-run logger.save_config_filepath=/out/ConfigContainer.yaml
```

Then a few steps on GPUs: `GPU_IDS=2,5,6,7 python3 bridge/models/launch.py bridge/models/gemma4/config/12b.yaml smoke
train_iters=10`.
