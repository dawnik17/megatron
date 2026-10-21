# kernels

Kernel-level functional check: every kernel our changes swap into a Megatron-Bridge model computes the same function
as the one it replaces, forward and backward, and is at least as accurate.

## What it checks

- **Kernels.** One forward pass of the pretrained model on FineWeb-Edu text records the arguments each changed
function receives, once per distinct shape. On those inputs, OOTB and ours both run in BF16 and are compared with
the exact result: OOTB's own code in float64, so every detail of Megatron's semantics (mask, window, scale, GQA
layout) is in the reference. Ours passes if, for the output and every input gradient, its error is at most
`MARGIN` (1.25) times OOTB's.
- **Gradients.** With every parameter trainable, the parameters that get no gradient must be exactly the ones
`train.py` freezes, with OOTB's functions and with ours.

```text
18 parameters frozen; without a gradient: OOTB 18, ours 18: PASS
_logit_softcapping [(1024, 1, 262144)]   PASS  out 2.71e-03 -> 1.66e-03  dlogits 3.08e-03 -> 1.62e-03
```

Read each row as OOTB's error, then ours, both relative to float64.

## Run

In the NeMo container, on one GPU, with the environment `container()` in `bridge/models/launch.py` sets (Megatron-Bridge
on `PYTHONPATH`, `HF_HOME`, `HF_HUB_OFFLINE=1`):

```bash
python bridge/validate/kernels --out kernels.json --model gemma4-e4b     # every change of gemma4-e4b
python bridge/validate/kernels --out kernels.json --only _gemma4_rms_norm  # one change on its own
```

From the host, at the repo root:

```bash
enroot start -e NVIDIA_VISIBLE_DEVICES=0 -e HOME=/tmp -e HF_HOME=/repo/.cache/hf -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM \
  -m "$PWD:/repo" pyxis_nemo-26-08 \
  python /repo/bridge/validate/kernels --out /repo/kernels.json
```

The container sees the repo at `/repo`, wherever it is on the host, and starts in `/workspace`, so the paths are
absolute. Add `-e GEMMA4_...` to change which patches `train.py` applies.

`--model` picks the config (default `gemma4-e4b`). `--only` takes the attribute names that start each row. The JSON
is what `bridge/validate/run.py` reads.

## Layout

| File | Role |
|---|---|
| `__init__.py` | `main()`: the steps in order |
| `inputs.py` | spy on the changed functions, capture their inputs |
| `eval.py` | ours and OOTB against float64, the pass rule |
| `grad.py` | the gradient check |
| `data.py` | FineWeb-Edu tokens |
| `config/` | everything model-specific: one `Config` per model |

## Add a model

Subclass `Config` in `config/<family>.py`, passing the recipe, the changed functions as `(owner, attribute)`, the
folder of the `train.py` under test, and the HF checkpoint. Then register it in `config/__init__.py`. Override
`load_model` or `config_flag_changes` only if the model needs it; `config/gemma4.py` shows both.

## Caveats

- **Memory.** One GPU, TP 1: Gemma 4 E4B at 4096 tokens peaks near 110 GB. Lower `seq_length` to fit an 80 GB card.
- **One call per shape.** Non-tensor arguments count only by type. If two layer types share shapes (say, sliding and
global attention with the same head dim), only the first is compared.
- **No rows, no FAIL.** A change the forward pass never calls through its registered attribute is silently
unchecked. Check that every change shows up in the output.
- **float64 OOTB.** The reference needs OOTB code that runs in float64, which plain PyTorch does and Transformer
Engine kernels don't.
