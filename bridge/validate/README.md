# Validation

Checks our changes to Megatron-Bridge's Gemma 4 E4B recipe against the recipe as shipped (OOTB), on real data, in
about 20 minutes on two free GPUs:

```bash
# on B300
python3 bridge/validate/run.py --gpus 2,3

# on A100s: config v2 (sequence 512, micro-batch 2) fits on 6 GPUs
python3 bridge/validate/run.py --config v2 --gpus 2,3,4,5,6,7
```

It prints one table with a verdict and exits 0 only if every check passes. Everything it ran stays in
`runs/validate/<date-time>/`: `report.md`, `kernels.log`, and each training run's `train.log` and `mfu_report.txt`.

**What is compared.** *Ours* is a training config, `bridge/models/gemma4/config/<name>.yaml` (`--config`, default
`v1`), run by `bridge/models/launch.py`: `train.py`'s runtime patches plus the config's `overrides`. *OOTB* is the same config
with `ootb=true`: Megatron-Bridge's own `run_recipe.py`, with neither. Both get the config's run settings: TP, the data,
the global batch, the sequence length, a 10-step warm-up and the seed, on the same GPUs for the same steps. OOTB keeps
every other recipe setting but the config's `ootb_overrides`: the recipe overlaps the gradient reduce, which fails on
Gemma 4 after the first step unless our freeze patch is in place, so OOTB runs without that overlap. A change is
validated as soon as it is in the config or `train.py`.

## Functional: does it compute and train the same model?

| Check | Passes when |
|---|---|
| Kernels | Each kernel we swap in is at most 1.25× as far from the exact result as the OOTB kernel it replaces, for its output and for every input gradient |
| Parameters without a gradient | Exactly the ones `train.py` freezes, with and without the patches |
| Same batches and learning rates | Identical at every step |
| No NaN or skipped steps | In both runs |
| Loss and gradient norm, steps 1–5 | Within 0.25% and 6% of OOTB's at every step |
| Mean loss over the last 20 steps | Within 10% of OOTB's |

**Kernels** (how: [`kernels/`](kernels/README.md); on one GPU, at 1024 tokens, as 4096 needs about 110 GB). No BF16
kernel is exact, so the rule is relative: ours must be no further from float64 than Megatron's own. At 1024 tokens
every kernel is at 0.2–1.0× OOTB's error: FlashAttention 2, the compiled softcap and fused RoPE are more accurate, the
compiled norms and TE's global attention equal. The deliberate-bug results (a sliding window one key short: 8.6× OOTB's
error; a 2% softmax-scale error: 6×) were measured on the earlier hand-built inputs and are not yet redone on captured
ones. The only parameters left without a gradient are the 18 `k_layernorm` weights of the KV-shared layers, with or
without the patches, and they are exactly what `train.py` freezes.

**Training.** OOTB and ours train 100 steps on FineWeb-Edu from the same random weights in the same batch order, the
way pretraining does: data-parallel across both GPUs with several micro-batches per step, gradient clipping, weight
decay, the distributed optimizer and a warm-up then cosine-decay schedule, all from the recipe. Two facts set what can
be checked:

- At step 1 both runs have the same weights and the same batch, so only rounding separates them, and the first updates
are small (warm-up). Over steps 1–5, equivalent runs (OOTB, OOTB at micro-batch 2, ours, ours again) were at most
0.076% apart in loss and 1.9% in gradient norm. A different forward or backward pass, gradient reduce or optimizer
step shows up here.
- From about step 9 every run goes through loss spikes: at random initialization the gradient norm is about 230,000,
and the learning rate peaks at step 10. Where the spikes fall depends on rounding, and so does where a 100-step run
ends: two runs of the very same code ended 3.2% apart in loss (and 141% in gradient norm, which is why that is not
checked at the end). So the end-of-run check catches training that stops working, not small effects; those are the
job of the kernel and steps 1–5 checks. A slower warm-up only moves the spikes later.

## Non-functional: is it faster, does it fit, and does it stay faster at scale?

| Check | Passes when |
|---|---|
| Step time | Ours is faster (median step time after step 20; MFU from `bridge/models/mfu.py`) |
| Peak memory per GPU | At most 95% of the GPU, with no allocator retries |
| Slow link: speedup kept | At least 80% of our speedup over OOTB remains |

**Slow link.** On this node the two GPUs talk over NVLink at about 490 GB/s. With `NCCL_P2P_DISABLE=1`, NCCL stays off
NVLink and goes through host memory, at 26 GB/s. A ring moves (N−1)/N of the gradient buffer through each rank: half
of it with two ranks, nearly all of it with many. So two ranks at 26 GB/s spend as long communicating per step as 64
data-parallel ranks on nodes that share a single 400 Gb/s NIC (about 50 GB/s of bus bandwidth). That is a worst case:
a fabric with a NIC per GPU gives each node about eight times that. Both runs are repeated this way, and the report
shows how much communication time each leaves exposed and how much of our speedup survives. Faster compute leaves less
time to hide the same communication, so some speedup is always lost; today ours keeps about 85% (88% with
`ddp.overlap_param_gather=true`, which is also no slower over NVLink). Below 80%, communication has become the
bottleneck, for example because a change stopped overlapping it. Part of what stays exposed in both runs cannot be
hidden: the embedding and per-layer-embedding weights, 3.5 of the model's 7.5 billion, get their gradients last in the
backward pass, so nothing is left to overlap their reduce with.

## The data

FineWeb-Edu `sample/10BT/000_00000.parquet`, pinned to a Hub revision and tokenized once by `bridge/data/prepare.py`
into `.cache/data/fineweb-edu/google--gemma-4-E4B-it/`: 726,000 documents and 745 million tokens, each document
`<bos> text <eos>` as Gemma expects. Running the script again gives the same files byte for byte. The configs train on
it (`data: fineweb-edu`, or `data=mock` for random tokens), so validation, benchmarks and real training read the same
data. To prepare it elsewhere (it downloads 2.15 GB once), run the enroot command in `prepare.py`'s docstring.

## When you change something

- **A new runtime patch in** `train.py`**:** add the function it replaces to `CHANGED` in `kernels/config/gemma4.py`; its
inputs are captured from the model. A config flag that switches kernels also needs its ours in
`config_flag_changes()`. The training and non-functional checks need nothing. Another model is a `Config` in
`kernels/config/`.
- **A new recipe override:** put it in the config's `overrides`. Everything else the config sets is shared by both runs.
- **A limit:** the constants at the top of `run.py`, and `MARGIN` in `kernels/eval.py`. The measurements behind them are above,
and their runs are in `runs/validate/calibration/` (with `mutate.py`, the deliberately broken kernels, and
`nccl_bw.py`, the link bandwidths); redo them if you change the steps, the batch or the schedule.

## What it does not show

- Convergence over a long run. 100 steps catch training that breaks, not effects smaller than the run-to-run spread
above.
- Other layouts. It runs data-parallel on two GPUs with TP 1, the production layout; TP > 1 and real multi-node runs are
not exercised, and the slow link only emulates their communication.
- Training from pretrained weights.
