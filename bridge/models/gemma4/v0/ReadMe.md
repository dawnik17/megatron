# Gemma 4 E4B: MFU benchmark on B300

These steps pre-train **Gemma 4 E4B** (text path) on dummy (mock) data with Megatron-Bridge on
`exp-blr-dgxb300-02` (8× B300), and turn the training log into **Model FLOPs Utilization (MFU)**.

Everything runs in the NeMo 26.08 container (`/sadata/nikkothari/Flipkart_Nemotron_35_lightning/nemo-26-08.sqsh`)
with the Megatron-Bridge 0.6.1 that ships inside it (`/opt/Megatron-Bridge`), using the upstream recipe
`gemma4_e4b_pretrain_config`. Nothing in the container is modified. The speed-ups come from recipe
overrides and runtime patches applied by `train.py`. This setup is adapted from a colleague's 8-GPU run on
`exp-blr-dgxb300-01` (the *reference run* below).

**Result (measured here):** 4× B300 (GPUs 4–7), micro batch 2, global batch 32, sequence 4096:
**33.1% MFU** (1.275 s/step, 25,699 tokens/s/GPU, 745 model TFLOP/s per GPU).
The reference run reached ≈ 30% on 8 GPUs (1.41 s/step at global batch 64); the unmodified recipe reached 3.9%.

```
train/gemma/megatron_mfu_optim/
├── ReadMe.md
├── run.sh            <- one command: checks the GPUs are free, trains in the container, writes the MFU report
├── train.py          <- runtime speed patches, then runs Bridge's run_recipe.py unchanged
└── compute_mfu.py    <- turns the training log into MFU
runs/train/gemma4-mfu/<run name>/
├── train.log         <- full training output, including the resolved config at the top
├── mfu_report.txt    <- the MFU result
└── tb_logs/          <- TensorBoard events
```

## Before you start

Nothing to install or clone:

- The container is unpacked on this node as `pyxis_nemo-26-08` (under `/raid/enroot`). On another node,
  `run.sh` unpacks it on first use (about 2 minutes, 40 GB).
- The recipe builds the model from the config of `google/gemma-4-E4B-it`, which is already in `.cache/hf`.
  `run.sh` reads it offline; no token or weights are needed, since mock pre-training starts from random weights.
- The container's `transformers` 5.12 and Megatron-Bridge 0.6.1 have everything `train.py` patches. The
  reference run cloned Bridge from GitHub and installed `transformers` ≥ 5.15 only because that newer Bridge
  needs it.

## Step 1: Pick free GPUs

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

Other users run jobs on this node outside Slurm, so pick GPUs that show about 0 MiB. `run.sh` refuses to start
on a GPU that already holds memory, because a shared GPU makes the step time, and so the MFU, meaningless.

MFU is per GPU, so any number of GPUs gives a comparable result as long as each GPU does the same work.
`run.sh` keeps 4 micro-batches per GPU per step, as in the reference run, by setting the global batch to
`MBS × GPUs / TP × 4` (32 on 4 GPUs, 64 on 8).

## Step 2: Train and measure

From the repo root:

```bash
cd /sadata/nikkothari/walmart
GPU_IDS=4,5,6,7 train/gemma/megatron_mfu_optim/run.sh baseline
```

The first step takes about 30 s while `torch.compile` builds its kernels; they are cached in
`/raid/$USER/torch-cache`, so later runs start faster. After that each step takes about 1.27 s, and a 100-step
run finishes in about 4 minutes. A healthy run prints the patches, then one line per step:

```
[gemma4-fast] RMSNorm + value norm: torch.compile
[gemma4-fast] final logit softcapping: torch.compile
[gemma4-fast] sliding-window attention: flex_attention (global layers unchanged)
[gemma4-fast] sliding-window attention: flex path in use (seq 4096, window [511, 0])
 iteration       42/     100 | elapsed time per iteration (ms): 1270.1 | ... | lm loss: ...
```

When training ends, `run.sh` prints the MFU report and saves it as `mfu_report.txt`:

```
Step time (median)     : 1275.0 ms
Throughput             : 102,798 tokens/s | 25,699 tokens/s/GPU
Model TFLOP/s per GPU  : 744.8
MFU                    : 33.10 %
```

To stop a run, press Ctrl+C. Don't use Ctrl+Z; that only pauses training, and it keeps holding GPU memory.

### Settings

Set these in front of `run.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `GPU_IDS` | required | GPUs to use, for example `4,5,6,7` |
| `TP` | `1` | Tensor parallel size. With TP=1 every GPU is data parallel (fastest for this model). |
| `MBS` | `2` | Micro batch size per GPU |
| `GBS` | `MBS × GPUs / TP × 4` | Global batch size. Must be a multiple of `MBS × GPUs / TP`. |
| `SEQ_LEN` | `4096` | Sequence length |
| `TRAIN_ITERS` | `100` | Training steps. MFU uses the median step time after the first 20. |
| `PEAK_TFLOPS` | `2250` | Dense BF16 peak per GPU used for MFU |
| `GEMMA4_FAST_NORM`, `GEMMA4_FAST_SOFTCAP`, `GEMMA4_FAST_ATTN` | `1` | Set to `0` to turn off one `train.py` patch |
| `GEMMA4_FREEZE_UNUSED` | `0` | Set to `1` to freeze the unused `k_layernorm` weights; needed for `ddp.overlap_grad_reduce=true` |
| `MASTER_PORT` | `29500` + first GPU id | `torch.distributed` port; differs per GPU set, so two runs on different GPUs don't clash |

Arguments after the run name go to `train.py`, which takes the same arguments as Bridge's
`scripts/training/run_recipe.py`. They come after `run.sh`'s defaults, so a later value replaces a default.
For example:

```bash
GPU_IDS=4,5,6,7 MBS=4 train/gemma/megatron_mfu_optim/run.sh mbs4
GPU_IDS=4,5,6,7 GEMMA4_FAST_ATTN=0 train/gemma/megatron_mfu_optim/run.sh no-flex
GPU_IDS=4,5,6,7 GEMMA4_FREEZE_UNUSED=1 train/gemma/megatron_mfu_optim/run.sh overlap \
    ddp.overlap_grad_reduce=true ddp.overlap_param_gather=true
```

### Running under Slurm instead

This node is in partition `req-628` (account `req-628`, QoS `qos-req-628`). Slurm doesn't know about jobs
started outside it, so a Slurm job can be given a GPU that someone else is using; `run.sh`'s busy check avoids
that. If you do use `srun`, pass `--cpus-per-task` (for example 256 with `--exclusive`): without it Slurm pins the
whole job to 2 CPU threads, and the reference run measured training 3–4× slower that way. `run.sh` starts the
container directly on the host, so all CPU threads are available.

## Results

Measured here, sequence 4096, median step time after warm-up:

| Run | GPUs | MBS | GBS | Step time | Tokens/s/GPU | MFU |
|---|---|---|---|---|---|---|
| `replicate-gpu4`: reference configuration | 4 (4–7) | 2 | 32 | 1.275 s | 25,699 | **33.1%** |

## Differences from the reference run

- **Container code instead of a GitHub clone.** Bridge 0.6.1 and `transformers` 5.12 from the image; no
  `pip install`.
- **`train.py`:** defaults to `BRIDGE_ROOT=/opt/Megatron-Bridge`; the flex-attention patch reads `softcap` with
  `getattr` because this Megatron-Core version's `DotProductAttention` has no `softcap` attribute (the
  original crashed on the first sliding layer); and it logs once when the flex path actually runs.
- **Output paths.** `run_recipe.py` changes into `/opt/Megatron-Bridge`, which is read-only in the container,
  so `run.sh` sends TensorBoard logs to `runs/train/gemma4-mfu/<run>/tb_logs`.
- **`ddp.grad_reduce_in_fp32=false` has no effect.** The `bf16_mixed` precision preset copies its own
  `grad_reduce_in_fp32=True` onto the DDP config after the command-line overrides, so gradients are still
  reduced in FP32 (the log line `Setting up DistributedDataParallel with config ... grad_reduce_in_fp32=True`
  shows it). This probably applied to the reference run as well. Turning it off for real needs a patch to the
  precision preset.
- **4 GPUs instead of 8,** with the same work per GPU. This run was 10% faster per GPU than the reference run.

## Performance changes in the reference run

Measured by the reference run on 8× B300 (`exp-blr-dgxb300-01`), GBS=64, sequence 4096, median step time after
warm-up. Nothing in Megatron-Bridge was changed.

| # | Change | Where | Step time | MFU |
|---|---|---|---|---|
| 0 | Upstream recipe as-is (MBS=1, 2 CPU threads) | – | 10.95 s | 3.9% |
| 1 | Give the job all CPUs | `srun --cpus-per-task=256` | 2.92 s | 14.4% |
| 2 | Fused RMSNorm, fused logit softcap, flex attention for sliding layers, MBS=2 | `train.py`, `MBS=2` | 1.54 s | 27.4% |
| 3 | Gradient-accumulation fusion, no NaN check (and the ineffective BF16 gradient reduce) | `run.sh` overrides | 1.50 s | 28.1% |
| 4 | Fused RoPE | `model.apply_rope_fusion=true` | **1.41 s** | **≈ 30%** |

What each change does:

1. **CPU threads (largest gain).** `srun --ntasks=1` without `--cpus-per-task` gives the job one core (2
   hardware threads). All training processes, their data-loader workers and the CUDA kernel-launch threads
   competed for those 2 threads, so the GPUs sat idle about 70% of the time. `run.sh` avoids this by running
   outside Slurm.
2. **Runtime patches in `train.py`.** Each can be turned off by setting its variable to `0`:
   - `GEMMA4_FAST_NORM`: E4B has 8 RMSNorms per layer (4 layer norms, Q/K/V norms, PLE norm). Bridge computes
     each in FP32 with about 6 separate kernels. `torch.compile` turns each into one fused kernel with the same
     math.
   - `GEMMA4_FAST_SOFTCAP`: the final `30·tanh(logits/30)` runs over a 4096 × 262,144-vocab logit tensor;
     compiled into one kernel.
   - `GEMMA4_FAST_ATTN`: the 35 sliding-window layers only look at 512 keys, but the plain attention computes
     and masks the full 4096 × 4096 matrix. `flex_attention` with a block-sparse window skips the masked
     blocks. Attention for these layers became 3× faster, and the output matches the original to bf16
     precision. The 7 global layers (head_dim 512) keep the original attention: no fast kernel in this
     container supports head_dim 512.
   - **MBS=2:** fewer, larger kernels per token. This cuts the CPU launch overhead and raises GEMM efficiency.
3. **Overrides.** `model.gradient_accumulation_fusion=true` adds weight gradients into the FP32 buffer inside
   the GEMM, which removes about 1,500 extra add kernels per step. `ddp.check_for_nan_in_grad=false` skips a
   per-step check. `ddp.grad_reduce_in_fp32=false` was meant to halve gradient traffic but has no effect (see
   above).
4. **`model.apply_rope_fusion=true`.** Uses the fused RoPE kernel instead of slice/rotate/`torch.cat`. Gemma 4's
   partial RoPE on the global layers stores zero frequencies for the non-rotated dimensions, so the fused
   kernel gives the same result. Losses matched the unfused run to within run-to-run noise.

**Tried and not kept:**

- `ddp.overlap_grad_reduce=true`: 3% slower there, because NVLink makes the gradient sync cheap. It also
  crashes unless `GEMMA4_FREEZE_UNUSED=1`; see Troubleshooting.
- `MBS=4`: out of memory on 8 GPUs. Here MBS=2 peaks at about 130 of the 268 GiB per GPU, so it may be worth
  retrying.
- Flash, cuDNN and memory-efficient SDPA: PyTorch reports "No available kernel" for these shapes.
- flex attention for the head_dim-512 global layers: crashes with a CUDA misaligned-address error.

**What limits MFU now** (reference run's profile of this configuration):

- GEMMs: 39% of GPU time.
- Element-wise ops: 24%.
- Flex attention: 20%.
- The 7 global layers' unfused attention (softmax and mask): about 6%.

Going well beyond this needs a Transformer Engine version of the Gemma 4 E4B layer (fused attention, fused
norm + linear, FP8, CUDA graphs). That means a change inside Megatron-Bridge.

## How MFU is computed

```
MFU = training FLOPs per step / (step time × number of GPUs × peak BF16 FLOP/s per GPU)
```

- **Training FLOPs** equal 3 × forward FLOPs (no activation recompute).
- **Forward FLOPs** are counted from the E4B architecture. The count covers:
  - Q/K/V/O projections (K/V skipped on the 18 KV-shared layers),
  - causal attention (512-token sliding window on 35 layers, 512-dim heads on the 7 global layers),
  - the gated GeLU MLP,
  - the Per-Layer Embeddings,
  - the 262,144-vocab LM head.

  At sequence 4096 this gives **28.98 GFLOP per training token**. The architecture constants in
  `compute_mfu.py` match the `text_config` of `.cache/models/gemma-4-E4B/config.json`.
- **Step time** is the median `elapsed time per iteration` after the first 20 iterations.
- **Peak** is `--peak-tflops 2250`, the dense BF16 figure assumed for B300 (`B300 SXM6 AC`). Check the datasheet
  and adjust if needed.
- Megatron-Bridge also logs its own `throughput per GPU (TFLOP/s/GPU)` (626 in the run above), shown for
  reference. It uses a generic GPT formula that doesn't model Gemma 4's architecture.

## CUDA and optimization flags

The settings in effect for `run.sh`'s defaults: recipe `gemma4_e4b_pretrain_config`, plus `run.sh`'s overrides,
plus the `train.py` patches, at TP=1. To confirm them for a run, check the config dump at the top of
`train.log`.

> **Important:** the Gemma 4 E4B layer definition (`get_gemma4_layer_spec` in
> `/opt/Megatron-Bridge/src/megatron/bridge/models/gemma/modeling_gemma4.py`) always builds the plain PyTorch
> ("local") version of each layer, not the Transformer Engine (TE) version. Features marked "Needs TE" can't be
> switched on with a flag today; they need a code change in Megatron-Bridge.

### Enabled

| Flag | What it does |
|---|---|
| Container started directly on the host (`run.sh`) | All CPU threads are available to the training processes. Under `srun` this needs `--cpus-per-task`. |
| `mixed_precision=bf16_mixed` | Computes in BF16, which roughly doubles speed over FP32. Optimizer state stays in FP32. |
| `ddp.grad_reduce_in_fp32` (forced by `bf16_mixed`) | Reduces gradients between GPUs in FP32, twice the traffic of BF16. The `ddp.grad_reduce_in_fp32=false` override doesn't change it. |
| `ddp.use_distributed_optimizer` | Splits the optimizer state (Adam moments, FP32 weights) across the GPUs to save memory. |
| `ddp.average_in_collective` | Averages gradients inside the communication call itself instead of in a separate step. |
| `model.gradient_accumulation_fusion` | Adds weight gradients into the FP32 buffer inside the matrix-multiply kernel, saving a memory pass. |
| `model.apply_rope_fusion` | Applies rotary position embeddings with one fused kernel. |
| `cross_entropy_loss_fusion` (`native`) | Computes the loss over the 262k-word vocabulary in one fused step, saving memory and time. |
| Compiled RMSNorm (`GEMMA4_FAST_NORM`) | The 8 RMSNorms per layer each run as one fused kernel (same math), via `torch.compile`. |
| Compiled logit softcap (`GEMMA4_FAST_SOFTCAP`) | `30·tanh(x/30)` on the logits runs as one fused kernel. |
| Flex attention for sliding layers (`GEMMA4_FAST_ATTN`) | The 35 sliding-window layers compute only the 512-key window, using a fused block-sparse kernel. |
| `train.manual_gc` (every 100 steps) | Runs Python garbage collection on a fixed schedule so it doesn't cause random stalls. |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | Makes the GPU run work in the order it was queued, so compute and communication overlap predictably. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Lets PyTorch grow memory blocks instead of fragmenting them, which reduces out-of-memory errors. |
| `TORCH_NCCL_AVOID_RECORD_STREAMS=1` | Frees communication buffers sooner. Already the default in this PyTorch, which prints a deprecation warning for it. |
| `TORCH_NCCL_HIGH_PRIORITY=1` | Runs GPU-to-GPU communication on high-priority streams so it isn't delayed by compute. |

### Not enabled

| Flag | What it does |
|---|---|
| `ddp.check_for_nan_in_grad` | Would check gradients for NaN on every step. Off for speed; turn it back on for real training. |
| `ddp.overlap_grad_reduce` | Sends gradients between GPUs during the backward pass. Measured 3% slower in the reference run. Also crashes unless `GEMMA4_FREEZE_UNUSED=1` (see Troubleshooting). |
| `ddp.overlap_param_gather` | Collects updated weights from other GPUs while the next forward pass starts. Off together with `overlap_grad_reduce`. |
| Fused attention for the 7 global layers | The global layers use head_dim 512, which no fast attention kernel in this container supports. They keep the plain attention, which stores a full 4096×4096 matrix per head. |
| `transformer_impl=transformer_engine` | Uses NVIDIA's TE-optimized layers (fused norm + linear, faster matrix multiplies). Not available for E4B today. Needs TE. |
| FP8 / MXFP8 (`mixed_precision=bf16_with_fp8_...`) | Matrix multiplies in 8-bit, about 2× BF16 throughput on B300. Needs TE. |
| CUDA graphs (`cuda_graph_impl`) | Records a whole step once and replays it, cutting CPU launch overhead. Needs TE. |
| `masked_softmax_fusion` | Fuses the attention mask and softmax into one kernel. The recipe turns it off for the plain layers. |
| `bias_activation_fusion` | Fuses the GeLU activation with the MLP's gating multiply. Megatron's fused kernel doesn't cover Gemma's tanh-GeLU. |
| `persist_layer_norm` | Apex LayerNorm kernel. Not used by Gemma 4's own RMSNorm class; the compiled RMSNorm covers it. |
| `bias_dropout_fusion` | Fuses bias + dropout + residual add. Little effect here: E4B has no bias and dropout is 0. |
| `attention_softmax_in_fp32` | Runs softmax in FP32 for accuracy. Off is faster. |
| `NCCL_NVLS_ENABLE` (set to 0 by the recipe) | NVLink SHARP: does part of the gradient all-reduce inside the NVSwitch. |
| `NCCL_GRAPH_REGISTER` (set to 0 by the recipe) | Pre-registers communication buffers with NCCL for faster transfers. |
| `optimizer.use_precision_aware_optimizer` | Keeps optimizer state in lower precision to save memory. |
| Activation recompute (`recompute_granularity`) | Recomputes activations in the backward pass instead of storing them: saves memory, costs about 30% extra compute. Off is right while memory allows. |
| Activation / CPU offloading | Moves activations to CPU memory to free GPU memory. Not needed with 268 GiB GPUs. |
| `model.sequence_parallel` | Splits norm and dropout work across tensor-parallel GPUs. Only matters with TP>1. |
| `tp_comm_overlap` | Overlaps tensor-parallel communication with matrix multiplies. Only matters with TP>1; needs TE. |
| Megatron FSDP (`use_megatron_fsdp`) | Fully shards weights, gradients and optimizer state across GPUs. Not needed for a 4B model. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GPUs already in use (index, MiB): ...` | Another process holds those GPUs. Pick free ones with `nvidia-smi` (step 1). |
| `Read-only file system: '/opt/Megatron-Bridge/...'` | A recipe output path points inside the container. Pass it as an absolute path under `runs/`, the way `run.sh` sets `logger.tensorboard_dir`. |
| Very slow steps (about 3–11 s) and low GPU utilization | Under `srun`, the job has only 2 CPU threads; add `--cpus-per-task`. Check with `grep Cpus_allowed_list /proc/self/status`. Otherwise check `nvidia-smi` for other processes on your GPUs. |
| `global batch size ... not divisible` | Make `GBS` a multiple of `MBS × GPUs / TP`. |
| CUDA out of memory | Use `MBS=1`. |
| `AssertionError` at `assert len(self.per_param_grad_ready_counts) == len(self.params)` after step 1 | Happens only with `ddp.overlap_grad_reduce=true`. The `k_layernorm` weights in the 18 KV-shared layers never get a gradient, because their K is discarded. Keep overlap off, or set `GEMMA4_FREEZE_UNUSED=1` to freeze those weights. |
| A `[gemma4-fast]` patch fails or you suspect it | Turn it off with `GEMMA4_FAST_NORM=0`, `GEMMA4_FAST_SOFTCAP=0` or `GEMMA4_FAST_ATTN=0`. |
| No `flex path in use` line | The sliding layers fell back to the plain attention, for example because a mask was passed. Expect slower steps. |
| `no 'iteration ...' lines found` | Training failed before it started logging; check `train.log` for the error. |
