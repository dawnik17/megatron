#!/usr/bin/env bash
# Pre-train Gemma 4 E4B on mock data in the NeMo 26.08 container, then compute MFU from the training log.
#   GPU_IDS=4,5,6,7 train/gemma/megatron_mfu_optim/run.sh <run name> [extra train.py args...]   results in runs/train/gemma4-mfu/<run name>/
#   GPU_IDS=4,5,6,7 train/gemma/megatron_mfu_optim/run.sh baseline
#   GPU_IDS=4,5,6,7 MBS=1 train/gemma/megatron_mfu_optim/run.sh mbs1
#   GPU_IDS=4,5,6,7 GEMMA4_FAST_ATTN=0 train/gemma/megatron_mfu_optim/run.sh no-flex model.apply_rope_fusion=false
set -euo pipefail
here=$(realpath "$(dirname "$0")")
cd "$here/../../.."   # the repo root

name=$1; shift        # everything after the run name goes to train.py
run=$PWD/runs/train/gemma4-mfu/$name
gpus=${GPU_IDS:?set GPU_IDS, e.g. GPU_IDS=4,5,6,7}
ngpu=$(tr ',' '\n' <<< "$gpus" | wc -l)
tp=${TP:-1} seq=${SEQ_LEN:-4096} mbs=${MBS:-2} iters=${TRAIN_ITERS:-100}
gbs=${GBS:-$((mbs * ngpu / tp * 4))}   # 4 micro-batches per data-parallel rank, as in the 8-GPU reference (GBS 64)
port=${MASTER_PORT:-$((29500 + ${gpus%%,*}))}
image=${IMAGE:-pyxis_nemo-26-08}
cache=/raid/$USER/torch-cache          # local disk, so torch.compile and Triton reuse kernels across runs
mkdir -p "$run" "$cache"

# A GPU shared with another process makes the step time, and so the MFU, meaningless.
busy=$(nvidia-smi -i "$gpus" --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 >= 1024')
[[ -z $busy ]] || { echo "GPUs already in use (index, MiB): $busy" >&2; exit 2; }
enroot list | grep -qx "$image" || enroot create -n "$image" /sadata/nikkothari/Flipkart_Nemotron_35_lightning/nemo-26-08.sqsh

echo "$name: GPUs $gpus, TP=$tp, MBS=$mbs, GBS=$gbs, seq $seq, $iters steps -> $run"

# run_recipe.py works from /opt/Megatron-Bridge, which is read-only in the container, so outputs need absolute paths.
enroot start -e NVIDIA_VISIBLE_DEVICES="$gpus" -e HOME=/tmp -e HF_HOME="$PWD/.cache/hf" -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e TORCHINDUCTOR_CACHE_DIR="$cache/inductor" -e TRITON_CACHE_DIR="$cache/triton" \
  -e GEMMA4_FAST_NORM="${GEMMA4_FAST_NORM:-1}" -e GEMMA4_FAST_SOFTCAP="${GEMMA4_FAST_SOFTCAP:-1}" \
  -e GEMMA4_FAST_ATTN="${GEMMA4_FAST_ATTN:-1}" -e GEMMA4_FREEZE_UNUSED="${GEMMA4_FREEZE_UNUSED:-0}" \
  -m "$PWD:$PWD" -m "$cache:$cache" "$image" \
  python -m torch.distributed.run --nproc_per_node "$ngpu" --master_port "$port" "$here/train.py" \
    --recipe gemma4_e4b_pretrain_config --mode pretrain --dataset mock \
    -tp "$tp" -sl "$seq" -mb "$mbs" -gb "$gbs" -ms "$iters" --warmup_iters 10 \
    checkpoint.save=null checkpoint.load=null checkpoint.pretrained_checkpoint=null \
    logger.tensorboard_dir="$run/tb_logs" \
    logger.log_interval=1 logger.log_throughput=true validation.eval_iters=0 \
    ddp.overlap_grad_reduce=false ddp.overlap_param_gather=false \
    ddp.grad_reduce_in_fp32=false ddp.check_for_nan_in_grad=false \
    model.gradient_accumulation_fusion=true model.apply_rope_fusion=true \
    "$@" 2>&1 | tee "$run/train.log"

python3 "$here/compute_mfu.py" --log "$run/train.log" --seq-len "$seq" --gbs "$gbs" --num-gpus "$ngpu" \
  --peak-tflops "${PEAK_TFLOPS:-2250}" --skip-iters 20 | tee "$run/mfu_report.txt"
