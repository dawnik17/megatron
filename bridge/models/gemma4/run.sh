#!/usr/bin/env bash
# Pre-train Gemma 4 E4B in the NeMo 26.08 container, then compute MFU from the training log.
#   GPU_IDS=2,3 bridge/models/gemma4/run.sh <run name> [extra train.py args...]   results in runs/train/gemma4-mfu/<run name>/
#   GPU_IDS=2,3 bridge/models/gemma4/run.sh best
#   GPU_IDS=2,3 MBS=2 bridge/models/gemma4/run.sh mbs2
#   GPU_IDS=2,3 DATA=mock bridge/models/gemma4/run.sh best-mock         # random tokens instead of FineWeb-Edu
#   GPU_IDS=2,3 OOTB=1 bridge/models/gemma4/run.sh ootb                 # Megatron-Bridge's recipe as shipped
#   GPU_IDS=2,3 PROFILE=1 bridge/models/gemma4/run.sh best-profile     # also profile steps 30-31 (PyTorch profiler)
#   GPU_IDS=2,3 PROFILE=nsys bridge/models/gemma4/run.sh best-nsys     # the same steps with Nsight Systems
set -euo pipefail
here=$(realpath "$(dirname "$0")")
cd "$here/../../.."   # the repo root

name=$1; shift        # everything after the run name goes to train.py
run=$(realpath -m "${RUN_DIR:-runs/train/gemma4-mfu/$name}")
gpus=${GPU_IDS:?set GPU_IDS, e.g. GPU_IDS=4,5,6,7}
ngpu=$(tr ',' '\n' <<< "$gpus" | wc -l)
tp=${TP:-1} seq=${SEQ_LEN:-4096} mbs=${MBS:-4} iters=${TRAIN_ITERS:-100}
gbs=${GBS:-$((8 * ngpu / tp))}         # 8 sequences per data-parallel rank per step, as in the 8-GPU reference (GBS 64)
port=${MASTER_PORT:-$((29500 + ${gpus%%,*}))}
image=${IMAGE:-pyxis_nemo-26-08} sqsh=${SQSH:-/sadata/nikkothari/Flipkart_Nemotron_35_lightning/nemo-26-08.sqsh}
# torch.compile and Triton reuse their kernels across runs from this cache: local disk (DGX's /raid) if there is one
user=${USER:-$(id -un)}; cache=$PWD/.cache/torch; [[ -d /raid/$user ]] && cache=/raid/$user/torch-cache
cache=${TORCH_CACHE:-$cache}
mkdir -p "$run" "$cache"

data=${DATA:-fineweb-edu}
if [[ $data == mock ]]; then           # the model takes the tokenizer's vocabulary: the mock one's 32,000 shrinks it
    dataset=(--dataset mock tokenizer.vocab_size=262144)
else                                   # tokenized once by bridge/data/prepare.py
    prefix=.cache/data/$data/google--gemma-4-E4B-it/${data}_text_document   # in the repo: /repo in the container
    [[ -f $prefix.idx ]] || { echo "$PWD/$prefix.idx not found; see bridge/data/prepare.py" >&2; exit 2; }
    dataset=(--dataset megatron-indexed dataset.data_path="/repo/$prefix" dataset.path_to_cache="/repo/${prefix%/*}/index"
    tokenizer.tokenizer_type=HuggingFaceTokenizer tokenizer.tokenizer_model=google/gemma-4-E4B-it)
fi

# Our changes to the recipe. OOTB=1 leaves them out and starts Megatron-Bridge's run_recipe.py instead of train.py,
# so none of its patches apply either. OOTB keeps every recipe setting but one: the recipe overlaps the gradient
# reduce, which fails after the first step on Gemma 4 unless train.py freezes the unused k_layernorm weights.
if [[ ${OOTB:-0} == 1 ]]; then
    what="OOTB recipe" entry=/opt/Megatron-Bridge/scripts/training/run_recipe.py mbs=recipe
    changes=(ddp.overlap_grad_reduce=false)
else
    what="our changes" entry=/repo/bridge/models/gemma4/train.py
    changes=(-mb "$mbs" ddp.overlap_grad_reduce=true ddp.overlap_param_gather=false ddp.check_for_nan_in_grad=false
    model.gradient_accumulation_fusion=true model.apply_rope_fusion=true)
fi
profile=() nsys=() trace=""            # rank 0, steps 30-31: PyTorch profiler (PROFILE=1) or Nsight Systems (PROFILE=nsys)
steps=(profiling.profile_step_start=30 profiling.profile_step_end=32)
case ${PROFILE:-0} in
    1) profile=(profiling.use_pytorch_profiler=true "${steps[@]}") trace=$run/torch_profile/rank-0.json.gz ;;
    nsys) profile=(profiling.use_nsys_profiler=true "${steps[@]}") trace=$run/nsys/trace.sqlite; mkdir -p "$run/nsys"
        nsys=(nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop)
        nsys+=(--export=sqlite --force-overwrite=true -o /out/nsys/trace) ;;
esac
switches=()                            # every GEMMA4_*, NCCL_* and PYTORCH_* variable set here reaches the container
for var in $(compgen -e | grep -E '^(GEMMA4|NCCL|PYTORCH)_' || true); do switches+=(-e "$var=${!var}"); done

# A GPU shared with another process makes the step time, and so the MFU, meaningless.
busy=$(nvidia-smi -i "$gpus" --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 >= 1024')
[[ -z $busy ]] || { echo "GPUs already in use (index, MiB): $busy" >&2; exit 2; }
enroot list | grep -qx "$image" || enroot create -n "$image" "$sqsh"

echo "$name: $what, GPUs $gpus, TP=$tp, MBS=$mbs, GBS=$gbs, seq $seq, $iters steps, data $data," \
    "switches: ${switches[*]:-none} -> $run" | tee "$run/train.log"

# The container sees the repo at /repo, the run's folder at /out and the compile cache at /cache, wherever they are on
# the host. run_recipe.py works from /opt/Megatron-Bridge, which is read-only, so outputs need absolute paths.
enroot start -e NVIDIA_VISIBLE_DEVICES="$gpus" -e HOME=/tmp -e HF_HOME=/repo/.cache/hf -e HF_HUB_OFFLINE=1 \
    -e PYTHONPATH=/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TRITON_CACHE_DIR=/cache/triton "${switches[@]}" \
    -m "$PWD:/repo" -m "$run:/out" -m "$cache:/cache" "$image" \
    "${nsys[@]}" python -m torch.distributed.run --nproc_per_node "$ngpu" --master_port "$port" "$entry" \
    --recipe gemma4_e4b_pretrain_config --mode pretrain "${dataset[@]}" \
    -tp "$tp" -sl "$seq" -gb "$gbs" -ms "$iters" --warmup_iters 10 \
    checkpoint.save=null checkpoint.load=null checkpoint.pretrained_checkpoint=null \
    logger.tensorboard_dir=/out/tb_logs \
    logger.log_interval=1 logger.log_throughput=true validation.eval_iters=0 \
    "${changes[@]}" "${profile[@]}" "$@" 2>&1 | tee -a "$run/train.log"

# bridge/models/mfu.py reads the run's settings from config.yaml next to train.log
printf 'model_family: gemma4\nmodel: google/gemma-4-E4B-it\nseq_len: %s\ngbs: %s\ngpus: "%s"\npeak_tflops: %s\n' \
    "$seq" "$gbs" "$gpus" "${PEAK_TFLOPS:-2250}" > "$run/config.yaml"
python3 "$here/../mfu.py" "$run" | tee "$run/mfu_report.txt"
if [[ -n $trace ]]; then
    python3 "$here/../../profile/breakdown" "$trace" | tee "$run/profile_breakdown.txt"
fi
