#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

require_var MODEL_NAME_OR_PATH
require_var SFT_DATA_JSONL
require_video_roots

export LOCAL_SFT_JSONL="${CACHE_DIR}/sft_local.jsonl"
localize_data "${SFT_DATA_JSONL}" "${LOCAL_SFT_JSONL}"

HOSTFILE=${HOSTFILE:-}
if [[ -f /etc/mpi/hostfile ]]; then
  if [[ ! -f /etc/mpi/hostfile_seq && -z "${HOSTFILE}" ]]; then
    echo "Please provide HOSTFILE or generate /etc/mpi/hostfile_seq with your launcher." >&2
    exit 1
  fi
  HOSTFILE=${HOSTFILE:-/etc/mpi/hostfile_seq}
fi

if [[ -n "${HOSTFILE}" ]]; then
  if [[ ! -f "${HOSTFILE}" ]]; then
    echo "Hostfile does not exist: ${HOSTFILE}" >&2
    exit 1
  fi
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    if [[ -n "${MY_NODE_IP:-}" ]]; then
      MASTER_ADDR=${MY_NODE_IP}
    else
      echo "Set MASTER_ADDR or MY_NODE_IP for multinode SFT." >&2
      exit 1
    fi
  fi
  NP=${NP:-$(awk '/slots=/ {for (i=1; i<=NF; i++) if ($i ~ /^slots=/) {split($i, a, "="); sum += a[2]}} END {print sum+0}' "${HOSTFILE}")}
else
  if [[ "${ALLOW_SINGLE_NODE_DEBUG:-0}" != "1" ]]; then
    echo "Reflect-R1 SFT is expected to run through a multi-node MPI hostfile." >&2
    echo "Set HOSTFILE, or set ALLOW_SINGLE_NODE_DEBUG=1 only for local debugging." >&2
    exit 1
  fi
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  if command -v nvidia-smi >/dev/null 2>&1; then
    NP=${NP:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
  else
    NP=${NP:-1}
  fi
fi

if [[ "${NP}" -lt 1 ]]; then
  echo "NP must be positive; got ${NP}" >&2
  exit 1
fi

export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-12349}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-16}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:16:8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export CUDA_HOME=${CUDA_HOME:-}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export LD_PRELOAD=${LD_PRELOAD:-}
export PYTHON
export MODEL_NAME_OR_PATH
export LOCAL_SFT_JSONL
export OUTPUT_DIR
export DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${ROOT_DIR}/scripts/zero3_offload_sft.json}
export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
export LEARNING_RATE
export LOGGING_STEPS=${LOGGING_STEPS:-1}
export REPORT_TO
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-4}
export RUN_NAME=${RUN_NAME:-reflect-r1-cold-start-sft}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-5}
export SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-false}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-1}
export DDP_TIMEOUT=${DDP_TIMEOUT:-1800}
export RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-false}
export MAX_STEPS=${MAX_STEPS:-}

MPIRUN=${MPIRUN:-mpirun}
SSH_PORT=${SSH_PORT:-$(awk 'tolower($1) == "port" {gsub(/"/, "", $2); port=$2} END {print port}' /etc/ssh/ssh_config 2>/dev/null)}
SSH_PORT=${SSH_PORT:-22}

WORKER="${ROOT_DIR}/scripts/open_source/_sft_mpi_worker.sh"
LOG_FILE=${LOG_FILE:-"${OUTPUT_DIR}/reflect_r1_cold_start_sft_$(date +%Y%m%d_%H%M%S).log"}

echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}, NP=${NP}, HOSTFILE=${HOSTFILE:-none}"
echo "Log file: ${LOG_FILE}"

if [[ -z "${HOSTFILE}" && "${SFT_SINGLE_NODE_LAUNCHER:-torchrun}" == "torchrun" ]]; then
  build_sft_python_command
  torchrun_cmd=(
    "${PYTHON}" -m torch.distributed.run
    --nproc_per_node="${NP}"
    --nnodes=1
    --node_rank=0
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
  )
  python_args=("${SFT_PYTHON_CMD[@]:1}")
  "${torchrun_cmd[@]}" "${python_args[@]}" "$@" 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

mpirun_cmd=(
  "${MPIRUN}"
  --allow-run-as-root
  -np "${NP}"
  -mca plm_rsh_args "-p ${SSH_PORT}"
)
if [[ -n "${HOSTFILE}" ]]; then
  mpirun_cmd+=(--hostfile "${HOSTFILE}")
fi
mpirun_cmd+=(
  -bind-to none
  -map-by slot
  --mca btl tcp,self
  -x MASTER_ADDR
  -x MASTER_PORT
  -x NCCL_TIMEOUT
  -x NCCL_IB_DISABLE
  -x NCCL_IB_GID_INDEX
  -x NCCL_IB_HCA
  -x NCCL_IB_QPS_PER_CONNECTION
  -x NCCL_IB_TIMEOUT
  -x NCCL_MIN_NCHANNELS
  -x NCCL_DEBUG
  -x CUDA_HOME
  -x PATH
  -x LD_LIBRARY_PATH
  -x LD_PRELOAD
  -x CUBLAS_WORKSPACE_CONFIG
  -x PYTORCH_CUDA_ALLOC_CONF
  -x TORCH_NCCL_AVOID_RECORD_STREAMS
  -x PYTHONPATH
  -x SIGLIP_URL
  -x NO_VERSION_CHECK
  -x PYTHON
  -x MODEL_NAME_OR_PATH
  -x LOCAL_SFT_JSONL
  -x OUTPUT_DIR
  -x DEEPSPEED_CONFIG
  -x PER_DEVICE_TRAIN_BATCH_SIZE
  -x GRADIENT_ACCUMULATION_STEPS
  -x LEARNING_RATE
  -x LOGGING_STEPS
  -x REPORT_TO
  -x ATTN_IMPLEMENTATION
  -x NUM_TRAIN_EPOCHS
  -x RUN_NAME
  -x SAVE_STEPS
  -x MAX_GRAD_NORM
  -x SAVE_ONLY_MODEL
  -x DATALOADER_NUM_WORKERS
  -x DDP_TIMEOUT
  -x RESUME_FROM_CHECKPOINT
  -x MAX_STEPS
  "${WORKER}"
)

"${mpirun_cmd[@]}" "$@" 2>&1 | tee -a "${LOG_FILE}"
