#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

PYTHON=${PYTHON:-python}
CACHE_DIR=${CACHE_DIR:-"${ROOT_DIR}/.cache/reflect-r1"}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT_DIR}/outputs/reflect-r1"}
CLIP_PORT=${CLIP_PORT:-52000}
REPORT_TO=${REPORT_TO:-none}
LEARNING_RATE=${LEARNING_RATE:-1e-6}

export SIGLIP_URL=${SIGLIP_URL:-"grpc://127.0.0.1:${CLIP_PORT}"}
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/clip_as_service/client:${ROOT_DIR}/clip_as_service/server:${PYTHONPATH:-}"
export NO_VERSION_CHECK=${NO_VERSION_CHECK:-1}

mkdir -p "${CACHE_DIR}" "${OUTPUT_DIR}"

require_var() {
  local name=$1
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 1
  fi
}

require_video_roots() {
  SHORT_VIDEO_DIR=${SHORT_VIDEO_DIR:-${VIDEO_R1_DIR:-}}
  LONG_VIDEO_DIR=${LONG_VIDEO_DIR:-${PANDA_DIR:-}}
  VIDEO_R1_DIR=${VIDEO_R1_DIR:-${SHORT_VIDEO_DIR:-}}
  PANDA_DIR=${PANDA_DIR:-${LONG_VIDEO_DIR:-}}
  require_var SHORT_VIDEO_DIR
  require_var LONG_VIDEO_DIR
  export SHORT_VIDEO_DIR LONG_VIDEO_DIR VIDEO_R1_DIR PANDA_DIR
}

localize_data() {
  local input=$1
  local output=$2
  require_video_roots
  "${PYTHON}" "${ROOT_DIR}/scripts/open_source/prepare_reflect_r1_data.py" localize \
    --input "${input}" \
    --output "${output}" \
    --short-dir "${SHORT_VIDEO_DIR}" \
    --long-dir "${LONG_VIDEO_DIR}" \
    --video-r1-dir "${VIDEO_R1_DIR}" \
    --panda-dir "${PANDA_DIR}" >&2
}

build_sft_python_command() {
  require_var MODEL_NAME_OR_PATH
  require_var LOCAL_SFT_JSONL

  local max_steps_args=()
  if [[ -n "${MAX_STEPS:-}" ]]; then
    max_steps_args=(--max_steps "${MAX_STEPS}")
  fi

  SFT_PYTHON_CMD=(
    "${PYTHON}"
    "${ROOT_DIR}/reflect_r1/sft_video_r1.py"
    --output_dir "${OUTPUT_DIR}"
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --dataset_name "${LOCAL_SFT_JSONL}"
    --deepspeed "${DEEPSPEED_CONFIG:-${ROOT_DIR}/scripts/zero3_offload_sft.json}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}"
    --learning_rate "${LEARNING_RATE}"
    --logging_steps "${LOGGING_STEPS:-1}"
    --bf16
    --report_to "${REPORT_TO}"
    --gradient_checkpointing true
    --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-4}"
    --run_name "${RUN_NAME:-reflect-r1-cold-start-sft}"
    --save_steps "${SAVE_STEPS:-1000}"
    --max_grad_norm "${MAX_GRAD_NORM:-5}"
    --save_only_model "${SAVE_ONLY_MODEL:-false}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-1}"
    --ddp_timeout "${DDP_TIMEOUT:-1800}"
    --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT:-false}"
    "${max_steps_args[@]}"
  )
}

prepare_grpo_dataset_yaml() {
  GRPO_SHORT_JSON=${GRPO_SHORT_JSON:-${GRPO_VIDEO_R1_JSON:-}}
  GRPO_LONG_JSON=${GRPO_LONG_JSON:-${GRPO_PANDA_JSON:-}}
  require_var GRPO_SHORT_JSON
  require_var GRPO_LONG_JSON
  local short_json="${CACHE_DIR}/grpo_short_local.json"
  local long_json="${CACHE_DIR}/grpo_long_local.json"
  local dataset_yaml="${CACHE_DIR}/grpo_dataset.yaml"

  localize_data "${GRPO_SHORT_JSON}" "${short_json}"
  localize_data "${GRPO_LONG_JSON}" "${long_json}"

  cat > "${dataset_yaml}" <<YAML
datasets:
  - json_path: ${short_json}
    sampling_strategy: ${GRPO_SHORT_SAMPLING:-${GRPO_VIDEO_R1_SAMPLING:-random:1000}}
  - json_path: ${long_json}
    sampling_strategy: ${GRPO_LONG_SAMPLING:-${GRPO_PANDA_SAMPLING:-random:8000}}
YAML
  echo "${dataset_yaml}"
}

build_torchrun_command() {
  TORCHRUN_CMD=(
    "${PYTHON}" -m torch.distributed.run
    --nproc_per_node="${NPROC_PER_NODE:-1}"
    --nnodes="${NNODES:-1}"
    --node_rank="${NODE_RANK:-0}"
    --master_addr="${MASTER_ADDR:-127.0.0.1}"
    --master_port="${MASTER_PORT:-12349}"
  )
}

run_grpo_entry() {
  local reward_func=$1
  shift
  require_var MODEL_NAME_OR_PATH
  local dataset_yaml
  dataset_yaml=$(prepare_grpo_dataset_yaml)

  local max_steps_args=()
  if [[ -n "${MAX_STEPS:-}" ]]; then
    max_steps_args=(--max_steps "${MAX_STEPS}")
  fi

  build_torchrun_command
  local cmd=(
    "${TORCHRUN_CMD[@]}"
    "${ROOT_DIR}/reflect_r1/train_VLLM_stage_1_split.py"
    --deepspeed "${DEEPSPEED_CONFIG:-${ROOT_DIR}/scripts/zero3_offload.json}"
    --output_dir "${OUTPUT_DIR}"
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --train_data_path "${dataset_yaml}"
    --video_folder "${SHORT_VIDEO_DIR}"
    --reward_func "${reward_func}"
    --prompt_template "${PROMPT_TEMPLATE:-v3}"
    --tool_name_list seek_video_frames
    --max_interaction_turns "${MAX_INTERACTION_TURNS:-4}"
    --max_prompt_length "${MAX_PROMPT_LENGTH:-24000}"
    --max_completion_length "${MAX_COMPLETION_LENGTH:-16000}"
    --max_completion_length_per_turn "${MAX_COMPLETION_LENGTH_PER_TURN:-512}"
    --total_video_tokens "${TOTAL_VIDEO_TOKENS:-10240}"
    --max_frames "${MAX_FRAMES:-734}"
    --min_per_frame_tokens "${MIN_PER_FRAME_TOKENS:-4}"
    --max_per_frame_tokens "${MAX_PER_FRAME_TOKENS:-192}"
    --num_generations "${NUM_GENERATIONS:-8}"
    --scale_rewards "${SCALE_REWARDS:-false}"
    --beta "${BETA:-0.005}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}"
    --steps_per_generation "${STEPS_PER_GENERATION:-1}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-1}"
    --dataloader_pin_memory "${DATALOADER_PIN_MEMORY:-true}"
    --logging_steps "${LOGGING_STEPS:-1}"
    --bf16
    --torch_dtype "${TORCH_DTYPE:-bfloat16}"
    --data_seed "${DATA_SEED:-42}"
    --gradient_checkpointing true
    --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
    --run_name "${RUN_NAME:-reflect-r1-grpo}"
    --report_to "${REPORT_TO}"
    --save_steps "${SAVE_STEPS:-200}"
    --save_only_model "${SAVE_ONLY_MODEL:-true}"
    --use_vllm "${USE_VLLM:-true}"
    --vllm_mode "${VLLM_MODE:-colocate}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.3}"
    --shuffle_dataset "${SHUFFLE_DATASET:-true}"
    --replay_buffer_type "${REPLAY_BUFFER_TYPE:-dapo}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
    --log_completions "${LOG_COMPLETIONS:-true}"
    --learning_rate "${LEARNING_RATE}"
    --use_counterfactual_reasoning "${USE_COUNTERFACTUAL_REASONING:-true}"
    "${max_steps_args[@]}"
    "$@"
  )
  "${cmd[@]}"
}
