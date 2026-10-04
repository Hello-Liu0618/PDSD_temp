#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/rp-opsd-swa}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/.cache/rp-opsd}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${ROOT_DIR}/configs/accelerate_zero2.yaml}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-32}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.4}"

[[ -n "${DATA_PATH}" ]] || {
  echo "Set DATA_PATH to the 500-example SWA OpenThoughts training JSON." >&2
  exit 2
}
[[ -f "${DATA_PATH}" ]] || {
  echo "Training data not found: ${DATA_PATH}" >&2
  exit 2
}
command -v accelerate >/dev/null || {
  echo "accelerate is not installed; create the environment from environment.yml." >&2
  exit 1
}

IFS=',' read -r -a DEVICES <<< "${CUDA_DEVICES}"
NUM_PROCESSES="${NUM_PROCESSES:-${#DEVICES[@]}}"
if (( EFFECTIVE_BATCH_SIZE % (PER_DEVICE_BATCH_SIZE * NUM_PROCESSES) != 0 )); then
  echo "EFFECTIVE_BATCH_SIZE must be divisible by PER_DEVICE_BATCH_SIZE * NUM_PROCESSES." >&2
  exit 2
fi
GRAD_ACCUMULATION_STEPS="$((EFFECTIVE_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NUM_PROCESSES)))"

mkdir -p "${OUTPUT_DIR}" "${CACHE_ROOT}"
export RP_OPSD_CACHE_ROOT="${CACHE_ROOT}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

cd "${ROOT_DIR}/src"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  --gradient_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  rp_opsd_train.py \
  --train_language SWA \
  --translated_data_path "${DATA_PATH}" \
  --model_name_or_path "${MODEL_NAME}" \
  --learning_rate 5e-6 \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_checkpointing \
  --gradient_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_config rp_opsd_swa_main \
  --max_steps 100 \
  --num_train_epochs 30 \
  --max_completion_length 2048 \
  --save_steps 5 \
  --save_total_limit 2 \
  --logging_steps 2 \
  --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --torch_dtype bfloat16 \
  --max_length 20000 \
  --beta 0 \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm_tensor_parallel_size 1 \
  --use_peft \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature 1.1 \
  --top_p 0.95 \
  --top_k 20 \
  --lmbda 1 \
  --student_enable_thinking \
  --rp_gate_beta 2.0 \
  --rp_gate_tau 0.0 \
  --rp_gate_min 0.05 \
  --rp_score_ema_decay 0.99 \
  --rp_score_z_clip 5.0 \
  --rp_gate_warmup_ratio 0.05 \
  --rp_gate_transition_ratio 0.05 \
  --rp_reference_lambda 0.2 \
  --wandb_project rp-opsd \
  "$@"
