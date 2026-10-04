#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK="${1:-afrimgsm}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
GPU_IDS="${GPU_IDS:-0}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/.cache/rp-opsd}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/eval}"

case "${BENCHMARK}" in
  afrimgsm)
    EVAL_BENCHMARK="afrimgsm"
    VAL_N="${VAL_N:-12}"
    SPLIT_ARGS=(--afrimgsm_split test)
    ;;
  polymath)
    EVAL_BENCHMARK="local_polymath"
    VAL_N="${VAL_N:-1}"
    SPLIT_ARGS=(--polymath_split all)
    ;;
  *)
    echo "Usage: $0 {afrimgsm|polymath}" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a DEVICES <<< "${GPU_IDS}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${#DEVICES[@]}}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/rp_opsd_swa_${BENCHMARK}.json}"
mkdir -p "${OUTPUT_DIR}" "${CACHE_ROOT}"

export RP_OPSD_CACHE_ROOT="${CACHE_ROOT}"
export OPSD_LOCAL_DATA_ROOT="${ROOT_DIR}/datasets"
export TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

CMD=(
  "${PYTHON_BIN:-python}" "${ROOT_DIR}/src/evaluate.py"
  --benchmark "${EVAL_BENCHMARK}"
  --base_model "${MODEL_NAME}"
  --language SWA
  --val_n "${VAL_N}"
  --seed "${SEED:-0}"
  --temperature "${TEMPERATURE:-1.0}"
  --top_p "${TOP_P:-0.95}"
  --top_k "${TOP_K:--1}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
  --max_model_len "${MAX_MODEL_LEN:-8192}"
  --max_new_tokens "${MAX_NEW_TOKENS:-2048}"
  --output_file "${OUTPUT_FILE}"
  --enable_thinking
  "${SPLIT_ARGS[@]}"
)

[[ -z "${CHECKPOINT_DIR}" ]] || CMD+=(--checkpoint_dir "${CHECKPOINT_DIR}")
[[ -z "${NUM_SAMPLES:-}" ]] || CMD+=(--num_samples "${NUM_SAMPLES}")

NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
"${CMD[@]}" "${@:2}"
