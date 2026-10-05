#!/usr/bin/env bash
# PDSD 一条龙：生成数据 → 分层划分 → 双臂训练 → 测试集评测（base / pdsd / rpopsd 同协议）
#
# 用法：
#   PY=/home/orion/miniconda3/envs/rp-opsd/bin/python N_TOTAL=1000 EPOCHS=3 MAXCOMP=2048 ./run_pipeline.sh
# 常用覆盖：
#   DATA=...           生成/使用的数据路径（已存在则跳过生成）
#   EXTRA_TRAIN_ARGS="--load-in-4bit --gradient-checkpointing"
#   SKIP_GEN=1 / SKIP_TRAIN=1 / SKIP_EVAL=1
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
DATA=${DATA:-data/generated_1000.jsonl}
N_TOTAL=${N_TOTAL:-1000}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-3}
MAXCOMP=${MAXCOMP:-2048}
N_TEST_PER_CLASS=${N_TEST_PER_CLASS:-20}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}
PIVOT_RHO=${PIVOT_RHO:-0.45}   # 解析不到 RP-OPSD 实测 ρ 时的回退值
REPAIR=${REPAIR:-1}            # 1=对被校验标为不一致的条目尝试重解修复后并入
USE_VLLM=${USE_VLLM:-1}        # 1=vLLM 批量生成（把生成提速一个数量级）；未装 vllm 会自动回退
VLLM_ARGS=""
[ "$USE_VLLM" = "1" ] && VLLM_ARGS="--use-vllm"
SKIP_GEN=${SKIP_GEN:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

echo "=== [1/5] 数据生成 ==="
if [ "$SKIP_GEN" = "1" ] || [ -f "$DATA" ]; then
  echo "[skip] $DATA 已存在或 SKIP_GEN=1"
else
  : "${DEEPSEEK_API_KEY:?需要 DEEPSEEK_API_KEY（或先备好数据并设 SKIP_GEN=1）}"
  "$PY" generate_data_deepseek.py --n-total "$N_TOTAL" --batch-size "$BATCH_SIZE" \
      --verify-model deepseek-reasoner --output "$DATA"
fi

echo "=== [2/5] 校验过滤 + 修复（剔除/救回参考答案有问题的记录）==="
"$PY" filter_verified.py --data "$DATA"
CLEAN=${DATA%.jsonl}.clean.jsonl
if [ ! -f "$CLEAN" ]; then
  echo "过滤失败：未生成 $CLEAN" >&2
  exit 1
fi
WORK="$CLEAN"
if [ "$REPAIR" = "1" ] && [ -f "${DATA%.jsonl}.flagged.jsonl" ] && [ -n "${DEEPSEEK_API_KEY:-}" ]; then
  "$PY" repair_flagged.py --data "$DATA"
  if [ -f "${DATA%.jsonl}.repaired.jsonl" ]; then
    cat "$CLEAN" "${DATA%.jsonl}.repaired.jsonl" > "${DATA%.jsonl}.clean_merged.jsonl"
    WORK="${DATA%.jsonl}.clean_merged.jsonl"
    echo ">>> 已并入修复条目 -> $WORK"
  fi
elif [ -f "${DATA%.jsonl}.flagged.jsonl" ]; then
  echo "[skip] 未设置 DEEPSEEK_API_KEY 或 REPAIR=0，跳过修复（flagged 已留存，未并入）"
fi
# 训练与测试**都**用 clean 数据（未闭合过滤是训练时动态做的，不在此处）
TRAIN=${WORK%.jsonl}.train.jsonl
TEST=${WORK%.jsonl}.test.jsonl

echo "=== [3/5] 分层划分 train/test ==="
"$PY" make_split.py --data "$WORK" --n-test-per-class "$N_TEST_PER_CLASS"

echo "=== [4/5] 双臂在线蒸馏（先 RP-OPSD 以取得 ρ，再让 PDSD 对齐）==="
if [ "$SKIP_TRAIN" = "1" ]; then
  echo "[skip] SKIP_TRAIN=1"
else
  echo "--- arm=rpopsd（决定监督预算 ρ）---"
  mkdir -p outputs
  "$PY" train_pdsd.py --data "$TRAIN" --arm rpopsd \
      --output-dir "outputs/rpopsd" --epochs "$EPOCHS" \
      --max-completion-length "$MAXCOMP" $VLLM_ARGS $EXTRA_TRAIN_ARGS 2>&1 | tee outputs/rpopsd.log

  # 取 rp_gate_mean（= mean(gate)，即监督预算）**稳态段**的均值作为 ρ：
  # 必须跳过 warmup+transition（默认各占 5%），否则那段 gate≈1 会把 ρ 高估。
  RHO=$(grep -o "rp_gate_mean': [0-9.]*" outputs/rpopsd.log \
        | grep -o "[0-9.]*$" \
        | awk '{v[n++]=$1} END {if (n==0) exit; s=int(n*0.10); t=0; c=0; for(i=s;i<n;i++){t+=v[i];c++} if(c>0) printf "%.4f", t/c}' || true)
  RHO=${RHO:-$PIVOT_RHO}
  echo ">>> RP-OPSD 实测 ρ = $RHO，PDSD 将按此对齐监督预算"

  echo "--- arm=pdsd（--pivot-rho $RHO）---"
  "$PY" train_pdsd.py --data "$TRAIN" --arm pdsd \
      --output-dir "outputs/pdsd" --epochs "$EPOCHS" \
      --max-completion-length "$MAXCOMP" --pivot-rho "$RHO" \
      $VLLM_ARGS $EXTRA_TRAIN_ARGS 2>&1 | tee outputs/pdsd.log
fi

echo "=== [5/5] 测试集评测（base / pdsd / rpopsd 同一协议）==="
if [ "$SKIP_EVAL" = "1" ]; then
  echo "[skip] SKIP_EVAL=1"
else
  mkdir -p outputs
  "$PY" label_difficulty.py --data "$TEST" --n-samples 3 --max-new-tokens "$MAXCOMP" \
      --out outputs/eval_base.jsonl
  for ARM in pdsd rpopsd; do
    "$PY" label_difficulty.py --data "$TEST" --n-samples 3 --max-new-tokens "$MAXCOMP" \
        --adapter "outputs/$ARM" --out "outputs/eval_$ARM.jsonl"
  done
  echo "评测结果：outputs/eval_{base,pdsd,rpopsd}.jsonl"
  "$PY" summarize_eval.py \
      base=outputs/eval_base.jsonl pdsd=outputs/eval_pdsd.jsonl rpopsd=outputs/eval_rpopsd.jsonl
fi

echo "=== 全部完成 ==="
