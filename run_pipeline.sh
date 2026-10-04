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
SKIP_GEN=${SKIP_GEN:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

TRAIN=${DATA%.jsonl}.train.jsonl
TEST=${DATA%.jsonl}.test.jsonl

echo "=== [1/4] 数据生成 ==="
if [ "$SKIP_GEN" = "1" ] || [ -f "$DATA" ]; then
  echo "[skip] $DATA 已存在或 SKIP_GEN=1"
else
  : "${DEEPSEEK_API_KEY:?需要 DEEPSEEK_API_KEY（或先备好数据并设 SKIP_GEN=1）}"
  "$PY" generate_data_deepseek.py --n-total "$N_TOTAL" --batch-size "$BATCH_SIZE" \
      --verify-model deepseek-reasoner --output "$DATA"
fi

echo "=== [2/4] 分层划分 train/test ==="
"$PY" make_split.py --data "$DATA" --n-test-per-class "$N_TEST_PER_CLASS"

echo "=== [3/4] 双臂在线蒸馏（先 RP-OPSD 以取得 ρ，再让 PDSD 对齐）==="
if [ "$SKIP_TRAIN" = "1" ]; then
  echo "[skip] SKIP_TRAIN=1"
else
  echo "--- arm=rpopsd（决定监督预算 ρ）---"
  mkdir -p outputs
  "$PY" train_pdsd.py --data "$TRAIN" --arm rpopsd \
      --output-dir "outputs/rpopsd" --epochs "$EPOCHS" \
      --max-completion-length "$MAXCOMP" $EXTRA_TRAIN_ARGS 2>&1 | tee outputs/rpopsd.log

  # 从训练日志里取全程 rp_gate_mean（= mean(gate)，即监督预算）的均值作为 ρ
  RHO=$(grep -o "rp_gate_mean': [0-9.]*" outputs/rpopsd.log \
        | grep -o "[0-9.]*$" \
        | awk '{s+=$1; n++} END {if (n>0) printf "%.4f", s/n}' || true)
  RHO=${RHO:-$PIVOT_RHO}
  echo ">>> RP-OPSD 实测 ρ = $RHO，PDSD 将按此对齐监督预算"

  echo "--- arm=pdsd（--pivot-rho $RHO）---"
  "$PY" train_pdsd.py --data "$TRAIN" --arm pdsd \
      --output-dir "outputs/pdsd" --epochs "$EPOCHS" \
      --max-completion-length "$MAXCOMP" --pivot-rho "$RHO" $EXTRA_TRAIN_ARGS 2>&1 | tee outputs/pdsd.log
fi

echo "=== [4/4] 测试集评测（base / pdsd / rpopsd 同一协议）==="
if [ "$SKIP_EVAL" = "1" ]; then
  echo "[skip] SKIP_EVAL=1"
else
  "$PY" label_difficulty.py --data "$TEST" --n-samples 3 --max-new-tokens "$MAXCOMP" \
      --out outputs/eval_base.jsonl
  for ARM in pdsd rpopsd; do
    "$PY" label_difficulty.py --data "$TEST" --n-samples 3 --max-new-tokens "$MAXCOMP" \
        --adapter "outputs/$ARM" --out "outputs/eval_$ARM.jsonl"
  done
  echo "评测结果：outputs/eval_{base,pdsd,rpopsd}.jsonl"
fi

echo "=== 全部完成 ==="
