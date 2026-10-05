#!/usr/bin/env bash
# PDSD 一条龙：生成数据 → 校验过滤/修复 → 分层划分 → 【base 基线标注】 → 双臂训练 → 训练后评测
#
# 用法（数据已备好，直接训练+标注）：
#   PY=python SKIP_PREP=1 MAXCOMP=2048 bash run_pipeline.sh
# 用法（从零开始）：
#   PY=python N_TOTAL=1000 MAXCOMP=2048 bash run_pipeline.sh
#
# 常用覆盖：
#   DATA=...                数据路径（默认 data/generated_1000.jsonl）
#   SKIP_PREP=1             跳过 [1-3]（生成/过滤/划分），直接用已有 clean_merged 数据
#   SKIP_TRAIN=1            只跑数据与评测，不训练
#   SKIP_EVAL=1             不跑训练后评测
#   USE_VLLM=0              关 vLLM（不推荐：生成会慢一个数量级）
#   REPAIR=0                不做被标记条目的重解修复
#   EXTRA_TRAIN_ARGS="..."  追加传给 train_pdsd.py
#   LIMIT=200               只训练前 N 条（快速验证）
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
LIMIT=${LIMIT:-0}
PIVOT_RHO=${PIVOT_RHO:-0.45}   # 解析不到 RP-OPSD 实测 ρ 时的回退值
REPAIR=${REPAIR:-1}            # 1=对被校验标为不一致的条目尝试重解修复后并入
USE_VLLM=${USE_VLLM:-1}        # 1=vLLM 批量生成；未装 vllm 会自动回退并告警
VLLM_ARGS=""
[ "$USE_VLLM" = "1" ] && VLLM_ARGS="--use-vllm"
LIMIT_ARGS=""
[ "$LIMIT" != "0" ] && LIMIT_ARGS="--limit $LIMIT"
LABEL_LIMIT=${LABEL_LIMIT:-0}   # 标注只用测试集前 N 条（0=全部）；快速自测用
LABEL_LIMIT_ARGS=""
[ "$LABEL_LIMIT" != "0" ] && LABEL_LIMIT_ARGS="--limit $LABEL_LIMIT"
SKIP_PREP=${SKIP_PREP:-0}
SKIP_GEN=${SKIP_GEN:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ---------- [1-3] 数据准备 ----------
if [ "$SKIP_PREP" = "1" ]; then
  echo "=== [1-3/6] 跳过数据准备（SKIP_PREP=1）==="
  WORK="${DATA%.jsonl}.clean_merged.jsonl"
  [ -f "$WORK" ] || { echo "缺少 $WORK —— SKIP_PREP=1 要求它已存在" >&2; exit 1; }
else
  echo "=== [1/6] 数据生成 ==="
  if [ "$SKIP_GEN" = "1" ] || [ -f "$DATA" ]; then
    echo "[skip] $DATA 已存在或 SKIP_GEN=1"
  else
    : "${DEEPSEEK_API_KEY:?需要 DEEPSEEK_API_KEY（或先备好数据并设 SKIP_PREP=1）}"
    "$PY" generate_data_deepseek.py --n-total "$N_TOTAL" --batch-size "$BATCH_SIZE" \
        --verify-model deepseek-reasoner --output "$DATA"
  fi

  echo "=== [2/6] 校验过滤 + 修复 ==="
  "$PY" filter_verified.py --data "$DATA"
  CLEAN=${DATA%.jsonl}.clean.jsonl
  [ -f "$CLEAN" ] || { echo "过滤失败：未生成 $CLEAN" >&2; exit 1; }
  WORK="$CLEAN"
  if [ "$REPAIR" = "1" ] && [ -f "${DATA%.jsonl}.flagged.jsonl" ] && [ -n "${DEEPSEEK_API_KEY:-}" ]; then
    "$PY" repair_flagged.py --data "$DATA"
    if [ -f "${DATA%.jsonl}.repaired.jsonl" ]; then
      cat "$CLEAN" "${DATA%.jsonl}.repaired.jsonl" > "${DATA%.jsonl}.clean_merged.jsonl"
      WORK="${DATA%.jsonl}.clean_merged.jsonl"
      echo ">>> 已并入修复条目 -> $WORK"
    fi
  elif [ -f "${DATA%.jsonl}.flagged.jsonl" ]; then
    echo "[skip] 未设置 DEEPSEEK_API_KEY 或 REPAIR=0，跳过修复"
  fi

  echo "=== [3/6] 分层划分 train/test ==="
  "$PY" make_split.py --data "$WORK" --n-test-per-class "$N_TEST_PER_CLASS"
fi

TRAIN=${WORK%.jsonl}.train.jsonl
TEST=${WORK%.jsonl}.test.jsonl
[ -f "$TRAIN" ] || { echo "缺少 $TRAIN" >&2; exit 1; }
[ -f "$TEST" ]  || { echo "缺少 $TEST"  >&2; exit 1; }
mkdir -p outputs

# ---------- [4] base 基线标注（放在训练前：先知道数据难不难） ----------
echo "=== [4/6] base 基线标注（测试集，同协议 n=3 temp=1.1 top_k=20）==="
"$PY" label_difficulty.py --data "$TEST" --n-samples 3 --max-new-tokens "$MAXCOMP" \
    --out outputs/eval_base.jsonl $LABEL_LIMIT_ARGS

# ---------- [5] 双臂训练（先 RP-OPSD 取 ρ，再让 PDSD 对齐） ----------
echo "=== [5/6] 双臂在线蒸馏 ==="
if [ "$SKIP_TRAIN" = "1" ]; then
  echo "[skip] SKIP_TRAIN=1"
else
  echo "--- arm=rpopsd（决定监督预算 ρ）---"
  "$PY" train_pdsd.py --data "$TRAIN" --arm rpopsd \
      --output-dir "outputs/rpopsd" --epochs "$EPOCHS" \
      --max-completion-length "$MAXCOMP" $VLLM_ARGS $LIMIT_ARGS $EXTRA_TRAIN_ARGS \
      2>&1 | tee outputs/rpopsd.log

  # 取 rp_gate_mean（= mean(gate)，即监督预算）**稳态段**均值作为 ρ：
  # 跳过 warmup+transition（默认各 5%），否则那段 gate≈1 会把 ρ 高估。
  RHO=$(grep -o "rp_gate_mean': [0-9.]*" outputs/rpopsd.log \
        | grep -o "[0-9.]*$" \
        | awk '{v[n++]=$1} END {if (n==0) exit; s=int(n*0.10); t=0; c=0; for(i=s;i<n;i++){t+=v[i];c++} if(c>0) printf "%.4f", t/c}' || true)
  RHO=${RHO:-$PIVOT_RHO}
  echo ">>> RP-OPSD 实测 ρ = $RHO，PDSD 将按此对齐监督预算"

  echo "--- arm=pdsd（--pivot-rho $RHO）---"
  "$PY" train_pdsd.py --data "$TRAIN" --arm pdsd \
      --output-dir "outputs/pdsd" --epochs "$EPOCHS" \
      --max-completion-length "$MAXCOMP" --pivot-rho "$RHO" \
      $VLLM_ARGS $LIMIT_ARGS $EXTRA_TRAIN_ARGS 2>&1 | tee outputs/pdsd.log
fi

# ---------- [6] 训练后评测 + 对比 ----------
echo "=== [6/6] 训练后评测（pdsd / rpopsd）==="
if [ "$SKIP_EVAL" = "1" ] || [ "$SKIP_TRAIN" = "1" ]; then
  echo "[skip] SKIP_EVAL=1 或 SKIP_TRAIN=1"
else
  for ARM in pdsd rpopsd; do
    [ -d "outputs/$ARM" ] || { echo "[skip] outputs/$ARM 不存在"; continue; }
    "$PY" label_difficulty.py --data "$TEST" --n-samples 3 --max-new-tokens "$MAXCOMP" \
        --adapter "outputs/$ARM" --out "outputs/eval_$ARM.jsonl" $LABEL_LIMIT_ARGS
  done
  "$PY" summarize_eval.py \
      base=outputs/eval_base.jsonl pdsd=outputs/eval_pdsd.jsonl rpopsd=outputs/eval_rpopsd.jsonl \
      || true
fi

echo "=== 全部完成 ==="
