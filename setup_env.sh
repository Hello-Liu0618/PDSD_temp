#!/usr/bin/env bash
# 一键配置 conda 环境（云端/新机器）。
#
# 用法：
#   bash setup_env.sh                        # 默认建 rp-opsd（含 vllm）
#   WITH_VLLM=0 bash setup_env.sh            # 不装 vllm（生成会慢一个数量级）
#   WITH_FLASH_ATTN=1 bash setup_env.sh      # 额外装 flash-attn（编译很慢，默认不装）
#   ENV_NAME=pdsd TORCH_CUDA=cu121 bash setup_env.sh
#
# 完成后：
#   conda activate rp-opsd
#   export HF_ENDPOINT=https://hf-mirror.com   # 国内加速（可写进 ~/.bashrc）
set -euo pipefail

ENV_NAME=${ENV_NAME:-rp-opsd}
PY_VER=${PY_VER:-3.10}
TORCH_CUDA=${TORCH_CUDA:-cu128}          # cu128 / cu121 / cu124 / cpu
TORCH_VER=${TORCH_VER:-2.8.0}
TV_VER=${TV_VER:-0.23.0}
WITH_VLLM=${WITH_VLLM:-1}                # 1=装 vllm（生成提速一个数量级）
WITH_FLASH_ATTN=${WITH_FLASH_ATTN:-0}    # flash-attn 需编译，可能耗时很久；不装则自动用 sdpa
HERE="$(cd "$(dirname "$0")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "找不到 conda。请先装 miniconda：https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[skip] 环境 $ENV_NAME 已存在，直接复用"
else
  echo "[1/4] 创建环境 $ENV_NAME (python $PY_VER)"
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

conda activate "$ENV_NAME"

echo "[2/4] 安装 torch（index: https://download.pytorch.org/whl/${TORCH_CUDA}）"
if [ "$TORCH_CUDA" = "cpu" ]; then
  pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
      --index-url https://download.pytorch.org/whl/cpu
else
  pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
      --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

echo "[3/4] 安装其余依赖"
pip install --upgrade pip
pip install -r "${HERE}/requirements.txt"

if [ "$WITH_VLLM" = "1" ]; then
  echo "[3b] 安装 vLLM（批量生成；若它与上面钉死的 torch 版本冲突，pip 会调整 torch，"
  echo "     装完请核对下面的自检输出，必要时重装 torch）"
  pip install vllm || echo "[警告] vllm 安装失败；训练时会自动回退到 HF generate（慢一个数量级）"
fi

if [ "$WITH_FLASH_ATTN" = "1" ]; then
  echo "[3c] 安装 flash-attn（编译耗时较长）"
  pip install flash-attn --no-build-isolation || \
    echo "[警告] flash-attn 安装失败；会自动退回 sdpa"
fi

echo
echo "==== 自检 ===="
python - <<'PY'
import torch, transformers, trl, peft, datasets, accelerate
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.is_available())
print("transformers", transformers.__version__, "| trl", trl.__version__,
      "| peft", peft.__version__, "| datasets", datasets.__version__,
      "| accelerate", accelerate.__version__)
try:
    import vllm
    print("vllm", vllm.__version__, "(生成加速可用)")
except Exception as e:
    print("vllm 不可用 ->", type(e).__name__, "(训练会回退 HF generate，慢一个数量级)")
try:
    import flash_attn
    print("flash-attn", flash_attn.__version__)
except Exception:
    print("flash-attn 不可用 -> 用 sdpa（可接受）")
PY
echo
echo "完成。别忘了： export HF_ENDPOINT=https://hf-mirror.com"
