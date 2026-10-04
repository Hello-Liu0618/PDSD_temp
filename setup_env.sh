#!/usr/bin/env bash
# 一键配置 conda 环境（云端/新机器）。
#
# 用法：
#   bash setup_env.sh                     # 默认 env=rp-opsd, python=3.10, cuda=cu128
#   ENV_NAME=pdsd TORCH_CUDA=cu121 bash setup_env.sh
#
# 完成后：
#   conda activate rp-opsd
#   export HF_ENDPOINT=https://hf-mirror.com   # 国内加速（可写入 ~/.bashrc）
set -euo pipefail

ENV_NAME=${ENV_NAME:-rp-opsd}
PY_VER=${PY_VER:-3.10}
TORCH_CUDA=${TORCH_CUDA:-cu128}          # cu128 / cu121 / cpu
TORCH_VER=${TORCH_VER:-2.8.0}
TV_VER=${TV_VER:-0.23.0}

if ! command -v conda >/dev/null 2>&1; then
  echo "找不到 conda。请先装 miniconda：https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[skip] 环境 $ENV_NAME 已存在，直接复用"
else
  echo "[1/3] 创建环境 $ENV_NAME (python $PY_VER)"
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

conda activate "$ENV_NAME"

echo "[2/3] 安装 torch（index: https://download.pytorch.org/whl/${TORCH_CUDA}）"
if [ "$TORCH_CUDA" = "cpu" ]; then
  pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
      --index-url https://download.pytorch.org/whl/cpu
else
  pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
      --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

echo "[3/3] 安装其余依赖"
pip install --upgrade pip
pip install -r "$(dirname "$0")/requirements.txt"

echo
echo "==== 自检 ===="
python - <<'PY'
import torch, transformers, trl, peft, datasets, accelerate
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.is_available())
print("transformers", transformers.__version__, "| trl", trl.__version__,
      "| peft", peft.__version__, "| datasets", datasets.__version__,
      "| accelerate", accelerate.__version__)
PY
echo
echo "完成。别忘了： export HF_ENDPOINT=https://hf-mirror.com"
