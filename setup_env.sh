#!/usr/bin/env bash
# 一键配置 conda 环境（云端/新机器）。
#
# 用法：
#   bash setup_env.sh                        # 默认建 rp-opsd（含 vllm）
#   USE_VENV=1 bash setup_env.sh             # conda 镜像不通时的救命选项：改用 python venv
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
USE_VENV=${USE_VENV:-0}                  # 1=用 python venv 代替 conda（conda 镜像不通时的救命选项）
VENV_DIR=${VENV_DIR:-$HOME/venvs/$ENV_NAME}
TORCH_CUDA=${TORCH_CUDA:-cu128}          # cu128 / cu121 / cu124 / cpu
TORCH_VER=${TORCH_VER:-2.9.0}   # 与 vllm 0.11.2 的强制要求一致（否则 pip 会报依赖冲突）
TV_VER=${TV_VER:-0.24.0}
WITH_VLLM=${WITH_VLLM:-1}                # 1=装 vllm（生成提速一个数量级）
VLLM_VER=${VLLM_VER:-0.11.2}             # 必须锁版本：TRL 0.26 只支持 0.10.2/0.11.0/0.11.1/0.11.2
WITH_FLASH_ATTN=${WITH_FLASH_ATTN:-0}    # flash-attn 需编译，可能耗时很久；不装则自动用 sdpa
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "$USE_VENV" = "1" ]; then
  echo "[1/4] 使用 venv：$VENV_DIR"
  if [ -d "$VENV_DIR" ]; then
    echo "[skip] 已存在，直接复用"
  else
    python3 -m venv "$VENV_DIR" || {
      echo "找不到可用的 python3。装一个： apt-get install -y python3 python3-venv" >&2; exit 1; }
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "找不到 conda。可改用： USE_VENV=1 bash setup_env.sh" >&2
    exit 1
  fi
  eval "$(conda shell.bash hook)"

  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[skip] 环境 $ENV_NAME 已存在，直接复用"
  else
    echo "[1/4] 创建环境 $ENV_NAME (python $PY_VER)"
    conda create -y -n "$ENV_NAME" "python=$PY_VER" || {
      echo ""
      echo "conda 创建失败——国内常见原因是 defaults 频道连不上 repo.anaconda.com。"
      echo "两个办法："
      echo "  1) 把 conda 指向清华镜像后重试："
      echo "     conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
      echo "     conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r"
      echo "     conda clean -i -y"
      echo "  2) 不用 conda，改走 venv： USE_VENV=1 bash setup_env.sh"
      exit 1; }
  fi
  conda activate "$ENV_NAME"
fi

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
  # 必须锁版本！TRL 0.26 只支持 vllm 0.10.2 / 0.11.0 / 0.11.1 / 0.11.2；
  # 裸装 `pip install vllm` 会拉到最新版，导致 `import trl` 直接报
  #   ModuleNotFoundError: No module named 'vllm.transformers_utils.tokenizer'
  # 并把 torch 顶到大版本（实测 0.30.0 会把 torch 升到 2.13）。
  echo "[3b] 安装 vLLM==${VLLM_VER}（TRL 支持的版本）"
  pip install "vllm==${VLLM_VER}" \
    || echo "[警告] vllm 安装失败；训练会回退 HF generate（慢一个数量级）"
  # requirements 里的 torch 已钉到 2.9.0（= vllm 0.11.2 的强制要求），二者一致，不会再冲突。
  echo "[3b] 校验版本组合（trl 能 import 才算通过）"
  python -c "import trl, vllm, torch, transformers; print('OK', torch.__version__, vllm.__version__, trl.__version__, transformers.__version__)" \
    || {
      echo "[警告] 与 TRL 不兼容！回退建议："
      echo "         pip uninstall -y vllm && pip install -r requirements.txt"
      echo "         然后用 USE_VLLM=0 训练（慢，但可用）"
    }
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
