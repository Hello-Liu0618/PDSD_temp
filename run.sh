#!/usr/bin/env bash
# 便捷运行：现场生成英文 CoT + 跑探针（全程走 hf-mirror）。
# 想改超参数时，直接对下面的 python 命令加 --xxx（见 config.py）。
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rp-opsd
export HF_ENDPOINT=https://hf-mirror.com

cd "$(dirname "$0")"

echo "==> 1) 现场生成英文 CoT"
python generate_data.py

echo "==> 2) 跑探针"
python pivot_probe.py
