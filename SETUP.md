# 上传 GitHub → 云端快速配置

## 0. 一次性准备（本地）

`RP-OPSD/RP-OPSD/` 自带 `.git`（嵌套仓库），会让外层 git 只记录一个 gitlink、**代码传不上去**。
先把它移出项目目录（可逆；上游 `github.com:NJUNLP/RP-OPSD` 仍有完整历史）：

```bash
mkdir -p ~/backup
mv RP-OPSD/RP-OPSD/.git ~/backup/RP-OPSD.git.bak     # 备份而非删除
```

然后初始化并推送：

```bash
cd PDSD
git init
git add .
git commit -m "PDSD: 激活突变枢轴探针 + PDSD/RP-OPSD 在线蒸馏"
git branch -M main
git remote add origin git@github.com:<你的账号>/<仓库名>.git
git push -u origin main
```

`.gitignore` 已排除 `outputs/`(299M)、大件生成数据、`*:Zone.Identifier` 等。仓库约 **25M**（含 21M 的 `RP-OPSD.pdf`）；想更精简可在 `.gitignore` 里加 `RP-OPSD/RP-OPSD.pdf`。

## 1. 云端配置环境

```bash
git clone <仓库地址> && cd PDSD
bash setup_env.sh                     # 创建 conda 环境 rp-opsd 并装依赖
conda activate rp-opsd
export HF_ENDPOINT=https://hf-mirror.com     # 国内加速；建议写进 ~/.bashrc
```

`setup_env.sh` 会按 `TORCH_CUDA`（默认 cu128）装 torch，再装 `requirements.txt`，最后自检打印版本与 GPU 可用性。其他 CUDA 版本：

```bash
TORCH_CUDA=cu121 bash setup_env.sh
```

## 2. 跑一条龙

```bash
export DEEPSEEK_API_KEY=sk-xxx                # 数据生成用
PY=python N_TOTAL=1000 EPOCHS=3 MAXCOMP=4096 bash run_pipeline.sh
```

流程（5 段）：**生成数据 → 校验过滤+修复 → 分层划分 → 先训 RP-OPSD（取 ρ）→ 再训 PDSD（按 ρ 对齐）→ 测试集评测 base/pdsd/rpopsd**。
数据已存在会自动跳过生成；各阶段可用 `SKIP_GEN=1 / SKIP_TRAIN=1 / SKIP_EVAL=1` 单独跳过；`REPAIR=0` 关闭对不一致条目的重解修复。

## 3. 显存选择（关键）

瓶颈是 KL 损失物化的 `[1, G, 151936]` 张量（G=完成长度）：

| G | 无量化 | +4bit(QLoRA) | 建议 |
|---|---|---|---|
| 1024 | ~6.2 GB | ~3.9 GB | 8GB 可用 |
| 2048 | ~9.0 GB | ~6.7 GB | 12GB / 8GB+4bit |
| 4096 | ~14.6 GB | ~12.3 GB | **24GB**（与已有难度标签口径一致） |

- `--load-in-4bit` 需先 `pip install bitsandbytes`（默认未装）。
- `--gradient-checkpointing` 只省模型激活，省不了 KL 大头。

## 4. 已知环境坑

- **deepspeed**：若机器装了 deepspeed 但缺 CUDA toolkit，`import deepspeed` 会崩并连带拖垮 Trainer 初始化。
  `train_pdsd.py` 已在顶部把 `accelerate.utils.other.is_deepspeed_available` 置为 `False` 绕过。
- **HF**：国内 `huggingface.co` 直连不通，需 `HF_ENDPOINT=https://hf-mirror.com`（须在 `import transformers` 之前设置）。
- **口径一致性**：`--max-completion-length` 必须与难度标注的 `max_new_tokens` 相同，否则难度不可比。
  详见 `TRAINER_SPEC.md` §9.1。

## 5. 目录速览

| 文件 | 作用 |
|---|---|
| `generate_data_deepseek.py` | 生成数学数据（题材格子 + 去重 + judge 校验） |
| `filter_verified.py` | 按 judge 结果切 clean/flagged（正确性过滤，训练与测试都做） |
| `repair_flagged.py` | 对 flagged 条目重解+重判，能救的并入 clean |
| `make_split.py` | 按大类分层划 test/train |
| `pdsd_collator.py` / `pdsd_gate.py` / `pdsd_trainer.py` / `train_pdsd.py` | 蒸馏（PDSD / RP-OPSD 双臂） |
| `label_difficulty.py` | 难度标注 / 评测（`--adapter` 评训练后模型） |
| `run_pipeline.sh` | 一条龙 |
| `TRAINER_SPEC.md` | 训练方案规范（含 ρ 对齐、口径一致性） |
| `pivot/`、`pivot_probe.py`、`sweep_*.py` | 方法一（激活突变探针）本体与扫参 |
