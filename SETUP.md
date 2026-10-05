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

**数据已生成完毕 → 一键跑「基线标注 + 双臂训练 + 增量评测」**：

```bash
PY=python SKIP_PREP=1 MAXCOMP=2048 bash run_pipeline.sh
```

**从零开始（含数据生成）**：

```bash
export DEEPSEEK_API_KEY=sk-xxx
PY=python N_TOTAL=1000 MAXCOMP=2048 bash run_pipeline.sh
```

流程（6 段）：**生成 → 校验过滤+修复 → 分层划分 → 〔base 基线标注〕→ 双臂训练（先 RP-OPSD 取 ρ，再 PDSD 按 ρ 对齐）→ 训练后评测对比**。

**基线标注特意排在训练之前**——先花 ~1 小时（测试集 120 条）确认数据难度可用，再投入几小时训练。

常用开关：

| 变量 | 作用 |
|---|---|
| `SKIP_PREP=1` | 跳过 [1-3]（数据已备好，直接用 `clean_merged`） |
| `SKIP_TRAIN=1` | 只做数据 + 基线标注 |
| `SKIP_EVAL=1` | 不跑训练后评测 |
| `LABEL_LIMIT=N` | 标注只用测试集前 N 条（快速自测） |
| `LIMIT=N` | 只训练前 N 条（快速验证） |
| `REPAIR=0` | 不做被标记条目的重解修复 |
| `USE_VLLM=0` | 关 vLLM（**不推荐**，慢一个数量级） |
| `EXTRA_TRAIN_ARGS="..."` | 追加传给 `train_pdsd.py` |

### 训练超参：已默认对齐原版 RP-OPSD

`train_pdsd.py` 的默认值直接取自原版 `RP-OPSD/scripts/train.sh`：

| 参数 | 值 | 为何重要 |
|---|---|---|
| `max_completion_length` | **2048** | 原文口径；显存/时间都与之线性 |
| `learning_rate` | **5e-6** | ⚠️ **有效性参数**：偏高会让训练发散，两臂比的就成了"谁更抗揍" |
| `max_grad_norm` | **0.1** | ⚠️ KL 梯度天然尖峰，需紧裁剪 |
| 有效 batch | **32** | 梯度噪声 + gate 的 EMA 统计稳定性 |
| `max_steps` | **100** | 训练总量（原文规模） |
| `lmbda` / `beta` | 1 / 0 | GOLD 字段，主路径不读，仅为字面一致 |
| `gradient_checkpointing` | 开 | 默认开（`--no-gradient-checkpointing` 可关） |
| 生成 | **vLLM colocate** | 决定"天 vs 小时" |
| attention | 自动（有 flash-attn 用 FA2，否则 sdpa） | |

改这些默认值等于**偏离基线**，除非有明确理由并两臂同步改。

## 3. 显存选择（关键，实测校正）

瓶颈是 KL 损失里 `F.log_softmax` 对**完整词表**的输出，且会是 **float32**：
`G × 151936 × 4` 字节 —— G=4096 时**每份 2.32 GiB**，而一次要好几份。

| 完成长度 G | 估计峰值 | 建议卡 |
|---|---|---|
| 1024 | ~6–7 GB | 8GB ✅（本地实测可用） |
| 2048 | ~16–18 GB | 24GB ✅ / 16GB 偏紧 |
| 3072 | ~23–25 GB | 32GB（留余量） |
| **4096** | **>32 GB**（**32GB 卡实测 OOM**） | **48GB+** |

- `--load-in-4bit`（QLoRA）约省 2.4 GB 权重，只够降一档，需先 `pip install bitsandbytes`。
- `--gradient-checkpointing` 只省模型激活（KL 大头省不掉），已实测与 `output_hidden_states` 兼容。
- 可加 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片，但**不足以**让 4096 上 32GB。
- **难度标注不受此限**（只做生成，无损失）——4096 的标注在 8GB 上都能跑；受限的只有**训练**。

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
