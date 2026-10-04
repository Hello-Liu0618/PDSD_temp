# 方法一探针：基于激活突变（activation shift）的推理枢轴检测

> **环境配置 / 云端部署**：见 [`SETUP.md`](SETUP.md)（一条命令 `bash setup_env.sh` 建环境）。
> **蒸馏训练方案**（PDSD vs RP-OPSD 双臂、口径一致性与监督预算对齐）：见 [`TRAINER_SPEC.md`](TRAINER_SPEC.md)。

## 目标

对一段英文推理轨迹（思维链 CoT），以**推理过程中激活分布的变化量**为标尺，筛选出
可能的"推理枢轴"（reasoning pivot）——即模型内部推理状态发生切换的位置。

具体做法：一次 teacher-forcing 前向拿到每层隐状态，把每层 `[T, D]` 压缩成 `[T, vec_dim]`
向量得到 profile，沿层维度 L2 归一化后用滑动窗口 cosine 距离算 `shift[t]`，轨迹内 z-score
找尖峰作为候选枢轴。

与 surprisal（下一 token 的负对数似然）的 Pearson 相关**只是前期排雷指标**：用于确认该
激活突变信号不是 surprisal 的翻版，而非本项目的目的。本探针本身不做任何下游（目标语言、
code-switch、训练等）。

## 环境

- conda 环境 `rp-opsd`（torch 2.8+cu128、transformers、accelerate、datasets、scipy、matplotlib）。
- 模型走 hf-mirror 下载（脚本内已 `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`）。
- GPU：RTX 4060 8GB，默认模型 Qwen3-1.7B（bf16 约 3.4GB，安全）。

## 运行

```bash
# 一步跑完（生成 + 探针）
bash run.sh

# 或分步：
python generate_data.py                 # 现场生成英文数学 CoT -> data/gsm8k_cot.jsonl
python generate_logic_data.py           # 现场生成逻辑演绎 CoT -> data/logic_cot.jsonl（ProofWriter）
python pivot_probe.py                   # 跑主探针 -> outputs/
python sweep_vec_dim.py --vec-dims 1,4,16,64,256   # 扫 vec_dim -> outputs/sweep/
python interactive_probe.py             # 交互式探针：输入文字看【】标注
```

## 文件结构

```
config.py              # 全部超参数（dataclass，命令行 --xxx 可覆盖）
generate_data.py       # GSM8K 数学题 -> Qwen 生成英文 CoT -> 抽答案校验 -> JSONL
generate_logic_data.py # ProofWriter 规则演绎 -> Qwen 生成 CoT -> 校验 True/False/Unknown -> JSONL
pivot_probe.py         # 主探针入口（shift + 尖峰 + surprisal 对照）
sweep_vec_dim.py       # 扫 vec_dim，量化峰值可分性 + 可视化
interactive_probe.py   # 交互式探针（REPL）
pivot/
  activation.py        # per-layer 隐状态压缩成 profile 矩阵（rms_chunk / random_proj）
  shift.py             # 列归一化 + 滑动窗口 cosine 距离
  peaks.py             # 轨迹内 z-score + 连续段峰值
  surprisal.py         # surprisal + Pearson 相关度
  separability.py      # 峰值可分性指标（contrast / prominence / 峰度）
  textmap.py           # token<->文本、子词合并、peak_label、【】标注
  viz.py               # matplotlib 可视化
  loader.py            # 模型/分词器加载
data/                  # 生成的轨迹 JSONL（gsm8k_cot / logic_cot / smoke）
outputs/
  summary.json         # 逐轨迹候选点 + 相关度 + 平均相关度
  annotated.md         # 尖峰最多的 N 条轨迹的【】切换点标注
  plots/               # 逐轨迹图 + 相关度直方图
outputs/sweep/         # vec_dim 扫描（aggregate + 每个 vec_dim 的报告/图/样本）
outputs/sweep_logic/   # 逻辑样本的 vec_dim 扫描
```

## 核心管线（严格按任务规范）

1. 一次前向（teacher-forcing，`output_hidden_states=True`）拿到 `hidden_states[1:]`，
   每层 `[T, D]` 压缩成 `[T, vec_dim]` 的向量 → profile `P[num_layers, vec_dim, T]`
   （先 `.float()` 防 bf16 溢出；`vec_dim=1` 且 `rms_chunk` 时即原来的逐层 RMS 标量）。
2. 关注层取 `[layer_lo_frac*L, layer_hi_frac*L)`（跳过输入/输出处理）；全层留作对照。
3. 把 profile 展平成 `[num_layers*vec_dim, T]`，每列沿该维度 L2 归一化；滑动窗口 k=5：
   `shift[t] = 1 - cos(mean(col[t-k:t]), mean(col[t:t+k]))`。
4. 轨迹内 z-score；`z > z_threshold` 且 `shift > shift_floor` 的连续段取峰值作为候选变点；
   映射回完整单词展示（空白/换行 token 显示为 `<空白>`/`<换行>`，不产生空 `【】`）。
5. surprisal：`-log_softmax(logits[t-1])[token_t]`，与 shift 在有效区间对齐算 Pearson 相关。

## 峰值可分性（判断"峰值是否真的凸出来"）

除 shift–surprisal 相关外，`separability.py` 额外量化候选峰值是否真的从背景里凸出：

| 指标 | 含义 |
|---|---|
| `contrast` | 峰值 shift 中位数 / 背景中位数，≈1 说明只是"平坦里挑高的"，>2~3 才是孤立尖峰 |
| `prominence` | 峰值相对两侧谷底的凸起高度（shift 单位） |
| `excess_kurtosis` | 有效段 shift 峰度（Fisher）：>0 重尾=有离群尖峰，<0 平坦 |

`outputs/sweep/aggregate.md` 汇总了不同 `vec_dim` 下的这些指标。

## 关键超参数（config.py，均可用 --xxx 覆盖）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model-name` | Qwen/Qwen3-1.7B | 分析模型 |
| `--dtype` | bf16 | bf16/fp16/fp32 |
| `--layer-lo-frac` / `--layer-hi-frac` | 0.10 / 0.90 | 关注层区间（跳过输入/输出处理） |
| `--layer-vec-dim` | 256 | 每层压缩成的向量维度（1=原标量） |
| `--layer-vec-method` | rms_chunk | rms_chunk / random_proj |
| `--window-k` | 5 | 滑动窗口半径 |
| `--z-threshold` | 1 | 尖峰阈值（轨迹内 z-score） |
| `--min-peak-sep` | 3 | 峰合并距离 |
| `--shift-floor` | 1e-4 | 绝对下限：shift 低于此值不算尖峰（滤平坦噪声） |
| `--span-extend` / `--span-max-len` / `--span-min-len` | true / 10 / 6 | 枢轴 span 向后延伸 |
| `--target-correct` | 40 | 保留的正确轨迹数（生成脚本） |
| `--do-sample` | false | 生成用 greedy（更稳） |

## 结论判定口径

平均 shift–surprisal Pearson 相关（关注层）：

- `> 0.8` → 基本是 surprisal 的翻版，价值存疑；
- `< 0.5` → 可能是独立信号，值得继续；
- `0.5 ~ 0.8` → 中等相关，需进一步区分。

注意这只是**前期排雷**：即使相关度低，也需结合峰值可分性指标与人工核查样本，判断
候选枢轴是否真的对应推理状态切换。
