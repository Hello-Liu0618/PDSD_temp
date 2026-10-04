# 蒸馏 Trainer 规范（PDSD vs RP-OPSD）

> 目的：把 PDSD（激活突变找枢轴）与 RP-OPSD（PRS gate）放在**完全相同的蒸馏管线**下对比，
> 唯一自变量是"枢轴寻找方式"。本文件规定必须共享的变量、在线采样制度、日志字段与评测协议。

## 1. 训练制度：纯在线

- **在线**：每个样本**每次访问时用当前策略重新生成** reference-free rollout（每 epoch 每样本一次）。
  - 不做"开局生成一次、全程复用"的离线缓存。
  - 两臂使用**同一**在线制度。（"每步重生成"与"每 epoch 刷新"在标准 SGD 下等价，因为每样本每 epoch 只访问一次；成本 = 离线 × E。）
- 采样参数（与难度标注、评测**严格一致**）：`temperature=1.1, top_p=0.95, top_k=20`。
- rollout 长度上限 `max_new_tokens=4096`（与难度标注口径一致；口径不同会让难度不可比——已踩过 2048 vs 4096 的坑）。
- 每样本 4 次前向（对齐 RP-OPSD）：`q_plus`(带参考)、`q_minus`(不带参考)、`p_ref`(冻结底座)、`student`；梯度只经 `student`。

## 2. 未闭合样本：全部参与

- **不按 `</think>` 过滤**。未闭合（截断/退化）样本同样进入 loss。
  - 理由：on-policy 的意义就是在模型自己访问的状态上训练；退化="卡在关键推理步反复犹豫"，教师（带参考）分布或许能纠正它。该假设**只有在线时才可能兑现**。
- 只用**长度上限**截断（算力/显存旋钮），不改变"是否参与"。
- 用日志字段**实证检验**该假设（见 §4）。

## 3. 两臂必须共享的清单（控制变量）

数据与 split、学生 prompt、采样参数、长度上限、在线制度、epochs、batch、LR、优化器、LoRA 配置、
未闭合策略、退化检测、评测协议、随机种子。**唯一不同的是枢轴寻找方式。**

## 4. 日志字段（每 step / 每 epoch）

| 字段 | 含义 | 用途 |
|---|---|---|
| `loss` | 训练损失 | 收敛 |
| `closure_rate` | 本批/累计 rollout 关闭 `</think>` 的比例 | 健康度；两臂比较 |
| `rollout_len_mean/p95` | rollout 长度 | 算力、退化 |
| `degenerate_rate` | 退化（复读）比例，见 §5 | 诊断 |
| `loss_closed` / `loss_degenerate` | **按组的 loss** | 验证"退化能否被纠正" |
| `closure_rate_over_time` | 闭合率随 step | 若上升 → 假设成立 |

## 5. 退化检测（仅用于分组统计，**不用于过滤**）

- 定义（建议）：rollout 末尾的 n-gram 重复率超过阈值（如末尾 200 token 内 n=8 的重复率 > 0.5），
  或末尾片段与更早片段的相似度超阈。
- 用途：把样本分成 闭合 / 未闭合-连贯 / 退化，分别记 loss 与闭合率变化。

## 6. 评测协议（训练前 = 训练后，两臂一致）

- 学生 JA prompt、`temp=1.1, top_p=0.95, top_k=20`、`max_new_tokens=4096`、`n=3`，报**均值**。
- 指标：test accuracy（`mean(diff_correct)`）、closure rate、答案抽取成功率。
- **baseline** = base 模型在同一协议下的得分（同一趟采样既做难度标注、也做 baseline）。
- 测试集：`make_split.py` 按大类分层划出（不过滤、固定种子）。baseline 与训练后成绩**都在测试集上**报。

## 7. 复现性

- 固定 seed；记录数据版本（`.meta.json`）、模型版本、代码 commit、采样参数。

## 8. 待定参数（实现前敲定）

- epochs `E`、batch size、LR、LoRA rank/alpha、优化器、warmup。
- 训练 rollout 是否也用 4096（长序列 × 4 前向在 8GB 上显存可能吃紧——需实测）。

## 9. 实施说明（代码已就位，实测记录）

**文件**
- `pdsd_collator.py` —— 日语三 prompt collator（`get_config("JA")`）
- `pdsd_gate.py` —— 向量化激活突变 shift（与 `pivot/shift.py` 数值一致，误差 1e-7）+ 峰检测 + gate（`spans`/`sigmoid` 两形态）
- `pdsd_trainer.py` —— `PDSDTrainer(RPOPSDTrainer)`，只换 gate；默认跳过 q_minus 前向
- `train_pdsd.py` —— 双臂入口（`--arm pdsd|rpopsd`），RP-OPSD 臂直接用**原版** `RPOPSDTrainer`
- `label_difficulty.py` —— 难度标注 / 评测（新增 `--adapter` 评训练后模型）
- `run_pipeline.sh` —— 生成→划分→双臂训练→评测 一条龙

**显存实测（8GB 卡）**：瓶颈在 `_token_kl_with_teacher` 物化的 `[1, G, 151936]` 张量——
G=1024→2.2G、1536→3.3G、2048→4.4G、4096→~10G。加模型权重 3.4G 后：**G≤1024 安全，1536 很紧，2048+ 需 QLoRA（`--load-in-4bit`）**。
分块计算 KL 几乎无用（autograd 仍存全量中间量）。

**环境坑**：本机 deepspeed 已装但 import 即崩（缺 CUDA toolkit）；`accelerate.unwrap_model` 会探测并 import 它 → 在 `train_pdsd.py` 里把 `accelerate.utils.other.is_deepspeed_available` 置为 `False` 绕过。

**口径一致性**：难度标签的 `max_new_tokens` 必须与训练的 `--max-completion-length` 相同；本地 8GB 跑不了 4096，若训练降预算，须在**该预算下重标难度**并在论文/记录里写明操作点。

### 9.1 采样口径一致性（易漏，必须逐次核对）

学生输出的**两个来源**必须同口径：

| | 难度标注/评测（`label_difficulty.py`） | 训练 rollout（`training_step`） |
|---|---|---|
| 模型 | 底座（或底座+`--adapter`） | **当前策略**（底座+LoRA，在线） |
| prompt | `seed_builder.build_student_prompt` | `pdsd_collator` 的 `student_prompts`（**同一套** `get_config("JA")` + `build_assistant_prefilled_prompt`） |
| 参考 | 无 | 无 |

**改协议时须同步三处**（否则口径漂移）：
1. `label_difficulty.py` 的 temp/top_p/top_k/max_new_tokens/n；
2. `train_pdsd.py` 的 `--temperature/--top-p/--top-k/--max-completion-length`；
3. 评测那一趟。

**已踩的坑**：`GOLDConfig.top_k` 默认 **0**（HF 含义 = **关闭** top-k 过滤），`train_pdsd.py` 曾漏传 → 训练生成静默变成不设 top-k。现已传 `top_k`，并在训练前打印 `[gen] rollout 采样: …` 供逐次核对（冒烟显示 `top_k=20`，与协议一致）。

> 注：底座 vs 当前策略不是"不一致"，而是在线蒸馏的定义；n=3（标注）vs n=1/次（训练）统计等价。

### 9.2 监督预算对齐（方案 b，必须开启）

损失里 `ρ = mean(g)` 直接决定"从参考答案学多少"。PDSD 的 `spans` 若用阈值法，ρ 是**涌现**的（实测 ~0.45，且随序列长度漂移），会让"选得准"与"给得多"混淆。

**做法**：PDSD 仍按 shift z **降序**取峰（保留其排序判据），但**只取到 ρ 达标为止** → 与 RP-OPSD 在**相同 ρ** 下比较"选得准不准"。

- 训练参数：`--pivot-rho <目标ρ>`（默认 0.45；设为 RP-OPSD 臂实测的 `rp_gate_mean`；`-1` 关闭回到阈值法）。
- 由 `g = g_min + (1−g_min)·span` 反解目标密度 `(ρ − g_min)/(1 − g_min)`；只在**有效完成段内**统计（右 padding 不计）。
- 日志里 `rp_gate_mean` 就是 ρ，**逐臂核对是否对齐**；`pdsd_gate_density` 是 span 密度。
- 粒度约 ±0.04（每段 span ~7 token）；ρ 过高（>0.75）时候选峰不足会略欠。
- 备选：`--pivot-mode sigmoid` —— 两臂用**同一 sigmoid 公式与超参**，只换输入信号（shift z vs PRS z），参数化完全一致。
