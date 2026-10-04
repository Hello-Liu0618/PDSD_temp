"""集中式配置：探针与数据生成的所有超参数。

设计目标：所有可调超参数集中在此，运行脚本时可用 --xxx 命令行覆盖，
便于扫参实验（改默认值或命令行覆盖即可，无需改核心逻辑）。

用法：
    from config import ProbeConfig, DataGenConfig
    cfg = ProbeConfig()          # 默认值
    cfg.window_k = 7             # 直接改
命令行：python pivot_probe.py --window-k 7 --z-threshold 2.5
"""
from dataclasses import dataclass, asdict

import torch


def _dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class ProbeConfig:
    """探针（激活突变检测）超参数。"""

    # ---- 模型 ----
    model_name: str = "Qwen/Qwen3-1.7B"
    dtype: str = "bf16"                 # bf16 / fp16 / fp32
    trust_remote_code: bool = True

    # ---- 层选择：关注除输入/输出处理外的所有层，[lo, hi] 按 num_layers 的比例 ----
    layer_lo_frac: float = 0.10         # 跳过最底部（输入处理）约 10%
    layer_hi_frac: float = 0.90         # 跳过最顶部（输出处理）约 10%

    # ---- 每层压缩成向量（而非单个 RMS 标量）----
    layer_vec_dim: int = 256             # 每层压缩成的向量维度（1 = 原来的逐层 RMS 标量）
    layer_vec_method: str = "rms_chunk"  # rms_chunk（分块 RMS）/ random_proj（随机投影）
    layer_vec_seed: int = 0             # random_proj 的随机种子

    # ---- 滑动窗口 cosine 距离 ----
    window_k: int = 5                   # t 前后各取 k 个 token 的 profile 均值

    # ---- 尖峰检测 ----
    z_threshold: float = 1            # 轨迹内 z-score 超过该值视为尖峰
    min_peak_sep: int = 3               # 合并相距 <= 该 token 数的尖峰
    shift_floor: float = 1e-4           # 绝对下限：shift 低于此值不算尖峰（滤平坦信号的数值噪声）

    # ---- 尖峰词向后延伸（枢轴 span 扩展）----
    span_extend: bool = True           # 是否把尖峰词向后延伸成一段"枢轴 span"
    span_max_len: int = 10              # 向后搜索停止点的最大长度（字符）
    span_min_len: int = 6               # 若 max 内未遇停止点，向后标记的最小长度（字符）

    raw_span: bool = False              # 只标尖峰 token 本身的字符，关闭一切合并/延伸修饰

    # ---- 输入输出 ----
    data_path: str = "data/gsm8k_cot.jsonl"
    output_dir: str = "outputs"
    n_annotate: int = 3                 # 用【】标注的轨迹条数

    # ---- 展示 ----
    context_n: int = 8                  # 尖峰上下文窗口 token 数

    @property
    def torch_dtype(self) -> torch.dtype:
        return _dtype(self.dtype)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataGenConfig:
    """现场生成英文 CoT 的超参数。"""

    # ---- 模型 ----
    model_name: str = "Qwen/Qwen3-1.7B"
    dtype: str = "bf16"
    trust_remote_code: bool = True

    # ---- 数据源 ----
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"         # GSM8K 有 ['main', 'socratic'] 两个 config
    dataset_split: str = "train"
    hf_endpoint: str = "https://hf-mirror.com"

    # ---- 生成规模 ----
    num_questions: int = 80             # 最多尝试的题目数
    target_correct: int = 40            # 目标保留的正确轨迹数
    start_index: int = 0                # 从数据集的第几条开始

    # ---- 解码 ----
    max_new_tokens: int = 1024          # thinking 模式输出较长，给足以免截断
    do_sample: bool = False             # False = greedy（更稳、正确率更高）
    temperature: float = 0.6
    top_p: float = 0.95
    seed: int = 0

    # ---- 输出 ----
    output_path: str = "data/gsm8k_cot.jsonl"

    @property
    def torch_dtype(self) -> torch.dtype:
        return _dtype(self.dtype)

    def to_dict(self) -> dict:
        return asdict(self)


def build_arg_parser(cfg_cls):
    """为给定 dataclass 生成 argparse，把所有字段变成可覆盖的 --xxx。"""
    import argparse
    p = argparse.ArgumentParser(description=cfg_cls.__name__)
    for name, f in cfg_cls.__dataclass_fields__.items():
        if not f.init:
            continue
        typ = f.type
        if typ is bool:
            typ = _str2bool
        p.add_argument(f"--{name.replace('_', '-')}", default=None, type=typ,
                       help=f"默认 {f.default!r}")
    return p


def apply_cli(cfg, args) -> None:
    """把命令行里显式给出的参数覆盖到 dataclass 实例上（仅覆盖非 None 项）。"""
    for name in cfg.__dataclass_fields__:
        v = getattr(args, name, None)
        if v is not None:
            setattr(cfg, name, v)
