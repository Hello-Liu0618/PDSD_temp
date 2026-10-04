"""激活摘要：把每层隐状态压缩成向量（而非单个 RMS 标量）。

对一次前向得到的 hidden_states 元组（长度 = num_layers + 1，第 0 个是 embedding 输出），
把每层 [T, D] 压缩成 [T, vec_dim] 的向量，得到 profile 张量 P [num_layers, vec_dim, T]。
vec_dim 是可调超参数；vec_dim=1 且 method="rms_chunk" 时退化为原来的逐层 RMS 标量。
"""
from __future__ import annotations

import torch


def compute_layer_profile(hidden_states, vec_dim: int = 1,
                          method: str = "rms_chunk", seed: int = 0) -> torch.Tensor:
    """把每层隐状态压缩成 vec_dim 维向量。

    两种压缩方式（用 method 选择）：
      - "rms_chunk"：把隐藏维度 D 均分成 vec_dim 段，各段算 RMS。完全确定、无随机性，
        vec_dim=1 时即原来的逐层 RMS 标量。
      - "random_proj"：固定种子的高斯随机投影 W[D, vec_dim]（列归一化），
        用随机方向保留几何结构，与神经元排列顺序无关。

    Returns:
        P: [num_layers, vec_dim, T] float32
    """
    hs = list(hidden_states)[1:]                     # 跳过 embedding
    num_layers = len(hs)
    T = hs[0].shape[1]
    D = hs[0].shape[2]
    dev = hs[0].device

    if vec_dim > D:
        raise ValueError(f"layer_vec_dim={vec_dim} 大于隐藏维度 D={D}，无法压缩")

    P = torch.empty((num_layers, vec_dim, T), dtype=torch.float32, device=dev)

    if method == "rms_chunk":
        for l, h in enumerate(hs):
            hf = h[0].float()                        # [T, D]，先 float 再平方防 bf16 溢出
            for j, seg in enumerate(torch.tensor_split(hf, vec_dim, dim=1)):
                P[l, j] = torch.sqrt(torch.mean(seg * seg, dim=-1))
    elif method == "random_proj":
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(D, vec_dim, generator=g, dtype=torch.float32).to(dev)
        W = W / torch.norm(W, dim=0, keepdim=True)   # 列归一化，各分量尺度可比
        for l, h in enumerate(hs):
            P[l] = (h[0].float() @ W).t()            # [vec_dim, T]
    else:
        raise ValueError(f"未知 layer_vec_method: {method!r}")
    return P


def flatten_profile(P: torch.Tensor) -> torch.Tensor:
    """把 [num_layers, vec_dim, T] 展平成 [num_layers*vec_dim, T]，供 shift 计算。"""
    return P.reshape(P.shape[0] * P.shape[1], P.shape[2])


def select_layer_range(P: torch.Tensor, lo_frac: float = 0.10,
                       hi_frac: float = 0.90):
    """从全层 profile 选出关注层子张量（沿第 0 维 = 层维度切片）。

    默认跳过最底部（输入处理）与最顶部（输出处理），保留其余所有层。
    P: [num_layers, ...]。返回 (P_focus, lo, hi)，lo/hi 为层下标（含左端点、右开）。
    """
    L = P.shape[0]
    lo = max(0, min(int(round(lo_frac * L)), L - 1))
    hi = max(lo + 1, min(int(round(hi_frac * L)), L))
    return P[lo:hi], lo, hi
