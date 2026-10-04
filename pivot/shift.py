"""shift 序列：滑动窗口 cosine 距离。

把 profile 的每个 token 列（沿"层"维度）L2 归一化后，对每个 t 取
"t 之前 k 个 token 的均值 profile" 与 "t 之后 k 个 token 的均值 profile" 的
余弦距离，作为该位置"激活突变"强度的度量。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def l2_normalize_cols(P: torch.Tensor) -> torch.Tensor:
    """沿第 0 维（层）把每个 token 列归一化为单位向量。P: [L, T] -> [L, T]."""
    norms = torch.norm(P, dim=0, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)
    return P / norms


def sliding_window_shift(Pn: torch.Tensor, k: int = 5):
    """Pn: [L, T]（已按列归一化）。

    shift[t] = 1 - cosine(mean(col[t-k:t]), mean(col[t:t+k]))，t in [k, T-k)。

    Returns:
        shift: [T] float32，窗口外位置为 0
        (lo, hi): 有效区间 [lo, hi) = [k, T-k)
    """
    L, T = Pn.shape
    shift = torch.zeros(T, dtype=torch.float32)
    for t in range(k, T - k):
        left = Pn[:, t - k:t].mean(dim=1)
        right = Pn[:, t:t + k].mean(dim=1)
        cos = F.cosine_similarity(left, right, dim=0)
        shift[t] = 1.0 - cos.item()
    return shift, (k, T - k)
