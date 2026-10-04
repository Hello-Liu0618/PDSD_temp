"""surprisal 计算与 shift/surprisal 相关度。"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


def compute_surprisal(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """logits: [1, T, V]；input_ids: [1, T]。

    surprisal[t] = -log p(token_t | token_<t)，即用 logits[t-1] 预测第 t 个 token。
    t 从 1 到 T-1；位置 0 置 0。
    """
    logp = F.log_softmax(logits[0].float(), dim=-1)        # [T, V]
    T = input_ids.shape[1]
    surp = torch.zeros(T, dtype=torch.float32)
    for t in range(1, T):
        surp[t] = -logp[t - 1, input_ids[0, t]].item()
    return surp


def pearson_corr(a, b):
    """两序列的 Pearson 相关系数 (r, p)。空/常数序列返回 (nan, nan)。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan"), float("nan")
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(a, b)
        return float(r), float(p)
    except Exception:
        r = float(np.corrcoef(a, b)[0, 1])
        return r, float("nan")
