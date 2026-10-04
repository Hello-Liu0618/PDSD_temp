"""尖峰检测：轨迹内 z-score + 连续段峰值。"""
from __future__ import annotations

from typing import List, Tuple

import torch


def zscore(x: torch.Tensor) -> torch.Tensor:
    """轨迹内 z-score。std 接近 0 时返回全 0，避免除零。"""
    x = x.float()
    sd = x.std()
    if sd.item() < 1e-8:
        return torch.zeros_like(x)
    return (x - x.mean()) / sd


def find_peaks(shift: torch.Tensor, z_threshold: float = 2.0,
               k: int = 5, min_sep: int = 3, shift_floor: float = 1e-4) -> Tuple[List[Tuple[int, float]], torch.Tensor]:
    """找候选变点（token 下标）。

    1) 轨迹内 z-score；
    2) 取 z > z_threshold 且 shift > shift_floor 的连续段（仅限有效窗口 [k, T-k)）；
       shift_floor 用于过滤近乎平坦信号被 z-score 放大的数值噪声；
    3) 每段取 z 峰值位置；
    4) 合并相距 <= min_sep 的峰（保留 z 更高的那个）。

    Returns:
        peaks: [(token_index, z_value), ...] 按 token 下标升序
        z: [T] z-score 序列
    """
    z = zscore(shift)
    T = shift.shape[0]
    lo, hi = k, T - k
    mask = torch.zeros(T, dtype=torch.bool)
    if hi > lo:
        mask[lo:hi] = (z[lo:hi] > z_threshold) & (shift[lo:hi] > shift_floor)

    raw = []
    i = 0
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            local = int(torch.argmax(z[i:j]))
            raw.append((i + local, float(z[i + local])))
            i = j
        else:
            i += 1

    if not raw:
        return [], z

    # 合并过近的峰
    peaks = [raw[0]]
    for idx, zv in raw[1:]:
        if idx - peaks[-1][0] <= min_sep:
            if zv > peaks[-1][1]:
                peaks[-1] = (idx, zv)
        else:
            peaks.append((idx, zv))
    return peaks, z
