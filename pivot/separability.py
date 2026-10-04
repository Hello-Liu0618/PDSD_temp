"""峰值可分性：量化"尖峰是否真的从背景里凸出来"。

回答的核心问题：检测到的峰值 token 的 shift，是明显高于背景的孤立尖峰，
还是只是"从相对平坦的数值里挑出的较高几个"。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.signal import peak_prominences
from scipy.stats import kurtosis as _kurtosis


def _valid_shift(shift, valid):
    lo, hi = valid
    return np.asarray(shift, dtype=np.float64)[lo:hi]


def prominences(shift, peak_indices: List[int]) -> np.ndarray:
    """每个峰值相对两侧谷底的凸起高度（scipy 的 peak_prominences 定义）。"""
    s = np.asarray(shift, dtype=np.float64)
    if not peak_indices:
        return np.array([])
    return peak_prominences(s, np.asarray(peak_indices, dtype=int))[0]


def summarize(shift, peaks: List[Tuple[int, float]], valid) -> dict:
    """单条轨迹的峰值可分性指标。

    指标含义：
      n_peaks          峰值个数
      peak_density     峰值占有效 token 的比例
      contrast         峰值 shift 中位数 / 背景 shift 中位数（≈1=峰值不突出；越大越凸出）
      prom_median      峰值 prominence 中位数（相对两侧谷底凸起多少，shift 单位）
      prom_max         峰值 prominence 最大值
      excess_kurtosis  有效段 shift 的峰度（Fisher）：>0 重尾=有离群尖峰；
                       ≈0 高斯=峰值只是正常分布的尾巴；<0 平坦=峰值无特殊性
    """
    s = _valid_shift(shift, valid)
    lo = valid[0]
    peak_idx = np.array([p - lo for p, _ in peaks], dtype=int)

    n = len(s)
    mask = np.ones(n, dtype=bool)
    if len(peak_idx):
        mask[peak_idx] = False
    bg = s[mask]

    report = {
        "n_peaks": int(len(peak_idx)),
        "peak_density": float(len(peak_idx) / n) if n else 0.0,
    }

    if len(peak_idx) and len(bg) and np.median(bg) > 0:
        report["contrast"] = float(np.median(s[peak_idx]) / np.median(bg))
    else:
        report["contrast"] = float("nan")

    prom = prominences(shift, [p for p, _ in peaks])
    report["prom_median"] = float(np.median(prom)) if len(prom) else float("nan")
    report["prom_max"] = float(np.max(prom)) if len(prom) else float("nan")

    report["excess_kurtosis"] = float(_kurtosis(s, fisher=True)) if n > 3 else float("nan")
    return report
