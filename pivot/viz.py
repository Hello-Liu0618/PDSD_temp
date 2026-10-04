"""可视化：shift 序列 + 尖峰、surprisal 对照、相关度分布、散点。"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# 中文字体：避免中文标题/标签渲染成方框（Noto Sans CJK 覆盖 CJK 字形）
_AVAILABLE_FONTS = {f.name for f in _fm.fontManager.ttflist}
for _cand in ("Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"):
    if _cand in _AVAILABLE_FONTS:
        plt.rcParams["font.sans-serif"] = [_cand, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_trajectory(shift, z, peaks, surprisal, z_threshold: float,
                    peak_labels: List[str], title: str, out_path: str) -> None:
    """单条轨迹三连图：shift、z-score、surprisal。"""
    T = len(shift)
    xs = np.arange(T)
    shift = np.asarray(shift, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    surprisal = np.asarray(surprisal, dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2, 1.2]})
    fig.suptitle(title, fontsize=10)

    ax = axes[0]
    ax.plot(xs, shift, lw=1.0, color="#1f77b4", label="activation shift (1−cos)")
    ax.set_ylabel("shift")
    ax.legend(loc="upper right", fontsize=8)
    for (pidx, _zv), lab in zip(peaks, peak_labels):
        ax.plot(pidx, shift[pidx], "o", color="#d62728", ms=7, zorder=5)
        ax.axvline(pidx, color="#d62728", ls=":", lw=0.7, alpha=0.6)
        ax.annotate(lab, (pidx, shift[pidx]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="#d62728")

    ax = axes[1]
    ax.plot(xs, z, lw=1.0, color="#2ca02c")
    ax.axhline(z_threshold, color="red", ls="--", lw=1, label=f"z={z_threshold}")
    ax.set_ylabel("z-score")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(xs, surprisal, lw=1.0, color="#9467bd", label="surprisal")
    ax.set_ylabel("surprisal")
    ax.set_xlabel("token position")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_corr_hist(corrs, out_path: str,
                   title: str = "shift–surprisal Pearson r (per trajectory)") -> None:
    corrs = [c for c in corrs if c == c]       # 去掉 nan
    if not corrs:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(corrs, bins=20, color="#1f77b4", alpha=0.8, edgecolor="white")
    mean = float(np.mean(corrs))
    ax.axvline(mean, color="red", ls="--", lw=2, label=f"mean={mean:.3f}")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_scatter(shift, surprisal, valid, corr, out_path: str,
                 title: str = "shift vs surprisal") -> None:
    s = np.asarray(shift, dtype=np.float64)[valid]
    u = np.asarray(surprisal, dtype=np.float64)[valid]
    if len(s) < 3:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(u, s, s=6, alpha=0.5, color="#1f77b4")
    m, b = np.polyfit(u, s, 1)
    xs_ = np.sort(u)
    ax.plot(xs_, m * xs_ + b, color="red", lw=1.5, label=f"r={corr:.3f}")
    ax.set_xlabel("surprisal")
    ax.set_ylabel("activation shift")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_separability(all_shift, all_peak_vals, report: dict, out_path: str) -> None:
    """峰值可分性总览图（4 面板），直观回答"峰值是否真的凸出来"。

    all_shift: list of 1D arrays（每条轨迹有效段的 shift）
    all_peak_vals: list of float（所有峰值位置的 shift 值）
    report: dict（聚合指标，含 vec_dim / method / per_trajectory）
    """
    flat = np.concatenate([np.asarray(s).ravel() for s in all_shift]) if all_shift else np.array([])
    peaks = np.asarray(all_peak_vals, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"峰值可分性（vec_dim={report['vec_dim']}, {report['method']}）", fontsize=11)

    ax = axes[0, 0]
    if flat.size:
        bins = np.linspace(0, np.percentile(flat, 99.5), 60)
        ax.hist(flat, bins=bins, color="#1f77b4", alpha=0.7, label="所有 token 的 shift")
        if peaks.size:
            ax.hist(peaks, bins=bins, color="#d62728", alpha=0.85, label="峰值 token 的 shift")
    ax.set_yscale("log")
    ax.set_xlabel("shift (1−cos)")
    ax.set_ylabel("频次 (log)")
    ax.legend(fontsize=8)
    ax.set_title("峰值是否落在分布最右端（红=峰值，蓝=全部）")

    ax = axes[0, 1]
    per_traj = report.get("per_trajectory", [])
    proms = [pt["prom_median"] for pt in per_traj if pt["prom_median"] == pt["prom_median"]]
    if proms:
        ax.hist(proms, bins=30, color="#2ca02c", alpha=0.8)
    ax.set_xlabel("峰值 prominence 中位数")
    ax.set_ylabel("轨迹数")
    ax.set_title("峰值相对两侧谷底凸起多少")

    ax = axes[1, 0]
    contrasts = [pt["contrast"] for pt in per_traj if pt["contrast"] == pt["contrast"]]
    if contrasts:
        ax.bar(range(len(contrasts)), sorted(contrasts), color="#9467bd", alpha=0.85)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("轨迹（按 contrast 排序）")
    ax.set_ylabel("contrast（峰值中位数 / 背景中位数）")
    ax.set_title("contrast≈1 说明峰值不突出，越大越凸出")

    ax = axes[1, 1]
    kurts = [pt["excess_kurtosis"] for pt in per_traj if pt["excess_kurtosis"] == pt["excess_kurtosis"]]
    if kurts:
        ax.hist(kurts, bins=30, color="#ff7f0e", alpha=0.8)
    ax.axvline(0.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("有效段 shift 峰度（Fisher）")
    ax.set_ylabel("轨迹数")
    ax.set_title("峰度>0=有离群尖峰，<0=平坦")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
