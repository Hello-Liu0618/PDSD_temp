"""PDSD 枢轴 gate：在 q_plus（参考答案条件）前向上算"激活突变" → 峰 → per-token gate。

与 RP-OPSD 的 PRS gate 对应：**损失结构完全相同**，只是 gate 的来源不同
（激活突变 shift vs KL(q_plus||q_minus)）。

两种 gate 形态（`mode`）：
  * "spans"  —— PDSD 的实际输出：找峰 + 前后延伸成枢轴段，段内 g=1、段外 g=g_min（稀疏、近二值）。
  * "sigmoid"—— 用与 RP-OPSD 相同的 sigmoid 公式作用在 shift 的 z 值上（同形不同信号，对比更干净）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------- 逐层压缩 + shift（向量化） ----------------

def _rms_chunk(hs_layer: torch.Tensor, vec_dim: int) -> torch.Tensor:
    """[B, T, D] -> [B, T, vec_dim]：把 D 均分 vec_dim 段，各段 RMS（先转 float32 防 bf16 溢出）。"""
    hf = hs_layer.float()
    return torch.stack([torch.sqrt((s * s).mean(-1)) for s in torch.tensor_split(hf, vec_dim, dim=-1)], dim=-1)


def layer_profile(hidden_states, vec_dim: int, layer_lo: float, layer_hi: float) -> torch.Tensor:
    """hidden_states: tuple，第 0 个是 embedding 输出，其余 [B, T, D]。

    关注层区间按 num_layers 比例 [layer_lo, layer_hi) 选取（与探针一致）。
    Returns: P [L', B, T, vec_dim] float32
    """
    hs = list(hidden_states)[1:]
    L = len(hs)
    lo = max(0, min(int(round(layer_lo * L)), L - 1))
    hi = max(lo + 1, min(int(round(layer_hi * L)), L))
    return torch.stack([_rms_chunk(h, vec_dim) for h in hs[lo:hi]], dim=0)   # [L', B, T, vec_dim]


def shift_sequence(P: torch.Tensor, window_k: int) -> torch.Tensor:
    """P: [L', B, T, vec_dim] -> shift [B, T]（全序列；窗口外为 0）。

    先在 (L', vec_dim) 上展平并按"层-块"维 L2 归一，再做滑动窗口 cosine 距离：
        shift[t] = 1 - cos( mean(col[t-k:t]), mean(col[t:t+k]) )
    全部向量化（原 pivot/shift.py 是逐 token 的 Python 循环）。
    """
    Lp, B, T, V = P.shape
    # 必须先把"层"与"vec_dim"排到相邻再展平（原 pivot/activation.py 的 P 是 [L, V, T] 才可直接 reshape）
    Pn = P.permute(0, 3, 1, 2).reshape(Lp * V, B, T)  # [N, B, T]
    Pn = Pn / Pn.norm(dim=0, keepdim=True).clamp_min(1e-8)

    shift = torch.zeros(B, T, dtype=torch.float32, device=P.device)
    k = window_k
    if T <= 2 * k:
        return shift
    U = Pn.unfold(dimension=-1, size=k, step=1)      # [N, B, T-k+1, k]
    left = U[..., 0:T - 2 * k, :].mean(-1)           # [N, B, T-2k]  对应 t-k
    right = U[..., k:T - k, :].mean(-1)              # [N, B, T-2k]  对应 t
    cos = F.cosine_similarity(left, right, dim=0)    # [B, T-2k]
    shift[:, k:T - k] = 1.0 - cos
    return shift


# ---------------- 峰检测（照抄 pivot/peaks.py 的语义） ----------------

def _zscore(x: torch.Tensor) -> torch.Tensor:
    sd = x.std()
    if sd.item() < 1e-8:
        return torch.zeros_like(x)
    return (x - x.mean()) / sd


def _peaks_ranked(shift: torch.Tensor, z: torch.Tensor, k: int, min_sep: int,
                  shift_floor: float):
    """返回按 z 降序排列的候选峰 [(idx, z), ...]（与 find_peak_spans 同法，但阈值取 0 以留足候选）。"""
    T = shift.shape[0]
    lo, hi = k, T - k
    mask = torch.zeros(T, dtype=torch.bool, device=shift.device)
    if hi > lo:
        mask[lo:hi] = (z[lo:hi] > 0) & (shift[lo:hi] > shift_floor)
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
        return []
    peaks = [raw[0]]
    for idx, zv in raw[1:]:
        if idx - peaks[-1][0] <= min_sep:
            if zv > peaks[-1][1]:
                peaks[-1] = (idx, zv)
        else:
            peaks.append((idx, zv))
    return sorted(peaks, key=lambda p: -p[1])


def find_peak_spans_budgeted(shift: torch.Tensor, target_rho: float, g_min: float, k: int,
                             min_sep: int, shift_floor: float,
                             extend_before: int, extend_after: int) -> torch.Tensor:
    """**方案 (b)**：按 shift z 从高到低加枢轴段，直到 mean(gate) 达到 target_rho。

    保留 PDSD 的排序判据（谁更像枢轴），但把"给多少监督"这个预算钉死，
    从而与 RP-OPSD 在相同的 ρ 下比较"选得准不准"。
    由 gate = g_min + (1-g_min)*span 反解出目标密度：density = (target_rho - g_min)/(1-g_min)。
    """
    T = shift.shape[0]
    z = _zscore(shift)
    target_density = max(0.0, min(1.0, (target_rho - g_min) / max(1e-6, 1.0 - g_min)))
    span = torch.zeros(T, dtype=torch.float32, device=shift.device)
    if target_density <= 0:
        return span
    for idx, _ in _peaks_ranked(shift, z, k, min_sep, shift_floor):
        a = max(0, idx - extend_before)
        b = min(T, idx + extend_after + 1)
        span[a:b] = 1.0
        if float(span.mean()) >= target_density:
            break
    return span


def find_peak_spans(shift: torch.Tensor, z_threshold: float, k: int, min_sep: int,
                    shift_floor: float, extend_before: int, extend_after: int):
    """在一维 shift [T] 上找峰并扩成 span。返回 mask [T]（1 表示枢轴段）。

    与 pivot/peaks.py.find_peaks 同语义：z>阈值 且 shift>floor 的连续段取 z 峰，合并过近峰；
    再把每个峰扩成 [p-extend_before, p+extend_after]。
    """
    z = _zscore(shift)
    T = shift.shape[0]
    lo, hi = k, T - k
    mask = torch.zeros(T, dtype=torch.bool, device=shift.device)
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
        return mask.new_zeros(T).float()
    peaks = [raw[0]]
    for idx, zv in raw[1:]:
        if idx - peaks[-1][0] <= min_sep:
            if zv > peaks[-1][1]:
                peaks[-1] = (idx, zv)
        else:
            peaks.append((idx, zv))

    span = torch.zeros(T, dtype=torch.float32, device=shift.device)
    for idx, _ in peaks:
        a = max(0, idx - extend_before)
        b = min(T, idx + extend_after + 1)
        span[a:b] = 1.0
    return span


# ---------------- 组装 per-token gate ----------------

@torch.no_grad()
def pdsd_gate(hidden_states, completion_mask, *, prompt_len: int, vec_dim: int,
              layer_lo: float, layer_hi: float, window_k: int, mode: str = "spans",
              z_threshold: float = 1.0, min_sep: int = 3, shift_floor: float = 1e-4,
              extend_before: int = 2, extend_after: int = 4,
              beta: float = 2.0, tau: float = 0.0, g_min: float = 0.05,
              target_rho: float | None = None) -> torch.Tensor:
    """在 q_plus 前向的 hidden_states 上算 gate，只覆盖完成段。

    Args:
        hidden_states: q_plus 前向输出（含 embedding 的第 0 项）
        completion_mask: [B, G] bool，完成 token 掩码（G = 生成长度）
        prompt_len: int，完成段在序列中的起点（hidden 下标 = prompt_len-1）
    Returns:
        (gate [B, G] 落在 [g_min,1]、z [B, G] shift 的逐例 z 值) —— 完成段外均为 0
    """
    B, G = completion_mask.shape
    P = layer_profile(hidden_states, vec_dim, layer_lo, layer_hi)   # [L', B, T, V]
    shift_full = shift_sequence(P, window_k)                        # [B, T]
    del P

    T = shift_full.shape[1]
    g0 = prompt_len - 1
    g1 = min(g0 + G, T)
    shift = shift_full.new_zeros(B, G)
    if g1 > g0:
        shift[:, :g1 - g0] = shift_full[:, g0:g1]                   # 完成段（含窗口上下文）

    gate = torch.full((B, G), float(g_min), dtype=torch.float32, device=shift.device)
    z_out = torch.zeros(B, G, dtype=torch.float32, device=shift.device)
    for b in range(B):
        v = int(completion_mask[b].sum())          # 有效完成长度（padding 在右端）
        if v == 0:
            continue
        s = shift[b, :v]                            # 只在该段内 z-score / 找峰 / 算密度
        z = _zscore(s)
        z_out[b, :v] = z
        if mode == "sigmoid":
            gate[b, :v] = g_min + (1.0 - g_min) * torch.sigmoid(beta * (z.clamp(-5.0, 5.0) - tau))
        elif mode == "spans":
            if target_rho is not None:      # 方案 (b)：按 z 降序取峰直到 ρ 达标
                span = find_peak_spans_budgeted(s, target_rho, g_min, window_k, min_sep,
                                                shift_floor, extend_before, extend_after)
            else:                            # 原阈值法
                span = find_peak_spans(s, z_threshold, window_k, min_sep, shift_floor,
                                       extend_before, extend_after)
            gate[b, :v] = g_min + (1.0 - g_min) * span
        else:
            raise ValueError(f"未知 gate mode: {mode!r}")

    m = completion_mask.to(gate.dtype)
    return gate * m, z_out * m
