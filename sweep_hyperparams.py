#!/usr/bin/env python3
"""扫 PDSD 超参数并可视化。

1. 生成 rollout + q_plus 前向只做一次（缓存 completion 隐状态，避免重复前向）；
2. 对每个超参组合（window_k × layer区间 × vec_dim），只重算 shift + 尖峰 + 可分性；
3. 输出到 outputs/sweep_hyperparams/：
   - summary.md   指标表（各组合 avg_contrast / avg_prom / avg_kurtosis / avg_n_peaks）
   - overview.png 平均 contrast 排序柱状图
   - traj/        尖峰最多的前 K 个组合的 shift 曲线图（人工看落点）

用法：
    python sweep_hyperparams.py --seed 0
    python sweep_hyperparams.py --window-ks 3,5,8 --layer-ranges 0.1-0.9,0.25-0.65 --vec-dims 16
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seed_builder import load_model, build_student_prompt, build_teacher_prompt
from pivot.activation import compute_layer_profile, flatten_profile, select_layer_range
from pivot.shift import l2_normalize_cols, sliding_window_shift
from pivot.peaks import find_peaks
from pivot.separability import summarize


def parse_layer_ranges(s: str):
    out = []
    for part in s.split(","):
        lo, hi = part.split("-")
        out.append((float(lo), float(hi)))
    return out


@torch.no_grad()
def generate_and_forward(model, tokenizer, problem_ja, problem_en, solution,
                         max_new_tokens, temperature, top_p, top_k):
    """生成 rollout + q_plus 前向，返回 (completion_ids[1,T], comp_hs tuple of [1,T,D])，缓存到 CPU。"""
    device = next(model.parameters()).device
    # 学生 rollout
    student_prompt = build_student_prompt(tokenizer, problem_ja)
    inputs = tokenizer(student_prompt, return_tensors="pt").to(device)
    gcfg = model.generation_config
    gcfg.do_sample = (temperature > 0)
    gcfg.temperature = temperature if temperature > 0 else None
    gcfg.top_p = top_p if temperature > 0 else None
    gcfg.top_k = top_k if temperature > 0 else None
    gcfg.max_new_tokens = max_new_tokens
    gcfg.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    out = model.generate(**inputs)
    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = out[:, prompt_len:].detach().cpu()
    # q_plus 前向
    teacher_prompt = build_teacher_prompt(tokenizer, problem_ja, problem_en, solution)
    teacher_ids = tokenizer(teacher_prompt, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([teacher_ids, completion_ids.to(device)], dim=1)
    fwd = model(input_ids=full_ids, output_hidden_states=True)
    hs = fwd.hidden_states
    teacher_len = teacher_ids.shape[1]
    comp_hs = tuple(h[:, teacher_len:, :].detach().cpu() for h in hs)
    return completion_ids, comp_hs


def compute_shift_from_hs(comp_hs, vec_dim, method, layer_lo, layer_hi, window_k):
    P = compute_layer_profile(comp_hs, vec_dim, method)
    P_mid, _, _ = select_layer_range(P, layer_lo, layer_hi)
    Pn = l2_normalize_cols(flatten_profile(P_mid))
    shift, (lo, hi) = sliding_window_shift(Pn, window_k)
    return shift, (lo, hi)


def _mean_skip_nan(xs):
    xs = [x for x in xs if x == x]
    return statistics.mean(xs) if xs else float("nan")


def _fmt(x, nd=2):
    return "nan" if x != x else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="扫 PDSD 超参数")
    ap.add_argument("--data-path", default="data/seed_ja.jsonl")
    ap.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--window-ks", default="2,3,5,8")
    ap.add_argument("--layer-ranges", default="0.1-0.9,0.25-0.65,0.4-0.8")
    ap.add_argument("--vec-dims", default="1,4,16,64,256")
    ap.add_argument("--vec-method", default="rms_chunk")
    ap.add_argument("--z-threshold", type=float, default=1.0)
    ap.add_argument("--min-peak-sep", type=int, default=3)
    ap.add_argument("--context-before", type=int, default=6, help="尖峰文本上下文：向前 token 数")
    ap.add_argument("--context-after", type=int, default=10, help="尖峰文本上下文：向后 token 数")
    ap.add_argument("--top-n", type=int, default=5, help="每条例题显示的尖峰数")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--top-k-combos", type=int, default=6, help="画 shift 曲线的前 K 个组合")
    ap.add_argument("--out-dir", default="outputs/sweep_hyperparams")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    window_ks = [int(x) for x in args.window_ks.split(",")]
    layer_ranges = parse_layer_ranges(args.layer_ranges)
    vec_dims = [int(x) for x in args.vec_dims.split(",")]

    records = [json.loads(l) for l in Path(args.data_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.num_samples is not None:
        records = records[: args.num_samples]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traj").mkdir(parents=True, exist_ok=True)

    print(f"加载模型 {args.model_name} ...")
    model, tokenizer = load_model(args.model_name, torch.bfloat16)

    # ---- 阶段 1：生成 + 前向（只做一次，缓存） ----
    print(f"阶段 1/2：生成 rollout + q_plus 前向（{len(records)} 条）...")
    cached = []
    for i, rec in enumerate(records):
        cids, chs = generate_and_forward(
            model, tokenizer, rec["problem_ja"], rec["problem"], rec["solution"],
            args.max_new_tokens, args.temperature, args.top_p, args.top_k,
        )
        cached.append({
            "problem_ja": rec["problem_ja"],
            "problem_en": rec["problem"],
            "solution": rec["solution"],
            "completion_ids": cids,
            "comp_hs": chs,
            "completion_text": tokenizer.decode(cids[0], skip_special_tokens=True),
        })
        print(f"  [{i + 1}/{len(records)}] {cids.shape[1]} tokens")

    # ---- 阶段 2：扫超参（只重算 shift） ----
    print(f"阶段 2/2：扫 {len(window_ks) * len(layer_ranges) * len(vec_dims)} 组超参 ...")
    rows = []
    for wk, (llo, lhi), vd in [(wk, lr, vd) for wk in window_ks for lr in layer_ranges for vd in vec_dims]:
        contrasts, proms, kurts, npeaks = [], [], [], []
        combo_pivots = []  # 该组合下各例的尖峰 [(ci, problem_ja, [(pidx, z, ctx)])]
        for ci, c in enumerate(cached):
            shift, (lo, hi) = compute_shift_from_hs(c["comp_hs"], vd, args.vec_method, llo, lhi, wk)
            peaks, _ = find_peaks(shift, args.z_threshold, wk, args.min_peak_sep, 1e-4)
            sep = summarize(shift.cpu().numpy(), peaks, (lo, hi))
            contrasts.append(sep["contrast"])
            proms.append(sep["prom_median"])
            kurts.append(sep["excess_kurtosis"])
            npeaks.append(sep["n_peaks"])
            # z 最高的 top-N 尖峰 + 文本上下文
            top = sorted(peaks, key=lambda p: -p[1])[: args.top_n]
            piv = []
            for pidx, zv in top:
                clo = max(0, pidx - args.context_before)
                chi = min(c["completion_ids"].shape[1], pidx + args.context_after + 1)
                ctx = tokenizer.decode(c["completion_ids"][0][clo:chi], skip_special_tokens=True)
                piv.append((pidx, zv, ctx))
            combo_pivots.append((ci, c["problem_ja"], piv))
        rows.append({
            "window_k": wk, "layer": f"[{llo},{lhi})", "vec_dim": vd,
            "avg_contrast": _mean_skip_nan(contrasts),
            "avg_prom": _mean_skip_nan(proms),
            "avg_kurt": _mean_skip_nan(kurts),
            "avg_n_peaks": statistics.mean(npeaks),
            "pivots": combo_pivots,
        })
        print(f"  k={wk} layers=[{llo},{lhi}) vec_dim={vd}  "
              f"contrast={_fmt(rows[-1]['avg_contrast'])} n_peaks={rows[-1]['avg_n_peaks']:.1f}")

    # ---- 汇总表 ----
    rows_sorted = sorted(rows, key=lambda r: (-(r["avg_contrast"] if r["avg_contrast"] == r["avg_contrast"] else -1)))
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# 超参数扫描汇总\n\n")
        f.write("| window_k | layer | vec_dim | avg_contrast | avg_prom | avg_kurt | avg_n_peaks |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows_sorted:
            f.write(f"| {r['window_k']} | {r['layer']} | {r['vec_dim']} | {_fmt(r['avg_contrast'])} "
                    f"| {_fmt(r['avg_prom'], 4)} | {_fmt(r['avg_kurt'], 1)} | {r['avg_n_peaks']:.1f} |\n")
        f.write("\n> avg_contrast 越大、avg_prom 越大，尖峰越凸出；avg_n_peaks 适中（不过多不过少）。\n")

    # ---- 枢轴文本（人工核查） ----
    with open(out_dir / "pivots.md", "w", encoding="utf-8") as f:
        f.write("# 枢轴位置人工核查\n\n")
        f.write(f"> 每条列出 z 最高的 top-{args.top_n} 个尖峰，附上下文文本（前 {args.context_before} 后 {args.context_after} token），"
                f"便于判断落点是否在'方法切换处'。\n\n")
        for r in rows_sorted:
            f.write(f"## k={r['window_k']}, layer={r['layer']}, vec_dim={r['vec_dim']} "
                    f"(avg_contrast={_fmt(r['avg_contrast'])})\n\n")
            for ci, problem, piv in r["pivots"]:
                f.write(f"### 例 {ci}：{problem[:40]}\n\n")
                if piv:
                    for pidx, zv, ctx in piv:
                        f.write(f"- t={pidx} z={zv:.2f} …{ctx}…\n")
                else:
                    f.write("（无尖峰）\n")
                f.write("\n")

    # ---- overview 柱状图（按 contrast 排序） ----
    labels = [f"k={r['window_k']},{r['layer']},d={r['vec_dim']}" for r in rows_sorted]
    vals = [r["avg_contrast"] if r["avg_contrast"] == r["avg_contrast"] else 0 for r in rows_sorted]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(labels)), vals, color="#1f77b4")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("avg contrast")
    ax.set_title("超参数组合 × 平均 contrast（越高尖峰越凸）")
    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=110)
    plt.close(fig)

    # ---- 前 K 个组合的 shift 曲线（人工看落点） ----
    top_rows = rows_sorted[: args.top_k_combos]
    for ri, r in enumerate(top_rows):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.ravel()
        for ci, c in enumerate(cached):
            shift, (lo, hi) = compute_shift_from_hs(c["comp_hs"], r["vec_dim"], args.vec_method,
                                                    float(r["layer"][1:-1].split(",")[0]),
                                                    float(r["layer"][1:-1].split(",")[1]), r["window_k"])
            s = shift.cpu().numpy()
            peaks, _ = find_peaks(shift, args.z_threshold, r["window_k"], args.min_peak_sep, 1e-4)
            ax = axes[ci]
            ax.plot(s, lw=0.8, color="#1f77b4")
            for pidx, _ in peaks:
                ax.plot(pidx, s[pidx], "o", color="#d62728", ms=5)
            ax.set_title(f"例{ci} ({len(peaks)} peaks)", fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axes[len(cached):]:
            ax.axis("off")
        fig.suptitle(f"k={r['window_k']} layer={r['layer']} vec_dim={r['vec_dim']} "
                     f"(avg_contrast={_fmt(r['avg_contrast'])})", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_dir / "traj" / f"top{ri + 1}_k{r['window_k']}_l{r['layer'].replace('[','').replace(')','').replace(',','-')}_d{r['vec_dim']}.png", dpi=100)
        plt.close(fig)

    print(f"\n完成。汇总见 {out_dir}/summary.md，图见 {out_dir}/overview.png 和 {out_dir}/traj/")


if __name__ == "__main__":
    main()
