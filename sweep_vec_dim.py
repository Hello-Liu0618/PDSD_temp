#!/usr/bin/env python3
"""扫描 vec_dim，逐个生成可视化 + 数据报告，并量化峰值可分性。

每个 vec_dim 产出（输出到 outputs/sweep/vec_dim_XXX/）：
  report.json       数据报告（峰值可分性 + 与 surprisal 相关度）
  report.md         可读文本报告
  plots/separability.png  峰值可分性总览图
  plots/example_traj.png  尖峰最多那条轨迹的 shift 曲线示例

并在 outputs/sweep/ 下写 aggregate.json / aggregate.md 做跨 vec_dim 对比。

用法：
    python sweep_vec_dim.py --vec-dims 1,4,16,64,256
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics

import numpy as np
import torch

from config import ProbeConfig
from pivot.loader import load_model_and_tokenizer
from pivot import separability, textmap, viz
from pivot_probe import process_trajectory


def _sanitize(o):
    """递归把 NaN/Inf 换成 None，保证 JSON 合法。"""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def _mean_skip_nan(xs):
    xs = [x for x in xs if x == x]
    return statistics.mean(xs) if xs else float("nan")


def _fmt(x, nd=3):
    if x != x:
        return "nan"
    return f"{x:.{nd}f}"


def _write_report_md(d: str, rep: dict) -> None:
    lines = []
    lines.append(f"# vec_dim = {rep['vec_dim']}（method={rep['method']}，{rep['n_trajectories']} 条轨迹）\n")
    lines.append("## 峰值可分性（核心：峰值是否真的凸出来）\n")
    lines.append("| 指标 | 均值 | 怎么读 |")
    lines.append("|---|---|---|")
    c = rep["avg_contrast"]
    lines.append(f"| contrast（峰值中位数/背景中位数） | {_fmt(c, 2)} | ≈1 说明峰值不突出；>2~3 说明明显凸出 |")
    lines.append(f"| prominence 中位数（相对两侧谷底凸起） | {_fmt(rep['avg_prom_median'], 4)} | 峰值比周围谷底高多少（shift 单位） |")
    lines.append(f"| 有效段 shift 峰度（Fisher） | {_fmt(rep['avg_excess_kurtosis'], 2)} | >0 重尾=有离群尖峰；<0 平坦 |")
    lines.append(f"| 平均峰值数 / 轨迹 | {rep['avg_n_peaks']:.1f} | — |")
    lines.append(f"| 峰值占有效 token 比例 | {_fmt(rep['avg_peak_density'] * 100, 1)}% | 越稀疏越像“决策点” |")
    lines.append("")
    lines.append("## 与 surprisal 对照（次要，仅排雷）\n")
    lines.append(f"- 平均 shift–surprisal Pearson 相关（关注层）：**{_fmt(rep['avg_corr_mid'], 3)}**\n")
    lines.append("## 逐轨迹\n")
    lines.append("| idx | n_peaks | contrast | prom_median | kurtosis | corr_mid |")
    lines.append("|---|---|---|---|---|---|")
    for pt in rep["per_trajectory"]:
        lines.append(f"| {pt['idx']} | {pt['n_peaks']} | {_fmt(pt['contrast'], 2)} | "
                     f"{_fmt(pt['prom_median'], 4)} | {_fmt(pt['excess_kurtosis'], 2)} | "
                     f"{_fmt(pt['corr_mid'], 3)} |")
    lines.append("")
    with open(os.path.join(d, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_aggregate_md(out_dir: str, agg: list) -> None:
    lines = ["# vec_dim 扫描汇总\n"]
    lines.append("| vec_dim | avg_corr_mid | avg_n_peaks | peak_density | avg_contrast | avg_prom_median | avg_kurtosis |")
    lines.append("|---|---|---|---|---|---|---|")
    for rep in agg:
        lines.append(f"| {rep['vec_dim']} | {_fmt(rep['avg_corr_mid'])} | {rep['avg_n_peaks']:.1f} | "
                     f"{_fmt(rep['avg_peak_density'] * 100, 1)}% | {_fmt(rep['avg_contrast'], 2)} | "
                     f"{_fmt(rep['avg_prom_median'], 4)} | {_fmt(rep['avg_excess_kurtosis'], 2)} |")
    lines.append("")
    lines.append("## 怎么读\n")
    lines.append("- **contrast**：峰值 shift 中位数是背景的几倍。≈1 说明只是“平坦里挑高的”；>2~3 才是真正的孤立尖峰。")
    lines.append("- **kurtosis（峰度）**：>0 重尾=分布有明显离群尖峰；≈0 高斯=峰值只是尾巴；<0 平坦=峰值无特殊性。")
    lines.append("- **prominence**：峰值相对两侧谷底凸起多少（shift 单位），越大越像离散事件。")
    lines.append("- **avg_corr_mid**：与 surprisal 的相关（次要，仅用于排雷“是否翻版”）。")
    lines.append("")
    with open(os.path.join(out_dir, "aggregate.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_samples_md(d: str, tokenizer, samples, vec_dim: int, cfg, context_n: int = 8) -> None:
    """samples: list of (idx, r, cot, problem)，按峰值数降序。尖峰词用【】标出。"""
    lines = [f"# vec_dim = {vec_dim}：人工核查样本（尖峰词用【】标出）\n"]
    lines.append("下面是该 vec_dim 下峰值最多的若干条轨迹，便于人工判断【】里的词是否像真正的推理枢轴。\n")
    for idx, r, cot, problem in samples:
        n = len(r["peaks"])
        lines.append(f"## 样本 traj {idx}（{n} 个峰值，corr_mid={r['corr_mid']:.3f}）\n")
        lines.append(f"**问题**：{problem}\n")
        annotated = textmap.annotate(cot, r["offsets"], [p for p, _ in r["peaks"]],
                                      cfg.span_extend, cfg.span_max_len, cfg.span_min_len,
                                      cfg.raw_span)
        lines.append("```text\n" + annotated + "\n```\n")
        lines.append("**峰值清单**：\n")
        for pidx, zv in r["peaks"]:
            w = textmap.peak_label(cot, r["offsets"], pidx)
            ctx = textmap.context_text(tokenizer, r["input_ids"], pidx, context_n, context_n)
            lines.append(f"- t={pidx} (z={zv:.2f}) 【{w}】 …{ctx}…")
        lines.append("")
    with open(os.path.join(d, "samples_annotated.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec-dims", default="1,4,16,64,256")
    ap.add_argument("--data-path", default="data/gsm8k_cot.jsonl")
    ap.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--method", default="rms_chunk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--out-dir", default="outputs/sweep")
    args = ap.parse_args()

    vec_dims = [int(x) for x in args.vec_dims.split(",")]
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    base = ProbeConfig(dtype=args.dtype, model_name=args.model_name)
    print(f"加载模型 {args.model_name} ...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, base.torch_dtype, True)

    with open(args.data_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    agg = []
    for vd in vec_dims:
        cfg = ProbeConfig(model_name=args.model_name, dtype=args.dtype,
                          layer_vec_dim=vd, layer_vec_method=args.method,
                          layer_vec_seed=args.seed, data_path=args.data_path)
        d = os.path.join(args.out_dir, f"vec_dim_{vd:03d}")
        plot_dir = os.path.join(d, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        sep_rows, corr_rows = [], []
        all_shift, all_peak_vals = [], []
        best = None  # (n_peaks, r, cot)
        results_all = []  # (n_peaks, idx, r, cot, problem)

        for i, rec in enumerate(records):
            r = process_trajectory(model, tokenizer, rec["cot"], cfg)
            s = separability.summarize(r["shift_mid"], r["peaks"], r["valid"])
            s["idx"] = i
            s["corr_mid"] = r["corr_mid"]
            sep_rows.append(s)
            corr_rows.append(r["corr_mid"])
            results_all.append((len(r["peaks"]), i, r, rec["cot"], rec.get("problem", "")))

            sh = np.asarray(r["shift_mid"], dtype=np.float64)[r["valid"][0]:r["valid"][1]]
            all_shift.append(sh)
            for pidx, _ in r["peaks"]:
                all_peak_vals.append(float(r["shift_mid"][pidx]))

            if best is None or len(r["peaks"]) > best[0]:
                best = (len(r["peaks"]), r, rec["cot"])

        report = {
            "vec_dim": vd,
            "method": args.method,
            "n_trajectories": len(records),
            "avg_corr_mid": _mean_skip_nan(corr_rows),
            "avg_n_peaks": statistics.mean([s["n_peaks"] for s in sep_rows]),
            "avg_peak_density": _mean_skip_nan([s["peak_density"] for s in sep_rows]),
            "avg_contrast": _mean_skip_nan([s["contrast"] for s in sep_rows]),
            "avg_prom_median": _mean_skip_nan([s["prom_median"] for s in sep_rows]),
            "avg_prom_max": _mean_skip_nan([s["prom_max"] for s in sep_rows]),
            "avg_excess_kurtosis": _mean_skip_nan([s["excess_kurtosis"] for s in sep_rows]),
            "per_trajectory": sep_rows,
        }
        agg.append(report)

        with open(os.path.join(d, "report.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize(report), f, ensure_ascii=False, indent=2)
        _write_report_md(d, report)

        viz.plot_separability(all_shift, all_peak_vals, report,
                              os.path.join(plot_dir, "separability.png"))

        # 示例轨迹（尖峰最多那条）
        if best is not None:
            n_peaks, r, cot = best
            peak_labels = []
            for pidx, _ in r["peaks"]:
                peak_labels.append(textmap.peak_label(cot, r["offsets"], pidx))
            title = f"vec_dim={vd} 尖峰最多轨迹（{n_peaks} 个峰值）：{cot[:40].replace(chr(10), ' ')}..."
            viz.plot_trajectory(r["shift_mid"], r["z_mid"], r["peaks"], r["surprisal"],
                                cfg.z_threshold, peak_labels, title,
                                os.path.join(plot_dir, "example_traj.png"))

        # 人工核查样本：峰值最多的 N 条，尖峰词用【】标出
        samples = sorted(results_all, key=lambda x: -x[0])[:args.n_samples]
        _write_samples_md(d, tokenizer,
                          [(idx, r, cot, prob) for (_, idx, r, cot, prob) in samples], vd, cfg)

        print(f"vec_dim={vd:3d}  corr_mid={report['avg_corr_mid']:.3f}  "
              f"peaks/traj={report['avg_n_peaks']:.1f}  "
              f"contrast={report['avg_contrast']:.2f}  "
              f"prom_med={report['avg_prom_median']:.4f}  "
              f"kurt={report['avg_excess_kurtosis']:.2f}")

    with open(os.path.join(args.out_dir, "aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(_sanitize(agg), f, ensure_ascii=False, indent=2)
    _write_aggregate_md(args.out_dir, agg)
    print(f"\n完成，汇总见 {args.out_dir}/aggregate.md")


if __name__ == "__main__":
    main()
