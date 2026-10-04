#!/usr/bin/env python3
"""方法一探针：基于激活突变的推理枢轴检测。

输入：轨迹 JSONL（{"problem","cot"}）+ 模型名。
输出：
  1) 每条轨迹的候选变点位置 + 对应文本片段；
  2) 每条轨迹 shift 与 surprisal 的 Pearson 相关；
  3) 所有轨迹的平均相关度（本探针最关键的数字）；
  4) 可视化图（outputs/plots/）+ 【】标注的轨迹样本（outputs/annotated.md）。

用法：
    python pivot_probe.py --data-path data/gsm8k_cot.jsonl --model-name Qwen/Qwen3-1.7B
"""
from __future__ import annotations

import json
import os
import statistics

import torch

from config import ProbeConfig, build_arg_parser, apply_cli
from pivot.loader import load_model_and_tokenizer, model_device
from pivot.activation import compute_layer_profile, flatten_profile, select_layer_range
from pivot.shift import l2_normalize_cols, sliding_window_shift
from pivot.peaks import find_peaks
from pivot.surprisal import compute_surprisal, pearson_corr
from pivot import textmap
from pivot import viz


@torch.no_grad()
def _forward(model, input_ids):
    return model(input_ids=input_ids, output_hidden_states=True,
                 return_dict=True, use_cache=False)


@torch.no_grad()
def process_trajectory(model, tokenizer, cot, cfg):
    """对单条轨迹做一次 teacher-forcing 前向，返回全部中间结果。"""
    input_ids_list, offsets = textmap.encode_with_offsets(tokenizer, cot)
    ids = torch.tensor([input_ids_list], dtype=torch.long, device=model_device(model))

    out = _forward(model, ids)

    P = compute_layer_profile(out.hidden_states, cfg.layer_vec_dim,
                              cfg.layer_vec_method, cfg.layer_vec_seed)   # [L, vec_dim, T]
    L, vec_dim, T = P.shape

    # 全层（对照）
    Pn_full = l2_normalize_cols(flatten_profile(P))
    shift_full, (lo, hi) = sliding_window_shift(Pn_full, cfg.window_k)

    # 中间层（主报告）
    P_mid, mlo, mhi = select_layer_range(P, cfg.layer_lo_frac, cfg.layer_hi_frac)
    Pn_mid = l2_normalize_cols(flatten_profile(P_mid))
    shift_mid, _ = sliding_window_shift(Pn_mid, cfg.window_k)

    # 尖峰（用中间层 shift）
    peaks, z_mid = find_peaks(shift_mid, cfg.z_threshold, cfg.window_k, cfg.min_peak_sep, cfg.shift_floor)

    # surprisal
    surprisal = compute_surprisal(out.logits, ids)

    # 相关度（有效区间 [lo, hi) 对齐）
    corr_full, _ = pearson_corr(shift_full[lo:hi], surprisal[lo:hi])
    corr_mid, _ = pearson_corr(shift_mid[lo:hi], surprisal[lo:hi])

    return {
        "shift_full": shift_full, "shift_mid": shift_mid,
        "z_mid": z_mid, "peaks": peaks, "surprisal": surprisal,
        "corr_full": corr_full, "corr_mid": corr_mid,
        "input_ids": input_ids_list, "offsets": offsets,
        "num_layers": L, "num_tokens": T, "layer_vec_dim": vec_dim, "valid": (lo, hi),
        "middle_range": (mlo, mhi),
    }


def _round_or_none(x, nd=4):
    if x != x:                       # NaN
        return None
    return round(float(x), nd)


def main() -> None:
    cfg = ProbeConfig()
    args = build_arg_parser(ProbeConfig).parse_args()
    apply_cli(cfg, args)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    print(f"加载模型 {cfg.model_name} (dtype={cfg.dtype}) ...")
    model, tokenizer = load_model_and_tokenizer(cfg.model_name, cfg.torch_dtype, cfg.trust_remote_code)
    L = model.config.num_hidden_layers
    lo_layer = int(round(cfg.layer_lo_frac * L))
    hi_layer = int(round(cfg.layer_hi_frac * L))
    print(f"  num_layers={L}  关注层(除输入/输出处理)=[{lo_layer},{hi_layer})")

    with open(cfg.data_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    plot_dir = os.path.join(cfg.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    results, corr_mid_list, corr_full_list = [], [], []

    for i, rec in enumerate(records):
        cot = rec["cot"]
        r = process_trajectory(model, tokenizer, cot, cfg)
        r["problem"] = rec.get("problem", "")
        results.append(r)
        corr_mid_list.append(r["corr_mid"])
        corr_full_list.append(r["corr_full"])

        peak_labels = []
        for pidx, _ in r["peaks"]:
            peak_labels.append(textmap.peak_label(cot, r["offsets"], pidx))

        print(f"\n[traj {i}] tokens={r['num_tokens']} peaks={len(r['peaks'])} "
              f"corr_mid={r['corr_mid']:.3f} corr_full={r['corr_full']:.3f}")
        for (pidx, zv), lab in zip(r["peaks"], peak_labels):
            ctx = textmap.context_text(tokenizer, r["input_ids"], pidx, cfg.context_n, cfg.context_n)
            print(f"    t={pidx:4d} z={zv:5.2f} 词「{lab}」  …{ctx}…")

        title = f"traj {i}: {cot[:50].replace(chr(10), ' ')}..."
        viz.plot_trajectory(
            r["shift_mid"], r["z_mid"], r["peaks"], r["surprisal"],
            cfg.z_threshold, peak_labels, title,
            os.path.join(plot_dir, f"traj_{i:03d}.png"))

    # ---- 汇总 ----
    valid_mid = [c for c in corr_mid_list if c == c]
    avg_mid = statistics.mean(valid_mid) if valid_mid else float("nan")
    valid_full = [c for c in corr_full_list if c == c]
    avg_full = statistics.mean(valid_full) if valid_full else float("nan")

    per_traj = []
    for i, r in enumerate(results):
        cot = records[i]["cot"]
        peaks_out = []
        for pidx, zv in r["peaks"]:
            peaks_out.append({"token": int(pidx), "z": round(float(zv), 3),
                              "word": textmap.peak_label(cot, r["offsets"], pidx)})
        per_traj.append({
            "idx": i, "n_peaks": len(r["peaks"]),
            "corr_mid": _round_or_none(r["corr_mid"]),
            "corr_full": _round_or_none(r["corr_full"]),
            "peaks": peaks_out,
        })

    summary = {
        "model": cfg.model_name,
        "num_trajectories": len(results),
        "avg_corr_mid": _round_or_none(avg_mid),
        "avg_corr_full": _round_or_none(avg_full),
        "config": {k: (str(v) if isinstance(v, torch.dtype) else v) for k, v in cfg.to_dict().items()},
        "per_trajectory": per_traj,
    }
    with open(os.path.join(cfg.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    viz.plot_corr_hist(corr_mid_list, os.path.join(plot_dir, "corr_hist.png"))

    # ---- 【】标注 N 条（选尖峰最多的） ----
    order = sorted(range(len(results)), key=lambda i: -len(results[i]["peaks"]))
    annotate_n = min(cfg.n_annotate, len(results))
    with open(os.path.join(cfg.output_dir, "annotated.md"), "w", encoding="utf-8") as f:
        f.write("# 方法一探针：候选推理枢轴切换点标注\n\n")
        f.write(f"模型 {cfg.model_name}，共 {len(results)} 条轨迹，"
                f"平均 corr_mid={avg_mid:.3f}。\n\n")
        for rank in range(annotate_n):
            i = order[rank]
            r = results[i]
            cot = records[i]["cot"]
            annotated = textmap.annotate(cot, r["offsets"], [p for p, _ in r["peaks"]],
                                         cfg.span_extend, cfg.span_max_len, cfg.span_min_len,
                                         cfg.raw_span)
            f.write(f"## 轨迹 {i}（{len(r['peaks'])} 个候选点，corr_mid={r['corr_mid']:.3f}）\n\n")
            f.write(f"**问题**：{records[i].get('problem', '')}\n\n")
            f.write("```text\n" + annotated + "\n```\n\n")
            for pidx, zv in r["peaks"]:
                ctx = textmap.context_text(tokenizer, r["input_ids"], pidx, cfg.context_n, cfg.context_n)
                f.write(f"- t={pidx} (z={zv:.2f}) 词「{textmap.peak_label(cot, r['offsets'], pidx)}」上下文：…{ctx}…\n")
            f.write("\n")

    # ---- 结论 ----
    print("\n" + "=" * 62)
    print(f"模型：{cfg.model_name}")
    print(f"轨迹数：{len(results)}")
    print(f"平均 shift–surprisal Pearson 相关（关注层，主报告）：{avg_mid:.3f}")
    print(f"平均 shift–surprisal Pearson 相关（全层，对照）：      {avg_full:.3f}")
    print("=" * 62)
    if avg_mid != avg_mid:
        print("结论：相关度为 NaN（有效数据不足）。")
    elif avg_mid > 0.8:
        print("结论：相关度 > 0.8 —— 激活突变信号基本是 surprisal 的翻版，价值存疑。")
    elif avg_mid < 0.5:
        print("结论：相关度 < 0.5 —— 可能是独立信号，值得继续。")
    else:
        print("结论：相关度介于 0.5~0.8 —— 与 surprisal 中等相关，需进一步区分。")
    print(f"\n详细结果：{cfg.output_dir}/summary.json 、 {cfg.output_dir}/annotated.md")


if __name__ == "__main__":
    main()
