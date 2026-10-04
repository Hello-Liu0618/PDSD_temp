#!/usr/bin/env python3
"""Phase 1 管线：跑通数据 + 扫 PDSD 超参数 + 观察难度。

对每条种子数据：
  1. 学生无参考生成日语 rollout；
  2. q_plus 前向（含英文参考 + output_hidden_states=True）；
  3. 在 completion 位置的隐状态上算 activation-shift；
  4. 报告：学生 rollout（看难度/是否正确/是否 rambling）、尖峰（位置+上下文）、可分性指标。

用法（可命令行扫超参）：
    python run_phase1.py --window-k 5 --layer-lo-frac 0.1 --layer-hi-frac 0.9 --vec-dim 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from seed_builder import load_model, build_student_prompt, build_teacher_prompt
from pivot.activation import compute_layer_profile, flatten_profile, select_layer_range
from pivot.shift import l2_normalize_cols, sliding_window_shift
from pivot.peaks import find_peaks
from pivot.separability import summarize


@torch.no_grad()
def generate_completion(model, tokenizer, prompt, max_new_tokens, temperature, top_p, top_k):
    """学生生成一次 rollout，返回 (full_ids, prompt_len)。"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gcfg = model.generation_config
    gcfg.do_sample = (temperature > 0)
    gcfg.temperature = (temperature if temperature > 0 else None)
    gcfg.top_p = (top_p if temperature > 0 else None)
    gcfg.top_k = (top_k if temperature > 0 else None)
    gcfg.max_new_tokens = max_new_tokens
    gcfg.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    out = model.generate(**inputs)
    return out, inputs["input_ids"].shape[1]


@torch.no_grad()
def compute_shift(model, tokenizer, teacher_prompt, completion_ids,
                  vec_dim, method, layer_lo_frac, layer_hi_frac, window_k):
    """在 [teacher_prompt + completion] 的 q_plus 前向上算 activation-shift（只算 completion 位置）。"""
    device = next(model.parameters()).device
    teacher_ids = tokenizer(teacher_prompt, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([teacher_ids, completion_ids], dim=1)
    out = model(input_ids=full_ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple，长度 num_layers+1，每个 [1, T, D]
    teacher_len = teacher_ids.shape[1]
    comp_hs = tuple(h[:, teacher_len:, :] for h in hs)  # 只取 completion 位置
    P = compute_layer_profile(comp_hs, vec_dim, method)  # [num_layers, vec_dim, T_gen]
    P_mid, _, _ = select_layer_range(P, layer_lo_frac, layer_hi_frac)
    Pn = l2_normalize_cols(flatten_profile(P_mid))
    shift, (lo, hi) = sliding_window_shift(Pn, window_k)
    return shift, (lo, hi)


def _fmt(x, nd=2):
    return "nan" if x != x else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 管线：跑数据 + 扫超参 + 看难度")
    ap.add_argument("--data-path", default="data/seed_ja.jsonl")
    ap.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--window-k", type=int, default=5)
    ap.add_argument("--layer-lo-frac", type=float, default=0.10)
    ap.add_argument("--layer-hi-frac", type=float, default=0.90)
    ap.add_argument("--vec-dim", type=int, default=16)
    ap.add_argument("--vec-method", default="rms_chunk")
    ap.add_argument("--z-threshold", type=float, default=1.0)
    ap.add_argument("--min-peak-sep", type=int, default=3)
    ap.add_argument("--extend-before", type=int, default=2, help="报告枢轴段时向前（更早 token）遮盖位数，建议 ≤2")
    ap.add_argument("--extend-after", type=int, default=4, help="报告枢轴段时向后（更晚 token）遮盖位数，盖内容")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="采样种子：扫超参时固定 seed，保证 rollout 相同")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    records = [json.loads(l) for l in Path(args.data_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.num_samples is not None:
        records = records[: args.num_samples]

    print(f"加载模型 {args.model_name} ...")
    model, tokenizer = load_model(args.model_name, torch.bfloat16)

    for i, rec in enumerate(records):
        problem_en, problem_ja, solution = rec["problem"], rec["problem_ja"], rec["solution"]

        # 1. 学生 rollout
        student_prompt = build_student_prompt(tokenizer, problem_ja)
        full_ids, prompt_len = generate_completion(
            model, tokenizer, student_prompt,
            args.max_new_tokens, args.temperature, args.top_p, args.top_k,
        )
        completion_ids = full_ids[:, prompt_len:]
        n = completion_ids.shape[1]
        completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True)

        # 2. q_plus 前向 + shift
        teacher_prompt = build_teacher_prompt(tokenizer, problem_ja, problem_en, solution)
        shift, (tok_lo, tok_hi) = compute_shift(
            model, tokenizer, teacher_prompt, completion_ids,
            args.vec_dim, args.vec_method, args.layer_lo_frac, args.layer_hi_frac, args.window_k,
        )

        # 3. 尖峰 + 可分性
        peaks, _z = find_peaks(shift, args.z_threshold, args.window_k, args.min_peak_sep, 1e-4)
        sep = summarize(shift.cpu().numpy(), peaks, (tok_lo, tok_hi))

        # 4. 报告
        closed = "</think>" in completion_text
        print(f"\n{'=' * 70}\n例 {i}")
        print(f"[题·日] {problem_ja}")
        print(f"[题·英] {problem_en}")
        print(f"[参考] {solution}")
        print(f"[学生 rollout · {n} tokens · 思考{'已关闭' if closed else '未关闭(rambling)'}]")
        print(completion_text[:500] + ("…(截断)" if len(completion_text) > 500 else ""))
        print(f"[可分性] contrast={_fmt(sep.get('contrast'))} "
              f"prom_median={_fmt(sep.get('prom_median'), 4)} "
              f"kurtosis={_fmt(sep.get('excess_kurtosis'), 1)} "
              f"n_peaks={sep.get('n_peaks')}")
        print(f"[尖峰 top-{args.top_n}] (k={args.window_k}, layers=[{args.layer_lo_frac},{args.layer_hi_frac}), vec_dim={args.vec_dim})")
        if peaks:
            for pidx, zv in peaks[: args.top_n]:
                lo = max(0, pidx - args.extend_before)
                hi = min(n, pidx + args.extend_after + 1)
                ctx = tokenizer.decode(completion_ids[0][lo:hi], skip_special_tokens=True)
                print(f"  t={pidx:4d} z={zv:5.2f}  [{lo}:{hi}] …{ctx}…")
        else:
            print("  （无尖峰）")

    print("\n完成。")


if __name__ == "__main__":
    main()
