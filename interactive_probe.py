#!/usr/bin/env python3
"""交互式探针：输入一段文字，输出尖峰词用【】包裹后的文本，便于探索启发式灵感。

模型只加载一次，进入 REPL 后可反复输入文字、随时改超参数。

用法：
    python interactive_probe.py [--model-name Qwen/Qwen3-1.7B] [--vec-dim 16] [--z-threshold 1.5]

REPL 内：
  文字（单行）              → 输出【】标注文本
  <<< ... >>>（多行）       → 多行文字输入（粘贴长文本）
  @vec_dim=64 @z=2.0 文字   → 本次临时覆盖超参数再跑（不改全局）
  set <参数> <值>            → 持久改超参数（set vec_dim 16 / set z_threshold 1.5）
  params                     → 查看当前超参数
  help / quit / exit         → 帮助 / 退出
"""
from __future__ import annotations

import argparse
import os

from config import ProbeConfig, _str2bool
from pivot.loader import load_model_and_tokenizer
from pivot_probe import process_trajectory
from pivot import textmap

ALIASES = {"vec_dim": "layer_vec_dim", "z": "z_threshold", "k": "window_k",
           "method": "layer_vec_method", "seed": "layer_vec_seed"}

HELP = """\
命令：
  文字（单行）             → 输出【】标注文本
  <<< ... >>>（多行）      → 多行文字
  @vec_dim=64 @z=2.0 文字  → 本次临时覆盖超参数后再跑
  set <参数> <值>           → 持久改超参数
  params                    → 查看当前超参数
  help / quit / exit        → 帮助 / 退出

可 set/临时覆盖的参数：vec_dim、z_threshold、window_k、min_peak_sep、
layer_lo_frac、layer_hi_frac、layer_vec_method、layer_vec_seed、
span_extend、span_max_len、span_min_len、raw_span
"""


def _coerce(field_type, raw: str):
    if field_type is bool:
        return _str2bool(raw)
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    return raw


def parse_inline(line: str):
    """解析行首 @key=value 前缀，返回 (overrides: dict[str,str], rest_text: str)。"""
    overrides = {}
    text = line
    while text.startswith("@"):
        head, _, tail = text.partition(" ")
        tok = head[1:]
        if "=" in tok:
            k, _, v = tok.partition("=")
            overrides[k] = v
        text = tail.strip()
    return overrides, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--vec-dim", type=int, default=16)
    ap.add_argument("--z-threshold", type=float, default=1.5)
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    cfg = ProbeConfig(model_name=args.model_name, dtype=args.dtype,
                      layer_vec_dim=args.vec_dim, z_threshold=args.z_threshold)

    print(f"加载模型 {cfg.model_name} (dtype={cfg.dtype}) ...")
    model, tokenizer = load_model_and_tokenizer(cfg.model_name, cfg.torch_dtype, True)
    fields = ProbeConfig.__dataclass_fields__
    print("模型就绪。输入 help 查看用法，params 看当前超参数。\n")

    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()

        if line in ("quit", "q", "exit"):
            break
        if line in ("help", "h", "?"):
            print(HELP)
            continue
        if line == "params":
            print(f"  vec_dim(layer_vec_dim)={cfg.layer_vec_dim}  "
                  f"z_threshold={cfg.z_threshold}  window_k={cfg.window_k}  "
                  f"min_peak_sep={cfg.min_peak_sep}")
            print(f"  层范围=[{cfg.layer_lo_frac},{cfg.layer_hi_frac})  "
                  f"method={cfg.layer_vec_method}")
            print(f"  span_extend={cfg.span_extend}  span_max_len={cfg.span_max_len}  "
                  f"span_min_len={cfg.span_min_len}  raw_span={cfg.raw_span}")
            continue
        if line.startswith("set "):
            parts = line.split()
            if len(parts) != 3:
                print("用法：set <参数> <值>，如 set vec_dim 16")
                continue
            name = ALIASES.get(parts[1], parts[1])
            if name not in fields or not fields[name].init:
                print(f"未知参数 {parts[1]}。可 set：vec_dim, z_threshold, window_k, "
                      f"min_peak_sep, layer_lo_frac, layer_hi_frac, layer_vec_method, "
                      f"layer_vec_seed, span_extend, span_max_len, span_min_len, raw_span")
                continue
            try:
                val = _coerce(fields[name].type, parts[2])
                setattr(cfg, name, val)
                print(f"  已设置 {name} = {val}")
            except ValueError as e:
                print(f"  解析失败 {parts[2]!r}: {e}")
            continue
        if line == "":
            continue

        # 多行输入
        if line == "<<<":
            buf = []
            while True:
                l = input("... ")
                if l.strip() == ">>>":
                    break
                buf.append(l)
            text = "\n".join(buf)
            overrides = {}
        else:
            overrides, text = parse_inline(line)

        if not text.strip():
            continue

        # 临时覆盖（@key=value）
        saved = {}
        for k, v in overrides.items():
            name = ALIASES.get(k, k)
            if name in fields and fields[name].init:
                try:
                    saved[name] = getattr(cfg, name)
                    setattr(cfg, name, _coerce(fields[name].type, v))
                except ValueError:
                    pass

        try:
            r = process_trajectory(model, tokenizer, text, cfg)
            annotated = textmap.annotate(text, r["offsets"], [p for p, _ in r["peaks"]],
                                         cfg.span_extend, cfg.span_max_len, cfg.span_min_len,
                                         cfg.raw_span)
            print(f"\n[峰值 {len(r['peaks'])} 个 | corr_mid={r['corr_mid']:.3f}]\n")
            print(annotated)
            if r["peaks"]:
                print("\n尖峰清单：")
                for pidx, zv in r["peaks"]:
                    w = textmap.peak_label(text, r["offsets"], pidx)
                    print(f"  t={pidx} z={zv:.2f} 【{w}】")
            else:
                print("（未检测到尖峰——文本可能太短，或阈值过高）")
            print()
        except Exception as e:
            print(f"处理出错：{e}\n")
        finally:
            for name, old in saved.items():
                setattr(cfg, name, old)

    print("再见。")


if __name__ == "__main__":
    main()
