#!/usr/bin/env python3
"""用 Qwen3-1.7B 在学生视角（日语题、无参考）上跑冒烟/小批数据，评判难度是否合适。

对每条：
  1. 构造 RP-OPSD 学生 prompt（日语题，无参考答案）；
  2. 生成 → 拆思考/答案 → 抽模型最终答案；
  3. 与标准答案比对（宽松：先 \boxed{}，再数值/分数归一化）；
  4. 统计：命中率、是否关闭 </think>、答案语言（日/英/中）、生成长度。

产出 outputs/eval_smoke/report.md（每条完整输出，供人工判断）与终端汇总表。

用法（在 rp-opsd 环境）：
    python eval_smoke.py                                  # 默认 data/pilot_v3.jsonl
    python eval_smoke.py --data data/seed_ja.jsonl
    python eval_smoke.py --temperature 1.1 --n-samples 3  # 对齐 RP-OPSD 采样
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# 必须在 import transformers 之前设置：huggingface_hub 在 import 时就读取这些变量
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 模型已缓存，直接离线，避免无谓的网络重试

import torch

from seed_builder import load_model, build_student_prompt, generate, split_thinking

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "outputs" / "eval_smoke"


# ---------------- 答案抽取与比对 ----------------

def _extract_boxed(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    i = idx + len("\\boxed{")
    depth, j = 1, i
    while j < len(text) and depth > 0:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return text[i:j - 1].strip() if depth == 0 else ""


def pred_answer(final: str) -> str:
    """抽模型最终答案：优先 \\boxed{}，否则取答案段最后一个数/分数。"""
    b = _extract_boxed(final)
    if b:
        return b
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+)?", final)
    return nums[-1].strip() if nums else ""


def _norm(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)  # \frac{a}{b}
    s = re.sub(r"\\[dt]?frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"(\1)/(\2)", s)  # \frac18 简写
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right|,|!|;|:)\s*", "", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    s = s.replace("{", "").replace("}", "").replace(" ", "").replace("\\", "")
    s = s.replace("，", ",").replace("、", ",")
    s = re.sub(r"^0+(\d)", r"\1", s)
    return s.lower()


def _as_number(s: str):
    s = s.strip().strip("()")
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:/\(?(-?\d+(?:\.\d+)?)\)?)?", s)
    if not m:
        return None
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) else 1.0
    return a / b if b != 0 else None


def match(gt: str, pred: str) -> str:
    if not gt or not pred:
        return "none"
    ng, np_ = _norm(gt), _norm(pred)
    if ng == np_:
        return "exact"
    g, p = _as_number(ng), _as_number(np_)
    if g is not None and p is not None and abs(g - p) < 1e-6:
        return "num"
    # 多答案（如 "4 and 8"）：都出现即算命中
    if "," in ng or "and" in ng:
        parts = [x for x in re.split(r",|and", ng) if x]
        if parts and all(x in np_ for x in parts):
            return "parts"
    return "none"


def lang_of(s: str) -> str:
    if not s:
        return "?"
    ja = len(re.findall(r"[\u3040-\u30ff]", s))
    zh = len(re.findall(r"[\u4e00-\u9fff]", s))
    en = len(re.findall(r"[A-Za-z]", s))
    if ja > 0 and ja >= en // 2:
        return "ja"
    if zh > en:
        return "zh"
    return "en"


# ---------------- 主流程 ----------------

@torch.no_grad()
def run_one(model, tokenizer, problem, args):
    prompt = build_student_prompt(tokenizer, problem)
    outs = []
    for _ in range(args.n_samples):
        raw = generate(model, tokenizer, prompt, args.max_new_tokens,
                       args.temperature, args.top_p, args.top_k)
        think, final = split_thinking(raw)
        closed = ("</think>" in raw)
        outs.append({"think": think, "final": final, "closed": closed})
    return outs


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3-1.7B 冒烟评测（难度判断）")
    ap.add_argument("--data", default="data/pilot_v3.jsonl")
    ap.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="默认 0.6（Qwen3 thinking 推荐）；1.1 对齐 RP-OPSD")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        recs = recs[:args.limit]

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"加载模型 {args.model_name} ({args.dtype})，共 {len(recs)} 条，n_samples={args.n_samples} ...")
    model, tokenizer = load_model(args.model_name, dtype)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = [f"# Qwen3-1.7B 难度评测\n",
          f"- 数据：`{args.data}`（{len(recs)} 条）",
          f"- 采样：temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, "
          f"max_new_tokens={args.max_new_tokens}, n={args.n_samples}\n"]
    n_hit = n_closed = hits_closed = 0
    rows = []
    for i, r in enumerate(recs):
        problem = r.get("problem_ja") or r["problem"]
        gt = r.get("answer") or _extract_boxed(r.get("solution", ""))
        outs = run_one(model, tokenizer, problem, args)
        best = "none"
        detail = []
        for o in outs:
            pred = pred_answer(o["final"])
            mt = match(gt, pred)
            o["pred"], o["match"] = pred, mt
            if mt != "none":
                best = mt
            detail.append(f"{pred!r}({mt})")
        ok = best != "none"
        n_hit += ok
        closed = sum(o["closed"] for o in outs)
        any_closed = closed > 0
        n_closed += any_closed
        hits_closed += (ok and any_closed)
        rows.append((i, r.get("topic", ""), gt, detail, ok))
        tag = "" if any_closed else "  ←思考未关闭(可能被截断)"
        print(f"  [{i:2}/{len(recs)}] {'✓' if ok else '✗'} gt={gt!r} pred={detail} closed={closed}/{len(outs)}{tag}")

        md.append(f"\n## {i}. [{r.get('topic','')}] {'✓ 命中' if ok else '✗ 未命中'}")
        md.append(f"**题（日）**：{problem}")
        md.append(f"**题（英）**：{r.get('problem','')}")
        md.append(f"**标准答案**：`{gt}`")
        for k, o in enumerate(outs):
            md.append(f"\n**样本 {k}**：预测 `{o['pred']}` → {o['match']}；关闭思考={o['closed']}")
            md.append(f"\n<details><summary>思考</summary>\n\n{o['think']}\n\n</details>")
            md.append(f"\n**最终答案段**：\n```\n{o['final']}\n```")

    (OUT_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")
    n = len(recs)
    print("\n==== 汇总 ====")
    print(f"关闭思考(给出答案)：{n_closed}/{n} = {n_closed/n:.0%}")
    print(f"总命中：{n_hit}/{n} = {n_hit/n:.0%}")
    if n_closed:
        print(f"在关闭思考的 {n_closed} 条中命中：{hits_closed}/{n_closed} = {hits_closed/n_closed:.0%}")
    print(f"报告：{OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
