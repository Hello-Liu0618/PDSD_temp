#!/usr/bin/env python3
"""用 Qwen3-1.7B 给生成数据打**难度标签**（学生视角、日语题、无参考）。

为什么用 1.7B 定义难度：我们的用途就是让 1.7B 在日语上做这些题，"它能不能解"才是难度的真定义。

对每题采样 k 条 rollout（默认 temp=1.1，对齐训练操作点），写入：
  diff_closed  = 关闭思考的比例（未闭合 = 被循环/截断困住）
  diff_correct = 答案正确的比例
  diff_n       = 采样条数

随后可按区间裁剪，使难度贴近早期 pilot-18 的水平。

用法（rp-opsd 环境）：
    python label_difficulty.py --data data/gen_v2_pilot.jsonl --n-samples 3
输出：<stem>.labeled.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

# 必须在 import transformers（经 seed_builder）之前设置：huggingface_hub 在 import 时读取这些变量。
# 否则会去连 huggingface.co 并反复重试（国内不可达，白等一分多钟）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 本地模型已缓存时可 `export HF_HUB_OFFLINE=1` 跳过联网检查；云端首次跑**不要**设（需要下载模型）。

import torch  # noqa: E402

from seed_builder import load_model, build_student_prompt, generate, split_thinking  # noqa: E402
from eval_smoke import pred_answer, match  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="用 Qwen3-1.7B 给数据打难度标签")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="默认 4096，与早期 pilot-18 的评测口径一致（预算不同会让难度不可比）")
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--adapter", default="", help="LoRA adapter 路径（评训练后模型）；空则评底座")
    ap.add_argument("--out", default="", help="输出路径（默认 <data>.labeled.jsonl）")
    args = ap.parse_args()

    path = Path(args.data)
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        recs = recs[:args.limit]

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"加载 {args.model_name}，{len(recs)} 题 × {args.n_samples} 采样，temp={args.temperature} ...")
    model, tokenizer = load_model(args.model_name, dtype)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        print(f"已加载 LoRA adapter: {args.adapter}")

    out = []
    for i, r in enumerate(recs):
        problem = r.get("problem_ja") or r["problem"]
        gt = r.get("answer") or ""
        closed_n = correct_n = 0
        for _ in range(args.n_samples):
            raw = generate(model, tokenizer, build_student_prompt(tokenizer, problem),
                           args.max_new_tokens, args.temperature, args.top_p, args.top_k)
            closed_n += ("</think>" in raw)
            _think, final = split_thinking(raw)
            if match(gt, pred_answer(final)) != "none":
                correct_n += 1
        r2 = dict(r)
        r2["diff_closed"] = round(closed_n / args.n_samples, 3)
        r2["diff_correct"] = round(correct_n / args.n_samples, 3)
        r2["diff_n"] = args.n_samples
        out.append(r2)
        print(f"  [{i + 1}/{len(recs)}] closed={r2['diff_closed']} correct={r2['diff_correct']}"
              f"  {problem[:50]}")

    out_path = Path(args.out) if args.out else path.with_name(path.stem + ".labeled.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n-> {out_path}")
    for key in ("diff_closed", "diff_correct"):
        c = Counter(r[key] for r in out)
        print(f"  {key} 分布: {dict(sorted(c.items()))}")
    hard = sum(1 for r in out if r["diff_correct"] == 0)
    easy = sum(1 for r in out if r["diff_correct"] == 1)
    mid = len(out) - hard - easy
    print(f"  难度分层: 全对(易) {easy} | 部分对(中) {mid} | 全错(难) {hard}")


if __name__ == "__main__":
    main()
