#!/usr/bin/env python3
"""汇总评测结果：把 base / pdsd / rpopsd 三份 label_difficulty 输出并列对比。

用法：
    python summarize_eval.py outputs/eval_base.jsonl outputs/eval_pdsd.jsonl outputs/eval_rpopsd.jsonl
每份文件由 `label_difficulty.py` 产出（字段 diff_correct / diff_closed）。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(p: Path):
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(recs)
    acc = sum(r["diff_correct"] for r in recs) / n if n else 0.0
    clo = sum(r["diff_closed"] for r in recs) / n if n else 0.0
    by_topic = defaultdict(list)
    for r in recs:
        by_topic[r.get("topic", "?")].append(r["diff_correct"])
    return n, acc, clo, {t: sum(v) / len(v) for t, v in by_topic.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="评测结果并列对比")
    ap.add_argument("files", nargs="+", help="形如 base=outputs/eval_base.jsonl 或直接给路径")
    args = ap.parse_args()

    rows = []
    for spec in args.files:
        name, _, path = spec.partition("=")
        if not path:
            path, name = name, Path(name).stem
        p = Path(path)
        if not p.exists():
            print(f"[跳过] {p} 不存在")
            continue
        rows.append((name, *load(p)))

    if not rows:
        raise SystemExit("没有可汇总的文件")

    print("\n==== 评测对比（测试集，同协议）====")
    print(f"{'臂':<12}{'条数':>6}{'准确率':>10}{'闭合率':>10}")
    for name, n, acc, clo, _ in rows:
        print(f"{name:<12}{n:>6}{acc:>10.3f}{clo:>10.3f}")

    base = next((r for r in rows if r[0] == "base"), None)
    if base:
        print("\n---- 相对 base 的提升 Δ准确率 ----")
        for name, n, acc, clo, _ in rows:
            if name == "base":
                continue
            print(f"  {name:<12}{acc - base[2]:+.3f}")

    topics = sorted({t for _, _, _, _, bt in rows for t in bt})
    if topics:
        print("\n---- 分大类准确率 ----")
        print(f"{'大类':<26}" + "".join(f"{r[0]:>10}" for r in rows))
        for t in topics:
            print(f"{t:<26}" + "".join(f"{r[4].get(t, float('nan')):>10.3f}" for r in rows))


if __name__ == "__main__":
    main()
