#!/usr/bin/env python3
"""从生成的全量数据里按大类分层划出测试集，其余作为训练池。

要点：
  * **不过滤**：测试集按原分布保留（含后续可能"未闭合"的难题），因为评测衡量真实能力。
  * 分层：每个大类抽相同数量，避免评测被大类的难易差异带偏。
  * 固定种子，可复现。
  * 过滤（未闭合）是**训练时逐 rollout 动态做**的，不在这里做。

用法：
    python make_split.py --data data/generated_1000.jsonl --n-test-per-class 20
输出：<stem>.test.jsonl / <stem>.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="按大类分层划分 train/test")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-test-per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    path = Path(args.data)
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_cls = defaultdict(list)
    for r in recs:
        by_cls[r["topic"]].append(r)

    rng = random.Random(args.seed)
    test, train = [], []
    for cls in sorted(by_cls):
        items = by_cls[cls][:]
        rng.shuffle(items)
        k = min(args.n_test_per_class, len(items))
        test += items[:k]
        train += items[k:]

    if not train:
        raise SystemExit(
            f"训练集为空：测试集把数据全拿走了（--n-test-per-class={args.n_test_per_class} 太大）。")
    short = [c for c in sorted(by_cls) if len(by_cls[c]) <= args.n_test_per_class]
    if short:
        print(f"[警告] 这些大类条数 <= n_test_per_class，其训练部分为空：{short}")

    def dump(p: Path, recs) -> None:
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    test_path = path.with_name(path.stem + ".test.jsonl")
    train_path = path.with_name(path.stem + ".train.jsonl")
    dump(test_path, test)
    dump(train_path, train)

    print(f"全量 {len(recs)} 条（大类分布 {dict(Counter(r['topic'] for r in recs))}）")
    print(f"测试集 {len(test)} 条 -> {test_path}  （每类 {args.n_test_per_class}）")
    print(f"训练池 {len(train)} 条 -> {train_path}")
    print(f"测试集大类分布：{dict(Counter(r['topic'] for r in test))}")


if __name__ == "__main__":
    main()
