#!/usr/bin/env python3
"""对已有数据**重跑 judge 校验**（不重新生成），用于修正校验模型/参数后重评。

用法（在已生成数据的基础上，省一次全量生成）：
    python reverify.py --data data/gen_v3_pilot.jsonl --verify-model deepseek-reasoner
输出：<stem>.reverified.jsonl + 统计
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from openai import OpenAI

from generate_data_deepseek import verify_answer   # 复用同一 judge


def main() -> None:
    ap = argparse.ArgumentParser(description="重跑 judge 校验")
    ap.add_argument("--data", required=True)
    ap.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    ap.add_argument("--verify-model", default="deepseek-reasoner")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("需要 DEEPSEEK_API_KEY（或 --api-key）")

    client = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")
    path = Path(args.data)
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        recs = recs[:args.limit]

    out = []
    for i, r in enumerate(recs):
        v, note = verify_answer(client, args.verify_model, r["problem"], r["answer"])
        r2 = dict(r)
        r2["verify"] = v
        r2["verify_note"] = note
        out.append(r2)
        mark = "" if v == "ok" else "  <<<"
        print(f"  [{i + 1}/{len(recs)}] {v:8}{mark}  ans={r['answer'][:40]!r}")
        if v != "ok" and note:
            print(f"        note: {note[:110]}")

    out_path = path.with_name(path.stem + ".reverified.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter(r["verify"] for r in out)
    print(f"\n-> {out_path}")
    print(f"judge: ok {c.get('ok', 0)} / mismatch {c.get('mismatch', 0)} / unknown {c.get('unknown', 0)}  共 {len(out)}")


if __name__ == "__main__":
    main()
