#!/usr/bin/env python3
"""尝试"救回"被 judge 标为不一致的条目：用更强模型重解 + 重判。

- 解答口误（如丢番图漏解）→ 重解后通常能通过；
- 题目本身矛盾（如"判别式为负"）→ 重解仍过不了，归入 stillbad，最终丢弃。

问题文本与日语翻译不变（题目没改），只重生成 solution/answer 并重判。

用法：
    python repair_flagged.py --data data/generated_1000.jsonl            # 读 <stem>.flagged.jsonl
    python repair_flagged.py --data data/generated_1000.jsonl --solve-model deepseek-reasoner
输出：<stem>.repaired.jsonl / <stem>.stillbad.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from generate_data_deepseek import extract_answer, generate_solution, verify_answer


def main() -> None:
    ap = argparse.ArgumentParser(description="重解+重判被标记的条目")
    ap.add_argument("--data", required=True, help="原始数据路径（用于定位 <stem>.flagged.jsonl）")
    ap.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    ap.add_argument("--solve-model", default="deepseek-reasoner", help="重解所用模型（更强）")
    ap.add_argument("--judge-model", default="deepseek-reasoner")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("需要 DEEPSEEK_API_KEY（或 --api-key）")

    path = Path(args.data)
    flag_path = path.with_name(path.stem + ".flagged.jsonl")
    if not flag_path.exists():
        raise SystemExit(f"找不到 {flag_path}（先跑 filter_verified.py）")
    recs = [json.loads(l) for l in flag_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"待修复 {len(recs)} 条 | 重解模型={args.solve_model}")

    client = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")
    repaired, stillbad = [], []
    for i, r in enumerate(recs, 1):
        try:
            solution, truncated = generate_solution(
                client, args.solve_model, r["problem"], max_tokens=args.max_tokens)
            answer = extract_answer(solution)
            if not answer:
                stillbad.append({**r, "_repair": "truncated" if truncated else "no_boxed"})
                print(f"  [{i}/{len(recs)}] 丢弃（{'截断' if truncated else '无boxed'}）")
                continue
            verify, note = verify_answer(client, args.judge_model, r["problem"], answer)
        except Exception as e:  # noqa: BLE001
            stillbad.append({**r, "_repair": f"error: {e}"})
            print(f"  [{i}/{len(recs)}] 出错: {e}")
            continue

        if verify == "ok":
            repaired.append({**r, "solution": solution, "answer": answer,
                             "verify": "ok", "verify_note": note, "_repaired": True})
            print(f"  [{i}/{len(recs)}] ✓ 已修复")
        else:
            stillbad.append({**r, "solution": solution, "answer": answer,
                             "verify": verify, "verify_note": note, "_repair": "still_bad"})
            print(f"  [{i}/{len(recs)}] ✗ 仍不合格: {note[:70]}")

    def dump(p: Path, rows) -> None:
        with open(p, "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    rep_path = path.with_name(path.stem + ".repaired.jsonl")
    bad_path = path.with_name(path.stem + ".stillbad.jsonl")
    if repaired:
        dump(rep_path, repaired)
    if stillbad:
        dump(bad_path, stillbad)
    print(f"\n修复成功 {len(repaired)} 条 -> {rep_path}")
    print(f"仍不合格（建议丢弃）{len(stillbad)} 条 -> {bad_path}")


if __name__ == "__main__":
    main()
