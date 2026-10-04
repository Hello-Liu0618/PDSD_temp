#!/usr/bin/env python3
"""按 judge 校验结果把数据分成 clean / flagged，供训练与评测使用。

为什么要做：参考答案若错，教师（q_plus）信号就是错的 → 标签噪声直接进蒸馏；
测试集若含错答案，会把答对的学生判错。所以**正确性**过滤对训练与测试都要做。

注意区分两种过滤：
  * 正确性过滤（本脚本）：按 `verify` 字段，**训练与测试都做**；
  * 未闭合过滤：**训练时逐 rollout 动态做**，测试集不做（见 TRAINER_SPEC §2）。

用法：
    python filter_verified.py --data data/generated_1000.jsonl
    python filter_verified.py --data ... --keep-unknown   # 把 unknown 也当 clean
输出：<stem>.clean.jsonl / <stem>.flagged.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="按 judge 校验结果切分 clean/flagged")
    ap.add_argument("--data", required=True)
    ap.add_argument("--keep-unknown", action="store_true",
                    help="把 verify=unknown（裁判没给出判定）也算作 clean；默认归入 flagged")
    ap.add_argument("--show", type=int, default=8, help="打印前 N 条 flagged 样例")
    args = ap.parse_args()

    path = Path(args.data)
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not recs:
        raise SystemExit(f"{path} 为空")

    dist = Counter(r.get("verify", "<缺字段>") for r in recs)
    print(f"共 {len(recs)} 条 | verify 分布: {dict(dist)}")

    n_missing = dist.get("<缺字段>", 0)
    if n_missing == len(recs):
        raise SystemExit(
            f"{path} 中所有记录都没有 verify 字段——可能是旧版生成的数据。\n"
            f"请先跑 reverify.py 补校验，或确认数据来源。")

    ok_values = {"ok", "unknown"} if args.keep_unknown else {"ok"}
    clean = [r for r in recs if r.get("verify") in ok_values]
    flagged = [r for r in recs if r.get("verify") not in ok_values]
    if not clean:
        raise SystemExit(
            f"clean 为空（flagged {len(flagged)}/{len(recs)}）——过滤后没有可用数据。"
            f"检查 {path} 的 verify 字段，或用 --keep-unknown。")

    def dump(p: Path, rows) -> None:
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    clean_path = path.with_name(path.stem + ".clean.jsonl")
    flag_path = path.with_name(path.stem + ".flagged.jsonl")
    dump(clean_path, clean)
    if flagged:
        dump(flag_path, flagged)

    print(f"clean  {len(clean):4} 条 -> {clean_path}")
    if flagged:
        print(f"flagged {len(flagged):4} 条 -> {flag_path}  (占比 {len(flagged)/len(recs):.1%})")
        print("\n--- flagged 样例（人工抽查用）---")
        for r in flagged[:args.show]:
            print(f"  [{r.get('topic','?')}/{r.get('subfield','?')}] verify={r.get('verify')}")
            print(f"    Q  : {r.get('problem','')[:90]}")
            print(f"    ans: {r.get('answer','')!r}")
            print(f"    why: {(r.get('verify_note') or '')[:120]}")
    else:
        print("无 flagged 记录。")


if __name__ == "__main__":
    main()
