#!/usr/bin/env python3
"""用 DeepSeek API 批量生成多步推理数学数据（英文题 + 日语翻译 + 英文参考解答 + 答案 + 校验）。

设计（2026-10-04 定稿）：
  * 目标是"非纯计算"的数学题：每题必须含一个非平凡的方法/定理/表示**选择点**。
  * **两级题材表 + 题型维度**：6 大类 × 4-6 子领域，交叉 5 种题型（不含证明题——难以自动判分）；
    给每个格子设配额，逐格生成 → 默认就分散，不靠模型自觉。
  * **难度校准**：prompt 要求"1.7B 可 2-4 步解出、答案须为单个数字或短闭式、不设长枚举"。
  * **批量出题**（每批 batch_size 条，默认 8）+ **随机指纹银行回灌**（每次从全覆盖清单里随机抽 ~40 条
    喂回 prompt：prompt 长度恒定，历次并集覆盖全库）。
  * **三层去重**：整句哈希 → 结构哈希(数字→N) → **场景指纹**（首句内容词集合）+ Jaccard≥0.6。
  * 参考解为**富含推理的自然语言**，显式宣布所用定理/规则，以 \\boxed{} 收尾；要求简洁（~100-200 词）。
    被截断（=题太难/模型绕圈）**立即放弃、不重试**（省时且筛掉过难题），非截断缺 box 才重试一次。
  * **答案 judge 校验**：给定题目 + 候选答案让模型判定对错（`--verify-model` 可换成更强模型），
    不一致的标 `verify="mismatch"` 并记 `verify_note`（不剔除，供人工抽查）。
  * 缺 \\boxed{} 的记录剔除到 `<output>.noanswer.jsonl`；同级写 `<output>.meta.json` 记录生成元信息。

字段：problem / problem_ja / solution / answer / topic(大类) / subfield / ptype / verify / verify_note

用法：
    export DEEPSEEK_API_KEY=sk-xxx
    python generate_data_deepseek.py --n-total 30            # 小样验证
    python generate_data_deepseek.py --n-total 1000 --output data/generated_1000.jsonl

依赖：pip install openai
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import List

from openai import OpenAI

BASE = Path(__file__).resolve().parent

# ---------------- 题材表（大类 → 子领域） ----------------

TAXONOMY = {
    "algebra": ("代数", [
        "linear equations", "systems of equations", "quadratics", "inequalities",
        "sequences and recurrences", "functions and graphs",
    ]),
    "geometry": ("幾何", [
        "triangles", "circles", "polygons", "coordinate geometry",
        "similarity and congruence", "solid geometry",
    ]),
    "combinatorics": ("組合せ", [
        "permutations and combinations", "counting and inclusion-exclusion",
        "pigeonhole principle", "elementary graph theory", "recursive counting",
    ]),
    "probability": ("確率", [
        "classical probability", "conditional probability and Bayes",
        "expectation", "geometric probability",
    ]),
    "number theory": ("整数論", [
        "divisibility and remainders", "congruences", "prime factorization",
        "divisors and their sums", "Diophantine equations",
    ]),
    "logic and applications": ("応用と論理", [
        "rate and work problems", "ratio and mixture problems",
        "logical deduction", "sequential reasoning puzzles",
    ]),
}

PROBLEM_TYPES = [
    "find a value", "count the number of ways", "find all possibilities",
    "optimize a quantity", "reason backwards from a target",
]

# 场景指纹用的停用词
_STOP = set(
    "a an the of to and or in on at by for with is are was were be been being "
    "it its this that these those what how many much if then than as from each per "
    "one two three four five six seven eight nine ten find give calculate determine "
    "compute total sum value number answer".split()
)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _structure_hash(s: str) -> str:
    return re.sub(r"\d+", "N", _norm(s))


def _scenario_fp(s: str) -> frozenset:
    """场景指纹：首句内容词集合（数字→N，去停用词）。情境通常在第一句给出。"""
    first = re.split(r"(?<=[.?!])\s+", s.strip())[0].lower()
    first = re.sub(r"\$?\d+(?:[.,]\d+)*", "N", first)
    return frozenset(w for w in re.findall(r"[a-z]+", first) if w not in _STOP and len(w) > 2)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------- API ----------------

def chat_full(client, model, messages, temperature=0.0, max_tokens=1024):
    """返回 (content, finish_reason)。finish_reason=='length' 表示被 max_tokens 截断。"""
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            )
            ch = r.choices[0]
            return ch.message.content.strip(), getattr(ch, "finish_reason", None)
        except Exception as e:  # noqa: BLE001
            print(f"    [retry {attempt + 1}] {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("DeepSeek API 调用多次失败")


def chat(client, model, messages, temperature=0.0, max_tokens=1024) -> str:
    return chat_full(client, model, messages, temperature, max_tokens)[0]


# ---------------- 出题（批量 + 题材格子 + 指纹银行） ----------------

def generate_problem_batch(client, model, cls, subfield, ptype, n, bank_sample) -> str:
    bank = ""
    if bank_sample:
        bank = (
            "\n\nIMPORTANT — the following scenarios have ALREADY been used. "
            "Your new problems must NOT reuse these scenarios, setups, objects, or unknowns:\n"
            + "\n".join(bank_sample)
        )
    ja = TAXONOMY[cls][0]
    prompt = (
        f"Generate {n} DIFFERENT math word problems.\n"
        f"Category: {cls} ({ja}). Subtopic: {subfield}. Preferred problem type: {ptype}.\n"
        f"Requirements for EVERY problem:\n"
        f"- It must require a NON-TRIVIAL strategic step: a choice of method, theorem, or "
        f"representation (e.g. applying a named theorem, factorizing, setting up equations, "
        f"using symmetry, counting cases, working backwards). Pure numerical computation of a "
        f"given expression is NOT acceptable.\n"
        f"- CALIBRATED DIFFICULTY: a competent small model (Qwen3-1.7B) should be able to solve it "
        f"in 2-4 reasoning moves. Avoid problems that need long case analysis, exhaustive search, "
        f"or heavy computation.\n"
        f"- The final answer must be a SINGLE number or a short closed-form expression "
        f"(for 'find all', a short finite list of at most ~6 items). Do NOT design problems whose "
        f"answer is a long list or requires enumerating many cases.\n"
        f"- Self-contained and unambiguous.\n"
        f"- All {n} problems must be clearly DISTINCT from each other: vary the scenario/setting, "
        f"the unknown quantity, the method used, and the number of steps. "
        f"Do not produce paraphrases of one another."
        f"{bank}\n\n"
        f"Output format: number them EXACTLY as `### Problem 1`, `### Problem 2`, ... , each "
        f"followed by the problem text on the next line(s). English only, no solutions, nothing else."
    )
    return chat(client, model, [{"role": "user", "content": prompt}], temperature=0.9, max_tokens=2000)


# 行首各种编号/装饰（必须 ^ 锚定，否则会误删句中 "1000." "3." 等）
_LEAD = re.compile(r"^(?:\s*#+\s*|[-*]\s*|\*\*|Problem\s*\d+\s*[:.)]?\s*|\d+\s*[:.)]\s*)+", re.I)


def parse_batch(text: str) -> List[str]:
    """从 `### Problem i` 编号块切出题目；解析失败时回退空行分段，并统一剥掉行首前缀。"""
    parts = re.split(r"#+\s*Problem\s*\d+\s*[:.]?\s*", text)
    cands = [p.strip() for p in parts[1:] if p.strip()]
    if len(cands) < 2:
        cands = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    cleaned = []
    for c in cands:
        c = _LEAD.sub("", c)
        c = re.sub(r"\s*\n\s*", " ", c).strip()
        c = c.strip("*").strip()
        if len(c) > 20 and not c.endswith(":"):
            cleaned.append(c)
    return cleaned


# ---------------- 解答（自然语言推理 + \boxed{} + 截断重试） ----------------

def extract_answer(solution: str) -> str:
    """抽取最后一个 \\boxed{...} 的内容（花括号配对，支持 \\frac{a}{b} 嵌套）。"""
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return ""
    i = idx + len("\\boxed{")
    depth, j = 1, i
    while j < len(solution) and depth > 0:
        if solution[j] == "{":
            depth += 1
        elif solution[j] == "}":
            depth -= 1
        j += 1
    return solution[i:j - 1].strip() if depth == 0 else ""


def generate_solution(client, model, problem, max_tokens=1536, max_tries=2):
    """自然语言推理式解答，以 \\boxed{} 收尾。返回 (解答, 是否截断)。

    * **被截断**（模型绕圈/题太难）→ 立即放弃、**不重试**：重试大概率还是截断，纯浪费时间；
      且这类题通常已超出目标难度。
    * 非截断但没 \\boxed{}（格式问题）→ 最多再重试一次。
    """
    last, truncated = "", False
    for t in range(max_tries):
        extra = "" if t == 0 else (
            " IMPORTANT: keep it short and make sure the final answer is enclosed in \\boxed{...}.")
        prompt = (
            "Write a natural, fluent English solution to the following problem. "
            "Narrate your reasoning: explain WHAT you do at each stage and WHY, justifying the steps "
            "so a reader can follow the logic. "
            "Name the theorem, rule, or principle you apply whenever you rely on one "
            "(e.g., 'By the Pythagorean theorem, ...'). "
            "Be CONCISE (roughly 100-200 words): do NOT restate the problem, do NOT explore dead "
            "ends, and do NOT hedge or second-guess yourself. "
            "End with the final answer inside \\boxed{}."
            + extra
            + f"\n\nProblem:\n{problem}"
        )
        last, finish = chat_full(client, model, [{"role": "user", "content": prompt}],
                                 temperature=0.0, max_tokens=max_tokens)
        truncated = (finish == "length")
        if extract_answer(last):
            return last, truncated
        if truncated:
            break           # 截断 → 视为过难，放弃；不浪费重试
    return last, truncated


# ---------------- 答案 judge 校验 ----------------

def verify_answer(client, model, problem, answer, max_tokens=4096, fallback="deepseek-chat"):
    """judge 式校验：给定题目 + 候选答案，判定是否正确。返回 (verdict, 首行理由)。

    比"再解一次"更稳：对多答案、散文式答案不敏感，也不要求裁判自己会完整解题。
    deepseek-reasoner 偶有 content 为空的情况（temp=0 下重试同样为空），故空回复时换 fallback 模型再判一次。
    """
    prompt = (
        "You are checking a proposed final ANSWER to a math problem.\n\n"
        f"Problem:\n{problem}\n\nProposed answer:\n{answer}\n\n"
        "Is the proposed answer correct? Judge the mathematical content, not the formatting: "
        "answers may be written in different but equivalent forms (e.g. 1/2 = 0.5, a set listed "
        "in a different order, or an equivalent unsimplified expression).\n"
        "Reply with EXACTLY one word on the first line — CORRECT or INCORRECT — then a one-line reason."
    )
    msgs = [{"role": "user", "content": prompt}]
    out = chat(client, model, msgs, temperature=0.0, max_tokens=max_tokens).strip()
    if not out and model != fallback:
        out = chat(client, fallback, msgs, temperature=0.0, max_tokens=1024).strip()
    if not out:
        return "unknown", "(judge returned empty)"
    first = out.splitlines()[0].strip().upper()
    if "INCORRECT" in first or "NOT CORRECT" in first:
        verdict = "mismatch"
    elif "CORRECT" in first:
        verdict = "ok"
    else:
        verdict = "unknown"           # 无法判定（≠ 判错）
    return verdict, out.strip().replace("\n", " ")[:200]


# ---------------- 翻译 ----------------

_LEAK_KEYS = ("翻訳して", "訳して", "変更せず", "Translate the following", "自然な日本語に訳")


def clean_translation(ja: str) -> str:
    """模型偶尔会把翻译指令一起吐出来（或以『問題：』做前缀）——剥掉，只留题面。"""
    lines = [ln for ln in ja.splitlines() if not any(k in ln for k in _LEAK_KEYS)]
    ja = "\n".join(lines).strip()
    ja = re.sub(r"^(?:問題|Problem|Question)\s*[:：]\s*", "", ja)
    return ja.strip()


def translate(client, model, problem) -> str:
    prompt = (
        "Translate the following math problem into natural Japanese. "
        "Keep all mathematical symbols (x, ^2, =, \\boxed, etc.) unchanged; translate only the words, "
        "using standard Japanese math terminology. "
        "Output ONLY the Japanese translation — no preamble, no labels like '問題：', no commentary, "
        "and do NOT repeat these instructions."
        "\n\nProblem:\n" + problem
    )
    out = chat(client, model, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=384)
    return clean_translation(out)


# ---------------- 配额 ----------------

def build_targets(n_total: int):
    """把总量按 大类 → 子领域 均分，返回 [(cls, subfield, quota), ...]。"""
    classes = list(TAXONOMY)
    per_class = n_total / len(classes)
    targets = []
    for cls in classes:
        subs = TAXONOMY[cls][1]
        per_sub = per_class / len(subs)
        for s in subs:
            targets.append((cls, s, max(1, round(per_sub))))
    return targets


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek 批量生成多步推理数学数据")
    ap.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--verify-model", default="", help="judge 校验所用模型（默认同 --model；可设 deepseek-reasoner）")
    ap.add_argument("--n-total", type=int, default=1000, help="目标总条数（按题材格子均分）")
    ap.add_argument("--batch-size", type=int, default=8, help="每次 API 调用请求几条题")
    ap.add_argument("--bank-size", type=int, default=40, help="每次回灌给模型的场景清单条数")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--output", default=str(BASE / "data" / "generated.jsonl"))
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("需要 DEEPSEEK_API_KEY（或 --api-key）")
    rng = random.Random(args.seed)
    client = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")

    seen_exact, seen_struct = set(), set()
    seen_fps: List[frozenset] = []
    bank_lines: List[str] = []      # 全覆盖清单（回灌时随机抽样）
    n_dup = 0
    out, rejected = [], []

    def register(problem: str) -> bool:
        nonlocal n_dup
        h_exact, h_struct = _norm(problem), _structure_hash(problem)
        fp = _scenario_fp(problem)
        if h_exact in seen_exact or h_struct in seen_struct or fp in seen_fps:
            n_dup += 1
            return False
        if any(_jaccard(fp, prev) >= 0.6 for prev in seen_fps):
            n_dup += 1
            return False
        seen_exact.add(h_exact)
        seen_struct.add(h_struct)
        seen_fps.append(fp)
        return True

    targets = build_targets(args.n_total)
    print(f"目标 {args.n_total} 条，共 {len(targets)} 个（大类×子领域）格子。")
    ptype_i = 0     # 全局题型计数器：跨格子轮转，避免小 n 时每格都从同一题型开始
    for cls, subfield, quota in targets:
        print(f"== {cls} / {subfield}  (目标 {quota}) ==")
        kept = attempts = 0
        while kept < quota and attempts < quota * 3 + 5:
            attempts += 1
            ptype = PROBLEM_TYPES[ptype_i % len(PROBLEM_TYPES)]
            ptype_i += 1
            want = min(args.batch_size, quota - kept)
            bank = rng.sample(bank_lines, min(args.bank_size, len(bank_lines)))
            try:
                batch = parse_batch(generate_problem_batch(
                    client, args.model, cls, subfield, ptype, want, bank))
            except Exception as e:  # noqa: BLE001
                print(f"  [出题出错] {e}")
                continue
            for problem in batch:
                if kept >= quota:
                    break
                if not register(problem):
                    continue
                try:
                    solution, truncated = generate_solution(client, args.model, problem)
                    answer = extract_answer(solution)
                    if not answer:
                        rejected.append({"problem": problem, "solution": solution,
                                         "_note": "truncated" if truncated else "no_boxed"})
                        print("    [丢弃:无boxed]")
                        continue
                    verify, verify_note = verify_answer(
                        client, args.verify_model or args.model, problem, answer)
                    problem_ja = translate(client, args.model, problem)
                except Exception as e:  # noqa: BLE001
                    print(f"    [出错] {e}")
                    continue
                out.append({"problem": problem, "problem_ja": problem_ja, "solution": solution,
                            "answer": answer, "topic": cls, "subfield": subfield,
                            "ptype": ptype, "verify": verify, "verify_note": verify_note})
                bank_lines.append(f"- [{cls}/{subfield}] {problem[:80]}")
                kept += 1
                flag = "" if verify == "ok" else "  <校验不一致!>"
                print(f"    [{kept}/{quota}] {problem[:60]}{flag}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, recs) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dump(out_path, out)
    meta = {"n": len(out), "n_total_requested": args.n_total, "model": args.model,
            "seed": args.seed, "batch_size": args.batch_size, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "taxonomy": {k: v[1] for k, v in TAXONOMY.items()}, "problem_types": PROBLEM_TYPES,
            "note": "生成时 DeepSeek 服务端采样不可完全复现；此 meta 记录批次与参数。"}
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    vc = Counter(r["verify"] for r in out)
    print(f"\n完成：{len(out)} 条 -> {out_path}")
    print(f"去重跳过 {n_dup} 条；缺 boxed 剔除 {len(rejected)} 条；"
          f"judge 校验: ok {vc.get('ok', 0)} / mismatch {vc.get('mismatch', 0)} / unknown {vc.get('unknown', 0)}")
    if rejected:
        rej_path = out_path.with_name(out_path.stem + ".noanswer.jsonl")
        dump(rej_path, rejected)
        print(f"缺 \\boxed{{}} 记录 -> {rej_path}")


if __name__ == "__main__":
    main()
