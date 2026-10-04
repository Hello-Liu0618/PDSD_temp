#!/usr/bin/env python3
"""现场生成逻辑推理 CoT（ProofWriter 规则演绎），只留答案正确的轨迹。

流程：加载 ProofWriter（按 depth 过滤）-> Qwen 生成逐步逻辑演绎 -> 抽取
True/False/Unknown 与 gold 比对 -> 只留正确 -> 存 JSONL（{"problem","cot","answer","gold","depth"}）。

用法：
    python generate_logic_data.py --depth depth-3 --target-correct 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

import torch
from datasets import load_dataset

from config import DataGenConfig
from pivot.loader import load_model_and_tokenizer
from generate_data import generate, clean_cot

SYSTEM = (
    "You are a careful logical reasoner. Given a set of facts and rules, determine "
    "whether a statement is True, False, or Unknown (insufficient information). "
    "Reason step by step, making each deduction explicit. End your solution with a "
    "final answer in the format '#### True', '#### False', or '#### Unknown'."
)

_ANS_RE = re.compile(r"####\s*(True|False|Unknown)", re.IGNORECASE)
_ANS_FALLBACK = re.compile(
    r"(?:the\s+)?(?:answer|conclusion)\s*(?:is|=|:)\s*(True|False|Unknown)", re.IGNORECASE)


def build_prompt(tokenizer, theory: str, question: str) -> str:
    user = (f"Facts and rules:\n{theory}\n\n"
            f"Question: {question}\n\n"
            f"Determine whether the statement is True, False, or Unknown.")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_answer(text: str):
    """抽取最终答案 True/False/Unknown。优先 '#### X'，其次 'answer is X'。"""
    m = _ANS_RE.search(text)
    if m:
        return m.group(1).capitalize()
    m = _ANS_FALLBACK.search(text)
    if m:
        return m.group(1).capitalize()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tasksource/proofwriter")
    ap.add_argument("--split", default="train")
    ap.add_argument("--depth", default="depth-3")
    ap.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--num-questions", type=int, default=120)
    ap.add_argument("--target-correct", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-path", default="data/logic_cot.jsonl")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = DataGenConfig(model_name=args.model_name, dtype=args.dtype,
                        max_new_tokens=args.max_new_tokens)

    print(f"[1/3] 加载模型 {args.model_name} ...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, cfg.torch_dtype, True)

    print(f"[2/3] 加载数据集 {args.dataset}/{args.split} (depth={args.depth}) ...")
    ds = load_dataset(args.dataset, split=args.split, trust_remote_code=True)
    ds = ds.filter(lambda ex: ex["config"] == args.depth).shuffle(seed=args.seed)
    print(f"  depth={args.depth} 共 {len(ds)} 条")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    kept = attempted = correct = 0
    out_f = open(args.output_path, "w", encoding="utf-8")

    for i, ex in enumerate(ds):
        if i >= args.num_questions or kept >= args.target_correct:
            break
        gold = str(ex["answer"]).capitalize()
        if gold not in ("True", "False", "Unknown"):
            continue
        attempted += 1
        prompt = build_prompt(tokenizer, ex["theory"], ex["question"])
        cot_raw = generate(model, tokenizer, prompt, cfg)
        pred = extract_answer(cot_raw)
        ok = (pred == gold)
        if ok:
            correct += 1
            rec = {
                "problem": f"Facts/rules:\n{ex['theory']}\n\nQuestion: {ex['question']}",
                "cot": clean_cot(cot_raw),
                "answer": pred, "gold": gold, "depth": args.depth,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            kept += 1
        print(f"  [{attempted}] correct={ok} (pred={pred} gold={gold}) kept={kept}")

    out_f.close()
    print(f"[3/3] 完成：attempted={attempted} correct={correct} kept={kept} -> {args.output_path}")


if __name__ == "__main__":
    main()
