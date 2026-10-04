#!/usr/bin/env python3
"""现场生成英文 CoT 轨迹（只做数据，不做下游）。

流程：加载 GSM8K 题目 -> Qwen 生成英文逐步推理 -> 抽取最终答案与 gold 比对
-> 只保留正确轨迹 -> 存 JSONL（{"problem","cot","answer","gold"}）。

用法：
    python generate_data.py
    python generate_data.py --target-correct 40 --num-questions 80 --do-sample true
"""
from __future__ import annotations

import json
import os
import re
import random

import torch
from datasets import load_dataset

from config import DataGenConfig, build_arg_parser, apply_cli
from pivot.loader import load_model_and_tokenizer, model_device

SYSTEM = (
    "You are a careful math problem solver. Solve the problem step by step, "
    "showing your full reasoning. End your solution with a final answer in the "
    "format: '#### <number>'."
)

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _strip_commas(text: str) -> str:
    # 去掉千分位逗号（避免 "3,500" 被拆成 3 和 500）
    return text.replace(",", "")


def extract_final_answer(text: str):
    """抽取生成文本中的最终数值答案（取最后一个数字）。无则返回 None。"""
    nums = _NUM_RE.findall(_strip_commas(text))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def gold_answer(answer_field: str):
    """GSM8K 的 answer 字段形如 '... #### 18'，取 '####' 后的数值。"""
    tail = answer_field.split("####")[-1] if "####" in answer_field else answer_field
    nums = _NUM_RE.findall(_strip_commas(tail))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def clean_cot(text: str) -> str:
    """抽取 <think>...</think> 之间的纯推理内容，剔除格式符号。

    Qwen3 默认开 thinking 模式，生成形如 "<think>…推理…</think>\n\n#### 72"。
    ＜think＞/＜/think＞/#### 属于"格式符号"，正是任务要区分于推理枢轴的表层文本，
    故只保留标签内的推理轨迹。

    三种情况：
    1) 完整 <think>...</think> -> 取标签内内容；
    2) 只有 <think> 没有 </think>（生成被 max_new_tokens 截断）-> 剥掉开头的 <think>；
    3) 无标签 -> 原样返回。
    """
    text = text.strip()
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    if text.startswith("<think>"):
        return text[len("<think>"):].strip()
    return text


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, cfg: DataGenConfig) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device(model))
    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if cfg.do_sample:
        gen_kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)
    out = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


def main() -> None:
    cfg = DataGenConfig()
    args = build_arg_parser(DataGenConfig).parse_args()
    apply_cli(cfg, args)

    os.environ.setdefault("HF_ENDPOINT", cfg.hf_endpoint)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"[1/3] 加载模型 {cfg.model_name} (dtype={cfg.dtype}) ...")
    model, tokenizer = load_model_and_tokenizer(cfg.model_name, cfg.torch_dtype, cfg.trust_remote_code)

    print(f"[2/3] 加载数据集 {cfg.dataset_name}/{cfg.dataset_config}/{cfg.dataset_split} ...")
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
    n_total = min(cfg.start_index + cfg.num_questions, len(ds))

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    kept = attempted = correct = 0
    out_f = open(cfg.output_path, "w", encoding="utf-8")

    for i in range(cfg.start_index, n_total):
        if kept >= cfg.target_correct:
            break
        ex = ds[i]
        gold = gold_answer(ex["answer"])
        if gold is None:
            continue
        attempted += 1
        prompt = build_prompt(tokenizer, ex["question"])
        cot_raw = generate(model, tokenizer, prompt, cfg)
        pred = extract_final_answer(cot_raw)
        ok = (pred is not None) and (abs(pred - gold) <= 1e-6)
        if ok:
            correct += 1
            rec = {"problem": ex["question"], "cot": clean_cot(cot_raw),
                   "answer": pred, "gold": gold}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            kept += 1
        print(f"  [{attempted}] correct={ok} (pred={pred} gold={gold}) kept={kept}")

    out_f.close()
    print(f"[3/3] 完成：attempted={attempted} correct={correct} kept={kept} -> {cfg.output_path}")


if __name__ == "__main__":
    main()
