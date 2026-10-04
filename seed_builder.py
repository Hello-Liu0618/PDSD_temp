#!/usr/bin/env python3
"""交互式种子模板构造器（Phase 0 工具）。

用 Qwen3-1.7B 快速验证 / 构造"定理应用数学"种子模板。

两种视角：
  - 学生视角：直接输入题目，看 1.7B 是否认出并宣布定理；
  - 教师视角：`@ref <英文参考> | <题目>`，参考在前，看 1.7B 是否跟着参考走。

种子库：list / seed / set / del / save / load（存到同目录 seed_templates.json）。

运行（在 rp-opsd 环境里）：
    python seed_builder.py                         # 默认 Qwen/Qwen3-1.7B, bf16, 采样(1.1，对齐 RP-OPSD)
    python seed_builder.py --temperature 0         # greedy（确定性，但会卡重复死循环，慎用）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED_FILE = Path(__file__).resolve().parent / "seed_templates.json"

# 复用 RP-OPSD 的 prompt 构造，保证 seed_builder 与蒸馏实验的 prompt 完全一致
# 注意：用 append（而非 insert(0)），避免 RP-OPSD 的 pivot.py 模块遮蔽本项目的 pivot/ 包
_RP_OPSD_SRC = Path(__file__).resolve().parent / "RP-OPSD" / "RP-OPSD" / "src"
if str(_RP_OPSD_SRC) not in sys.path:
    sys.path.append(str(_RP_OPSD_SRC))
from prompt_template_utils import build_assistant_prefilled_prompt
from pivot_languages import get_config

# 主力 9 类定理（日语名 -> 英语名），作为默认种子库的骨架
DEFAULT_THEOREMS = {
    "解と係数の関係": "Vieta's formulas",
    "二項定理": "Binomial theorem",
    "因数定理": "Factor theorem",
    "鳩の巣原理": "Pigeonhole principle",
    "包除原理": "Inclusion-exclusion principle",
    "三平方の定理": "Pythagorean theorem",
    "余弦定理": "Law of cosines",
    "順列・組み合わせ": "Permutation / combination",
    "ベイズの定理": "Bayes' theorem",
}

HELP = """\
命令：
  直接输入题目                       → 学生视角：1.7B 自己解（RP-OPSD 学生 prompt）
  @ref <英文参考> | <英文题> | <日语题>  → 教师视角（q_plus，RP-OPSD 教师 prompt）
  list                               → 列出所有定理类 + 是否已有种子
  seed <定理名>                      → 显示该定理的种子
  set <定理名> | <日语题>            → 给某定理设置日语题（可继续 set 补 problem_en / solution）
  del <定理名>                       → 删除某定理的种子内容
  save / load                        → 保存 / 重新加载种子库
  help / quit                        → 帮助 / 退出
"""


# --------------------------------------------------------------------------- #
# 模型与生成
# --------------------------------------------------------------------------- #

def load_model(model_name: str, torch_dtype):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch_dtype, trust_remote_code=True, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def build_student_prompt(tokenizer, problem: str) -> str:
    """RP-OPSD 学生视角 prompt（与 data_collator 的学生构造完全一致，语言=JA）。"""
    cfg = get_config("JA")
    labels = cfg["labels"]
    message = f"{labels['problem_target']}: {problem}\n\n{cfg['student_instruction']}"
    return build_assistant_prefilled_prompt(
        tokenizer, [{"role": "user", "content": message}],
        enable_thinking=True, think_prefix=cfg["think_prefix"],
    )


def build_teacher_prompt(tokenizer, problem: str, problem_en: str, solution: str) -> str:
    """RP-OPSD 教师视角（q_plus）prompt（与 data_collator 的教师构造完全一致，语言=JA）。"""
    cfg = get_config("JA")
    labels = cfg["labels"]
    shared = f"{labels['problem_target']}: {problem}\n\n{labels['problem_english']}: {problem_en}"
    full = f"{shared}\n\n{labels['solution_english']}:\n{labels['ref_begin']}\n{solution}\n{labels['ref_end']}"
    message = f"{full}\n\n{cfg['transition_prompt']}\n{cfg['teacher_final_instruction']}"
    return build_assistant_prefilled_prompt(
        tokenizer, [{"role": "user", "content": message}],
        enable_thinking=True, think_prefix=cfg["think_prefix"],
    )


def split_thinking(text: str):
    """把 Qwen3 的 <think>...</think> 和最终答案分开。"""
    if "<think>" in text:
        if "</think>" in text:
            think = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
            answer = text.split("</think>", 1)[1].strip()
            return think, answer
        # 有 <think> 但没 </think>：思考未关闭（模型卡在思考阶段，一直 rambling）
        think = text.split("<think>", 1)[1].strip()
        return think, "（思考未关闭，未生成最终答案）"
    return "", text.strip()


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int,
             temperature: float, top_p: float, top_k: int) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # 直接改写模型的 generation_config，彻底覆盖 Qwen3 默认的 do_sample=True
    gcfg = model.generation_config
    gcfg.do_sample = (temperature > 0)
    gcfg.max_new_tokens = max_new_tokens
    gcfg.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    gcfg.temperature = (temperature if temperature > 0 else None)
    gcfg.top_p = (top_p if temperature > 0 else None)
    gcfg.top_k = (top_k if temperature > 0 else None)
    out = model.generate(**inputs)
    prompt_len = inputs["input_ids"].shape[1]
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # 从 <think> 开始截取（<think> 在 prompt 预填里，后面才是模型生成），否则思考会显示成空
    idx = full.find("<think>")
    if idx != -1:
        return full[idx:].strip()
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# 种子库
# --------------------------------------------------------------------------- #

def load_seeds() -> dict:
    if SEED_FILE.exists():
        data = json.loads(SEED_FILE.read_text(encoding="utf-8"))
        # 合并：保证默认骨架里的定理都在
        for name, en in DEFAULT_THEOREMS.items():
            data.setdefault(name, {"theorem_en": en, "problem": "", "problem_en": "", "solution": ""})
        return data
    return {name: {"theorem_en": en, "problem": "", "problem_en": "", "solution": ""}
            for name, en in DEFAULT_THEOREMS.items()}


def save_seeds(seeds: dict) -> None:
    SEED_FILE.write_text(json.dumps(seeds, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 主循环
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="交互式种子模板构造器")
    ap.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.1,
                    help="采样温度（RP-OPSD 默认 1.1）；0=greedy，会卡重复死循环，慎用")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"加载模型 {args.model_name} (dtype={args.dtype}) ...")
    model, tokenizer = load_model(args.model_name, dtype)
    seeds = load_seeds()
    print("模型就绪。输入 help 查看用法，list 查看种子库。\n")

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if line in ("quit", "q", "exit"):
            break
        if line in ("help", "h", "?"):
            print(HELP)
            continue
        if line == "":
            continue

        # ---- 种子库命令 ----
        if line == "list":
            for name, s in seeds.items():
                mark = "✓" if s.get("problem") else "·"
                print(f"  {mark} {name}（{s.get('theorem_en', '')}）")
            continue
        if line == "save":
            save_seeds(seeds)
            print(f"  已保存到 {SEED_FILE}")
            continue
        if line == "load":
            seeds = load_seeds()
            print("  已重新加载种子库。")
            continue
        if line.startswith("seed "):
            name = line[5:].strip()
            s = seeds.get(name)
            if s is None:
                print(f"  未知定理 {name!r}（用 list 查看）。")
                continue
            print(f"  [{name}] 日语题: {s.get('problem', '')}")
            print(f"  [{name}] 英文题: {s.get('problem_en', '')}")
            print(f"  [{name}] 英文参考解答: {s.get('solution', '')}")
            continue
        if line.startswith("del "):
            name = line[4:].strip()
            if name in seeds:
                seeds[name]["problem"] = seeds[name]["problem_en"] = seeds[name]["solution"] = ""
                print(f"  已清空 {name} 的种子内容。")
            else:
                print(f"  未知定理 {name!r}。")
            continue
        if line.startswith("set "):
            rest = line[4:].strip()
            if "|" not in rest:
                print("  用法：set <定理名> | <日语题>（可再 set <定理名> | en:<英文题> 或 sol:<英文参考>）")
                continue
            name, value = rest.split("|", 1)
            name, value = name.strip(), value.strip()
            if name not in seeds:
                seeds[name] = {"theorem_en": "", "problem": "", "problem_en": "", "solution": ""}
            if value.startswith("en:"):
                seeds[name]["problem_en"] = value[3:].strip()
                print(f"  已设置 {name} 的英文题。")
            elif value.startswith("sol:"):
                seeds[name]["solution"] = value[4:].strip()
                print(f"  已设置 {name} 的英文参考解答。")
            else:
                seeds[name]["problem"] = value
                print(f"  已设置 {name} 的日语题。用 save 保存。")
            continue

        # ---- 教师视角（q_plus）：@ref <英文参考> | <英文题> | <日语题> ----
        if line.startswith("@ref"):
            rest = line[4:].strip()
            parts = [p.strip() for p in rest.split("|")]
            if len(parts) != 3:
                print("  用法：@ref <英文参考> | <英文题> | <日语题>")
                continue
            solution, problem_en, problem = parts
            prompt = build_teacher_prompt(tokenizer, problem, problem_en, solution)
            out = generate(model, tokenizer, prompt, args.max_new_tokens,
                           args.temperature, args.top_p, args.top_k)
            think, answer = split_thinking(out)
            print(f"\n[教师视角 · q_plus]\n--- 思考 ---\n{think}\n--- 答案 ---\n{answer}\n")
            continue

        # ---- 学生视角：直接输入题目 ----
        prompt = build_student_prompt(tokenizer, line)
        out = generate(model, tokenizer, prompt, args.max_new_tokens,
                       args.temperature, args.top_p, args.top_k)
        think, answer = split_thinking(out)
        print(f"\n[学生视角]\n--- 思考 ---\n{think}\n--- 答案 ---\n{answer}\n")

    print("再见。")


if __name__ == "__main__":
    main()
