"""Offline pivot-finding tool for RP-OPSD (no training / no optimization step).

Given the translated training JSON and a base model (optionally + a trained LoRA
checkpoint), this tool reproduces the *pivot-finding* half of RP-OPSD exactly:

  1. Build the three prompts -- student, solution-conditioned teacher (``q_plus``)
     and ablation teacher (``q_minus``) -- exactly as
     ``RPOPSDSelfDistillationDataCollator`` does, reusing the same constants and
     prompt builder so the text is byte-identical.
  2. Sample ONE on-policy completion from the student prompt, using the same
     generation config as training (temperature / top_p / top_k / max_new_tokens).
  3. Append that same completion to all three views and compute the per-token pivot
     score ``KL(q_plus || q_minus)`` (``pivot.compute_pivot_score``).
  4. Normalize with the EMA gate (``pivot.PivotGate``) and emit, per token:
     its text, ``score``, ``z`` (z-score against the running EMA) and ``gate``, plus
     a step-level aggregation (split on newlines) so the pivot can be separated out
     of the chain-of-thought.

No weights are updated and no file in the original project is modified.

Run from the ``src/`` directory so the flat imports resolve:

    cd src
    python find_pivot.py \\
        --data_path /path/to/translated_swa.json \\
        --model_name_or_path Qwen/Qwen3-1.7B \\
        --output_file pivot_results.json

With a trained adapter:

    python find_pivot.py --data_path ... --checkpoint_dir outputs/rp-opsd-swa/checkpoint-100

Notes
-----
* Warmup is intentionally disabled (``step=None``): the gate is the pure
  ``sigmoid(beta * (z - tau))`` mapping, i.e. training's post-warmup regime. This is
  what you want for *finding* pivots -- the early ``gate=1`` warmup would mask the
  signal.
* The EMA gate normalizes across examples *in file order*, one example at a time.
  The first example initializes the EMA mean/std; later examples use the decaying
  EMA. This mirrors ``_update_rp_score_ema`` but with batch size 1 instead of 32.
* Generation uses ``transformers`` (not vLLM) for simplicity. Both sample from the
  same logits, so the pivot score is unchanged; only the sampled completion differs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from prompt_template_utils import build_assistant_prefilled_prompt
from pivot_languages import canonicalize as canonicalize_language_code, get_config
from pivot import compute_pivot_score, PivotGate, build_full_sequences


# --------------------------------------------------------------------------- #
# Data loading (faithful to rp_opsd_train.py: has_target_translation / add_target_language)
# --------------------------------------------------------------------------- #

def has_target_translation(example: dict, lang: str, require_ok: bool = True) -> bool:
    lang = lang.lower()
    q_key = f"problem_{lang}"
    ok_key = f"problem_{lang}_ok"

    if q_key not in example:
        return False
    if not isinstance(example[q_key], str) or not example[q_key].strip():
        return False
    if require_ok and ok_key in example:
        return example[ok_key] is True
    return True


def load_rows(path: str) -> list[dict]:
    """Load the translated JSON as a list of dicts (handles both array and JSONL)."""
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # JSONL fallback (one object per line)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def prepare_examples(path: str, lang: str, require_ok: bool) -> list[dict]:
    rows = load_rows(path)
    out = []
    for ex in rows:
        if not has_target_translation(ex, lang=lang, require_ok=require_ok):
            continue
        ex = dict(ex)
        # add_target_language(): store the English source under `problem_en`.
        ex["target_lang"] = lang
        ex["problem_en"] = ex["problem"]
        out.append(ex)
    return out


# --------------------------------------------------------------------------- #
# Prompt construction (faithful to RPOPSDSelfDistillationDataCollator.__call__)
# --------------------------------------------------------------------------- #

def build_prompts(tokenizer, example: dict, language: str = "SWA", enable_thinking: bool = True):
    """Return (student_text, teacher_text, ablation_text) for one example."""
    cfg = get_config(language)
    labels = cfg["labels"]

    problem_en = example.get("problem_en", example.get("problem"))
    problem_target = example[cfg["data_field"]]
    solution_en = example["solution"]

    student_message = (
        f"{labels['problem_target']}: {problem_target}\n\n"
        f"{cfg['student_instruction']}"
    )
    student_prompt = build_assistant_prefilled_prompt(
        tokenizer,
        [{"role": "user", "content": student_message}],
        enable_thinking=enable_thinking,
        think_prefix=cfg["think_prefix"],
    )

    shared_context = (
        f"{labels['problem_target']}: {problem_target}\n\n"
        f"{labels['problem_english']}: {problem_en}"
    )
    full_context = (
        f"{shared_context}\n\n"
        f"{labels['solution_english']}:\n"
        f"{labels['ref_begin']}\n{solution_en}\n{labels['ref_end']}"
    )
    teacher_prompt = build_assistant_prefilled_prompt(
        tokenizer,
        [
            {
                "role": "user",
                "content": (
                    f"{full_context}\n\n{cfg['transition_prompt']}\n"
                    f"{cfg['teacher_final_instruction']}"
                ),
            }
        ],
        enable_thinking=enable_thinking,
        think_prefix=cfg["think_prefix"],
    )
    ablation_prompt = build_assistant_prefilled_prompt(
        tokenizer,
        [{"role": "user", "content": f"{shared_context}\n\n{cfg['ablation_instruction']}"}],
        enable_thinking=enable_thinking,
        think_prefix=cfg["think_prefix"],
    )

    return student_prompt, teacher_prompt, ablation_prompt


def tokenize_prompt(tokenizer, text: str, max_length: int) -> torch.Tensor:
    """Tokenize a single prompt (no padding), returning 1-D ``input_ids``."""
    enc = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"][0]


# --------------------------------------------------------------------------- #
# Per-token decoding + step segmentation
# --------------------------------------------------------------------------- #

def decode_token_texts(tokenizer, completion_ids: torch.Tensor) -> list[str]:
    ids = completion_ids.tolist()
    texts = []
    for tok in ids:
        txt = tokenizer.decode([tok], skip_special_tokens=True)
        texts.append(txt)
    return texts


def segment_steps(token_texts: list[str]) -> list[dict]:
    """Split a token sequence into steps on newlines (a first-order 'reasoning step')."""
    steps = []
    cur = {"start": 0, "parts": []}
    for i, t in enumerate(token_texts):
        if not cur["parts"]:
            cur["start"] = i
        cur["parts"].append(t)
        if "\n" in t:
            cur["end"] = i
            steps.append(cur)
            cur = {"start": i + 1, "parts": []}
    if cur["parts"]:
        cur["end"] = len(token_texts) - 1
        steps.append(cur)
    return steps


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model_and_tokenizer(model_name_or_path: str, checkpoint_dir: str | None,
                             torch_dtype, attn_implementation: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        dtype=torch_dtype,
        trust_remote_code=True,
        use_cache=True,
        device_map="auto",
    )
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    except Exception as e:  # pragma: no cover - fallback when flash-attn is missing
        if attn_implementation and "flash" in attn_implementation:
            print(f"[warn] failed to load with {attn_implementation} ({e}); "
                  f"retrying with sdpa")
            load_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        else:
            raise

    if checkpoint_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_dir)
        print(f"Loaded LoRA adapter from {checkpoint_dir}")

    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def run(args) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    examples = prepare_examples(args.data_path, args.language, args.require_translation_ok)
    if args.num_samples is not None:
        end = min(args.sample_start + args.num_samples, len(examples))
    else:
        end = args.sample_end if args.sample_end is not None else len(examples)
    examples = examples[args.sample_start:end]

    if not examples:
        print("No usable examples found.")
        sys.exit(2)

    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path, args.checkpoint_dir,
        args.torch_dtype, args.attn_implementation,
    )

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    if (hasattr(model.generation_config, "eos_token_id")
            and model.generation_config.eos_token_id is not None):
        gen_config.eos_token_id = model.generation_config.eos_token_id

    gate_fn = PivotGate(
        beta=args.rp_gate_beta,
        tau=args.rp_gate_tau,
        g_min=args.rp_gate_min,
        ema_decay=args.rp_score_ema_decay,
        z_clip=args.rp_score_z_clip,
    )

    results = []
    for idx, ex in enumerate(examples):
        student_text, teacher_text, ablation_text = build_prompts(
            tokenizer, ex, language=args.language, enable_thinking=args.enable_thinking
        )

        device = model.device
        student_ids = tokenize_prompt(tokenizer, student_text, args.max_length).to(device)
        teacher_ids = tokenize_prompt(tokenizer, teacher_text, args.max_length).to(device)
        ablation_ids = tokenize_prompt(tokenizer, ablation_text, args.max_length).to(device)

        batch = {
            "student_prompts": student_ids.unsqueeze(0),
            "student_prompt_length": len(student_ids),
            "student_prompt_lengths_per_example": torch.tensor([len(student_ids)]),
            "teacher_prompts": teacher_ids.unsqueeze(0),
            "teacher_prompt_length": len(teacher_ids),
            "ablation_teacher_prompts": ablation_ids.unsqueeze(0),
            "ablation_teacher_prompt_length": len(ablation_ids),
        }

        # On-policy generation from the student prompt.
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=student_ids.unsqueeze(0).to(model.device),
                attention_mask=torch.ones_like(student_ids.unsqueeze(0)).to(model.device),
                generation_config=gen_config,
                return_dict_in_generate=True,
                use_cache=True,
            )
        generated_ids = gen_out.sequences  # [1, L_s + gen_len]

        # Assemble the three full sequences + completion mask (faithful to training_step).
        seqs = build_full_sequences(batch, generated_ids, tokenizer.pad_token_id)

        # Pivot score = KL(q_plus || q_minus) per completion token.
        score, _, _ = compute_pivot_score(
            model,
            full_input_ids=seqs["full_input_ids"],
            full_attention_mask=seqs["full_attention_mask"],
            full_prompt_len=seqs["full_prompt_len"],
            ablation_input_ids=seqs["ablation_input_ids"],
            ablation_attention_mask=seqs["ablation_attention_mask"],
            ablation_prompt_len=seqs["ablation_prompt_len"],
            completion_mask=seqs["completion_mask"],
        )

        completion_mask = seqs["completion_mask"]
        gate = gate_fn(score, completion_mask)  # updates EMA, then z-score + sigmoid

        # z-score (recomputed against the post-update EMA, same as inside the gate).
        z = (score - gate_fn.mean) / (gate_fn.std + 1e-6)
        z = z.clamp(-args.rp_score_z_clip, args.rp_score_z_clip)

        gen_ids_1d = seqs["generation_ids"][0]
        completion_text = tokenizer.decode(gen_ids_1d, skip_special_tokens=True)
        token_texts = decode_token_texts(tokenizer, gen_ids_1d)

        score_list = score[0].float().tolist()
        z_list = z[0].float().tolist()
        gate_list = gate[0].float().tolist()
        n = len(gen_ids_1d)

        tokens = [
            {
                "i": i,
                "text": token_texts[i],
                "token_id": int(gen_ids_1d[i]),
                "score": round(score_list[i], 6),
                "z": round(z_list[i], 6),
                "gate": round(gate_list[i], 6),
            }
            for i in range(n)
        ]

        steps = []
        for seg in segment_steps(token_texts):
            lo, hi = seg["start"], seg["end"]
            seg_scores = score_list[lo:hi + 1]
            seg_gates = gate_list[lo:hi + 1]
            steps.append({
                "text": "".join(seg["parts"]),
                "start": lo,
                "end": hi,
                "n_tokens": hi - lo + 1,
                "mean_score": round(sum(seg_scores) / len(seg_scores), 6),
                "max_score": round(max(seg_scores), 6),
                "sum_score": round(sum(seg_scores), 6),
                "mean_gate": round(sum(seg_gates) / len(seg_gates), 6),
            })

        valid_scores = [s for s in score_list]
        results.append({
            "example_id": args.sample_start + idx,
            "problem_en": ex.get("problem_en"),
            "problem_swa": ex.get("problem_swa"),
            "solution_en": ex.get("solution"),
            "student_prompt": student_text,
            "teacher_prompt": teacher_text,
            "ablation_prompt": ablation_text,
            "completion_text": completion_text,
            "num_completion_tokens": n,
            "tokens": tokens,
            "steps": steps,
            "summary": {
                "mean_score": round(sum(valid_scores) / max(1, len(valid_scores)), 6),
                "max_score": round(max(valid_scores), 6) if valid_scores else 0.0,
                "sum_score": round(sum(valid_scores), 6),
                "mean_gate": round(sum(gate_list) / max(1, len(gate_list)), 6),
                "n_pivot_tokens_z_gt_0": int(sum(1 for v in z_list if v > 0)),
                "n_pivot_tokens_gate_gt_half": int(sum(1 for g in gate_list if g > 0.5)),
            },
        })

        print(f"[{idx + 1}/{len(examples)}] example {args.sample_start + idx}: "
              f"{n} completion tokens, mean_score={results[-1]['summary']['mean_score']:.4f}, "
              f"pivot_tokens(z>0)={results[-1]['summary']['n_pivot_tokens_z_gt_0']}")

        del generated_ids, gen_ids_1d, score, gate, z
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = {
        "metadata": {
            "model": args.model_name_or_path,
            "checkpoint_dir": args.checkpoint_dir,
            "data_path": args.data_path,
            "language": args.language,
            "num_examples": len(results),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "rp_gate_beta": args.rp_gate_beta,
            "rp_gate_tau": args.rp_gate_tau,
            "rp_gate_min": args.rp_gate_min,
            "rp_score_ema_decay": args.rp_score_ema_decay,
            "rp_score_z_clip": args.rp_score_z_clip,
            "warmup_disabled": True,
        },
        "results": results,
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved pivot results for {len(results)} examples to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Offline RP-OPSD pivot finder (no training).")
    p.add_argument("--data_path", required=True, help="Path to the translated SWA JSON.")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-1.7B")
    p.add_argument("--checkpoint_dir", default=None,
                   help="Optional trained LoRA checkpoint dir (adapter_model.*).")
    p.add_argument("--output_file", default="pivot_results.json")
    p.add_argument("--language", default="SWA")
    p.add_argument("--require_translation_ok", action="store_true", default=True)

    p.add_argument("--num_samples", type=int, default=None,
                   help="Cap number of examples (None = all).")
    p.add_argument("--sample_start", type=int, default=0)
    p.add_argument("--sample_end", type=int, default=None)

    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enable_thinking", action="store_true", default=True)
    p.add_argument("--no_thinking", dest="enable_thinking", action="store_false")

    p.add_argument("--max_length", type=int, default=20000,
                   help="Prompt truncation limit (matches training).")
    p.add_argument("--torch_dtype", default="bfloat16")
    p.add_argument("--attn_implementation", default="flash_attention_2")

    # Gate hyperparameters (mirror train.sh defaults).
    p.add_argument("--rp_gate_beta", type=float, default=2.0)
    p.add_argument("--rp_gate_tau", type=float, default=0.0)
    p.add_argument("--rp_gate_min", type=float, default=0.05)
    p.add_argument("--rp_score_ema_decay", type=float, default=0.99)
    p.add_argument("--rp_score_z_clip", type=float, default=5.0)
    return p.parse_args()


def main():
    args = parse_args()
    args.language = canonicalize_language_code(args.language)
    dtype_map = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
    }
    args.torch_dtype = dtype_map.get(args.torch_dtype.lower(), torch.bfloat16)
    run(args)


if __name__ == "__main__":
    main()
