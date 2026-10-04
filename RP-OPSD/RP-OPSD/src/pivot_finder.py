"""Convenient high-level API + visual report for RP-OPSD pivot finding.

A friendlier layer on top of :mod:`pivot` and :mod:`find_pivot`. It can be run from
any directory (no ``cd src`` needed) and lets you find pivots with a single call,
then renders an HTML report that colour-codes every completion token by its pivot
gate, so you can see at a glance which parts of the chain-of-thought the reference
solution actually drives.

Quick start (from anywhere)::

    import sys
    sys.path.insert(0, "/path/to/RP-OPSD/src")
    from pivot_finder import find_pivots, render_html, save_json

    res = find_pivots(
        data_path="translated_swa.json",
        model_name_or_path="Qwen/Qwen3-1.7B",
        checkpoint_dir="outputs/rp-opsd-swa/checkpoint-100",   # optional
        num_samples=10,
    )
    render_html(res, "pivot_report.html")
    save_json(res, "pivot_results.json")

    # programmatic access
    ex0 = res["results"][0]
    print(ex0["completion_text"])
    print(ex0["steps"][0]["mean_gate"])        # per-step pivot strength
    print(ex0["summary"]["n_pivot_tokens_z_gt_0"])

CLI::

    python pivot_finder.py --data_path translated_swa.json --num_samples 10
"""

from __future__ import annotations

import argparse
import html as _html
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
from transformers import GenerationConfig

from pivot import compute_pivot_score, PivotGate, build_full_sequences
from find_pivot import (
    build_prompts,
    decode_token_texts,
    load_model_and_tokenizer,
    prepare_examples,
    segment_steps,
    tokenize_prompt,
)
from pivot_languages import LANGUAGES, canonicalize as canonicalize_language_code

__all__ = ["PivotFinder", "find_pivots", "render_html", "save_json"]


# --------------------------------------------------------------------------- #
# Colour mapping (low gate -> white, high gate -> red)
# --------------------------------------------------------------------------- #

def _gate_style(g: float, g_min: float = 0.05, g_max: float = 1.0) -> str:
    t = (g - g_min) / (g_max - g_min)
    t = max(0.0, min(1.0, t))
    r = 255
    gg = int(255 - 150 * t)
    b = int(255 - 230 * t)
    fg = "#111" if t < 0.55 else "#fff"
    return f"background-color: rgb({r},{gg},{b}); color: {fg};"


# --------------------------------------------------------------------------- #
# Core finder
# --------------------------------------------------------------------------- #

class PivotFinder:
    """Load a model once, then find pivots across examples with a shared EMA gate.

    The :class:`PivotGate` is created once per finder and shared across all examples
    in a ``find`` call (order matters -- the EMA normalizes across the corpus). Call
    :meth:`reset_gate` to start a fresh normalization for an independent pass.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-1.7B",
        checkpoint_dir: str | None = None,
        language: str = "SWA",
        require_translation_ok: bool = True,
        temperature: float = 1.1,
        top_p: float = 0.95,
        top_k: int = 20,
        max_new_tokens: int = 2048,
        seed: int = 0,
        enable_thinking: bool = True,
        max_length: int = 20000,
        torch_dtype=torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        gate_kwargs: dict | None = None,
    ):
        self.language = canonicalize_language_code(language)
        self.require_translation_ok = require_translation_ok
        self.enable_thinking = enable_thinking
        self.max_length = max_length
        self.model_name_or_path = model_name_or_path
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed

        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name_or_path, checkpoint_dir, torch_dtype, attn_implementation
        )
        self.gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        if (hasattr(self.model.generation_config, "eos_token_id")
                and self.model.generation_config.eos_token_id is not None):
            self.gen_config.eos_token_id = self.model.generation_config.eos_token_id

        self._gate_kwargs = gate_kwargs or {}
        self.gate_fn = PivotGate(**self._gate_kwargs)
        self._sampling = dict(
            temperature=temperature, top_p=top_p, top_k=top_k,
            max_new_tokens=max_new_tokens, seed=seed,
        )

    def reset_gate(self) -> None:
        """Reset the EMA gate state for an independent pass."""
        self.gate_fn = PivotGate(**self._gate_kwargs)

    def _find_one(self, example: dict, example_id: int) -> dict:
        student_text, teacher_text, ablation_text = build_prompts(
            self.tokenizer, example, language=self.language, enable_thinking=self.enable_thinking
        )
        device = self.model.device
        student_ids = tokenize_prompt(self.tokenizer, student_text, self.max_length).to(device)
        teacher_ids = tokenize_prompt(self.tokenizer, teacher_text, self.max_length).to(device)
        ablation_ids = tokenize_prompt(self.tokenizer, ablation_text, self.max_length).to(device)

        batch = {
            "student_prompts": student_ids.unsqueeze(0),
            "student_prompt_length": len(student_ids),
            "student_prompt_lengths_per_example": torch.tensor([len(student_ids)]),
            "teacher_prompts": teacher_ids.unsqueeze(0),
            "teacher_prompt_length": len(teacher_ids),
            "ablation_teacher_prompts": ablation_ids.unsqueeze(0),
            "ablation_teacher_prompt_length": len(ablation_ids),
        }

        with torch.no_grad():
            gen_out = self.model.generate(
                input_ids=student_ids.unsqueeze(0),
                attention_mask=torch.ones_like(student_ids.unsqueeze(0)),
                generation_config=self.gen_config,
                return_dict_in_generate=True,
                use_cache=True,
            )
        generated_ids = gen_out.sequences  # [1, L_s + G]

        seqs = build_full_sequences(batch, generated_ids, self.tokenizer.pad_token_id)
        score, _, _ = compute_pivot_score(
            self.model,
            full_input_ids=seqs["full_input_ids"],
            full_attention_mask=seqs["full_attention_mask"],
            full_prompt_len=seqs["full_prompt_len"],
            ablation_input_ids=seqs["ablation_input_ids"],
            ablation_attention_mask=seqs["ablation_attention_mask"],
            ablation_prompt_len=seqs["ablation_prompt_len"],
            completion_mask=seqs["completion_mask"],
        )

        completion_mask = seqs["completion_mask"]
        gate = self.gate_fn(score, completion_mask)  # updates EMA, then z-score + sigmoid
        z = (score - self.gate_fn.mean) / (self.gate_fn.std + 1e-6)
        z = z.clamp(-self.gate_fn.z_clip, self.gate_fn.z_clip)

        gen_ids_1d = seqs["generation_ids"][0]
        completion_text = self.tokenizer.decode(gen_ids_1d, skip_special_tokens=True)
        token_texts = decode_token_texts(self.tokenizer, gen_ids_1d)

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
            ss = score_list[lo:hi + 1]
            gg = gate_list[lo:hi + 1]
            steps.append({
                "text": "".join(seg["parts"]),
                "start": lo,
                "end": hi,
                "n_tokens": hi - lo + 1,
                "mean_score": round(sum(ss) / len(ss), 6),
                "max_score": round(max(ss), 6),
                "sum_score": round(sum(ss), 6),
                "mean_gate": round(sum(gg) / len(gg), 6),
            })

        return {
            "example_id": example_id,
            "problem_en": example.get("problem_en"),
            "problem_swa": example.get("problem_swa"),
            "solution_en": example.get("solution"),
            "student_prompt": student_text,
            "teacher_prompt": teacher_text,
            "ablation_prompt": ablation_text,
            "completion_text": completion_text,
            "num_completion_tokens": n,
            "tokens": tokens,
            "steps": steps,
            "summary": {
                "mean_score": round(sum(score_list) / max(1, n), 6),
                "max_score": round(max(score_list), 6) if n else 0.0,
                "sum_score": round(sum(score_list), 6),
                "mean_gate": round(sum(gate_list) / max(1, n), 6),
                "n_pivot_tokens_z_gt_0": int(sum(1 for v in z_list if v > 0)),
                "n_pivot_tokens_gate_gt_half": int(sum(1 for g in gate_list if g > 0.5)),
            },
        }

    def find(
        self,
        data_path: str | None = None,
        examples: list[dict] | None = None,
        num_samples: int | None = None,
        sample_start: int = 0,
        sample_end: int | None = None,
    ) -> dict:
        """Find pivots over a corpus; returns ``{"metadata": ..., "results": [...]}``.

        Provide either ``data_path`` (translated JSON) or ``examples`` (in-memory
        list of dicts with ``problem``/``problem_swa``/``solution``).
        """
        if data_path is not None:
            exs = prepare_examples(data_path, self.language, self.require_translation_ok)
        else:
            exs = [dict(e) for e in (examples or [])]

        end = len(exs) if sample_end is None else min(sample_end, len(exs))
        if num_samples is not None:
            end = min(end, sample_start + num_samples)
        exs = exs[sample_start:end]

        torch.manual_seed(self.seed)
        results = []
        for i, ex in enumerate(exs):
            rid = sample_start + i
            r = self._find_one(ex, rid)
            results.append(r)
            print(
                f"[{i + 1}/{len(exs)}] example {rid}: {r['num_completion_tokens']} tokens, "
                f"mean_score={r['summary']['mean_score']:.4f}, "
                f"pivot(z>0)={r['summary']['n_pivot_tokens_z_gt_0']}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "metadata": {
                "model": self.model_name_or_path,
                "checkpoint_dir": self.checkpoint_dir,
                "language": self.language,
                "num_examples": len(results),
                **self._sampling,
                "gate": {
                    k: getattr(self.gate_fn, k)
                    for k in ("beta", "tau", "g_min", "ema_decay", "z_clip")
                },
                "warmup_disabled": True,
            },
            "results": results,
        }


def find_pivots(
    data_path: str | None = None,
    examples: list[dict] | None = None,
    model_name_or_path: str = "Qwen/Qwen3-1.7B",
    checkpoint_dir: str | None = None,
    num_samples: int | None = None,
    sample_start: int = 0,
    sample_end: int | None = None,
    **kwargs,
) -> dict:
    """One-shot convenience wrapper around :class:`PivotFinder`."""
    finder = PivotFinder(model_name_or_path, checkpoint_dir=checkpoint_dir, **kwargs)
    return finder.find(
        data_path=data_path, examples=examples, num_samples=num_samples,
        sample_start=sample_start, sample_end=sample_end,
    )


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #

def save_json(results: dict, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to {p}")
    return str(p)


def _build_html(res: dict) -> str:
    meta = res["metadata"]
    results = res["results"]
    gate = meta.get("gate", {})

    css = """
    body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; max-width: 920px;
           margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
    h2 { margin-bottom: .2rem; }
    .meta { color: #666; font-size: .85rem; margin-bottom: 1rem; }
    .legend { display: flex; align-items: center; gap: .5rem; margin: 1rem 0; flex-wrap: wrap; }
    .swatch { padding: .15rem .6rem; border-radius: 4px; font-size: .8rem; }
    .example { border-top: 1px solid #ddd; padding: 1rem 0; }
    .completion { white-space: pre-wrap; line-height: 1.8; font-family: ui-monospace,
                  Menlo, Consolas, monospace; font-size: .92rem; border: 1px solid #eee;
                  border-radius: 6px; padding: .75rem; background: #fafafa; }
    details { margin-top: .6rem; }
    summary { cursor: pointer; color: #555; }
    pre { white-space: pre-wrap; font-size: .8rem; color: #444; }
    table { border-collapse: collapse; width: 100%; font-size: .85rem; margin-top: .4rem; }
    th, td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid #eee; vertical-align: top; }
    th { color: #555; font-weight: 600; }
    """

    out = ["<!doctype html><html><head><meta charset='utf-8'>",
           f"<style>{css}</style></head><body>"]
    out.append(f"<h2>RP-OPSD Pivot Report</h2>")
    out.append(
        f"<div class='meta'>model: {_html.escape(str(meta.get('model')))} "
        f"&nbsp;|&nbsp; checkpoint: {_html.escape(str(meta.get('checkpoint_dir')))} "
        f"&nbsp;|&nbsp; examples: {meta.get('num_examples')} "
        f"&nbsp;|&nbsp; temp={meta.get('temperature')} top_p={meta.get('top_p')} top_k={meta.get('top_k')}</div>"
    )

    # Legend
    out.append("<div class='legend'>gate:")
    for g in (0.05, 0.25, 0.5, 0.75, 1.0):
        out.append(f"<span class='swatch' style='{_gate_style(g)}'>{g:.2f}</span>")
    out.append("<span style='font-size:.8rem;color:#666'>← weak pivot … strong pivot →</span></div>")

    for ex in results:
        eid = ex.get("example_id", "?")
        out.append(f"<div class='example'><h3>Example {eid}</h3>")

        spans = []
        for t in ex["tokens"]:
            txt = _html.escape(t["text"])
            spans.append(f"<span style='{_gate_style(t['gate'])}'>{txt}</span>")
        out.append(f"<div class='completion'>{''.join(spans)}</div>")

        s = ex.get("summary", {})
        out.append(
            f"<div style='font-size:.82rem;color:#555;margin-top:.4rem'>"
            f"mean_gate={s.get('mean_gate')} · pivot(z&gt;0)={s.get('n_pivot_tokens_z_gt_0')}"
            f"/{ex.get('num_completion_tokens')} tokens</div>"
        )

        rows = "".join(
            f"<tr><td>{_html.escape(st['text'][:120])}</td><td>{st['n_tokens']}</td>"
            f"<td>{st['mean_gate']:.3f}</td><td>{st['mean_score']:.3f}</td>"
            f"<td>{st['sum_score']:.3f}</td></tr>"
            for st in ex["steps"]
        )
        out.append(
            f"<details><summary>Steps ({len(ex['steps'])})</summary>"
            f"<table><tr><th>step</th><th>tokens</th><th>mean gate</th>"
            f"<th>mean score</th><th>sum score</th></tr>{rows}</table></details>"
        )

        prompts = (
            f"<details><summary>prompts</summary>"
            f"<p><b>student</b><pre>{_html.escape(ex['student_prompt'])}</pre></p>"
            f"<p><b>teacher (q_plus)</b><pre>{_html.escape(ex['teacher_prompt'])}</pre></p>"
            f"<p><b>ablation (q_minus)</b><pre>{_html.escape(ex['ablation_prompt'])}</pre></p>"
            f"</details>"
        )
        out.append(prompts)
        out.append("</div>")

    out.append("</body></html>")
    return "".join(out)


def render_html(results: dict, path: str | None = None) -> str:
    """Render a self-contained HTML report colouring tokens by pivot gate."""
    html_str = _build_html(results)
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html_str, encoding="utf-8")
        print(f"Saved HTML report to {p}")
    return html_str


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="RP-OPSD pivot finder (convenient CLI).")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--language", default="SWA", choices=sorted(LANGUAGES),
                    help="Target language for the pivot prompts.")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--sample_start", type=int, default=0)
    ap.add_argument("--sample_end", type=int, default=None)
    ap.add_argument("--output_json", default="pivot_results.json")
    ap.add_argument("--report", default="pivot_report.html")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_thinking", dest="enable_thinking", action="store_false", default=True)
    ap.add_argument("--torch_dtype", default="bfloat16")
    ap.add_argument("--attn_implementation", default="flash_attention_2")
    a = ap.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
    }.get(a.torch_dtype.lower(), torch.bfloat16)

    res = find_pivots(
        data_path=a.data_path,
        model_name_or_path=a.model_name_or_path,
        checkpoint_dir=a.checkpoint_dir,
        language=a.language,
        num_samples=a.num_samples,
        sample_start=a.sample_start,
        sample_end=a.sample_end,
        temperature=a.temperature,
        top_p=a.top_p,
        top_k=a.top_k,
        max_new_tokens=a.max_new_tokens,
        seed=a.seed,
        enable_thinking=a.enable_thinking,
        torch_dtype=dtype,
        attn_implementation=a.attn_implementation,
    )
    save_json(res, a.output_json)
    render_html(res, a.report)


if __name__ == "__main__":
    main()
