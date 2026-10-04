#!/usr/bin/env python3
"""Build the RP-OPSD Swahili training JSON from an OpenThoughts math subset.

The original repo's training data is described only as "500-example,
OpenThoughts-derived". The exact 500 selection and the translation pipeline are not
published in this repository, so this is a *faithful reconstruction*, not a
byte-exact reproduction.

Output: a JSONL file (one object per line) where each row is::

    {"problem": "<English question>",
     "problem_swa": "<Swahili translation>",
     "solution": "<English reference solution>"}

and, if ``--mark_ok`` is given, additionally ``"problem_swa_ok": true``.

Run inside the ``rp-opsd`` conda env (needs ``datasets`` + ``transformers`` + a GPU):

    conda activate rp-opsd
    python scripts/build_train_data.py --n 500 --translator nllb

Dry-run (validate the source + selection + schema without any translation):

    python scripts/build_train_data.py --n 5 --translator none
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SRC_LANG = "eng_Latn"
TGT_LANG = "swh_Latn"


# --------------------------------------------------------------------------- #
# Source
# --------------------------------------------------------------------------- #

def load_source(source: str, n: int, seed: int | None):
    from datasets import load_dataset

    ds = load_dataset(source, split="train")
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    rows = ds.select(range(min(n, len(ds))))

    for col in ("problem", "solution"):
        if col not in rows.column_names:
            raise KeyError(f"source '{source}' has no '{col}' column; "
                           f"columns are {rows.column_names}")
    return rows


# --------------------------------------------------------------------------- #
# Translators
# --------------------------------------------------------------------------- #

def translate_nllb(texts: list[str], mt_model: str, batch_size: int) -> list[str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mt_model, src_lang=SRC_LANG)
    mt = AutoModelForSeq2SeqLM.from_pretrained(mt_model).to("cuda")
    mt.eval()
    bos = tok.lang_code_to_id[TGT_LANG]

    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")
        gen = mt.generate(**enc, forced_bos_token_id=bos, max_new_tokens=512)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  translated {min(i + batch_size, len(texts))}/{len(texts)}")
    return out


def translate_llm(texts: list[str], model_name: str) -> list[str]:
    """Basic LLM translation (higher quality, slower). Override the prompt as needed."""
    from transformers import pipeline

    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    tpl = (
        "Translate the following math problem into Swahili. "
        "Output only the Swahili translation, with no explanation.\n"
        "English: {t}\nSwahili:"
    )
    out: list[str] = []
    for t in texts:
        r = pipe(tpl.format(t=t), max_new_tokens=512, do_sample=False)[0]["generated_text"]
        out.append(r.split("Swahili:", 1)[-1].strip())
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Generate the RP-OPSD SWA training JSON.")
    ap.add_argument("--source", default="open-r1/OpenThoughts-114k-math",
                    help="HF dataset id with clean problem/solution columns.")
    ap.add_argument("--n", type=int, default=500, help="Number of examples (default 500).")
    ap.add_argument("--seed", type=int, default=0, help="Selection shuffle seed.")
    ap.add_argument("--output", default="translated_swa.json", help="Output JSONL path.")
    ap.add_argument("--translator", default="nllb", choices=["nllb", "llm", "none"])
    ap.add_argument("--mt_model", default="facebook/nllb-200-distilled-600M",
                    help="MT model when --translator nllb.")
    ap.add_argument("--llm_model", default="Qwen/Qwen3-1.7B",
                    help="Instruct model when --translator llm.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--mark_ok", action="store_true",
                    help="Write 'problem_swa_ok': true on every row (else omit the field).")
    args = ap.parse_args()

    rows = load_source(args.source, args.n, args.seed)
    problems = [r["problem"] for r in rows]
    solutions = [r["solution"] for r in rows]

    if args.translator == "none":
        print(f"[dry-run] loaded {len(rows)} rows from {args.source}. Schema preview:")
        for i in range(min(3, len(rows))):
            print(f"\n--- row {i} ---\nproblem (EN): {problems[i][:200]!r}\n"
                  f"solution (EN): {solutions[i][:200]!r}")
        print("\nNo file written. Re-run with --translator nllb to generate data.")
        return

    print(f"Translating {len(problems)} problems "
          f"({SRC_LANG} -> {TGT_LANG}) with {args.translator} ...")
    if args.translator == "nllb":
        translated = translate_nllb(problems, args.mt_model, args.batch_size)
    else:  # llm
        translated = translate_llm(problems, args.llm_model)

    records = []
    for i, (prob, sol, swa) in enumerate(zip(problems, solutions, translated)):
        # Skip rows the training filter would drop (empty problem or translation).
        if not (isinstance(prob, str) and prob.strip()):
            continue
        if not (isinstance(sol, str) and sol.strip()):
            continue
        if not (isinstance(swa, str) and swa.strip()):
            continue
        rec = {"problem": prob, "problem_swa": swa, "solution": sol}
        if args.mark_ok:
            rec["problem_swa_ok"] = True
        records.append(rec)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} rows to {out_path} "
          f"(dropped {len(translated) - len(records)} empty rows).")


if __name__ == "__main__":
    main()
