"""Extensible language configurations for the offline pivot finder.

The original RP-OPSD artifact is SWA-only (``language_config.py`` /
``data_collator.py``). This module generalizes that so the pivot finder can run on any
language Qwen can reason about, chosen to maximize the *information gap* between the
model's unaided reasoning and the English reference solution.

To add a language, add ONE entry to :data:`LANGUAGES` with:

* ``aliases``    -- accepted CLI codes (e.g. "JA", "JPN").
* ``data_field`` -- the example key holding the target-language text (e.g. "problem_ja").
* the prompt fields ``student_instruction`` / ``think_prefix`` / ``transition_prompt`` /
  ``teacher_final_instruction`` / ``ablation_instruction`` / ``labels``.

Nothing else needs to change -- ``canonicalize`` and ``get_config`` both derive from the
registry. Kept torch-free so the configs can be inspected without a GPU.
"""

from language_config import LANGUAGE_CONFIG


# Mirrors data_collator.ABLATION_INSTRUCTION (inlined to keep this module torch-free).
_SWA_ABLATION_INSTRUCTION = (
    "Tatua tatizo mwenyewe kutoka mwanzo.\n"
    "Kwanza hakikisha umeelewa swali asilia na tafsiri yake ya Kiingereza.\n"
    "Kisha, kwa kutumia maneno yako mwenyewe na hoja huru, suluhisha swali asilia kwa Kiswahili.\n"
    "Fikiri hatua kwa hatua, jaribu mbinu tofauti, na usiogope kurudi nyuma au kufikiria upya "
    "ikiwa kitu hakifanyi kazi.\n\n"
    "Tafadhali fikiri hatua kwa hatua kwa Kiswahili, na uweke jibu lako la mwisho ndani ya "
    "\\boxed{}."
)


def _swa_config() -> dict:
    cfg = dict(LANGUAGE_CONFIG["SWA"])
    cfg["ablation_instruction"] = _SWA_ABLATION_INSTRUCTION
    cfg["data_field"] = "problem_swa"
    cfg["aliases"] = ["SW", "SWA", "SWAHILI"]
    return cfg


LANGUAGES = {
    "SWA": _swa_config(),
    "ZH": {
        "aliases": ["ZH", "CHN", "ZHO", "ZHS", "CHINESE", "ZH-CN"],
        "data_field": "problem_zh",
        "student_instruction": "请一步一步推理，并将最终答案放在 \\boxed{} 中。",
        "think_prefix": "应要求，我将开始用中文思考。",
        "transition_prompt": (
            "阅读上面的英文参考解答后，请确保你真正理解了每一步的逻辑——不要只是照抄或复述。"
            "现在，用你自己的话和独立的推理，用中文解答原题。一步一步思考，尝试不同方法，"
            "如果某一步行不通，不要害怕回头或重新思考："
        ),
        "teacher_final_instruction": "请一步一步用中文推理，并将最终答案放在 \\boxed{} 中。",
        "ablation_instruction": (
            "请从头开始独立解答这道题。\n"
            "首先确保你理解了原题及其英文翻译。\n"
            "然后，用你自己的话和独立推理，用中文解答原题。\n"
            "一步一步思考，尝试不同方法，如果某一步行不通，不要害怕回头或重新思考。\n\n"
            "请一步一步用中文推理，并将最终答案放在 \\boxed{} 中。"
        ),
        "labels": {
            "problem_target": "题目",
            "problem_english": "题目的英文翻译",
            "solution_english": "英文参考解答",
            "ref_begin": "=== 参考解答开始 ===",
            "ref_end": "=== 参考解答结束 ===",
        },
    },
    "JA": {
        "aliases": ["JA", "JPN", "JAPANESE"],
        "data_field": "problem_ja",
        "student_instruction": "段階的に考え、最終的な答えを \\boxed{} の中に入れてください。",
        "think_prefix": "要望があれば、日本語で考え始めます。",
        "transition_prompt": (
            "上の英語の参考解答を読んだ後、各ステップの論理を本当に理解したことを確認してください。"
            "単に写したり言い換えたりしないでください。次に、自分の言葉と独立した推論で、"
            "元の問題を日本語で解いてください。段階的に考え、さまざまな方法を試し、"
            "うまくいかない場合はためらわずに戻って考え直してください："
        ),
        "teacher_final_instruction": "段階的に日本語で推論し、最終的な答えを \\boxed{} の中に入れてください。",
        "ablation_instruction": (
            "最初から独立してこの問題を解いてください。\n"
            "まず、元の問題とその英語訳を理解したことを確認してください。\n"
            "次に、自分の言葉と独立した推論で、元の問題を日本語で解いてください。\n"
            "段階的に考え、さまざまな方法を試し、うまくいかない場合はためらわずに戻って考え直してください。\n\n"
            "段階的に日本語で推論し、最終的な答えを \\boxed{} の中に入れてください。"
        ),
        "labels": {
            "problem_target": "問題",
            "problem_english": "問題の英語訳",
            "solution_english": "英語の参考解答",
            "ref_begin": "=== 参考解答の開始 ===",
            "ref_end": "=== 参考解答の終了 ===",
        },
    },
}


def canonicalize(lang) -> str:
    """Normalize a language code to a canonical key in :data:`LANGUAGES`."""
    if lang is None:
        return "SWA"
    code = str(lang).strip().upper().replace("-", "").replace("_", "")
    for key, cfg in LANGUAGES.items():
        aliases = [a.strip().upper().replace("-", "").replace("_", "") for a in cfg["aliases"]]
        if code in aliases:
            return key
    raise ValueError(
        f"Unsupported language {lang!r}. Supported: {sorted(LANGUAGES)} "
        f"(add an entry to pivot_languages.LANGUAGES to extend)."
    )


def get_config(lang: str) -> dict:
    """Return a copy of the full prompt config for ``lang``."""
    return dict(LANGUAGES[canonicalize(lang)])
