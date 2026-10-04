"""日语版自蒸馏数据 collator：构造 学生 / 教师(带参考) / 消融教师(q_minus) 三组 prompt。

RP-OPSD 自带的 `data_collator.py` 硬编码斯瓦希里语；这里照抄其结构，改用
`pivot_languages.get_config(lang)` 的全部 prompt 字段，因此与 `seed_builder.py` 的日语 prompt 完全一致。
输出键名与 RP-OPSD collator 保持相同，便于 trainer 直接对接。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# 复用 RP-OPSD 的 prompt 构造（append 而非 insert，避免其 pivot.py 遮蔽本项目的 pivot/ 包）
_RP_SRC = Path(__file__).resolve().parent / "RP-OPSD" / "RP-OPSD" / "src"
if str(_RP_SRC) not in sys.path:
    sys.path.append(str(_RP_SRC))

from prompt_template_utils import build_assistant_prefilled_prompt  # noqa: E402
from pivot_languages import get_config  # noqa: E402


class SelfDistillationDataCollator:
    """给定语言（默认 JA）构造三组 prompt：student / teacher(参考) / ablation(q_minus)。"""

    def __init__(self, tokenizer, max_length: int = 4096, lang: str = "JA",
                 student_enable_thinking: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.student_enable_thinking = student_enable_thinking
        self.cfg = get_config(lang)
        self.labels = self.cfg["labels"]
        self.data_field = self.cfg["data_field"]        # e.g. "problem_ja"
        self.tokenizer.padding_side = "right"

    def _tokenize_with_batch_max(self, texts):
        encoded_no_pad = self.tokenizer(
            texts, padding=False, truncation=True, max_length=self.max_length)
        lengths = [len(ids) for ids in encoded_no_pad["input_ids"]]
        max_len = max(lengths)
        encoded = self.tokenizer(
            texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        return encoded, lengths, max_len

    def _build_chat_prompt(self, user_message, enable_thinking=True):
        return build_assistant_prefilled_prompt(
            self.tokenizer, [{"role": "user", "content": user_message}],
            enable_thinking=enable_thinking, think_prefix=self.cfg["think_prefix"])

    def __call__(self, features):
        cfg, labels = self.cfg, self.labels
        student_prompts, teacher_prompts, ablation_prompts = [], [], []

        for feature in features:
            problem_en = feature.get("problem_en") or feature.get("problem")
            problem_tgt = feature.get(self.data_field)
            solution_en = feature.get("solution")
            if not all(isinstance(v, str) and v.strip() for v in (problem_en, problem_tgt, solution_en)):
                raise ValueError(
                    f"每条数据需含非空的 problem_en、{self.data_field}、solution，实际得到 "
                    f"{list(feature.keys())}")

            student_msg = f"{labels['problem_target']}: {problem_tgt}\n\n{cfg['student_instruction']}"
            student_prompts.append(
                self._build_chat_prompt(student_msg, enable_thinking=self.student_enable_thinking))

            shared = f"{labels['problem_target']}: {problem_tgt}\n\n{labels['problem_english']}: {problem_en}"
            full = (f"{shared}\n\n{labels['solution_english']}:\n"
                    f"{labels['ref_begin']}\n{solution_en}\n{labels['ref_end']}")
            teacher_prompts.append(self._build_chat_prompt(
                f"{full}\n\n{cfg['transition_prompt']}\n{cfg['teacher_final_instruction']}"))
            ablation_prompts.append(self._build_chat_prompt(
                f"{shared}\n\n{cfg['ablation_instruction']}"))

        student_enc, student_lengths, student_max = self._tokenize_with_batch_max(student_prompts)
        teacher_enc, teacher_lengths, teacher_max = self._tokenize_with_batch_max(teacher_prompts)
        abla_enc, abla_lengths, abla_max = self._tokenize_with_batch_max(ablation_prompts)

        return {
            "student_prompts": student_enc["input_ids"],
            "student_prompt_attention_mask": student_enc["attention_mask"],
            "student_prompt_length": student_max,
            "student_prompt_lengths_per_example": torch.tensor(student_lengths, dtype=torch.long),
            "teacher_prompts": teacher_enc["input_ids"],
            "teacher_prompt_attention_mask": teacher_enc["attention_mask"],
            "teacher_prompt_length": teacher_max,
            "teacher_prompt_lengths_per_example": torch.tensor(teacher_lengths, dtype=torch.long),
            "ablation_teacher_prompts": abla_enc["input_ids"],
            "ablation_teacher_prompt_attention_mask": abla_enc["attention_mask"],
            "ablation_teacher_prompt_length": abla_max,
            "ablation_teacher_prompt_lengths_per_example": torch.tensor(abla_lengths, dtype=torch.long),
            "lang_codes": [self.cfg.get("aliases", ["JA"])[0]] * len(features),
        }
