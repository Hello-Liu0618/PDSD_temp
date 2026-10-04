"""Build the three matched prompts needed by the RP-OPSD objective."""

import torch

from language_config import LANGUAGE_CONFIG, canonicalize_language_code
from prompt_template_utils import build_assistant_prefilled_prompt, chat_template_family


ABLATION_INSTRUCTION = (
    "Tatua tatizo mwenyewe kutoka mwanzo.\n"
    "Kwanza hakikisha umeelewa swali asilia na tafsiri yake ya Kiingereza.\n"
    "Kisha, kwa kutumia maneno yako mwenyewe na hoja huru, suluhisha swali asilia kwa Kiswahili.\n"
    "Fikiri hatua kwa hatua, jaribu mbinu tofauti, na usiogope kurudi nyuma au kufikiria upya "
    "ikiwa kitu hakifanyi kazi.\n\n"
    "Tafadhali fikiria hatua kwa hatua kwa Kiswahili, na uweke jibu lako la mwisho ndani ya "
    "\\boxed{}."
)


class RPOPSDSelfDistillationDataCollator:
    """Construct student, solution-conditioned, and ablated teacher prompts."""

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=False,
        student_enable_thinking=True,
        include_problem_en=True,
        include_reference_solution_en=True,
    ):
        if reason_first:
            raise ValueError("The RP-OPSD main experiment does not use reason_first.")
        if not include_problem_en or not include_reference_solution_en:
            raise ValueError("The RP-OPSD main experiment requires both English privileged fields.")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.student_enable_thinking = student_enable_thinking
        self.chat_template_family = chat_template_family(tokenizer)
        self.tokenizer.padding_side = "right"

    def _tokenize_with_batch_max(self, texts):
        encoded_no_pad = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        lengths = [len(ids) for ids in encoded_no_pad["input_ids"]]
        max_len = max(lengths)
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return encoded, lengths, max_len

    def _build_chat_prompt(self, user_message, enable_thinking=True):
        return build_assistant_prefilled_prompt(
            self.tokenizer,
            [{"role": "user", "content": user_message}],
            enable_thinking=enable_thinking,
            think_prefix=LANGUAGE_CONFIG["SWA"]["think_prefix"],
        )

    def __call__(self, features):
        cfg = LANGUAGE_CONFIG["SWA"]
        labels = cfg["labels"]
        student_prompts = []
        teacher_prompts = []
        ablation_teacher_prompts = []

        for feature in features:
            lang = canonicalize_language_code(feature.get("target_lang", "SWA"))
            if lang != "SWA":
                raise ValueError(f"This artifact supports SWA only, got {lang}.")

            problem_en = feature.get("problem_en", feature.get("problem"))
            problem_swa = feature.get("problem_swa")
            solution_en = feature.get("solution")
            if not all(isinstance(value, str) and value.strip() for value in (
                problem_en,
                problem_swa,
                solution_en,
            )):
                raise ValueError(
                    "Each training row must contain non-empty problem/problem_en, problem_swa, and solution."
                )

            student_message = (
                f"{labels['problem_target']}: {problem_swa}\n\n"
                f"{cfg['student_instruction']}"
            )
            student_prompts.append(
                self._build_chat_prompt(
                    student_message,
                    enable_thinking=self.student_enable_thinking,
                )
            )

            shared_context = (
                f"{labels['problem_target']}: {problem_swa}\n\n"
                f"{labels['problem_english']}: {problem_en}"
            )
            full_context = (
                f"{shared_context}\n\n"
                f"{labels['solution_english']}:\n"
                f"{labels['ref_begin']}\n{solution_en}\n{labels['ref_end']}"
            )
            teacher_prompts.append(
                self._build_chat_prompt(
                    f"{full_context}\n\n{cfg['transition_prompt']}\n"
                    f"{cfg['teacher_final_instruction']}"
                )
            )
            ablation_teacher_prompts.append(
                self._build_chat_prompt(f"{shared_context}\n\n{ABLATION_INSTRUCTION}")
            )

        student_encoded, student_lengths, student_max = self._tokenize_with_batch_max(
            student_prompts
        )
        teacher_encoded, teacher_lengths, teacher_max = self._tokenize_with_batch_max(
            teacher_prompts
        )
        ablation_encoded, ablation_lengths, ablation_max = self._tokenize_with_batch_max(
            ablation_teacher_prompts
        )

        return {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": student_max,
            "student_prompt_lengths_per_example": torch.tensor(student_lengths, dtype=torch.long),
            "teacher_prompts": teacher_encoded["input_ids"],
            "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
            "teacher_prompt_length": teacher_max,
            "teacher_prompt_lengths_per_example": torch.tensor(teacher_lengths, dtype=torch.long),
            "ablation_teacher_prompts": ablation_encoded["input_ids"],
            "ablation_teacher_prompt_attention_mask": ablation_encoded["attention_mask"],
            "ablation_teacher_prompt_length": ablation_max,
            "ablation_teacher_prompt_lengths_per_example": torch.tensor(
                ablation_lengths, dtype=torch.long
            ),
            "lang_codes": ["SWA"] * len(features),
        }


# Kept as the import used by the trainer.
RPSelfDistillationDataCollator = RPOPSDSelfDistillationDataCollator
