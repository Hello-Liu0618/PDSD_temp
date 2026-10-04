"""Swahili prompt configuration used by the RP-OPSD artifact."""


def canonicalize_language_code(lang_code):
    if lang_code is None:
        return "SWA"
    code = str(lang_code).strip().upper()
    if code == "SW":
        return "SWA"
    if code != "SWA":
        raise ValueError(f"This artifact supports SWA only, got {lang_code!r}.")
    return code


LANGUAGE_CONFIG = {
    "SWA": {
        "student_instruction": (
            "Tafadhali fikiri hatua kwa hatua, na uweke jibu lako la mwisho ndani ya \\boxed{}."
        ),
        "think_prefix": "Kwa ombi, nitaanza kufikiria kwa Kiswahili.",
        "transition_prompt": (
            "Baada ya kusoma suluhisho la rejeleo la Kiingereza hapo juu, hakikisha umeelewa "
            "kweli mantiki ya kila hatua—usilinakili wala kulifafanua upya tu. Sasa, kwa kutumia "
            "maneno yako mwenyewe na hoja huru, tatua swali la asili kwa Kiswahili. Fikiri hatua "
            "kwa hatua, jaribu mbinu tofauti, na usiogope kurudi nyuma au kufikiria upya ikiwa "
            "jambo fulani halifanyi kazi:"
        ),
        "teacher_final_instruction": (
            "Tafadhali fikiri hatua kwa hatua kwa Kiswahili, na uweke jibu lako la mwisho ndani "
            "ya \\boxed{}."
        ),
        "labels": {
            "problem_target": "Swali",
            "problem_english": "Tafsiri ya Kiingereza ya swali",
            "solution_english": "Suluhisho sahihi la rejeleo kwa Kiingereza",
            "ref_begin": "=== Mwanzo wa Suluhisho la Rejeleo ===",
            "ref_end": "=== Mwisho wa Suluhisho la Rejeleo ===",
        },
    }
}
