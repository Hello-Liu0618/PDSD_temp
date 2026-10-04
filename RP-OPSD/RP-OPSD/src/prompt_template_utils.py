"""Model-family-aware chat prompt construction for RP-OPSD rollouts."""


def chat_template_family(tokenizer) -> str:
    """Return the prompt family needed to preserve Qwen3's thinking contract."""
    template = str(getattr(tokenizer, "chat_template", "") or "")
    if (
        "<|im_start|>" in template
        and "<|im_end|>" in template
        and "enable_thinking" in template
    ):
        return "qwen3"
    return "native"


def build_assistant_prefilled_prompt(
    tokenizer,
    messages,
    *,
    enable_thinking: bool,
    think_prefix: str = "",
) -> str:
    """Render a generation prompt without injecting another model's control tokens.

    Qwen3's template owns the ``<think>`` contract.  Existing RP-OPSD runs also
    prefilled a target-language sentence inside that block, so that behavior is
    retained exactly.  Other model families (including Phi-4-mini-reasoning)
    use their native assistant marker and receive only the plain-text language
    prefill.  This keeps Phi prompts in the documented ``<|assistant|>`` format.
    """
    family = chat_template_family(tokenizer)
    template_kwargs = {}
    if family == "qwen3":
        template_kwargs["enable_thinking"] = enable_thinking

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )

    if not enable_thinking:
        return prompt

    prefix = str(think_prefix or "").strip()
    if family == "qwen3":
        return prompt + "<think>\n" + prefix
    return prompt + prefix
