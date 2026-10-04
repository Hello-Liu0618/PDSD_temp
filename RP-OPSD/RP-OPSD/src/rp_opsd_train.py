import os

os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

DEFAULT_CACHE_ROOT = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "rp-opsd",
)
CACHE_ROOT = os.environ.get(
    "RP_OPSD_CACHE_ROOT",
    os.environ.get("COPSD_CACHE_ROOT", DEFAULT_CACHE_ROOT),
)

# ====== Hugging Face ======
os.environ.setdefault("HF_HOME", f"{CACHE_ROOT}/huggingface")
os.environ.setdefault("HF_HUB_CACHE", f"{CACHE_ROOT}/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{CACHE_ROOT}/huggingface/transformers")
os.environ.setdefault("HF_DATASETS_CACHE", f"{CACHE_ROOT}/huggingface/datasets")

# ====== vLLM======
os.environ.setdefault("VLLM_CACHE_ROOT", f"{CACHE_ROOT}/vllm")

from dataclasses import dataclass, field
from pathlib import Path

import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig, TrainerCallback

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from rp_opsd_trainer import RPOPSDTrainer
from language_config import canonicalize_language_code

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Arguments for the RP-OPSD main experiment."""

    translated_data_path: str = field(
        default="",
        metadata={
            "help": "Path to the per-language translated JSON file. "
            "If empty, defaults to ./translated_opsd/translated_full_<lang>.json"
        },
    )
    train_language: str = field(
        default="SWA",
        metadata={"help": "This artifact accepts SWA only."},
    )
    require_translation_ok: bool = field(
        default=True,
        metadata={
            "help": "If True, filter out rows where problem_<lang>_ok is present and not True."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    student_enable_thinking: bool = field(
        default=True,
        metadata={"help": "Enable the Qwen3 thinking block for student rollouts."},
    )
    rp_gate_beta: float = field(default=2.0, metadata={"help": "RP gate sigmoid beta."})
    rp_gate_tau: float = field(default=0.0, metadata={"help": "RP gate z-score threshold."})
    rp_gate_min: float = field(default=0.05, metadata={"help": "RP gate lower bound."})
    rp_score_ema_decay: float = field(default=0.99, metadata={"help": "EMA decay for score normalization."})
    rp_score_z_clip: float = field(default=5.0, metadata={"help": "Absolute z-score clipping threshold."})
    rp_gate_warmup_ratio: float = field(default=0.05, metadata={"help": "Fraction of steps using gate=1."})
    rp_gate_transition_ratio: float = field(
        default=0.05,
        metadata={"help": "Fraction of steps interpolating from gate=1 to gated loss."},
    )
    rp_reference_lambda: float = field(
        default=0.2,
        metadata={"help": "Frozen-reference anchoring coefficient."},
    )


STATE_FILE_GLOBS = (
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "rng_state*.pth",
    "latest",
    "zero_to_fp32.py",
)


def prune_checkpoint_state(checkpoint_dir: Path) -> None:
    """Keep adapter/tokenizer files but remove large resume-only state from old checkpoints."""
    for state_dir in checkpoint_dir.glob("global_step*"):
        if state_dir.is_dir():
            import shutil

            shutil.rmtree(state_dir, ignore_errors=True)

    for pattern in STATE_FILE_GLOBS:
        for path in checkpoint_dir.glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)


class KeepOnlyLatestResumeStateCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            return control

        output_dir = Path(args.output_dir)
        current_checkpoint = output_dir / f"checkpoint-{state.global_step}"
        for checkpoint_dir in output_dir.glob("checkpoint-*"):
            if checkpoint_dir.is_dir() and checkpoint_dir != current_checkpoint:
                prune_checkpoint_state(checkpoint_dir)
        return control


def resolve_translated_data_path(script_args) -> str:
    if script_args.translated_data_path:
        return script_args.translated_data_path
    raise ValueError("--translated_data_path is required.")


def add_target_language(example, lang):
    example["target_lang"] = lang
    # English source question is stored in `problem` for your per-language JSONs.
    example["problem_en"] = example["problem"]
    return example


def has_target_translation(example, lang, require_ok=True):
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


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    script_args.train_language = canonicalize_language_code(script_args.train_language)
    target_lang = script_args.train_language
    translated_data_path = resolve_translated_data_path(script_args)

    ################
    # WandB Run Name & Output Directory
    ################
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_{target_lang}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(
                Path(training_args.output_dir) / f"{script_args.run_config}_{target_lang.lower()}"
            )
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        full_wandb_run_config = (
            f"rp_opsd_{model_name}_{target_lang.lower()}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )
        training_args.output_dir = str(
            Path(training_args.output_dir) / target_lang.lower()
        )

    print(f"\n{'='*80}")
    print("RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Target Language: {target_lang}")
    print("Objective: gated forward KL + frozen-reference anchoring")
    print(f"Translated Data Path: {translated_data_path}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "translated_data_path": translated_data_path,
                "train_language": target_lang,
                "require_translation_ok": script_args.require_translation_ok,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "rp_gate_beta": script_args.rp_gate_beta,
                "rp_gate_tau": script_args.rp_gate_tau,
                "rp_gate_min": script_args.rp_gate_min,
                "rp_score_ema_decay": script_args.rp_score_ema_decay,
                "rp_score_z_clip": script_args.rp_score_z_clip,
                "rp_gate_warmup_ratio": script_args.rp_gate_warmup_ratio,
                "rp_gate_transition_ratio": script_args.rp_gate_transition_ratio,
                "rp_loss_variant": "recommended",
                "rp_kl_direction": "forward",
                "rp_reference_lambda": script_args.rp_reference_lambda,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    training_args.presence_penalty = script_args.presence_penalty
    
    # IMPORTANT: skip SFTTrainer's default text-field preprocessing.
    # Our multilingual collator consumes raw rows directly.
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    
    
    # IMPORTANT: keep all raw columns for the custom multilingual collator.
    # Without this, Trainer may remove columns like problem_ewe before batching.
    training_args.remove_unused_columns = False
    
    # if not hasattr(training_args, "dataset_text_field") or training_args.dataset_text_field is None:
    #     training_args.dataset_text_field = "problem"

    print(f"Loading translated JSON dataset from: {translated_data_path}")
    dataset = load_dataset("json", data_files=translated_data_path)
    train_dataset = dataset["train"]

    before_count = len(train_dataset)
    train_dataset = train_dataset.filter(
        lambda ex: has_target_translation(
            ex,
            lang=target_lang,
            require_ok=script_args.require_translation_ok,
        )
    )

    train_dataset = train_dataset.map(
        lambda ex: add_target_language(ex, target_lang)
    )

    after_count = len(train_dataset)

    print(f"\n{'='*80}")
    print("DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"Original examples: {before_count}")
    print(f"Usable examples for {target_lang}: {after_count}")
    print(f"Dropped examples: {before_count - after_count}")
    print(f"{'='*80}\n")

    if after_count == 0:
        raise ValueError(
            f"No usable examples found for target language {target_lang}. "
            f"Check your translated JSON and the problem_{target_lang.lower()} columns."
        )

    trainer = RPOPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        student_enable_thinking=script_args.student_enable_thinking,
        rp_gate_beta=script_args.rp_gate_beta,
        rp_gate_tau=script_args.rp_gate_tau,
        rp_gate_min=script_args.rp_gate_min,
        rp_score_ema_decay=script_args.rp_score_ema_decay,
        rp_score_z_clip=script_args.rp_score_z_clip,
        rp_gate_warmup_ratio=script_args.rp_gate_warmup_ratio,
        rp_gate_transition_ratio=script_args.rp_gate_transition_ratio,
        rp_reference_lambda=script_args.rp_reference_lambda,
    )

    trainer.add_callback(KeepOnlyLatestResumeStateCallback())

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    resume_from_checkpoint = getattr(training_args, "resume_from_checkpoint", None)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
