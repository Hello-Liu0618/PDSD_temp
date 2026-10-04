#!/usr/bin/env python3
"""PDSD / RP-OPSD 双臂在线蒸馏入口（同一脚本，`--arm` 选择臂）。

两臂共享：数据、日语 prompt collator、在线 rollout、p_ref(disable_adapter)、KL 门控损失、超参、评测协议。
**唯一差异**：枢轴寻找方式（PDSD 激活突变 gate vs RP-OPSD PRS gate）。

冒烟（验证能跑通，极小配置）：
    python train_pdsd.py --data data/gen_v3_pilot.reverified.jsonl --smoke
正式：
    python train_pdsd.py --data data/generated_1000.train.jsonl --arm pdsd \
        --output-dir outputs/pdsd_ja --max-completion-length 4096 --epochs 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

# 本机 deepspeed 已安装但 import 即崩（无 CUDA toolkit → CUDA_HOME 缺失）。
# accelerate 的 unwrap_model 会在 `is_deepspeed_available()` 为真时 import 它，从而炸掉 Trainer 初始化。
# 我们不用 deepspeed，直接让该探测返回 False。
import accelerate.utils.other as _acc_other  # noqa: E402

_acc_other.is_deepspeed_available = lambda: False

import dataclasses  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl.experimental.gold import GOLDConfig  # noqa: E402

BASE = Path(__file__).resolve().parent
_RP_SRC = BASE / "RP-OPSD" / "RP-OPSD" / "src"
if str(_RP_SRC) not in sys.path:
    sys.path.append(str(_RP_SRC))

from rp_opsd_trainer import RPOPSDTrainer  # noqa: E402
from pdsd_collator import SelfDistillationDataCollator  # noqa: E402
from pdsd_trainer import PDSDTrainer  # noqa: E402

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser(description="PDSD/RP-OPSD 在线蒸馏")
    p.add_argument("--data", required=True)
    p.add_argument("--arm", choices=["pdsd", "rpopsd"], default="pdsd")
    p.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--output-dir", default="outputs/distill")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--max-completion-length", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--reference-lambda", type=float, default=0.2)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="QLoRA：4bit 量化底座，省显存（8GB 卡想上 2048 预算时用）")
    p.add_argument("--attn-impl", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--report-to", default="none", help="none / wandb")
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条（冒烟）")
    p.add_argument("--seed", type=int, default=0)
    # PDSD 枢轴参数
    p.add_argument("--pivot-mode", choices=["spans", "sigmoid"], default="spans")
    p.add_argument("--pivot-rho", type=float, default=0.45,
                   help="方案(b)：目标 mean(gate)，按 shift z 降序取峰直到达标；"
                        "应设为 RP-OPSD 臂实测的 rp_gate_mean 以对齐监督预算。-1 关闭（回到阈值法）")
    p.add_argument("--window-k", type=int, default=5)
    p.add_argument("--layer-lo", type=float, default=0.25)
    p.add_argument("--layer-hi", type=float, default=0.65)
    p.add_argument("--vec-dim", type=int, default=1)
    p.add_argument("--z-threshold", type=float, default=1.0)
    p.add_argument("--extend-before", type=int, default=2)
    p.add_argument("--extend-after", type=int, default=4)
    p.add_argument("--smoke", action="store_true", help="极小配置，仅验证能跑通")
    return p.parse_args()


def build_config(a) -> GOLDConfig:
    wanted = dict(
        output_dir=a.output_dir,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        max_length=a.max_length,
        max_completion_length=a.max_completion_length,
        temperature=a.temperature,
        top_p=a.top_p,
        top_k=a.top_k,
        gradient_checkpointing=a.gradient_checkpointing,
        logging_steps=1,
        save_strategy="no",
        report_to=[] if a.report_to == "none" else [a.report_to],
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        bf16=True,
        use_vllm=False,
        seed=a.seed,
        data_seed=a.seed,
    )
    # 只保留 GOLDConfig 真正存在的字段（不同 trl 版本字段有出入）
    valid = {f.name for f in dataclasses.fields(GOLDConfig)}
    kwargs = {k: v for k, v in wanted.items() if k in valid}
    dropped = sorted(set(wanted) - set(kwargs))
    if dropped:
        print(f"[warn] GOLDConfig 无这些字段，已忽略: {dropped}")
    cfg = GOLDConfig(**kwargs)

    init_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": a.attn_impl,
        "use_cache": not a.gradient_checkpointing,
    }
    if a.load_in_4bit:      # QLoRA：4bit 量化底座，省 ~2.4GB 权重显存
        from transformers import BitsAndBytesConfig

        init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        init_kwargs["device_map"] = "auto"
        print("[4bit] QLoRA：底座 4bit 量化")
    cfg.model_init_kwargs = init_kwargs
    return cfg


def main():
    a = parse_args()
    if a.smoke:
        a.max_length = min(a.max_length, 1024)
        a.max_completion_length = min(a.max_completion_length, 256)
        a.epochs = 1
        a.batch_size = 1
        a.grad_accum = 1
        a.limit = a.limit or 2
        print("[smoke] 极小配置：2 条数据、256 生成长度、1 步")

    torch.manual_seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        a.model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=a.data)["train"]
    ds = ds.map(lambda ex: {"problem_en": ex.get("problem_en") or ex["problem"]})
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
    print(f"数据: {len(ds)} 条 | arm={a.arm} | rollout 上限={a.max_completion_length}")

    collator = SelfDistillationDataCollator(
        tokenizer, max_length=a.max_length, lang="JA", student_enable_thinking=True)
    peft_config = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
    cfg = build_config(a)

    common = dict(
        model=a.model_name, args=cfg, train_dataset=ds, processing_class=tokenizer,
        peft_config=peft_config, data_collator=collator,
        student_enable_thinking=True, rp_reference_lambda=a.reference_lambda,
    )
    if a.arm == "pdsd":
        pivot_cfg = dict(vec_dim=a.vec_dim, layer_lo=a.layer_lo, layer_hi=a.layer_hi,
                         window_k=a.window_k, mode=a.pivot_mode, z_threshold=a.z_threshold,
                         extend_before=a.extend_before, extend_after=a.extend_after)
        if a.pivot_rho is not None and a.pivot_rho >= 0:
            pivot_cfg["target_rho"] = a.pivot_rho
        trainer = PDSDTrainer(pivot_cfg=pivot_cfg, **common)
        print(f"PDSD 枢轴: mode={a.pivot_mode} k={a.window_k} layers=[{a.layer_lo},{a.layer_hi}) "
              f"vec_dim={a.vec_dim} target_rho={a.pivot_rho}")
    else:
        trainer = RPOPSDTrainer(**common)
        print("RP-OPSD：PRS gate")

    gc = trainer.generation_config
    print(f"[gen] rollout 采样: max_new_tokens={gc.max_new_tokens} temperature={gc.temperature} "
          f"top_p={gc.top_p} top_k={gc.top_k} do_sample={gc.do_sample}")
    print(f"[gen] 评测协议须与之一致: temp=1.1 top_p=0.95 top_k=20 n_samples=3 "
          f"max_new_tokens={a.max_completion_length}")

    trainer.train()
    trainer.save_model(a.output_dir)
    print(f"训练完成 -> {a.output_dir}")


if __name__ == "__main__":
    main()
