#!/usr/bin/env python3
"""PDSD / RP-OPSD 双臂在线蒸馏入口（同一脚本，`--arm` 选择臂）。

两臂共享：数据、日语 prompt collator、在线 rollout、p_ref(disable_adapter)、KL 门控损失、超参、评测协议。
**唯一差异**：枢轴寻找方式（PDSD 激活突变 gate vs RP-OPSD PRS gate）。

默认超参**对齐原版 RP-OPSD 的 `scripts/train.sh`**：
    max_completion_length=2048, lr=5e-6, max_grad_norm=0.1, 有效 batch=32,
    max_steps=100, lmbda=1, beta=0, gradient_checkpointing 常开, vLLM colocate。
对齐理由见 TRAINER_SPEC.md §9.2；lr/grad_norm 是**有效性**参数，不对齐则比较失去意义。

冒烟（极小配置，验证能跑通）：
    python train_pdsd.py --data data/gen_v3_pilot.reverified.jsonl --smoke
正式：
    python train_pdsd.py --data data/generated_1000.clean_merged.train.jsonl --arm pdsd \
        --output-dir outputs/pdsd --use-vllm
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
# KL 损失会产生大量大块张量，碎片容易导致 OOM；此设置开销很小
# torch 2.9 起改名，两个都设以兼容不同版本
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 有些机器 deepspeed 已安装但 import 即崩（无 CUDA toolkit → CUDA_HOME 缺失）。
# accelerate 的 unwrap_model 会在 `is_deepspeed_available()` 为真时 import 它，从而炸掉 Trainer 初始化。
# 本实验不用 deepspeed（单卡不需要 ZeRO），直接让该探测返回 False。
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


def _best_attn() -> str:
    """装了 flash-attn 就用它（与原版一致），否则退回 sdpa。"""
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:  # noqa: BLE001
        return "sdpa"


def parse_args():
    p = argparse.ArgumentParser(description="PDSD/RP-OPSD 在线蒸馏（默认对齐原版 RP-OPSD）")
    p.add_argument("--data", required=True)
    p.add_argument("--arm", choices=["pdsd", "rpopsd"], default="pdsd")
    p.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--output-dir", default="outputs/distill")

    # ---- 与原版 train.sh 对齐的默认值 ----
    p.add_argument("--max-length", type=int, default=20000)
    p.add_argument("--max-completion-length", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-grad-norm", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=100, help="-1 = 用 --epochs 跑满")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=1, help="per_device")
    p.add_argument("--grad-accum", type=int, default=32, help="有效 batch = 该值 × batch-size × 进程数")
    p.add_argument("--lmbda", type=float, default=1.0, help="GOLD 参数（主路径不读，仅为与原文一致）")
    p.add_argument("--beta", type=float, default=0.0, help="同上")
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--reference-lambda", type=float, default=0.2)
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="默认开启（与原版一致）；关掉更慢但省一次重算")
    p.add_argument("--use-vllm", action="store_true", help="vLLM colocate 批量生成（把生成提速一个数量级）")
    p.add_argument("--steps-per-generation", type=int, default=1,
                   help="每多少次梯度累积才做一次 vLLM 生成（>1 则一次批量生成多prompt，吞吐更高）。"
                        "TRL 要求它是 gradient_accumulation_steps 的整数倍，例如 grad_accum=32 时填 32 或 64。")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3,
                   help="vLLM 与训练共卡时的显存预留比例。原文用 0.4（卡更大）；"
                        "32GB 卡务必 ≤0.3，否则 vLLM 预留过多、训练必 OOM。注意 vLLM 默认是 0.9！")
    p.add_argument("--load-in-4bit", action="store_true", help="QLoRA 4bit 底座（省 ~2.4GB）")
    p.add_argument("--attn-impl", default=None, choices=[None, "sdpa", "eager", "flash_attention_2"])
    p.add_argument("--report-to", default="none", help="none / wandb")
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条（冒烟）")
    p.add_argument("--seed", type=int, default=0)

    # ---- PDSD 枢轴参数 ----
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


def build_config(a, grad_ckpt: bool, attn_impl: str) -> GOLDConfig:
    wanted = dict(
        output_dir=a.output_dir,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        learning_rate=a.lr,
        max_grad_norm=a.max_grad_norm,
        lmbda=a.lmbda,
        beta=a.beta,
        max_length=a.max_length,
        max_completion_length=a.max_completion_length,
        temperature=a.temperature,
        top_p=a.top_p,
        top_k=a.top_k,
        gradient_checkpointing=grad_ckpt,
        logging_steps=2,
        save_strategy="no",
        report_to=[] if a.report_to == "none" else [a.report_to],
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        bf16=True,
        use_vllm=a.use_vllm,
        steps_per_generation=a.steps_per_generation,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=a.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=1,
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
        "attn_implementation": attn_impl,
        "use_cache": not grad_ckpt,
    }
    if a.load_in_4bit:      # QLoRA：4bit 量化底座
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
    grad_ckpt = not a.no_gradient_checkpointing
    attn_impl = a.attn_impl or _best_attn()

    if a.use_vllm:
        try:
            import vllm  # noqa: F401
        except Exception:  # noqa: BLE001
            print("[警告] 指定了 --use-vllm 但 vllm 未安装 → 已回退到 HF generate。"
                  "生成会慢一个数量级！安装： pip install vllm")
            a.use_vllm = False

    if a.smoke:
        a.max_length = min(a.max_length, 1024)
        a.max_completion_length = min(a.max_completion_length, 256)
        a.max_steps, a.epochs = 2, 1
        a.batch_size, a.grad_accum = 1, 1
        a.limit = a.limit or 2
        a.use_vllm = False          # 冒烟不用 vLLM（省去引擎初始化）
        print("[smoke] 极小配置：2 条数据、256 生成长度、2 步、关 vLLM")

    # 显存提示：KL 里 log_softmax 对完整词表输出，随完成长度线性增长
    est = {1024: 7, 1536: 12, 2048: 18, 3072: 25, 4096: 35}.get(a.max_completion_length)
    note = ""
    if a.max_completion_length > 2048 and not a.load_in_4bit:
        note = "  ⚠ 32GB 卡在 4096 会 OOM（实测）；需 48GB+ 或 --load-in-4bit"
    print(f"[配置] 显存估计 ~{est or '?'}GB | attn={attn_impl} | grad_ckpt={grad_ckpt} | "
          f"vLLM={a.use_vllm}{note}")
    print(f"[对齐原文] lr={a.lr} max_grad_norm={a.max_grad_norm} "
          f"有效batch={a.batch_size * a.grad_accum} max_steps={a.max_steps} "
          f"rollout={a.max_completion_length}")

    torch.manual_seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        a.model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=a.data)["train"]
    ds = ds.map(lambda ex: {"problem_en": ex.get("problem_en") or ex["problem"]})
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
    print(f"[数据] {len(ds)} 条 | arm={a.arm}")

    collator = SelfDistillationDataCollator(
        tokenizer, max_length=a.max_length, lang="JA", student_enable_thinking=True)
    peft_config = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
    cfg = build_config(a, grad_ckpt, attn_impl)

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
        print(f"[PDSD] mode={a.pivot_mode} k={a.window_k} layers=[{a.layer_lo},{a.layer_hi}) "
              f"vec_dim={a.vec_dim} target_rho={a.pivot_rho}")
    else:
        trainer = RPOPSDTrainer(**common)
        print("[RP-OPSD] PRS gate（原版 trainer）")

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
