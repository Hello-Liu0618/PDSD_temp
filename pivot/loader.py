"""模型与分词器加载（生成与探针共用）。"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str, torch_dtype: torch.dtype,
                             trust_remote_code: bool = True):
    """加载模型（bf16/fp16、device_map=auto）与分词器，模型置 eval。

    优先用本地缓存离线加载（不受代理/网络影响），缓存缺失时才联网下载。

    Returns:
        (model, tokenizer)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, trust_remote_code=trust_remote_code,
            device_map="auto", local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, trust_remote_code=trust_remote_code,
            device_map="auto")
    model.eval()
    return model, tokenizer


def model_device(model) -> torch.device:
    """取模型参数所在设备（device_map 下也稳健）。"""
    return next(model.parameters()).device
