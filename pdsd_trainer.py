"""PDSDTrainer：与 RPOPSDTrainer 完全相同的损失结构，只把 per-token gate 换成"激活突变"来源。

相对父类的唯一实质改动：
  1. q_plus 前向加 `output_hidden_states=True`，在该前向的逐层 hidden_states 上算 PDSD gate
     （`pdsd_gate.pdsd_gate`，在完成段内做 z-score/找峰）；
  2. 默认**跳过** q_minus（消融）前向 —— PDSD 的 gate 不需要反事实对比（RP-OPSD 需要）。
其余一切（在线 rollout、p_ref = disable_adapter、KL 损失与门控结构、日志）完全复用父类，
从而保证两臂的**唯一变量是"枢轴寻找方式"**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_RP_SRC = Path(__file__).resolve().parent / "RP-OPSD" / "RP-OPSD" / "src"
if str(_RP_SRC) not in sys.path:
    sys.path.append(str(_RP_SRC))

from rp_opsd_trainer import RPOPSDTrainer  # noqa: E402

from pdsd_gate import pdsd_gate  # noqa: E402


def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class PDSDTrainer(RPOPSDTrainer):
    def __init__(self, *args, pivot_cfg: dict | None = None,
                 use_ablation_forward: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pivot_cfg = dict(pivot_cfg or {})      # 传给 pdsd_gate：vec_dim/layer_lo/layer_hi/window_k/mode/...
        self.use_ablation_forward = use_ablation_forward

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        completion_mask = inputs["labels"][:, student_prompt_len:] != -100

        if return_outputs:
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()

        with torch.no_grad():
            # ---- q_plus：多取 hidden_states，用于 PDSD 的激活突变 ----
            outputs_full_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
                output_hidden_states=True,
            )
            hidden_states = outputs_full_teacher.hidden_states
            teacher_logits_for_loss = outputs_full_teacher.logits[
                :, teacher_prompt_len - 1 : -1, :].detach()
            del outputs_full_teacher
            _empty_cache()

            gate, score_z = pdsd_gate(
                hidden_states, completion_mask, prompt_len=teacher_prompt_len, **self.pivot_cfg)
            del hidden_states
            _empty_cache()

            # 与 RP-OPSD 臂保持一致的 gate warmup（同 schedule，见父类 _gate_warmup_alpha）。
            # 这是训练稳定性日程，不属于"枢轴寻找方式"，两臂必须相同。
            alpha = self._gate_warmup_alpha()
            if alpha < 1.0:
                gate = ((1.0 - alpha) + alpha * gate) * completion_mask.to(gate.dtype)

            log_q_full = F.log_softmax(teacher_logits_for_loss, dim=-1).detach()
            q_full = log_q_full.exp()
            if self.use_ablation_forward:
                outputs_ab = model(
                    input_ids=inputs["ablation_teacher_input_ids"],
                    attention_mask=inputs["ablation_teacher_attention_mask"],
                )
                ab_logits = outputs_ab.logits[
                    :, inputs["ablation_teacher_prompt_length"] - 1 : -1, :]
                log_q_ab = F.log_softmax(ab_logits, dim=-1).detach()
                score_a = torch.sum(q_full * (log_q_full - log_q_ab), dim=-1).detach()
                score_a = score_a * completion_mask.to(score_a.dtype)
                del outputs_ab, ab_logits, log_q_ab
            else:
                score_a = score_z            # PDSD 不用反事实；日志里用 shift 的 z 充当"枢轴分数"
            del q_full, log_q_full
            _empty_cache()

        # ---- p_ref：同一模型关掉 LoRA adapter ----
        with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
            outputs_reference = model(
                input_ids=inputs["student_input_ids"],
                attention_mask=inputs["student_attention_mask"],
            )
            reference_logits_for_loss = outputs_reference.logits[
                :, student_prompt_len - 1 : -1, :].detach()
            del outputs_reference
            _empty_cache()

        # ---- student：带梯度 ----
        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]
        kl_student_full = self._token_kl_with_teacher(student_logits, teacher_logits_for_loss)
        kl_student_ref = self._token_kl_with_teacher(student_logits, reference_logits_for_loss)

        mask_float = completion_mask.to(kl_student_full.dtype)
        gate_float = gate.to(kl_student_full.dtype)
        full_loss_weight = mask_float * gate_float
        reference_loss_weight = mask_float * float(self.rp_reference_lambda) * (1.0 - gate_float)
        loss_weight = full_loss_weight + reference_loss_weight
        denominator = mask_float.sum().clamp_min(1.0)
        loss = (
            (full_loss_weight * kl_student_full).sum()
            + (reference_loss_weight * kl_student_ref).sum()
        ) / denominator

        self._record_rp_metrics(
            score_a=score_a.detach(),
            gate=gate.detach(),
            kl_student_full=kl_student_full.detach(),
            completion_mask=completion_mask,
            loss=loss.detach(),
            loss_weight=loss_weight.detach(),
            kl_student_ref=kl_student_ref.detach(),
            reference_loss_weight=reference_loss_weight.detach(),
        )
        # PDSD 专属日志：枢轴段密度、shift z 的均值
        n_valid = max(1, int(completion_mask.sum()))
        self._metrics["train"]["pdsd_gate_density"].append(
            float((((gate > 0.5) & completion_mask).sum()) / n_valid))
        self._metrics["train"]["pdsd_shift_z_mean"].append(
            float(score_z[completion_mask].mean()) if completion_mask.any() else 0.0)

        del (outputs_student, student_logits, teacher_logits_for_loss, reference_logits_for_loss,
             score_a, gate, loss_weight, reference_loss_weight, kl_student_full, kl_student_ref)
        _empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return loss, minimal_output
        return loss
