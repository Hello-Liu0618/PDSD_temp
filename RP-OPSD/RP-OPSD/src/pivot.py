"""Self-contained extraction of the RP-OPSD "pivot" (reference-solution) filter.

This module re-implements, in isolation, the two pieces that together select how
much the reference solution should steer the student at each completion token:

  1. :func:`compute_pivot_score` -- the raw per-token signal
     ``KL(q_plus || q_minus)``, i.e. how much conditioning on the English
     reference solution perturbs the model's next-token distribution relative to
     conditioning on the English translation alone.
  2. :class:`PivotGate` -- the EMA-normalized, sigmoid gate that maps that signal
     into a per-token weight in ``[g_min, 1]``.

It mirrors ``compute_loss`` / ``_compute_rp_gate`` / ``_update_rp_score_ema`` /
``_global_score_stats`` / ``_gate_warmup_alpha`` from ``rp_opsd_trainer.py`` and is
meant to be numerically identical when fed the same tensors, while carrying no
dependency on ``trl`` or on the trainer itself.

Reuse contract (inherited from the trainer, see ``training_step``):

* The three prompts are produced by ``RPOPSDSelfDistillationDataCollator``
  (``student_prompts``, ``teacher_prompts``, ``ablation_teacher_prompts``).
* A single completion is sampled from the **student** prompt (on-policy), then the
  *same* completion is appended to all three prompts before scoring. This is what
  makes the three views comparable -- the teachers re-score the student's own
  tokens under different conditioning, they do not generate their own.
* The logits slice ``prompt_len - 1 : -1`` aligns the teacher/ablation logits with
  the shared completion tokens (logits at position ``i`` predict token ``i+1``).

Typical usage::

    from pivot import compute_pivot_score, PivotGate, build_full_sequences

    full, ablation, completion_mask, _ = build_full_sequences(
        batch, generated_ids, pad_token_id
    )
    gate_fn = PivotGate()  # defaults match the manuscript: beta=2, tau=0, g_min=0.05
    score, q_full, log_q_full = compute_pivot_score(
        model, **full_and_ablation_view(full, ablation, completion_mask)
    )
    gate = gate_fn(score, completion_mask, step=state.global_step, max_steps=100)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "compute_pivot_score",
    "PivotGate",
    "build_full_sequences",
]


@torch.no_grad()
def compute_pivot_score(
    model,
    *,
    full_input_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    full_prompt_len: int,
    ablation_input_ids: torch.Tensor,
    ablation_attention_mask: torch.Tensor,
    ablation_prompt_len: int,
    completion_mask: torch.Tensor,
):
    """Return the per-token pivot score ``KL(q_plus || q_minus)``.

    Parameters
    ----------
    model:
        The student policy (any module returning ``.logits``). Called under
        ``no_grad``; the LoRA/adapters are expected to be **enabled** here, since
        both views are evaluations of the current policy.
    full_input_ids / full_attention_mask:
        ``[teacher_prompt][generation]`` -- the solution-conditioned view ``q_plus``.
    full_prompt_len:
        Batch-max length of the teacher prompt (same role as
        ``inputs["teacher_prompt_length"]``).
    ablation_input_ids / ablation_attention_mask:
        ``[ablation_prompt][generation]`` -- the no-reference view ``q_minus``.
    ablation_prompt_len:
        Batch-max length of the ablation prompt.
    completion_mask:
        ``[B, gen_len]`` bool/int mask, ``1`` on real completion tokens, ``0`` on
        padding. Shared by all three views because the completion is shared.

    Returns
    -------
    score:
        ``[B, gen_len]`` per-token ``KL(q_plus || q_minus)``, zeroed on padding.
    q_full:
        ``[B, gen_len, V]`` the ``q_plus`` distribution (returned so callers can
        reuse it, e.g. for a ``KL(q_plus || p_ref)`` contrast).
    log_q_full:
        ``[B, gen_len, V]`` ``log q_plus``, likewise for downstream reuse.
    """
    out_full = model(input_ids=full_input_ids, attention_mask=full_attention_mask)
    full_logits = out_full.logits[:, full_prompt_len - 1 : -1, :].detach()  # [B, gen_len, V]
    log_q_full = F.log_softmax(full_logits, dim=-1).detach()
    q_full = log_q_full.exp()

    out_ablation = model(
        input_ids=ablation_input_ids, attention_mask=ablation_attention_mask
    )
    ablation_logits = out_ablation.logits[:, ablation_prompt_len - 1 : -1, :].detach()
    log_q_ablation = F.log_softmax(ablation_logits, dim=-1).detach()

    score = torch.sum(q_full * (log_q_full - log_q_ablation), dim=-1)  # [B, gen_len]
    score = score * completion_mask.to(score.dtype)
    return score, q_full, log_q_full


class PivotGate:
    """EMA-normalized, sigmoid gate over per-token pivot scores.

    Mirrors ``_update_rp_score_ema`` + ``_compute_rp_gate`` + ``_gate_warmup_alpha``.
    The score statistics (mean/std) are tracked with an exponential moving average
    across batches; each batch is then z-scored against that EMA and mapped through
    ``sigmoid(beta * (z - tau))`` into ``[g_min, 1]``.

    Note on multi-rank training: the original ``_global_score_stats`` all-reduces the
    sum / sum-of-squares / count across processes before the EMA update so that all
    ranks share one normalization. This single-process version uses the local batch
    statistics only; callers running DDP/DeepSpeed should reduce those three scalars
    themselves before calling :meth:`update` if cross-rank consistency is required.
    """

    def __init__(
        self,
        beta: float = 2.0,
        tau: float = 0.0,
        g_min: float = 0.05,
        ema_decay: float = 0.99,
        z_clip: float = 5.0,
        warmup_ratio: float = 0.05,
        transition_ratio: float = 0.05,
    ):
        self.beta = beta
        self.tau = tau
        self.g_min = g_min
        self.ema_decay = ema_decay
        self.z_clip = z_clip
        self.warmup_ratio = warmup_ratio
        self.transition_ratio = transition_ratio

        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self._initialized = False

    def update(self, score: torch.Tensor, mask: torch.Tensor):
        """Fold the current batch's score statistics into the running EMA.

        Uses the float64 sum / sum-of-squares / count formulation from
        ``_global_score_stats`` for numerical stability, then applies the EMA decay.
        The first non-empty batch initializes ``mean``/``std`` directly.
        """
        valid = score[mask.bool()].float()
        if valid.numel() == 0:
            if self.mean is None:
                self.mean = torch.zeros((), dtype=score.dtype, device=score.device)
                self.std = torch.ones((), dtype=score.dtype, device=score.device)
            return self.mean, self.std

        v64 = valid.to(torch.float64)
        s = v64.sum()
        ss = (v64 * v64).sum()
        n = torch.tensor(v64.numel(), dtype=torch.float64, device=score.device)

        count = n.clamp_min(1.0)
        batch_mean = (s / count).to(score.dtype).detach()
        variance = (ss / count - (s / count) * (s / count)).clamp_min(1e-12)
        batch_std = torch.sqrt(variance).to(score.dtype).detach().clamp_min(1e-6)

        if not self._initialized or self.mean is None:
            self.mean = batch_mean
            self.std = batch_std
            self._initialized = True
        else:
            d = self.ema_decay
            self.mean = (d * self.mean + (1.0 - d) * batch_mean).detach()
            self.std = (d * self.std + (1.0 - d) * batch_std).detach().clamp_min(1e-6)

        return self.mean, self.std

    def warmup_alpha(self, step: int, max_steps: int) -> float:
        """Return the interpolation coefficient for the gate warmup schedule.

        ``0.0`` during the uniform warmup (gate forced to 1), rising linearly to
        ``1.0`` over the transition window, then ``1.0`` afterwards. Mirrors
        ``_gate_warmup_alpha``.
        """
        if max_steps <= 0:
            return 1.0
        warmup = max(1, int(round(max_steps * self.warmup_ratio)))
        transition = max(1, int(round(max_steps * self.transition_ratio)))
        if step < warmup:
            return 0.0
        if step < warmup + transition:
            return (step - warmup + 1) / transition
        return 1.0

    def __call__(
        self,
        score: torch.Tensor,
        mask: torch.Tensor,
        *,
        step: int | None = None,
        max_steps: int = 0,
        update: bool = True,
    ) -> torch.Tensor:
        """Map a pivot-score tensor to a per-token gate in ``[g_min, 1]``.

        By default this also updates the EMA (mirroring ``_compute_rp_gate``, which
        always updates). Pass ``update=False`` to score held-out data without
        mutating state (e.g. offline analysis). ``step``/``max_steps`` enable the
        warmup schedule; omit ``step`` to skip it.
        """
        if update:
            self.update(score, mask)
        if self.mean is None:
            self.update(score, mask)

        z = (score - self.mean) / (self.std + 1e-6)
        z = z.clamp(-self.z_clip, self.z_clip)
        g = self.g_min + (1.0 - self.g_min) * torch.sigmoid(self.beta * (z - self.tau))

        if step is not None:
            alpha = self.warmup_alpha(int(step), int(max_steps))
            if alpha < 1.0:
                g = (1.0 - alpha) + alpha * g

        return g.detach() * mask.to(g.dtype)


def build_full_sequences(
    batch: dict,
    generated_ids: torch.Tensor,
    pad_token_id: int | None,
) -> dict:
    """Reconstruct the three full sequences from a collator batch + on-policy gen.

    This reproduces the sequence-assembly block of ``training_step`` (the part that
    builds ``student_input_ids`` / ``teacher_input_ids`` /
    ``ablation_teacher_input_ids`` and ``labels``) so the pivot score can be computed
    offline or in a different harness without touching the trainer.

    Parameters
    ----------
    batch:
        Output of ``RPOPSDSelfDistillationDataCollator``. Needs ``student_prompts``,
        ``student_prompt_length``, ``teacher_prompts``, ``teacher_prompt_length``,
        ``ablation_teacher_prompts``, ``ablation_teacher_prompt_length`` and (for a
        faithful ``completion_mask``) ``student_prompt_lengths_per_example``.
    generated_ids:
        ``[B, student_prompt_len + gen_len]`` -- the full ``model.generate`` output
        from ``student_prompts`` (i.e. prompt already included).
    pad_token_id:
        Tokenizer pad id used to mask padded completion tokens; may be ``None``.

    Returns
    -------
    dict with keys:
        ``full_input_ids``, ``full_attention_mask``, ``full_prompt_len``,
        ``ablation_input_ids``, ``ablation_attention_mask``, ``ablation_prompt_len``,
        ``completion_mask`` (``[B, gen_len]``), ``generation_ids`` (``[B, gen_len]``),
        ``student_prompt_len``. Ready to spread into :func:`compute_pivot_score`.
    """
    student_prompt_len = int(batch["student_prompt_length"])
    teacher_prompt_len = int(batch["teacher_prompt_length"])
    ablation_prompt_len = int(batch["ablation_teacher_prompt_length"])

    generation_ids = generated_ids[:, student_prompt_len:]  # [B, gen_len]

    full_input_ids = torch.cat([batch["teacher_prompts"], generation_ids], dim=1)
    ablation_input_ids = torch.cat([batch["ablation_teacher_prompts"], generation_ids], dim=1)

    def _mask(ids: torch.Tensor) -> torch.Tensor:
        m = torch.ones_like(ids)
        if pad_token_id is not None:
            m[ids == pad_token_id] = 0
        return m

    # Completion mask over the shared generation: prompt tokens are masked out by
    # per-example lengths, padding by the pad token id.
    labels = generated_ids.clone()
    per_example = batch.get("student_prompt_lengths_per_example")
    if per_example is not None:
        for i in range(labels.shape[0]):
            actual = int(per_example[i].item())
            labels[i, :actual] = -100
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    completion_mask = (labels[:, student_prompt_len:] != -100)

    return {
        "full_input_ids": full_input_ids,
        "full_attention_mask": _mask(full_input_ids),
        "full_prompt_len": teacher_prompt_len,
        "ablation_input_ids": ablation_input_ids,
        "ablation_attention_mask": _mask(ablation_input_ids),
        "ablation_prompt_len": ablation_prompt_len,
        "completion_mask": completion_mask,
        "generation_ids": generation_ids,
        "student_prompt_len": student_prompt_len,
    }
