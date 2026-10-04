# RP-OPSD

This repository provides the training and evaluation code for RP-OPSD, together
with the Swahili (SWA) AfriMGSM and PolyMath evaluation data used by the main
experiment.

## Contents

```text
src/                         RP-OPSD training and evaluation code
scripts/train.sh             Main SWA training entry point
scripts/evaluate.sh          SWA AfriMGSM/PolyMath evaluation entry point
configs/accelerate_zero2.yaml
datasets/afrimgsm/swa/test/  250 AfriMGSM test examples
datasets/polymath/swa/       1,000 PolyMath examples in four difficulty splits
environment.yml
```

The main objective implemented here is
`g * KL(q_plus || p_student) + 0.2 * (1 - g) * KL(p_ref || p_student)`.
The two teacher views are stop-gradient evaluations of the current policy. The
reference distribution is the frozen base policy obtained by disabling the LoRA
adapter. The gate uses EMA-normalized `KL(q_plus || q_minus)` with the manuscript
settings (`beta=2`, `tau=0`, `g_min=0.05`, EMA decay `0.99`, 5% uniform warmup,
and 5% interpolation).

## Environment

```bash
conda env create -f environment.yml
conda activate rp-opsd
pip install flash-attn --no-build-isolation
```

Use a CUDA-capable NVIDIA GPU. The default model is `Qwen/Qwen3-1.7B`; override it
with `MODEL_NAME`.

## Training data

Training data is not bundled so that the archive contains only the requested SWA
AfriMGSM and PolyMath benchmark data. Set `DATA_PATH` to the 500-example,
OpenThoughts-derived SWA training JSON. Each row must contain:

```json
{
  "problem": "English problem",
  "problem_swa": "Swahili translation",
  "solution": "English reference solution"
}
```

If a row contains `problem_swa_ok`, it must be `true`.

## Train

```bash
DATA_PATH=/path/to/translated_swa.json \
CUDA_DEVICES=0,1,2,3 \
bash scripts/train.sh
```

The launcher fixes the manuscript's main configuration: 100 steps, effective batch
size 32, learning rate `5e-6`, bfloat16, LoRA rank 64/alpha 128, maximum completion
length 2,048, temperature 1.1, top-p 0.95, and top-k 20. Outputs are written to
`outputs/rp-opsd-swa` by default.

## Evaluate

```bash
CHECKPOINT_DIR=outputs/rp-opsd-swa/checkpoint-100 \
GPU_IDS=0 \
bash scripts/evaluate.sh afrimgsm

CHECKPOINT_DIR=outputs/rp-opsd-swa/checkpoint-100 \
GPU_IDS=0 \
bash scripts/evaluate.sh polymath
```

AfriMGSM uses pass@12. PolyMath uses one generation per problem and reports
difficulty-weighted ACC@1 (weights: low 1, medium 2, high 3, top 4). For a quick
integration check, add `NUM_SAMPLES=2 MAX_NEW_TOKENS=64`.

The bundled benchmark files originate from the public
[AfriMGSM](https://huggingface.co/datasets/masakhane/afrimgsm) and
[PolyMath](https://huggingface.co/datasets/Qwen/PolyMath) releases; their original
licenses and terms apply.

## Repository scope

The repository contains no machine-specific absolute paths, experiment logs,
checkpoints, or online tracking identifiers. Weights & Biases is disabled by default.
