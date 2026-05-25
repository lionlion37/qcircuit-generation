# SWAP-Count Conditioning via Fine-tuning from Baseline

**Date:** 2026-05-25
**Status:** training pending
**Experiment ID:** `unitary_swap_finetuned`

---

## Motivation

The from-scratch SWAP-conditioned model (`unitary_swap_count_conditioning`) achieved near-perfect
SWAP compliance but suffered a ~30 percentage-point accuracy drop even at `swap_count=1` — the
baseline's natural output. The diagnosis: adding the `; swap_count=N` suffix changes the CLIP
embedding enough to disrupt the compilation signal, and training from scratch forced the model to
rebuild compilation ability under a harder dual objective, which it failed to do fully.

**Hypothesis:** Starting from baseline weights (96.1% exact_found_rate) should preserve the
compiled compilation knowledge. The model only needs to adapt to the shifted CLIP embedding
distribution introduced by the swap_count suffix, not relearn compilation from scratch.

---

## Training Setup

| Parameter | From-scratch | Fine-tuned |
|---|---|---|
| Init weights | random | baseline (`paper_unitary`) |
| Learning rate | 3e-4 | **3e-5** (10× lower) |
| LR scheduler | OneCycleLR | **CosineAnnealingLR** (gentler) |
| Epochs | 150 | **75** |
| Dataset | swap-constrained | swap-constrained (same) |
| Architecture | identical | identical |
| Batch size | 256 | 256 |

Lower learning rate and CosineAnnealingLR avoid overwriting the pretrained weights aggressively.
75 epochs should be sufficient to adapt the embedding distribution shift without catastrophic
forgetting.

## Training command

```bash
export WANDB_MODE=offline
python scripts/train_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  training=unitary_swap_finetuned
```

---

## Evaluation

Reuse `notebooks/evaluation/unitary/swap_constrained_evaluation.ipynb` with two config changes:

```python
SWAP_MODEL_DIR = "./artifacts/models/unitary-swap-finetuned/unitary_swap_finetuned"
ARTIFACT_SUBDIR = "unitary-swap-finetuned"
```

---

## What to Look For

**Primary question:** Does fine-tuning recover compilation accuracy while keeping SWAP compliance?

| Metric | From-scratch | Target for fine-tuned |
|---|---|---|
| exact_found_rate (count=0) | 0.648 | > 0.80 |
| exact_found_rate (count=1) | 0.633 | > 0.85 |
| exact_found_rate (count=2) | 0.672 | > 0.85 |
| SWAP compliance (all counts) | ~1.000 | ~1.000 |
| median infidelity (wrong) | 0.938 | < 0.80 |

**Diagnostic questions:**

1. If accuracy recovers to ~90%+ at counts 0–2: fine-tuning from baseline is the right strategy;
   CLIP conditioning is not fundamentally broken, just needs a warm start.

2. If accuracy recovers partially (70–85%) but compliance stays near 100%: confirms the
   from-scratch result is a training-strategy issue, not an architecture issue. The model can
   learn both objectives given a good start, but there is still some irreducible tension.

3. If accuracy barely improves over from-scratch: the CLIP embedding shift is the root cause
   regardless of init; a dedicated scalar conditioning channel would be needed.

4. If compliance drops noticeably below 100%: the fine-tuned model is prioritising compilation
   over SWAP adherence — may indicate LR is too high or too many epochs.

---

## Results

*(to be filled after training and evaluation)*

| Model / requested count | exact_found_rate | exact_compliance_rate | mean_actual_swap |
|---|---|---|---|
| baseline | 0.961 | — | 0.988 |
| from-scratch, count=0 | 0.648 | 1.000 | 0.000 |
| from-scratch, count=1 | 0.633 | 1.000 | 1.000 |
| from-scratch, count=2 | 0.672 | 0.999 | 2.000 |
| finetuned, count=0 | — | — | — |
| finetuned, count=1 | — | — | — |
| finetuned, count=2 | — | — | — |
