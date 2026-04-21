# Curriculum Learning Investigation Notes

**Date:** 2026-04-21
**Status:** In progress — stage2 underperforms baseline

## Current Results (128 eval unitaries, 128 samples each)

| Model | exact_found_rate | mean_distinct_correct | mean_best_model_distance |
|---|---|---|---|
| baseline | 0.617 | 23.80 | 0.078 |
| curriculum-stage2 | 0.531 | 9.09 | 0.461 |

## Dataset Sizes (actual stored tensors)

| Dataset | num_samples (config) | final samples | ratio to baseline |
|---|---|---|---|
| baseline train | 4,000,000 | 2,118,117 | 1.0x |
| stage1 | 3,000,000 | 295,820 | 0.14x |
| **stage2** | **500,000** | **163,944** | **0.08x** |

Stage2 has **13x fewer samples** than baseline. This is the dominant factor.

## Root Causes (ranked)

### 1. Stage2 dataset far too small
`num_samples: 500000` in `unitary_curriculum_stage2_qiskit.yaml` vs `4000000` in baseline.
After dedup + balance-by-gate-length pipeline, 500K raw -> 164K final vs 4M raw -> 2.1M final.
The model gets ~13x less training data during the critical fine-tuning phase.

### 2. OneCycleLR inappropriate for fine-tuning
Both stages use `OneCycleLR`. This scheduler ramps LR up then anneals to near-zero — designed for from-scratch training. For fine-tuning from pretrained weights, the warmup phase disrupts learned features. Literature recommends cosine annealing without warmup or constant-then-decay.

### 3. Gate-set change is not true curriculum learning
Removing CCX from stage1 doesn't simplify the task — it changes the distribution. The model learns "CCX never appears" (the CCX token embedding gets no useful gradient). In stage2, this becomes negative transfer. Recent diffusion curriculum papers (arXiv:2403.10348, arXiv:2405.13637) show that effective curricula vary difficulty within the same distribution (e.g., denoising timestep difficulty, circuit length), not between distributions.

### 4. Backend mismatch between stages
Stage1 uses quditkit, stage2 uses qiskit. Even for the same gate set, the backends produce circuits with different statistical properties. Unique-sample ratio at 100K: 69.4% (quditkit optimized) vs 75.9% (qiskit optimized).

## Training Config Comparison

| Parameter | Stage1 | Stage2 | Baseline |
|---|---|---|---|
| num_epochs | 150 | 150 | 150 |
| learning_rate | 3e-4 | 5e-5 | 3e-4 |
| batch_size | 256 | 256 | 256 |
| lr_scheduler | OneCycleLR | OneCycleLR | OneCycleLR |
| dataset samples | 295,820 | 163,944 | 2,118,117 |
| gradient steps/epoch | ~1,156 | ~640 | ~8,274 |

## Proposed Fixes

### A. Fix the current approach
1. Increase `num_samples` in stage2 config to 4,000,000 (match baseline data budget)
2. Switch stage2 LR scheduler to CosineAnnealingLR (no warmup) for fine-tuning
3. Consider more stage2 epochs or early stopping

### B. Redesign curriculum (same distribution)
- Stage1: full gate set {H, CX, Z, X, CCX, SWAP}, short circuits (2-6 gates)
- Stage2: full gate set, full range (2-12 gates)
- Both stages use same backend (qiskit), no distribution shift

### C. Timestep curriculum (single stage)
Schedule which denoising timesteps are sampled during training: start with high-noise (easy), gradually include low-noise (hard). No dataset changes needed.

### D. Mixed/replay training
During stage2, mix stage1 samples into batches (e.g., 20/80 split) to prevent catastrophic forgetting and increase data volume.

## References
- arXiv:2403.10348 — Denoising Task Difficulty-based Curriculum for Diffusion Models
- arXiv:2405.13637 — Curriculum Direct Preference Optimization (CVPR 2025)
- arXiv:2101.10382 — Curriculum Learning: A Survey
