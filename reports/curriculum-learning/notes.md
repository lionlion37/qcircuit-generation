# Curriculum Learning Investigation Notes

**Date:** 2026-05-26
**Status:** Complete — curriculum pretraining does not improve over optimally-trained from-scratch model on same data budget

## Results (128 eval unitaries, 128 samples each, guidance_scale=7.5, sample_steps=20)

### Paper eval dataset — full gate set {h, cx, z, x, ccx, swap}
(`artifacts/datasets/unitary-baseline-reproduction/eval/qiskit`, float64 targets via re-decode)

| Model | exact_found_rate | mean_distinct_correct | mean_exact_count |
|---|---|---|---|
| remote (paper weights) | **0.984** | 30.9 | 86.3 |
| baseline (our reproduction) | **0.969** | 32.9 | 83.8 |
| stage2-scratch (lr=3e-4, OneCycleLR) | **0.938** | 13.7 | 59.2 |
| curriculum-stage2 (from stage1, lr=5e-5) | **0.906** | 19.0 | 62.5 |

### Stage1 eval dataset — Clifford-only {h, cx, z, x, swap}, no ccx
(`artifacts/datasets/unitary-curriculum-learning/stage1_eval`, qiskit backend, float64 U)

| Model | exact_found_rate | mean_distinct_correct | mean_exact_count |
|---|---|---|---|
| remote (paper weights) | **1.000** | 36.8 | 91.3 |
| baseline (our reproduction) | **1.000** | 41.3 | 87.9 |
| curriculum-stage2 (from stage1, lr=5e-5) | **0.953** | 25.0 | 69.0 |
| stage2-scratch (lr=3e-4, OneCycleLR) | **0.945** | 16.0 | 55.0 |

### Curriculum learning verdict
The from-scratch model (stage2-scratch) outperforms curriculum-stage2 on the primary task
(93.8% vs 90.6% on full gate set). Curriculum pretraining slightly helps on Clifford-only
circuits (95.3% vs 94.5%) — the stage1 warm-up provides a small benefit there — but this
comes at the cost of full-gate performance. The curriculum approach as implemented does not
provide a net benefit. Root causes are documented below.

## Dataset Sizes (actual stored tensors)

Verified directly from safetensors files.

| Dataset | num_samples (config) | final samples | ratio to baseline |
|---|---|---|---|
| baseline train | 4,000,000 | 2,118,117 | 1.0x |
| stage1 train | 3,000,000 | 295,820 | 0.14x |
| stage2 train | 4,000,000 | 663,862 | 0.31x |

Total curriculum (stage1 + stage2): ~960K samples — approximately **2.2x fewer** than baseline.
Stage2 alone is ~3x fewer than baseline.

## Root Causes (ranked)

### 1. Stage2 dataset smaller than baseline
`num_samples: 4000000` in both configs, but after dedup + balance pipeline: 664K (stage2) vs 2.1M (baseline).
Stage2 gets ~3x less training data during fine-tuning. Note: the stage2 dataset already contains 27.1%
Clifford-only circuits (no CCX in gate pool), because each circuit is generated with a random subset of
the full gate pool as its prompt — so subset prompts without CCX occur naturally. This means data replay
of stage1 Clifford circuits into stage2 training is already happening implicitly and would not provide
additional benefit. The data shortage is driven by higher deduplication rates at scale.

### 2. OneCycleLR inappropriate for fine-tuning
Both stages use `OneCycleLR`. This scheduler ramps LR up then anneals to near-zero — designed for from-scratch training. For fine-tuning from pretrained weights, the warmup phase disrupts learned features. Literature recommends cosine annealing without warmup or constant-then-decay.

### 3. Gate-set change is not true curriculum learning
Removing CCX from stage1 doesn't simplify the task — it changes the distribution. The model learns "CCX never appears" (the CCX token embedding gets no useful gradient). In stage2, this becomes negative transfer. Recent diffusion curriculum papers (arXiv:2403.10348, arXiv:2405.13637) show that effective curricula vary difficulty within the same distribution (e.g., denoising timestep difficulty, circuit length), not between distributions.

### 4. Backend mismatch between stages
Stage1 uses quditkit, stage2 uses qiskit. Even for the same gate set, the backends produce circuits with different statistical properties. Unique-sample ratio at 100K: 69.4% (quditkit optimized) vs 75.9% (qiskit optimized).

## Stage2 Dataset Composition

The stage2 dataset uses random gate-pool subsets per circuit (uniform random subset of the full 6-gate pool,
size 1–6). This means the distribution of CCX exposure is not binary:

| Gate pool subset type | Count | Fraction |
|---|---|---|
| Contains CCX (full-gate circuits) | 483,780 | 72.9% |
| Clifford-only (no CCX) | 180,082 | 27.1% |
| **Total** | **663,862** | |

Gate pool size distribution: 2 gates (10.6%), 3 gates (21.4%), 4 gates (28.2%), 5 gates (30.8%), 6 gates (9.0%).

**Implication**: Stage2 already provides implicit replay of Clifford-only circuits throughout training. There is
no catastrophic forgetting of Clifford knowledge — the model sees Clifford-only examples in every epoch.
The performance gap vs baseline is entirely explained by the 3x data volume difference.

## Training Config Comparison

| Parameter | Stage1 | Stage2 | Baseline |
|---|---|---|---|
| num_epochs | 150 | 150 | 150 |
| learning_rate | 3e-4 | 5e-5 | 3e-4 |
| batch_size | 256 | 256 | 256 |
| lr_scheduler | OneCycleLR | OneCycleLR | OneCycleLR |
| dataset samples | 295,820 | 663,862 | 2,118,117 |
| gradient steps/epoch | ~1,156 | ~2,593 | ~8,274 |

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
~~During stage2, mix stage1 samples into batches (e.g., 20/80 split) to prevent catastrophic forgetting.~~
**Note: this is redundant.** The stage2 dataset already contains 27.1% Clifford-only circuits by design
(random gate-pool subsets per circuit), so replay is already built in. The real bottleneck is total data volume,
not distribution mixing. To increase volume, fix issue #1 instead.

## References
- arXiv:2403.10348 — Denoising Task Difficulty-based Curriculum for Diffusion Models
- arXiv:2405.13637 — Curriculum Direct Preference Optimization (CVPR 2025)
- arXiv:2101.10382 — Curriculum Learning: A Survey
