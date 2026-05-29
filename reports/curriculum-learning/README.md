# Curriculum Learning for Unitary Compilation

**Experiment ID**: `unitary-curriculum-learning`  
**Task**: Unitary compilation (3-qubit, gate set `{H, CX, Z, X, CCX, SWAP}`)  
**Evaluation**: 128 target unitaries × 128 generated circuits, guidance_scale=7.5, DDIM 20 steps, max_gates=12

---

## Key Finding

Curriculum pre-training on Clifford-only circuits (stage 1) followed by fine-tuning on the full gate set (stage 2) **does not improve** over training from scratch on the same stage-2 data budget.

| Model | Exact-found rate | Mean distinct correct |
|---|---|---|
| Remote (paper weights) | **98.4%** | 30.9 |
| Baseline (reproduced) | **96.9%** | 32.9 |
| Stage-2 from scratch | **93.8%** | 13.7 |
| Curriculum stage-2 | **92.2%** | 24.3 |

The curriculum model trails stage-2-scratch by 1.6 pp on exact-found rate, despite the stage-1 pre-training on ~1.1M Clifford circuits. However, it finds nearly **2× more diverse solutions** per target (24.3 vs 13.7 distinct correct circuits), suggesting broader coverage of the solution space when it does succeed.

---

## Figures

| File | Description |
|---|---|
| `accuracy_diversity_comparison.png` | (a) Exact-found rate and (b) mean distinct correct circuits per model — main result figure |
| `exact_count_boxplot.png` | Distribution of exact solutions found per target unitary (box plots, outliers hidden) |
| `training_loss_curves.png` | (a) Validation loss curves for all models; (b) full curriculum trajectory (stage-1 → stage-2) versus baseline and scratch |
| `failure_rate.png` | Fraction of target unitaries for which zero exact solutions were found |
| `summary.csv` | Numeric summary of all evaluation metrics |

---

## Root Causes

1. **Stage-2 data volume** — Stage-2 dataset has 664K samples vs 2.1M for baseline (3× less). This is the dominant factor; the stage-1 warm-up cannot compensate for the data deficit in stage-2.
2. **Distribution shift between stages** — Stage-1 (Clifford-only) and stage-2 (full gate set including CCX) have different unitary distributions. The CCX token gets no gradient in stage-1, so the transition to stage-2 requires unlearning the "CCX never appears" prior. This is negative transfer, not facilitated learning.
3. **Gate-set curriculum ≠ difficulty curriculum** — Removing CCX from stage-1 changes the *distribution*, not just the *difficulty*. Effective diffusion curricula vary difficulty within the same distribution (e.g., circuit depth, denoising timestep difficulty).

The 27.1% Clifford-only circuits naturally present in the stage-2 dataset (random gate-pool subsets per circuit) mean that catastrophic forgetting of Clifford knowledge is not an issue — Clifford replay is already built in.

---

## Training Configuration

| Parameter | Stage 1 | Stage 2 (curriculum) | Stage 2 (scratch) | Baseline |
|---|---|---|---|---|
| Dataset size | ~1.1M (subsamp.) | 664K | 664K | 2.1M |
| num_epochs | 150 | 150 | 150 | 150 |
| learning_rate | 3e-4 | 5e-5 | 3e-4 | 3e-4 |
| lr_scheduler | OneCycleLR | CosineAnnealingLR | OneCycleLR | OneCycleLR |
| init_from | random | stage-1 weights | random | random |
| Gate set | Clifford only | Full (+ CCX) | Full (+ CCX) | Full (+ CCX) |
| Backend | quditkit | qiskit | qiskit | qiskit |
