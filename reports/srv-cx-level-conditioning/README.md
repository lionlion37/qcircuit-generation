# SRV CX-Level Conditioning

**Date:** 2026-05-27 — 2026-05-28  
**Status:** complete  
**Experiment ID:** `srv_noise_finetuned`

---

## Motivation

Near-term quantum hardware is noise-limited: circuits with more two-qubit (CX) gates accumulate
more error. The SRV task generates circuits of a specified entanglement structure, but the
original model provides no control over CX density. Adding CX-level conditioning gives the user
an inference-time lever to request low-, medium-, or high-noise circuits for a given target SRV.

The SRV task is structurally well-suited for this: because many circuits implement each SRV
(mean ~5,200 circuits per unique SRV), the dataset contains genuine variation in CX content for
each target — unlike the unitary task where each target has a fixed minimum-CX compilation.

---

## Experimental Design

### Noise proxy

**cx_ratio = n_cx / n_gates** — the fraction of gate steps that are CX gates. This normalises
for circuit length, so a 5-CX circuit in a 10-gate depth and a 5-CX circuit in a 50-gate depth
are correctly treated as "medium" and "low" respectively.

### Classification

CX levels are assigned **per SRV using tertiles of cx_ratio**:

| Level  | Definition                                      |
|--------|-------------------------------------------------|
| low    | cx_ratio ≤ 33rd percentile within this SRV      |
| medium | 33rd < cx_ratio ≤ 67th percentile within this SRV |
| high   | cx_ratio > 67th percentile within this SRV      |

Per-SRV tertiles ensure "low" means "low CX for how hard this target is", not globally sparse.

### Label format

```
Generate SRV: [1, 2, 2]; cx_level=low
```

The suffix `; cx_level=<level>` is appended to every existing SRV label. Dataset derived from
`artifacts/datasets/qc_srv_dataset_3to8qubit` via
`notebooks/datasets/srv_cx_level_relabelling.ipynb`.

### Training

Fine-tuned from the existing SRV stage-2 weights (paper baseline). Identical model architecture.

| Hyperparameter       | Value                  |
|----------------------|------------------------|
| Base model           | `srv-baseline-reproduction/paper_stage_2` |
| Learning rate        | 3e-5                   |
| Epochs               | 50                     |
| Batch size           | 256                    |
| Padding mode         | bucket                 |
| Guidance train p     | 0.1                    |
| CLIP text cache      | disabled (new labels)  |

---

## Results

### Overall accuracy and CX compliance

| Condition | Overall accuracy | Mean cx_ratio |
|-----------|-----------------|---------------|
| baseline (no suffix) | 6.6% | 0.699 |
| low       | 91.2%           | 0.366         |
| medium    | 86.6%           | 0.632         |
| high      | 95.6%           | 0.790         |

The 6.6% baseline accuracy is expected: after fine-tuning on suffixed labels, the model requires
a cx_level suffix to generate meaningful circuits. The original unsuffixed prompts fall outside
the training distribution.

### Per-qubit-count accuracy

| Condition | 3q     | 4q     | 5q     | 6q     |
|-----------|--------|--------|--------|--------|
| baseline  | 13.1%  | 12.5%  | 5.8%   | 5.3%   |
| low       | 65.0%  | 81.8%  | 91.5%  | 95.2%  |
| medium    | 82.5%  | 81.0%  | 89.0%  | 87.1%  |
| high      | 97.8%  | 96.7%  | 97.1%  | 94.5%  |

### Key observations

1. **CX conditioning works.** Mean cx_ratio varies systematically: 0.37 (low) vs 0.63 (medium)
   vs 0.79 (high). The model reliably produces circuits of different CX densities on request.

2. **`high` is the easiest condition** (95.6%). High CX density is the "natural" regime the
   baseline was trained on — fine-tuning preserves this capability at near-baseline performance.

3. **`low` improves strongly with qubit count** (65% at 3q → 95% at 6q). Low-CX circuits for
   small, low-entanglement targets are genuinely sparse and unusual in the training distribution;
   at larger qubit counts the model has more low-CX examples to learn from.

4. **`medium` is the hardest condition** (86.6%). The middle tertile is the narrowest range to
   satisfy — neither trivially achievable nor the model's default mode.

5. **No accuracy collapse.** Unlike the SWAP-count experiment on the unitary task (96.1% → 63%),
   CX-level conditioning on the SRV task preserves strong compilation accuracy. The SRV task's
   many-to-one structure and categorical (non-numeric) label format are both factors: CLIP
   discriminates "low"/"medium"/"high" well, and the text conditioning channel is not overloaded
   by an interfering numeric suffix.

---

## Comparison to SWAP-Count Conditioning (Unitary Task)

| Aspect                     | SRV cx_level (this work) | Unitary swap_count |
|----------------------------|--------------------------|--------------------|
| Conditioning signal        | Categorical (low/med/high) | Numeric integer  |
| Compliance rate            | Implicit (ratio shift)   | ~100% exact       |
| Accuracy vs baseline       | 87–96% (−4 to −9 pp)     | 63–67% (−29 pp)   |
| CLIP discriminability      | Good (categorical words) | Poor (small L2 between integers) |
| Task structure             | Many circuits per target | One-to-one target |

The structural advantage of the SRV task — and the use of natural-language categorical labels
rather than numeric suffixes — explains why noise conditioning succeeds here where it degraded
compilation quality in the unitary task.

---

## Figures

| File | Description |
|------|-------------|
| `accuracy_by_entanglement_per_cx_level.png` | 3-panel: accuracy vs entangled qubits per cx_level, lines by qubit count |
| `mean_accuracy_by_qubit_count.png` | Mean accuracy by qubit count, one line per condition |
| `cx_ratio_and_accuracy_by_condition.png` | Bar charts: accuracy and mean cx_ratio per condition |
| `cx_ratio_distributions_by_qubit_count.png` | cx_ratio distributions by qubit count per condition |
| `per_srv_low_vs_high.png` | Per-SRV scatter: median cx_ratio in low vs high bin |

---

## Key Artifacts

| Artifact | Path |
|----------|------|
| Relabelling notebook | `notebooks/datasets/srv_cx_level_relabelling.ipynb` |
| Relabelled dataset | `artifacts/datasets/srv-noise-conditioning/` |
| Training config | `conf/training/srv_noise_finetuned.yaml` |
| Trained model | `artifacts/models/srv-noise-conditioning/srv_noise_finetuned/` |
| Evaluation notebook | `notebooks/evaluation/srv/cx_level_evaluation.ipynb` |
| Evaluation results | `artifacts/evaluations/srv-noise-conditioning/cx_level_eval/` |
