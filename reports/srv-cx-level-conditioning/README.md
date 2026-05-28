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

| Level  | Definition                                         |
|--------|----------------------------------------------------|
| low    | cx_ratio ≤ 33rd percentile within this SRV         |
| medium | 33rd < cx_ratio ≤ 67th percentile within this SRV |
| high   | cx_ratio > 67th percentile within this SRV         |

Per-SRV tertiles ensure "low" means "low CX for how hard this target is", not globally sparse.

### Label format

```
Generate SRV: [1, 2, 2]; cx_level=low
```

The suffix `; cx_level=<level>` is appended to every existing SRV label. Dataset derived from
`artifacts/datasets/qc_srv_dataset_3to8qubit` via
`notebooks/datasets/srv_cx_level_relabelling.ipynb`.

### Training

Fine-tuned from the existing SRV stage-2 weights (paper baseline).

| Hyperparameter       | Value                                             |
|----------------------|---------------------------------------------------|
| Base model           | `srv-baseline-reproduction/paper_stage_2`         |
| Learning rate        | 3e-5                                              |
| Epochs               | 50                                                |
| Batch size           | 256                                               |
| Padding mode         | bucket                                            |
| CLIP text cache      | disabled (new labels)                             |

---

## Evaluation Protocol

Identical to the main `model_evaluation.ipynb` protocol for direct comparability.

| Parameter             | Value                                          |
|-----------------------|------------------------------------------------|
| Qubit counts          | 3, 4, 5, 6, 7, 8                              |
| Samples per entanglement bucket | 8,192                                |
| Sampling strategy     | Stratified by entanglement bucket, seed 1234   |
| Guidance scale        | 7.5                                            |
| max_gates             | 16                                             |
| True baseline model   | `srv-baseline-reproduction/paper_stage_2`      |
| Fine-tuned model      | `srv-noise-conditioning/srv_noise_finetuned`   |

**Conditions:**
- `true_baseline` — original `paper_stage_2` model, prompts without any cx_level suffix
- `low` / `medium` / `high` — fine-tuned model with `; cx_level=<level>` appended to each prompt

The `true_baseline` numbers match the existing `reports/summary.csv` baseline exactly,
confirming protocol equivalence.

---

## Results

### Accuracy by qubit count

| Condition      | 3q    | 4q    | 5q    | 6q    | 7q    | 8q    |
|----------------|-------|-------|-------|-------|-------|-------|
| true_baseline  | 78.8% | 83.7% | 88.8% | 91.0% | 92.0% | 91.4% |
| low            | 68.6% | 84.2% | 92.2% | 94.9% | 95.1% | 93.6% |
| medium         | 79.7% | 83.9% | 88.3% | 87.7% | 86.5% | 83.6% |
| high           | 97.7% | 98.2% | 96.0% | 92.4% | 87.1% | 81.0% |

### Key observations

1. **`high` excels at small qubit counts, degrades at large.** At 3–4q, `high` is dramatically
   better than baseline (97–98% vs 79–84%). But this inverts by 7–8q where `high` (81–87%)
   falls significantly below baseline (91–92%). At high qubit counts, forcing many CX gates
   makes it harder to place them correctly for complex entanglement targets.

2. **`low` shows the opposite trend.** Starts below baseline (68.6% at 3q) but surpasses it
   from 5q onward, reaching 95.1% at 7q vs 92.0% baseline. At larger scales, sparse-CX
   circuits are abundant and learnable; the constraint becomes less restrictive.

3. **`medium` tracks baseline at 3–5q then diverges.** At 6–8q, `medium` drops 3–8 pp below
   baseline. The middle tertile is the narrowest range to satisfy and the model has the least
   training signal for it at large qubit counts.

4. **No catastrophic accuracy collapse.** Unlike the SWAP-count experiment (96.1% → 63%),
   fine-tuning with categorical CX-level labels preserves strong performance across all
   conditions. The worst-case gap vs baseline is −10 pp (`high` at 8q), not −33 pp.

5. **Conversion rate is unaffected.** All conditions maintain 96–99% conversion rate,
   confirming circuit syntax is preserved throughout.

---

## Comparison to SWAP-Count Conditioning (Unitary Task)

| Aspect                     | SRV cx_level (this work)       | Unitary swap_count         |
|----------------------------|--------------------------------|----------------------------|
| Conditioning signal        | Categorical (low/med/high)     | Numeric integer            |
| Compliance                 | Implicit (cx_ratio shift)      | ~100% exact                |
| Worst-case accuracy drop   | −10 pp vs baseline             | −33 pp vs baseline         |
| CLIP discriminability      | Good (categorical words)       | Poor (small L2 per integer)|
| Task structure             | Many circuits per target       | One-to-one target          |

---

## Figures

| File | Description |
|------|-------------|
| `accuracy_by_entanglement_bucket.png` | 4-panel paper figure: accuracy vs entangled qubits per condition, Oranges lines by qubit count (3–8q) |
| `summary_comparison.png` | Mean accuracy and conversion rate by qubit count, one line per condition |
| `cx_density_comparison.png` | CX density (ratio and absolute count) per condition per qubit count; threshold lines at t33/t67 |
| `threshold_adherence.png` | Box plots of generated cx_ratio per condition overlaid on low/medium/high zone shading (256 samples/condition/qubit count, max_gates=52) |

### Note on threshold visualisation

The two dashed lines in `threshold_adherence.png` (t33 = 0.583, t67 = 0.667) are **global
tertile boundaries** computed over the entire training dataset. They appear close together because
the training data's cx_ratio is concentrated around 0.62, making the global medium band narrow
(width ≈ 0.08).

This does **not** reflect how the cx_level labels were actually assigned. Labelling used
**per-SRV tertiles**: within each individual SRV's distribution, exactly one third of circuits
are labelled low, one third medium, one third high, regardless of the SRV's absolute cx_ratio
range. For a given SRV whose circuits span cx_ratio 0.4–0.9, the medium zone would cover roughly
0.57–0.73 — much wider than the global band. The narrow global lines are a summary reference
only and should not be interpreted as the effective width of the medium class during training.

---

## Key Artifacts

| Artifact | Path |
|----------|------|
| Relabelling notebook | `notebooks/datasets/srv_cx_level_relabelling.ipynb` |
| Relabelled dataset | `artifacts/datasets/srv-noise-conditioning/` |
| Training config | `conf/training/srv_noise_finetuned.yaml` |
| Trained model | `artifacts/models/srv-noise-conditioning/srv_noise_finetuned/` |
| Stratified evaluation notebook | `notebooks/evaluation/srv/cx_level_evaluation_stratified.ipynb` |
| Evaluation results | `artifacts/evaluations/srv-noise-conditioning/cx_level_stratified/` |
| Exploratory evaluation (3–6q, per-SRV) | `artifacts/evaluations/srv-noise-conditioning/cx_level_eval/` |
