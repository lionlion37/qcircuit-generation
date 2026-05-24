# SWAP-Count Conditioning for Unitary Compilation

**Date:** 2026-05-24  
**Status:** in_progress — dataset generation and training pending  
**Experiment ID:** `unitary_swap_count_conditioning`

---

## Motivation

SWAP gates are the most expensive 2-qubit primitive on near-term hardware: high error rates (~10×
single-qubit error), no direct physical implementation on most architectures (requires 3 CNOTs).
Giving the model a SWAP budget at inference time is a practical noise-reduction handle. The
scientific question is: can a diffusion model trained with exact SWAP counts in the prompt learn
to honour those counts without degrading compilation accuracy?

---

## Experimental Design

### Label format

```
Compile using: ['h', 'cx', 'z', 'x', 'ccx', 'swap']; swap_count=2
```

- Suffix is appended **only when `"swap"` is in the gate pool** for that circuit. Circuits generated
  without swap in the pool trivially have 0 SWAPs, so appending `swap_count=0` there would dilute
  the meaningful signal (swap available but model chose not to use it).
- Exact count semantics (`swap_count=N`, not `SWAP <= N`). Training on exact counts is cleaner:
  every example is a valid instance of satisfying its own constraint, and the model learns the full
  distribution per count. At inference time, requesting `swap_count=0` steers the model toward
  zero-SWAP compilations.

### Code changes (minimal)

| File | Change |
|------|--------|
| `src/my_genQC/platform/circuits_generation.py` | `include_swap_count: bool = False` added to `get_rnd_encoded_circuits()` and `generate_circuit_dataset()`; `qc` unpacked (not discarded) from inner loop; suffix appended conditionally |
| `src/quantum_diffusion/data/dataset.py` | `include_swap_count` propagated through `DatasetGenerator.generate_dataset()` → `generation_kwargs` (UNITARY branch only) |
| `conf/datasets/unitary_swap_constrained_qiskit.yaml` | New config; identical to baseline except `include_swap_count: true` |
| `conf/training/unitary_swap_constrained.yaml` | New config; identical to `paper_unitary.yaml` except paths and run_name |

**Backward compatibility:** `include_swap_count=False` default leaves all existing pipelines unchanged.

### Why the baseline model (not curriculum)

Adding SWAP conditioning on top of curriculum learning would conflate two variables. The baseline
(`paper_unitary.yaml`) is the simpler, more reproducible reference: any performance delta between
the baseline and the swap-conditioned model is attributable solely to the new conditioning.

---

## Status Quo Analysis (2026-05-24)

### Dataset SWAP distribution (baseline unitary training dataset)

| Metric | Value |
|--------|-------|
| Fraction of circuits containing 'swap' in gate pool | ~70.9% |
| SWAP gates as fraction of all gates | ~18.3% |
| Mean SWAP count (over all circuits) | ~1.99 |
| Typical range | 0–4 per circuit |

### Baseline model SWAP behaviour (64 unitaries × 64 generated circuits)

| Metric | Value |
|--------|-------|
| Mean generated SWAP count | 0.99 |
| Dataset mean SWAP count | ~1.99 |
| Mean SWAPs in correct compilations | 1.13 |
| Mean SWAPs in incorrect compilations | 0.87 |

The model generates roughly half as many SWAPs as the dataset mean. Correct compilations use
slightly more SWAPs than incorrect ones — plausibly because SWAP gates enable more expressive
circuit structure (reaching target unitaries that are hard without them).

### CLIP text embedding discriminability

CLIP (ViT-B-32, embedding norm ≈ 17.7) has limited numerical discrimination for the swap_count
suffix tokens. L2 distances between adjacent encodings:

| Pair | L2 distance |
|------|-------------|
| swap_count=0 → swap_count=1 | 0.517 |
| swap_count=1 → swap_count=2 | 0.304 |
| swap_count=2 → swap_count=3 | 0.296 |

The `0 → 1` gap is meaningfully larger (model gets a clear "zero vs non-zero" signal). For
counts 1–3 the gaps are small (~0.30), meaning the CLIP conditioning signal for distinguishing
specific counts in that range is weak.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CLIP cannot discriminate counts 1–3 (small L2 gap ~0.30) | Medium | Measure `mean_abs_swap_error`; if > 0.5 consider categorical labels or dedicated numerical encoder |
| Training data imbalance (most circuits have 0–2 SWAPs; 4+ rare) | Medium | Dataset uses `balance_after_generation=true, balance_max=60000` — check SWAP bucket counts in analysis notebook |
| Accuracy drops for extreme counts (swap_count=0 hardest: forces avoidance) | Medium | Document in evaluation; acceptable if swap_count=0 compliance is substantially better than baseline |
| SWAP budget coherence: model may ignore suffix if CLIP guidance is weak | Medium | The zero-vs-nonzero gap (0.517) is the most actionable; even partial compliance at swap_count=0 is a positive result |

**Most likely failure mode:** The model learns `swap_count=0` reasonably well (the CLIP gap is
largest there) but fails to distinguish counts 1, 2, 3 reliably. This would still be practically
useful — `swap_count=0` is the most hardware-relevant constraint.

---

## Key Evaluation Metrics

Run `notebooks/evaluation/unitary/swap_constrained_evaluation.ipynb` after training:

1. **Exact-count compliance rate** per requested count: `fraction(actual_swap == requested)`
2. **Mean absolute swap error** per requested count: `E[|actual - requested|]`
3. **Requested vs actual heatmap**: diagonal = perfect compliance; off-diagonal reveals systematic bias
4. **Compilation accuracy** (`exact_found_rate`) per requested count — must not degrade more than ~5% from baseline
5. **Baseline comparison**: same eval set, baseline model without swap_count suffix

**Success criteria (proposed):**
- `swap_count=0` compliance rate > 50% (baseline ≈ 0% since model has no reason to avoid SWAPs)
- `mean_abs_swap_error` < 1.0 for counts 0–3
- `exact_found_rate` within 5% of baseline

---

## Pending Steps

1. Generate dataset: `python scripts/generate_dataset.py hydra.run.dir=. hydra.output_subdir=null datasets=unitary_swap_constrained_qiskit`
2. Train model: `python scripts/train_model.py hydra.run.dir=. hydra.output_subdir=null training=unitary_swap_constrained`
3. Run `notebooks/datasets/swap_gate_analysis.ipynb` (populate outputs)
4. Run `notebooks/evaluation/unitary/swap_constrained_evaluation.ipynb`
5. If compliance for counts 1–3 is weak: implement categorical swap labels or dedicated numerical encoder head

---

## Commands

```bash
# Dataset generation (~hours on 8 workers)
conda run -n qc python scripts/generate_dataset.py \
  hydra.run.dir=. hydra.output_subdir=null \
  datasets=unitary_swap_constrained_qiskit

# Training (~hours on GPU)
conda run -n qc python scripts/train_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  training=unitary_swap_constrained
```
