# SWAP-Count Conditioning for Unitary Compilation

**Date:** 2026-05-24 — 2026-05-25
**Status:** complete
**Experiment ID:** `unitary_swap_count_conditioning`

---

## Motivation

SWAP gates are the most expensive 2-qubit primitive on near-term hardware: high error rates (~10×
single-qubit error), no direct physical implementation on most architectures (requires 3 CNOTs,
each introducing noise). The hypothesis: if a diffusion model can be conditioned to respect an
exact SWAP count in its text prompt, the user gains a direct inference-time lever for trading
compilation success rate against hardware noise.

The broader thesis context is **noise-awareness** in circuit-generating diffusion models. SWAP
count is a practical proxy for circuit noise — circuits with fewer SWAPs have lower expected gate
error on most near-term devices.

---

## Experimental Design

### Label format

```
Compile using: ['h', 'cx', 'z', 'x', 'ccx', 'swap']; swap_count=2
```

The suffix `; swap_count=N` is appended **only when `"swap"` is in the gate pool** for that
circuit. Circuits without swap in the pool trivially have 0 SWAPs; appending the suffix there
would dilute the meaningful signal (swap is available but the model chose not to use it).

**Why exact count, not upper bound (`SWAP <= N`):** An upper-bound label cannot be trained
cleanly from circuit-by-circuit data. Every training circuit has its own exact SWAP count; if
labelled `SWAP <= K` where K equals the actual count, the model still learns exact-count
behaviour. Truly learning upper bounds would require augmenting each circuit with all valid upper
bounds ≥ its actual count, complicating generation with no clear benefit. Exact-count semantics
are honest and trainable.

### Dataset and training

Same generation parameters as the baseline (`paper_unitary.yaml`): 4 M generated circuits, 3
qubits, gate set `['h', 'cx', 'z', 'x', 'ccx', 'swap']`, 2–12 gates, Qiskit backend with
optimization, balanced to 60 000 per CLIP-label bucket. After balancing: ~2.1 M circuits.

Training: identical hyperparameters to baseline (150 epochs, batch 256, OneCycleLR, lr=3e-4).
Trained from scratch (not fine-tuned from baseline).

### Code changes (minimal, backward-compatible)

| File | Change |
|------|--------|
| `src/my_genQC/platform/circuits_generation.py` | `include_swap_count: bool = False` in `get_rnd_encoded_circuits()` / `generate_circuit_dataset()`; `qc` unpacked from inner loop; suffix appended conditionally |
| `src/quantum_diffusion/data/dataset.py` | `include_swap_count` propagated through `DatasetGenerator.generate_dataset()` → `generation_kwargs` (UNITARY branch only) |
| `conf/datasets/unitary_swap_constrained_qiskit.yaml` | New config; identical to baseline except `include_swap_count: true` |
| `conf/training/unitary_swap_constrained.yaml` | New config; identical to `paper_unitary.yaml` except paths/run_name |

---

## Status Quo: Baseline SWAP Behaviour

From `notebooks/datasets/swap_gate_analysis.ipynb` on the baseline training and eval datasets.

### Training dataset distribution

| Metric | Value |
|--------|-------|
| Total circuits | 2,118,117 |
| Circuits with 'swap' in gate pool | 71.5% |
| Mean SWAP count (swap-in-pool circuits) | 1.985 |
| Zero-SWAP fraction (swap-in-pool) | 13.9% |

### Baseline model SWAP output (128 unitaries × 64 samples)

| Metric | Value |
|--------|-------|
| Mean generated SWAP count | 0.984 |
| Mean SWAPs in correct compilations | 1.060 |
| Mean SWAPs in incorrect compilations | 0.911 |

The unconditioned model naturally produces ~1 SWAP on average — already below the training
distribution mean of 1.985. Correct compilations use slightly more SWAPs than incorrect ones,
suggesting SWAPs help the model reach hard target unitaries.

### CLIP text embedding discriminability

CLIP (ViT-B-32, embedding norm ≈ 17.7) encodes the entire prompt into a single 512-dim vector.
L2 distances between adjacent swap_count encodings:

| Pair | L2 distance |
|------|-------------|
| swap_count=0 → swap_count=1 | 0.517 |
| swap_count=1 → swap_count=2 | 0.304 |
| swap_count=2 → swap_count=3 | 0.296 |

The 0→1 gap is larger; adjacent integer gaps from 1 onwards are small (~0.30), giving the model
only a weak conditioning signal to distinguish specific counts.

---

## Evaluation Results

From `notebooks/evaluation/unitary/swap_constrained_evaluation.ipynb`.
Evaluated on 128 target unitaries from `artifacts/datasets/unitary-baseline-reproduction/eval/qiskit`,
64 samples per unitary, guidance_scale=7.5, 20 DDIM steps.

All numbers use the **float64-corrected** evaluation protocol (see note below).

### Compilation accuracy and SWAP compliance

| Model / requested count | exact_found_rate | exact_compliance_rate | mean_actual_swap | valid_decode_rate |
|---|---|---|---|---|
| baseline (no suffix) | **0.961** | — | 0.988 | 0.998 |
| swap_constrained, count=0 | 0.648 | **1.000** | 0.000 | 0.995 |
| swap_constrained, count=1 | 0.633 | **1.000** | 1.000 | 0.996 |
| swap_constrained, count=2 | 0.672 | **0.999** | 2.000 | 0.995 |
| swap_constrained, count=3 | 0.469 | **1.000** | 3.000 | 0.996 |
| swap_constrained, count=6 | 0.477 | **0.998** | 6.000 | 0.996 |

### Infidelity of wrong circuits

For circuits that did not exactly implement the target unitary, how large was the infidelity?

| Model / requested count | mean infidelity (wrong) | median infidelity (wrong) |
|---|---|---|
| baseline | 0.805 | 0.750 |
| swap_constrained, count=0 | 0.848 | 0.938 |
| swap_constrained, count=1 | 0.846 | 0.938 |
| swap_constrained, count=2 | 0.857 | 0.938 |
| swap_constrained, count=3 | 0.865 | 0.938 |
| swap_constrained, count=6 | 0.912 | 0.938 |

---

## Findings and Interpretation

### 1. SWAP compliance is near-perfect

The conditioned model respects the requested SWAP count with ~100% compliance across all tested
values. `mean_actual_swap` tracks the diagonal exactly (requested 0 → generated 0.000, requested
2 → generated 2.000). The model has fully internalised the conditioning signal.

### 2. Compilation accuracy dropped significantly — but the constraint is not the cause

Accuracy fell from 96.1% (baseline) to 63–67% for counts 0–2 and to 47–48% for counts 3–6.
The accuracy drop at high counts (3, 6) can be explained by the constraint being genuinely
restrictive: very few correct compilations of a 3-qubit unitary require exactly 6 SWAPs in at
most 12 gates.

**However, counts 0–2 cannot be explained by constraint difficulty.** The baseline already
produces a mean of 0.988 SWAPs unconditioned — requesting `swap_count=1` should be no harder
than what the baseline does naturally, yet accuracy drops from 96.1% to 63.3%. The constraint
itself is not the bottleneck.

### 3. The prompt suffix disrupts the compilation signal in CLIP

The CLIP text encoder is **frozen** and encodes the entire prompt into a single vector. Adding
`; swap_count=1` to the prompt produces a different embedding than the baseline prompt, even
though the intended circuit structure should be similar. The UNet was trained on this shifted
embedding distribution and, by learning to use the SWAP count as its primary conditioning
signal, partially displaced the compiled unitary information.

**This reveals that the CLIP text conditioning channel is not modular.** A new numeric constraint
added via text suffix modifies the same embedding vector that encodes the compilation target,
creating interference between the two objectives. The model resolves this interference by
prioritising the simpler, more learnable objective (exact count compliance) over the harder one
(unitary correctness).

### 4. Wrong circuits are not near-misses — they are completely wrong

Median infidelity of wrong circuits from the conditioned model is **0.938**, near the maximum
possible value of 1.0. The model produces syntactically valid, SWAP-compliant circuits that
implement an essentially random unitary — it does not struggle and nearly get the answer right;
it confidently gives the wrong answer.

This is confirmed by the `valid_decode_rate` staying near-identical to the baseline (~99.6%):
the model has not forgotten circuit syntax, only the compilation objective.

### 5. Baseline wrong circuits are genuinely hard

After the float64 evaluation fix (see below), baseline wrong circuits have median infidelity 0.75.
These are genuine failures on hard unitaries, not evaluation artefacts.

---

## Evaluation Bug: Float32 / Complex64 Cascade

Two related precision bugs were discovered and fixed during evaluation.

**Bug 1 — Float32 target quantization:** The eval dataset stores target unitaries in float32.
`1/√2` cannot be represented exactly in float32 (~1.2e-8 per-element error), causing H-gate
circuits to appear "wrong" under the 1e-8 threshold even when the model found the correct
circuit.

**Fix:** Decode the stored reference circuit from `eval_dataset.x[idx]` using the tokenizer,
re-simulate at float64 via the Qiskit backend, and use the resulting float64 unitary as the
comparison target. The float32 split is still passed to the model for conditioning.

**Bug 2 — Complex64 comparison:** After fix 1, the distance computation still cast both tensors
to `torch.complex64` (float32 arithmetic), reintroducing ~1.19e-7 quantization error and
pushing near-exact circuits back above the 1e-8 threshold. This was visible as baseline wrong
circuits having median infidelity exactly at float32 machine epsilon.

**Fix:** Cast both tensors to `torch.complex128` (float64 arithmetic) for the infidelity
computation.

Both fixes are applied in `notebooks/evaluation/unitary/swap_constrained_evaluation.ipynb`.
The correct evaluation protocol is also documented in
`reports/unitary-compilation/notes.md` and implemented in
`notebooks/evaluation/unitary/unitary_model_evaluation.ipynb`.

---

## Architectural Conclusion

CLIP-based text conditioning is **not well-suited for adding independent numerical constraints**
to an existing diffusion model. Because the text encoder is frozen and maps the entire prompt to
a single vector, a numeric suffix modifies the same embedding that encodes the compilation
target. The two signals compete in a shared, undifferentiated conditioning space.

A clean architecture for noise-aware conditioning would use a **dedicated scalar channel** for
the SWAP budget, projected independently into the UNet conditioning space and kept separate from
the CLIP text channel. This would preserve the baseline's compilation ability while adding the
noise-budget control.

Alternatively, **fine-tuning from the baseline checkpoint** (rather than training from scratch)
may partially preserve compilation ability while adapting to the new CLIP embedding distribution.
This is the most accessible next step within the current architecture.

---

## Key Artifacts

| Artifact | Path |
|----------|------|
| Dataset config | `conf/datasets/unitary_swap_constrained_qiskit.yaml` |
| Training config | `conf/training/unitary_swap_constrained.yaml` |
| Training dataset | `artifacts/datasets/unitary-swap-constrained/train` |
| Trained model | `artifacts/models/unitary-swap-constrained/unitary_swap_constrained/` |
| Status quo notebook | `notebooks/datasets/swap_gate_analysis.ipynb` |
| Evaluation notebook | `notebooks/evaluation/unitary/swap_constrained_evaluation.ipynb` |
| Evaluation results | `artifacts/evaluations/unitary-swap-constrained/swap_count_evaluation/` |
