# Unitary Compilation Evaluation Notes

**Date:** 2026-05-24
**Status:** Baseline established — evaluation bug fixed

## Corrected Baselines

| Model | exact_found_rate | mean_exact_count | mean_distinct_correct |
|---|---|---|---|
| remote (paper weights) | **0.984** | 86.3 | 30.9 |
| our trained baseline | **0.969** | 83.8 | 32.9 |
| paper reported (arXiv:2311.02041) | 0.926 | — | — |

Evaluated on 128 unitaries from `artifacts/datasets/unitary-baseline-reproduction/eval/qiskit`,
128 samples per unitary, guidance_scale=7.5, sample_steps=20, max_gates=12.

Artifacts: `artifacts/evaluations/paper-baseline-comparison/remote_vs_baseline_paper_eval/`

## Float32 Evaluation Bug (now fixed)

**Symptom:** both models appeared to achieve only ~62–65% exact_found_rate; extensive ablation
over guidance mode, DDIM steps, token indexing, prompt style, and sample count changed nothing.

**Root cause:** `artifacts/datasets/unitary-baseline-reproduction/eval/qiskit` stores target
unitaries in float32. Clifford gates involving H produce matrix entries of ±1/√2, which float32
cannot represent exactly (~1.2e-8 per-element error). This accumulated to a systematic infidelity
of ~2.14e-4 between the stored and true algebraic unitary. With `EXACT_DISTANCE_TOL = 1e-8`, every
test circuit that involved an H gate was a **false negative**: the model correctly found the circuit
(Frobenius distance ~4.56e-8 to the target), but the comparison failed.

**Evidence:** perfect correlation across 128 test cases — all 48 "not found" circuits had float32
quantization error (~2.14e-4 infidelity between stored and decoded reference), all 80 "found"
circuits had exact float32 representations. 46 of the 48 had best_model_distance < 1e-7 (correct
circuit found, tolerance too strict); only 2 were genuine failures.

**Fix:** `notebooks/evaluation/unitary/unitary_model_evaluation.ipynb` now decodes the stored
reference circuit (`dataset.x[idx]`) and re-simulates at float64 to obtain the exact algebraic
target unitary for comparison. The float32 stored value is still passed to the model for
conditioning (consistent with the paper's likely inference setup).

**Secondary fix:** `generate_circuit_dataset` default changed from `torch.float16` to
`torch.float32` to avoid even worse quantization in newly generated datasets.

## Secondary Fix: Complex64 Comparison Arithmetic

Even after fix 1, casting both tensors to `torch.complex64` for the infidelity computation
reintroduces float32 quantization error (~1.19e-7 per comparison). This is exactly float32
machine epsilon and pushes near-exact circuits back above the 1e-8 threshold.

**Fix:** Cast both the generated and target unitary tensors to `torch.complex128` before calling
`UnitaryInfidelityNorm.distance()`. This ensures the comparison happens in float64 arithmetic.

This was discovered during the SWAP-count conditioning evaluation: the baseline still showed
0.617 exact_found_rate after fix 1, and wrong circuits had median infidelity 1.19e-7 — a clear
float32-arithmetic signature. After applying fix 2, baseline exact_found_rate = 0.961, consistent
with the result in this notebook.

**Summary of correct evaluation protocol:**
1. Re-simulate reference circuit from `dataset.x[idx]` at float64 to get exact target unitary
2. Pass float32 split to the model for conditioning (as during training)
3. Compare generated unitary to float64 target using `torch.complex128` arithmetic

---

## Gap vs Paper (92.6%)

Our models achieve ~97–98% on this dataset vs the paper's 92.6% on their original test set.
The difference is expected: the paper's test unitaries are a different random draw and may include
harder cases. The paper likely computed target unitaries at float64 at runtime (no intermediate
float32 storage), so they were not affected by this bug.
