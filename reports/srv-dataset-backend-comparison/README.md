# Quditkit vs Qiskit — Dataset Generation Comparison

Artifacts from `artifacts/evaluations/srv_dataset_backend_comparison/backend_comparison/`.
Full run details in `notebooks/datasets/dataset_generation_quditkit_vs_qiskit.ipynb`.

Second run includes native quditkit circuit optimizer (`circuit_optimizer.py`) replacing
the previous qiskit round-trip for quditkit-optimized cases.

---

## SRV task (5 qubits, gates: h/cx, min 4 – max 20)

| Backend | Optimized | unique_ratio 10k | unique_ratio 100k | time 100k |
|---------|-----------|-----------------|-------------------|-----------|
| qiskit  | No        | 0.998            | 0.997             | 101.9 s   |
| qiskit  | Yes       | 0.980            | 0.944             | 154.4 s   |
| quditkit| No        | 0.998            | 0.997             | 41.3 s    |
| quditkit| Yes       | 0.988            | **0.966**         | **40.9 s**|

- quditkit (optimized) is **3.8× faster** than qiskit (optimized) at 100k.
- Unique ratio is comparable to qiskit optimized, slightly better at 100k (0.966 vs 0.944).

## Unitary task (3 qubits, gates: h/cx/z/x/swap, min 2 – max 12)

| Backend | Optimized | unique_ratio 10k | unique_ratio 100k | time 100k |
|---------|-----------|-----------------|-------------------|-----------|
| qiskit  | No        | 0.938            | 0.867             | 43.9 s    |
| qiskit  | Yes       | 0.867            | 0.759             | 100.5 s   |
| quditkit| No        | 0.935            | 0.866             | 34.0 s    |
| quditkit| Yes       | 0.812            | **0.694**         | **29.7 s**|

- quditkit (optimized) is **3.4× faster** than qiskit (optimized) at 100k.
- quditkit (optimized) is faster than quditkit (unoptimized) — shorter circuits mean cheaper unitary computation.
- Lower unique ratio (0.694 vs 0.759): the native optimizer is more aggressive than qiskit's level-1 transpiler; more random circuits collapse to the same canonical form. Expected behavior, not a bug. For a 3M training run with post-generation balancing this is acceptable.

## Figures

- `srv_generation_time.png` — wall-clock time by backend/size/optimization
- `srv_unique_ratio.png` — unique sample ratio for SRV
- `unitary_generation_time.png` — wall-clock time for unitary task
- `unitary_unique_ratio.png` — unique sample ratio for unitary task
- `srv_summary.csv` / `unitary_summary.csv` — per-task summary tables
- `all_runs.csv` — full results across all 16 cases
