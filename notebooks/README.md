# Notebooks

Notebook content is organized by workflow rather than chronology.

## Layout
- `datasets/`: dataset exploration, audits, and backend-generation comparisons.
- `evaluation/srv/`: SRV task evaluations, including repaired decode analysis.
- `evaluation/unitary/`: unitary-compilation evaluation notebooks.
- `training/bucket_padding/`: bucket-padding and alignment analysis.
- `noise_awareness/`: noise-aware dataset generation and analysis.
- `shared/`: helper utilities reused by notebooks.

## Main Thesis Notebooks
- `evaluation/srv/model_evaluation.ipynb`
- `evaluation/srv/model_evaluation_with_repairs.ipynb`
- `evaluation/unitary/unitary_model_evaluation.ipynb`
- `datasets/dataset_generation_quditkit_vs_qiskit.ipynb`
- `training/bucket_padding/bucket_deep_dive.ipynb`
- `training/bucket_padding/bucket_training_alignment_check.ipynb`

## Conventions
- Generated notebook outputs should be written to `artifacts/evaluations/...`.
- Shared helper code belongs in `shared/` or next to the workflow it supports.
- Notebook imports should resolve the repo root dynamically instead of assuming a flat `notebooks/` layout.
