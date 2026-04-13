# Experiment Tracking

This repo already has the code and config pieces for thesis experiments. The main problem is bookkeeping: datasets, checkpoints, evaluations, notebooks, and local W&B fragments are not tied together in one auditable place.

## What Is Tracked
- Source of truth for experiment scope: `experiments/registry.yaml`
- Large generated outputs: `artifacts/`
- Thesis-ready summaries and tables: `reports/thesis/`

## Why This Split
- `conf/`, `scripts/`, and `src/` define how an experiment runs.
- `artifacts/` stores what a run produced.
- `experiments/registry.yaml` records what should exist and what is currently missing.
- `reports/thesis/` is where you turn raw outputs into material you can cite in the thesis.

## Minimal Workflow
1. Pick or add an experiment entry in `experiments/registry.yaml`.
2. Run the relevant Hydra command with explicit output overrides into `artifacts/.../<experiment-id>/...`.
3. Save evaluation summaries or extracted tables under `reports/thesis/notes/`.
4. Run `python scripts/audit_experiments.py` to see which expected files or directories are still missing.
5. Update the experiment status in the registry from `partial` to `ready` only when the main artifacts and notes exist.

## Recommended Naming
- Use stable experiment IDs such as `srv-baseline-reproduction` or `unitary-curriculum-learning`.
- Keep stage names explicit when there are multiple phases: `stage1`, `stage2`, `clip`, `cloob`, `qiskit`, `quditkit`.
- Avoid unnamed directories like `final_run`, `new_run`, `test2`, or notebook-only evidence without a matching registry entry.

## Audit Script
The audit script checks every path declared in `experiments/registry.yaml` and reports:
- which configs and evidence files exist,
- which expected artifact directories are present,
- which items are still missing.

Run it from the repo root:

```bash
python scripts/audit_experiments.py
```

For a markdown table:

```bash
python scripts/audit_experiments.py --format markdown
```
