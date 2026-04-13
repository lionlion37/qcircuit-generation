# Experiment Registry

This directory is the thesis-facing source of truth for experiments in this repo.

## Files
- `registry.yaml`: canonical list of experiments, expected artifacts, and current recovery status.

## How To Use It
1. Add or update the matching entry in `registry.yaml` before starting a new experiment.
2. Save large generated artifacts under `artifacts/` and keep only summaries, tables, and small metadata in git.
3. Point Hydra outputs to explicit experiment-specific paths instead of relying on scattered defaults.
4. After a run, record what was produced, what metrics are trustworthy, and what still needs rerunning.

## Recommended Artifact Layout
Use Hydra overrides so outputs land in a stable location:

```text
artifacts/
  datasets/<experiment-id>/
  models/<experiment-id>/
  evaluations/<experiment-id>/
  logs/<experiment-id>/
```

Keep thesis-ready material in tracked files:

```text
experiments/
  registry.yaml
reports/
  thesis/
    tables/
    figures/
    notes/
```

## Example Overrides
Dataset generation:

```bash
python scripts/generate_dataset.py \
  hydra.run.dir=. hydra.output_subdir=null \
  datasets=srv_paper_dataset_qiskit_optimized \
  datasets.output_path=./artifacts/datasets/srv-datasets/qiskit
```

Training:

```bash
python scripts/train_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  training=paper_stage_1 \
  training.general.dataset=./artifacts/datasets/srv-datasets/qiskit \
  training.general.output_path=./artifacts/models/srv-baseline-reproduction
```

Evaluation:

```bash
python scripts/evaluate_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  evaluation=paper_srv \
  evaluation.dataset=./artifacts/datasets/srv-datasets/qiskit/srv_8q_dataset \
  evaluation.model_dir=./artifacts/models/srv-baseline-reproduction/paper_stage_2 \
  evaluation.save_folder=./artifacts/evaluations/srv-baseline-reproduction
```
