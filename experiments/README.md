# Experiment Registry

This directory is the thesis-facing source of truth for experiments in this repo.

## Files
- `registry.yaml`: canonical list of experiments, expected artifacts, and current recovery status.

## How To Use It
1. Add or update the matching entry in `registry.yaml` before starting a new experiment.
2. Save large generated artifacts under `artifacts/`.
3. Save thesis-ready summaries, tables, and figures under `reports/thesis/`.
4. Point Hydra outputs to explicit experiment-specific paths instead of relying on scattered defaults.
5. After a run, record what was produced, what metrics are trustworthy, and what still needs rerunning.

## Current Layout
Use explicit paths so outputs land in stable, meaningful locations:

```text
artifacts/
  datasets/
    srv-datasets/
      qiskit/
      quditkit/
    unitary-baseline-reproduction/
      train/
      eval/
    unitary-curriculum-learning/
      stage1/
      stage2/
  models/
    encoders/
    checkpoints/
    srv-baseline-reproduction/
    srv-text-encoder-ablation/
    unitary-baseline-reproduction/
    unitary-curriculum-learning/
  evaluations/
    <experiment-id>/<run-name>/
  logs/
```

Keep thesis-ready material under `reports/`:

```text
experiments/
  registry.yaml
reports/
  thesis/
    notes/
    tables/
    figures/
    appendix/
notebooks/
  datasets/
  evaluation/srv/
  evaluation/unitary/
  training/bucket_padding/
```

## Workflow
1. Use `registry.yaml` to define the experiment and expected outputs.
2. Generate or train into `artifacts/datasets/...` and `artifacts/models/...`.
3. Run scripted or notebook-based evaluation into `artifacts/evaluations/.../<run-name>/`.
4. Promote final tables/figures into `reports/thesis/...`.
5. Add a short note in `reports/thesis/notes/` that links the thesis claim to the artifact directory.

## Examples
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

Notebook-based evaluation now follows the same pattern. The main evaluation notebooks save:
- raw evaluation outputs to `artifacts/evaluations/...`
- thesis-ready figures/tables can then be copied or regenerated into `reports/thesis/...`

## Notes
- `scripts/audit_experiments.py` checks `registry.yaml` against the filesystem and shows what is still missing.
- The registry tracks experiment status, but it does not replace judgment. If a path exists but the results are not trustworthy, keep the experiment marked `partial`.
