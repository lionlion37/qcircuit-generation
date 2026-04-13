# QuantumDiffusion

A master's thesis research codebase for **quantum circuit generation via diffusion models**. The repo covers dataset generation (SRV and unitary tasks), diffusion model training, and evaluation — all driven by [Hydra](https://hydra.cc) configs. It vendors the genQC stack in `src/my_genQC` and ships a qudit stabilizer backend in `src/quditkit` so runs are fully self-contained.

---

## Setup

**Python version:** 3.10–3.13 recommended. Hydra 1.3.x crashes on Python 3.14 during CLI setup, so stay on 3.12 or 3.13.

```bash
# Install main dependencies
pip install -r requirements.txt

# Install the bundled qudit stabilizer backend
pip install -e quditkit-main_schmidt

# Or use the Makefile shortcut
make env
```

Run all scripts from the **repo root** so `src/` is on `sys.path`. To keep Hydra artifacts beside the repo instead of under `outputs/<date>/<time>/`, always pass:

```
hydra.run.dir=. hydra.output_subdir=null
```

---

## Repository Layout

```
conf/                        # Hydra config root
  config.yaml                # Top-level defaults
  datasets/                  # Dataset presets (SRV, unitary, curriculum)
  training/                  # Training presets (paper stages, CLOOB/CLIP ablation)
  evaluation/                # Evaluation presets

scripts/                     # Entrypoints
  generate_dataset.py
  train_model.py
  evaluate_model.py
  audit_experiments.py       # Checks registry.yaml against the filesystem

src/
  quantum_diffusion/         # Dataset helpers, training loop, evaluation utilities, logging
  my_genQC/                  # Vendored genQC models, pipelines, tokenizers, schedulers

artifacts/                   # All generated outputs (gitignored large files)
  datasets/
    srv-datasets/            # qiskit/ and quditkit/ variants
    unitary-baseline-reproduction/
    unitary-curriculum-learning/
  models/
    srv-baseline-reproduction/
    srv-text-encoder-ablation/
    unitary-baseline-reproduction/
    unitary-curriculum-learning/
  evaluations/               # Per-experiment evaluation outputs
  logs/

experiments/
  registry.yaml              # Canonical experiment registry (status, configs, expected artifacts)

reports/                     # Thesis-ready curated outputs
  srv-dataset-backend-comparison/
  srv-text-encoder-ablation/
  thesis/
    notes/                   # Short summaries linking thesis claims to artifact dirs
    tables/
    figures/
    appendix/

notebooks/                   # Exploratory and evaluation notebooks
  datasets/
  evaluation/srv/
  evaluation/unitary/
  training/bucket_padding/

docs/                        # Pipeline walkthroughs
  dataset_generation.md
  training_pipeline.md
  evaluation_pipeline.md
  experiment_tracking.md
  curriculum_learning.md

tests/                       # Minimal unit tests
```

---

## Configuration Presets

`conf/config.yaml` sets the defaults:

```yaml
defaults:
  - datasets: default
  - training: default
  - evaluation: default
```

Override presets and individual fields on the CLI. Example: `training=paper_stage_1 training.training.num_epochs=5`.

### Dataset presets (`conf/datasets/`)

| Preset | Description |
|---|---|
| `default` | 5-qubit SRV, Qiskit backend |
| `srv_paper_dataset_qiskit_optimized` | Hydra multirun, 3–8 qubits, Qiskit |
| `srv_paper_dataset_quditkit_not_optimized` | Same sweep, quditkit backend |
| `unitary_paper_dataset_qiskit_optimized` | Unitary task dataset, Qiskit |
| `unitary_curriculum_stage1_quditkit` | Curriculum stage 1 (no CCX), quditkit |
| `unitary_curriculum_stage2_qiskit` | Curriculum stage 2, Qiskit |

### Training presets (`conf/training/`)

| Preset | Description |
|---|---|
| `paper_stage_1` / `paper_stage_2` | SRV baseline, two-stage paper schedule |
| `paper_unitary` | Unitary baseline |
| `unitary_curriculum_stage1` / `stage2` | Curriculum learning for unitary task |
| `cloob_rn50_stage_1/2` | CLOOB RN50 text encoder ablation |
| `cloob_rn50x4_stage_1/2` | CLOOB RN50x4 text encoder ablation |
| `clip_rn50_stage_1/2` | CLIP RN50 text encoder ablation |
| `clip_rn50x4_stage_1/2` | CLIP RN50x4 text encoder ablation |

### Evaluation presets (`conf/evaluation/`)

| Preset | Description |
|---|---|
| `default` | Standard local model evaluation |
| `comprehensive` | Extended metrics sweep |
| `paper_srv` | Paper-style SRV evaluation |
| `paper_unitary` | Paper-style unitary evaluation |
| `remote_model` | Pulls a pipeline from Hugging Face |

---

## Workflows

### Generate a dataset

Paper-style SRV sweep across 3–8 qubits (Hydra multirun):

```bash
python scripts/generate_dataset.py \
  hydra.run.dir=. hydra.output_subdir=null \
  datasets=srv_paper_dataset_qiskit_optimized \
  datasets.output_path=./artifacts/datasets/srv-datasets/qiskit
```

Single-condition example (3-qubit Clifford unitary):

```bash
python scripts/generate_dataset.py \
  hydra.run.dir=. hydra.output_subdir=null \
  datasets=clifford_3q_unitary \
  datasets.num_samples=2048 \
  datasets.output_path=./artifacts/datasets/clifford_3q_unitary
```

### Train a model

SRV baseline reproduction (two-stage training):

```bash
# Stage 1
python scripts/train_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  training=paper_stage_1 \
  training.general.dataset=./artifacts/datasets/srv-datasets/qiskit \
  training.general.output_path=./artifacts/models/srv-baseline-reproduction

# Stage 2
python scripts/train_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  training=paper_stage_2 \
  training.general.dataset=./artifacts/datasets/srv-datasets/qiskit \
  training.general.output_path=./artifacts/models/srv-baseline-reproduction
```

If you supply a parent directory containing multiple dataset folders, each is loaded and combined for training.

### Evaluate a model

Local model:

```bash
python scripts/evaluate_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  evaluation=paper_srv \
  evaluation.dataset=./artifacts/datasets/srv-datasets/qiskit/srv_8q_dataset \
  evaluation.model_dir=./artifacts/models/srv-baseline-reproduction/paper_stage_2 \
  evaluation.save_folder=./artifacts/evaluations/srv-baseline-reproduction
```

Hosted Hugging Face pipeline:

```bash
python scripts/evaluate_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  evaluation=remote_model \
  evaluation.dataset=./artifacts/datasets/srv-datasets/qiskit/srv_8q_dataset \
  evaluation.model_dir=null
```

---

## Experiments

Experiments are tracked in `experiments/registry.yaml`. Run the audit script to check which artifacts are present:

```bash
python scripts/audit_experiments.py
```

| Experiment | Task | Status |
|---|---|---|
| `srv_dataset_backend_comparison` | Quditkit vs Qiskit dataset generation | partial |
| `bucket_padding_variants` | Bucket padding impact on training | partial |
| `srv_baseline_reproduction` | Reproduce paper SRV baseline | partial |
| `unitary_baseline_reproduction` | Reproduce paper unitary compilation baseline | partial |
| `srv_text_encoder_cloob` | CLOOB vs CLIP text encoder ablation (SRV) | **ready** |
| `unitary_curriculum_learning` | Curriculum learning for unitary task | partial |

Thesis-ready outputs (figures, tables) go to `reports/`. Raw generated artifacts (datasets, checkpoints, evaluation JSONs) live in `artifacts/`. See `docs/experiment_tracking.md` for the full workflow.

---

## Notes

- GPU is optional but strongly recommended for training.
- Pipelines rely on PyTorch plus either Qiskit or quditkit as the circuit simulation backend.
- For thesis work, prefer explicit Hydra path overrides into `./artifacts/...` and record experiment status in `experiments/registry.yaml`.
- See `docs/*.md` for in-depth pipeline documentation.
