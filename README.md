# QuantumDiffusion

QuantumDiffusion generates quantum circuit datasets, trains diffusion models on those circuits, and evaluates the trained models with fidelity and circuit-level statistics. The repository vendors the genQC stack under `src/my_genQC`, so every script can run locally without extra services.

## Setup
- Use Python 3.10+ inside a virtual environment.
- Install dependencies with `pip install -r requirements.txt`; this provides PyTorch, Qiskit, Hydra, genQC, and supporting tooling.
- Run every command from the repository root so the scripts can import `src/` and Hydra can locate `conf/`.
- Hydra writes runs into `./outputs/<date>/<time>` by default. Pass `hydra.run.dir=.` `hydra.output_subdir=null` or use absolute paths in the configs when you want artifacts to land directly in the repo.

## Configuration system
Hydra drives every entrypoint through `conf/config.yaml`:

```yaml
defaults:
  - datasets: default
  - training: default
  - evaluation: default
```

- `conf/datasets/` – parameters accepted by `DatasetGenerator` (gate sets, qubits, SRV/UNITARY/BOTH conditioning, sample counts, output paths, etc.).
- `conf/training/` – `general`, `model`, `training`, `scheduler`, and `text_encoder` blocks used by `DiffusionTrainer` and the dataset loader.
- `conf/evaluation/` – dataset/model locations plus guidance, sampling, and Hugging Face overrides for `evaluate_model.py`.

Select a different preset with `datasets=clifford_3q_unitary`, `training=quick_test_srv`, `evaluation=comprehensive`, etc. Any field can be overridden inline, e.g. `training.training.num_epochs=5`. To add a reusable configuration, drop a new YAML file inside the corresponding folder and refer to it by name.

## Workflow
### 1. Generate circuits
`scripts/generate_dataset.py` reads `cfg["datasets"]` and calls `DatasetGenerator`. Use Hydra overrides to pick a preset and tweak parameters:

```bash
python scripts/generate_dataset.py \
  hydra.run.dir=. hydra.output_subdir=null \
  datasets=clifford_3q_unitary \
  datasets.num_samples=2048 \
  datasets.output_path=./datasets/clifford_3q_unitary
```

The generator emits one folder per requested condition (SRV, UNITARY, or BOTH), each containing `config.yaml` plus the serialized tensors under `dataset/ds/`.

### 2. Train diffusion models
`scripts/train_model.py` consumes `cfg["training"]`. The `general` block wires up the dataset path, output location, and experiment names, while `model`, `training`, `scheduler`, and `text_encoder` map directly onto the trainer. CLI flags `--device`, `--resume`, and `--verbose` are still available.

```bash
python scripts/train_model.py \
  --device cuda \
  hydra.run.dir=. hydra.output_subdir=null \
  training=quick_test_srv \
  training.general.dataset=./datasets/srv_dataset \
  training.general.output_path=./models/quick_test
```

Artifacts (model weights, embedder cache, metadata, logs) are written to `training.general.output_path`, and `ModelManager` registers the run using `general.model_name`.

### 3. Evaluate models
`scripts/evaluate_model.py` evaluates a saved pipeline (local or Hugging Face) against a dataset produced by step 1. Configure it through `cfg["evaluation"]`:

```bash
python scripts/evaluate_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  evaluation=comprehensive \
  evaluation.dataset=./datasets/srv_dataset \
  evaluation.model_dir=./models/quick_test/default_model_srv \
  evaluation.num_samples=128 \
  evaluation.model_params.sample_steps=50
```

Set `evaluation.hf_repo=<org/model>` if you want to pull a remote pipeline; leave `evaluation.model_dir` empty in that case. The script reports fidelity metrics, gate statistics, and entanglement histograms while reusing genQC’s native tooling.

## Repository layout
- `conf/`: Hydra defaults plus dataset/training/evaluation presets.
- `src/quantum_diffusion/`: dataset generation, training, evaluation, and utility modules that wrap genQC.
- `src/my_genQC/`: vendored genQC stack used by every pipeline.
- `scripts/`: entrypoints (`generate_dataset.py`, `train_model.py`, `evaluate_model.py`) and sample artifacts under `scripts/datasets`, `scripts/models`, `scripts/logs`, and `scripts/outputs`.
- `notebooks/`: exploratory workflows and debugging references.
- `tests/`: regression tests for the core package.
