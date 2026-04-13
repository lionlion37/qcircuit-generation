# QuantumDiffusion

Hydra-driven scripts for quantum circuit dataset generation, diffusion-model training, and evaluation. The repository vendors the genQC stack in `src/my_genQC` and ships a stabilizer backend in `quditkit-main_schmidt` so runs are self-contained.

## Setup
- Python 3.10-3.13 recommended; Hydra 1.3.x currently crashes on Python 3.14 during CLI setup, so keep the project on 3.12/3.13.
- Install dependencies with `pip install -r requirements.txt`. For the bundled qudit stabilizer backend, also run `pip install -r quditkit-main_schmidt/requirements.txt` and `pip install -e quditkit-main_schmidt` (mirrors the `make env` target).
- Run commands from the repo root so `src/` is on `sys.path`. Hydra writes to `outputs/<date>/<time>`; add `hydra.run.dir=. hydra.output_subdir=null` to keep artifacts beside the repo.

## Repository layout
- `conf/` – Hydra defaults (`config.yaml`) plus dataset/training/evaluation presets.
- `scripts/` – entrypoints: `generate_dataset.py`, `train_model.py`, `evaluate_model.py` (previous Hydra runs live under `scripts/outputs`, `scripts/multirun`, etc.).
- `src/quantum_diffusion/` – dataset loading/generation helpers, training loop, evaluation utilities, and logging.
- `src/my_genQC/` – vendored genQC models, pipelines, tokenizers, schedulers, and datasets.
- `artifacts/` – canonical location for datasets, trained models, evaluation outputs, logs, and local W&B runs.
- `experiments/` – thesis experiment registry and bookkeeping.
- `quditkit-main_schmidt/` – local qudit stabilizer simulator used by notebooks/evaluation.
- `docs/` – pipeline walkthroughs (`dataset_generation.md`, `training_pipeline.md`, `evaluation_pipeline.md`).
- `notebooks/`, `tests/` – exploratory work and minimal decoding tests.

## Configuration presets
Hydra defaults live in `conf/config.yaml`:
```yaml
defaults:
  - datasets: default
  - evaluation: default
  - training: default
```
Pick presets with `datasets=<name>`, `training=<name>`, or `evaluation=<name>` and override any field inline (`training.training.num_epochs=5`).

- Dataset configs: `default` (5-qubit SRV on Qiskit), `clifford_3q_unitary`, `clifford_4q_srv`, `srv_paper_dataset_qiskit_optimized` (Hydra multirun over 3–8 qubits with Qiskit), `srv_paper_dataset_quditkit_not_optimized` (multirun with the quditkit backend).
- Training configs: `default`, `quick_test_srv`, `quick_test_unitary`, `standard_unitary`, `large_model_unitary`.
- Evaluation configs: `default`, `comprehensive`, `remote_model` (pulls a pipeline from Hugging Face).

See `docs/*.md` for deeper pipeline details and config field explanations.

## Workflows
### Generate datasets
`scripts/generate_dataset.py` calls `quantum_diffusion.data.DatasetGenerator` to build genQC-compatible datasets. Paper-style SRV sweep (Hydra multirun):
```bash
python scripts/generate_dataset.py \
  hydra.run.dir=. hydra.output_subdir=null \
  datasets=srv_paper_dataset_qiskit_optimized
```
Single-condition example (3-qubit Clifford unitary dataset):
```bash
python scripts/generate_dataset.py \
  hydra.run.dir=. hydra.output_subdir=null \
  datasets=clifford_3q_unitary \
  datasets.num_samples=2048 \
  datasets.output_path=./artifacts/datasets/clifford_3q_unitary
```

### Train a diffusion model
`scripts/train_model.py` uses `DatasetLoader` to read a genQC dataset folder (or combine multiple folders if you point at a directory). Example SRV training on bundled paper data:
```bash
python scripts/train_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  training=quick_test_srv \
  training.general.dataset=./artifacts/datasets/srv-paper-datasets/qiskit \
  training.general.output_path=./artifacts/models/quick_test
```
If you supply a parent directory containing multiple datasets, the current script trims each to the first 1000 samples before combining (debug behavior in code).

### Evaluate a model or HF pipeline
`scripts/evaluate_model.py` reuses genQC sampling/metrics on a stored dataset:
```bash
python scripts/evaluate_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  evaluation=default \
  evaluation.dataset=./artifacts/datasets/srv-paper-datasets/quditkit/srv_3q_dataset \
  evaluation.model_dir=./artifacts/models/default/default_model_srv \
  evaluation.num_samples=256 \
  evaluation.model_params.guidance_scale=1.5
```
For a hosted pipeline, use the HF preset:
```bash
python scripts/evaluate_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  evaluation=remote_model \
  evaluation.dataset=./artifacts/datasets/srv-paper-datasets/qiskit/srv_8q_dataset \
  evaluation.model_dir=null
```

## Notes
- The repo now treats `./artifacts/` as the canonical root for generated datasets, checkpoints, evaluation outputs, logs, and local W&B runs.
- Pipelines rely on PyTorch plus Qiskit or quditkit; GPU is optional but recommended for training.
- For thesis work, prefer explicit Hydra overrides into `./artifacts/...` and track experiment status in `experiments/registry.yaml`. See `docs/experiment_tracking.md`.
