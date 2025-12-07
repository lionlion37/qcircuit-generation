# QuantumDiffusion

Hydra-driven scripts for quantum circuit generation, diffusion-model training, and evaluation. The repository vendors the genQC stack in `src/my_genQC` (plus a stabilizer backend in `quditkit-main_schmidt`) so experiments can run locally without external services.

## Setup
- Python 3.10+ is recommended; use a virtual environment.
- Install dependencies with `pip install -r requirements.txt` (PyTorch, Qiskit/Aer, Hydra, ruamel, genQC, etc.).
- Run commands from the repo root so `src/` is on `sys.path` and Hydra finds `conf/`.
- Hydra writes runs to `outputs/<date>/<time>`; add `hydra.run.dir=.` `hydra.output_subdir=null` to keep artifacts beside the repo.

## Repository layout
- `conf/` – Hydra defaults (`config.yaml`) plus presets under `datasets/`, `training/`, and `evaluation/`.
- `scripts/` – entrypoints: `generate_dataset.py`, `train_model.py`, and `evaluate_model.py`.
- `src/quantum_diffusion/` – wrappers for dataset gen/loading (`data/`), training (`training/`), evaluation (`evaluation/`), and logging utilities.
- `src/my_genQC/` – vendored genQC models, pipelines, tokenizers, schedulers, and datasets.
- `data/paper/` – pre-generated SRV datasets for 3–8 qubits matching the paper settings.
- `quditkit-main_schmidt/` – qudit stabilizer simulator used in notebooks.
- `notebooks/` – exploratory notebooks for dataset generation and training.

## Configuration presets
Hydra defaults live in `conf/config.yaml`:
```yaml
defaults:
  - datasets: default
  - evaluation: default
  - training: default
```
Pick presets with `datasets=<name>`, `training=<name>`, or `evaluation=<name>` and override any field inline (`training.training.num_epochs=5`).

- Dataset configs: `default`, `clifford_3q_unitary`, `clifford_4q_srv`, `srv_paper_dataset` (multi-run over 3–8 qubits).
- Training configs: `default`, `quick_test_srv`, `quick_test_unitary`, `standard_unitary`, `large_model_unitary`.
- Evaluation configs: `default`, `comprehensive`, `remote_model` (Hugging Face pull).

## Workflows
### Generate datasets
`scripts/generate_dataset.py` calls `quantum_diffusion.data.DatasetGenerator` to build genQC-compatible datasets. Example: run the paper SRV sweep across 3–8 qubits (multi-run via Hydra):
```bash
python scripts/generate_dataset.py \
  hydra.run.dir=. hydra.output_subdir=null \
  datasets=srv_paper_dataset
```
Single-condition example (unitary conditioning on 3 qubits):
```bash
python scripts/generate_dataset.py \
  hydra.run.dir=. hydra.output_subdir=null \
  datasets=clifford_3q_unitary \
  datasets.num_samples=2048 \
  datasets.output_path=./datasets/clifford_3q_unitary
```

### Train a diffusion model
`scripts/train_model.py` uses `DatasetLoader` to read a genQC dataset folder, builds pipelines from `src/my_genQC`, and trains according to `cfg["training"]`:
```bash
python scripts/train_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  training=quick_test_srv \
  training.general.dataset=data/paper/srv_3q_dataset \
  training.general.output_path=./models/quick_test
```
If `training.general.dataset` points to a directory containing multiple datasets, the loader will combine them with padding/balancing.

### Evaluate a model or HF pipeline
`scripts/evaluate_model.py` reuses genQC sampling/metrics on a stored dataset:
```bash
python scripts/evaluate_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  evaluation=remote_model \
  evaluation.dataset=data/paper/srv_3q_dataset \
  evaluation.hf_repo=Floki00/qc_srv_3to8qubit \
  evaluation.num_samples=300
```
For local checkpoints, set `evaluation.model_dir=<path>` and clear `evaluation.hf_repo`.

## Notes
- Default paths in configs (`./datasets/...`, `./models/...`) assume you copy or generate data into those locations; the paper SRV datasets under `data/` are the only bundled artifacts.
- Pipelines rely on PyTorch and Qiskit; GPU is optional but recommended for training.
