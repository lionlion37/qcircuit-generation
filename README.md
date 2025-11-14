# QuantumDiffusion

QuantumDiffusion generates quantum circuit datasets, trains diffusion models on those circuits, and evaluates the trained models with fidelity and circuit-level statistics. The repository vendors the genQC stack under `src/my_genQC`, so every script can run locally without extra services.

## Setup
1. Use Python 3.10+ inside a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`; this provides PyTorch, Qiskit, Hydra, genQC, and supporting tooling.
3. Execute all commands from the repository root so the scripts can import the modules under `src/`.

## Workflow
### 1. Generate circuits
Run `python scripts/generate_dataset.py --config configs/datasets/clifford_unitary.yaml --output ./datasets/clifford_unitary`. The script instantiates `DatasetGenerator`, calls into the genQC simulator/tokenizer, and writes tensors plus a `config.yaml` per condition (SRV, UNITARY, or BOTH). Override gates, qubits, samples, and conditioning through CLI flags, or switch to a preset such as `--preset clifford_3q_unitary`.

### 2. Train diffusion models
Use `python scripts/train_model.py --dataset ./datasets/clifford_unitary/unitary --config configs/training/compilation_training.yaml --output ./models/compilation_run`. Training loads the condition-specific dataset, builds the requested UNet and scheduler, and stores checkpoints in `./models`. Flags like `--preset quick_test`, `--epochs`, `--batch-size`, `--learning-rate`, `--resume`, and `--device` override the config on the fly.

### 3. Evaluate models
Evaluate with `python scripts/evaluate_model.py --model ./models/compilation_run --dataset ./datasets/clifford_unitary/unitary --config configs/evaluation/comprehensive.yaml --output ./evaluation/compilation_run`. The evaluator can regenerate samples or reuse held-out data, computes fidelity, gate-depth statistics, diversity metrics, and optionally plots results. Limit the workload with `--num-samples`, pick specific `--metrics`, or skip figures via `--no-plots`.

## Configuration
Configuration files live under `configs/` (`datasets/`, `training/`, `evaluation/`). Every script accepts `--config` with a YAML file or `--preset` names defined in `ConfigManager` (e.g., `clifford_3q_unitary`, `quick_test`, `comprehensive`). CLI flags always override the loaded configuration, which makes it straightforward to sweep parameters without editing the YAML files.

## Repository layout
- `src/quantum_diffusion/`: dataset generation, training, evaluation, and utility modules that wrap genQC.
- `src/my_genQC/`: minimal, vendored copy of genQC used by the pipelines.
- `scripts/`: runnable entrypoints for dataset generation, training, and evaluation (default outputs are `./datasets`, `./models`, and `./evaluation`).
- `configs/`: ready-to-use YAML configs referenced in the commands above.
- `notebooks/`: exploratory workflows and debugging references.
- `tests/`: regression tests for the core package.
