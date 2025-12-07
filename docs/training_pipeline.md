# Training Pipeline

How `scripts/train_model.py` trains a genQC diffusion model using Hydra configs and the dataset format saved by the generator.

## Configure
- Pick a preset in `conf/training/*.yaml` (default via `conf/config.yaml`). Key fields: `general.dataset`, `general.output_path`, `general.device` (`auto` chooses CUDA if available), `training.batch_size`, `training.num_epochs`, optimizer/lr, and model/scheduler params under `model.*` and `scheduler.*`.
- Point `general.dataset` to a folder containing `dataset/ds/` + `config.yaml` (single set) or a directory of multiple such folders (for mixed qubit counts). Keep Hydra outputs local with `hydra.run.dir=. hydra.output_subdir=null`.

## Execution Flow (code: `scripts/train_model.py`, `src/quantum_diffusion/training/training.py`, `src/quantum_diffusion/data/dataset.py`)
1) **Bootstrap**: Resolve device, initialize logging/ExperimentLogger, read `cfg["training"]`.
2) **Dataset load**: `DatasetLoader.load_dataset` reconstructs tensors/tokenizer from `config.yaml`. If `general.dataset` is a directory of sub-datasets, they are loaded and merged via `DatasetLoader.combine_datasets` (padding/balancing applied).
3) **Dataloaders**: `DatasetLoader.get_dataloaders` builds train/val loaders with optional text encoder instantiation (from stored `gate_pool`).
4) **Model setup**: `DiffusionTrainer.setup_model` selects tokenizer, injects vocab size into UNet params, configures unitary encoder if present, builds the scheduler, and instantiates the diffusion pipeline (`DiffusionPipeline` or compilation variant).
5) **Compile**: `DiffusionTrainer.compile_model` wires optimizer/loss and learning rate into the pipeline.
6) **Train**: `pipeline.fit` runs for `training.num_epochs`, driven by the dataloaders; ExperimentLogger tracks steps/metrics.
7) **Persist**: `DiffusionTrainer.save_model` writes the pipeline checkpoint/config to `general.output_path[/model_name]`, plus `training_config.yaml` and `metadata.yaml`. `ModelManager` can register the run in-memory for later reference.

## Running it
```bash
python scripts/train_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  training=quick_test_srv \
  training.general.dataset=./datasets/srv_dataset \
  training.general.output_path=./models/quick_test \
  training.training.num_epochs=5
```
