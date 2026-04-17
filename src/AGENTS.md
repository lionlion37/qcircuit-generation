# Source packages

Three packages live under `src/`. The dependency flow is:

```
quantum_diffusion  -->  my_genQC  -->  quditkit-main_schmidt
(high-level API)       (core)         (vendored simulator)
```

---

## `my_genQC` -- core framework

### `dataset/`
Dataset management for quantum circuits.

| Module                  | Role                                                              |
|-------------------------|-------------------------------------------------------------------|
| `config_dataset.py`     | Base dataset class with YAML/SafeTensors load/save                |
| `cached_dataset.py`     | Adds pre-computed CLIP text embeddings to speed up training       |
| `circuits_dataset.py`   | Circuit-specific dataset (gate pools, qubit configs)              |
| `mixed_cached_dataset.py` | Combines multiple datasets with bucket or max padding           |
| `dataset_helper.py`     | Deduplication, shuffling, uniquification utilities                |
| `balancing.py`          | Quantile-based balancing by gate count                            |

### `models/`
Neural network architectures for diffusion-based generation.

| Module                           | Role                                                        |
|----------------------------------|-------------------------------------------------------------|
| `unet_qc.py`                    | U-Net for circuit generation (`QC_Cond_UNet`, `QC_Compilation_UNet`) |
| `unitary_encoder.py`            | CNN + transformer encoder for unitary matrix conditions      |
| `frozen_open_clip.py`           | Frozen CLIP/CLOOB text encoder wrapper                      |
| `layers.py`                     | Shared building blocks (residual, down/up blocks)           |
| `position_encoding.py`          | RoPE and learned positional embeddings                      |
| `config_model.py`               | Base module with checkpoint I/O                             |
| `clip/unitary_clip.py`          | Contrastive pre-training for unitary + circuit encoders     |
| `embedding/base_embedder.py`    | Abstract embedding interface                                |
| `embedding/rotational_preset_embedder.py` | Multimodal rotational embedder                   |
| `transformers/attention.py`     | Self-attention and cross-attention blocks                    |
| `transformers/transformers.py`  | Full transformer encoder/decoder with spatial attention      |
| `transformers/cirdit_multimodal.py` | Circuit Diffusion Transformer (CirDiT)                  |

### `pipeline/`
Training and inference pipelines.

| Module                             | Role                                                    |
|------------------------------------|---------------------------------------------------------|
| `pipeline.py`                      | Base training loop with callbacks and checkpointing     |
| `diffusion_pipeline.py`            | Diffusion training with scheduler and guidance          |
| `diffusion_pipeline_special.py`    | Unitary-conditioned variant                             |
| `compilation_diffusion_pipeline.py`| Conditions generation on target unitaries               |
| `multimodal_diffusion_pipeline.py` | Parametrized gate support with separate schedulers      |
| `unitary_clip_pipeline.py`         | Contrastive learning for unitary/circuit encoders       |
| `callbacks.py`                     | Callback mechanism (epoch/batch/fit events)             |
| `metrics.py`                       | Training metrics                                        |

### `platform/`
Backend abstraction and circuit simulation.

| Module                              | Role                                                   |
|-------------------------------------|--------------------------------------------------------|
| `simulation.py`                     | `Simulator` class -- unified interface over all backends |
| `circuits_instructions.py`          | Data structures for quantum instructions               |
| `circuits_generation.py`            | Random circuit dataset generation (SRV / unitary targets) |
| `backends/base_backend.py`          | Abstract backend interface                             |
| `backends/circuits_qiskit.py`       | Qiskit backend                                         |
| `backends/circuits_quditkit.py`     | QuditKit backend                                       |
| `backends/circuits_cudaq.py`        | CUDA-Q backend (GPU-accelerated)                       |
| `backends/circuits_pennylane.py`    | PennyLane backend                                      |
| `backends/circuit_optimizer.py`     | Native qudit circuit optimizer                         |
| `tokenizer/base_tokenizer.py`      | Abstract tokenizer interface                           |
| `tokenizer/circuits_tokenizer.py`   | Circuit-to-tensor tokenizer with parameter normalization |
| `tokenizer/tensor_tokenizer.py`     | Gate-pair tokenizer for hierarchical decompositions    |

### `scheduler/`
Diffusion noise schedulers.

| Module              | Role                                              |
|---------------------|---------------------------------------------------|
| `scheduler.py`      | Base scheduler (timestep management, noise addition) |
| `scheduler_ddpm.py` | DDPM with configurable beta schedules             |
| `scheduler_ddim.py` | DDIM for faster deterministic sampling            |
| `scheduler_dpm.py`  | DPM solver for improved sample quality            |

### `inference/`

| Module                | Role                                                       |
|-----------------------|------------------------------------------------------------|
| `sampling.py`         | Batched model sampling with guidance and parameter constraints |
| `evaluation_helper.py`| Parallel SRV and unitary computation from backend objects  |
| `eval_metrics.py`     | `UnitaryFrobeniusNorm`, `UnitaryInfidelityNorm` metrics    |

### `utils/`

| Module            | Role                                                          |
|-------------------|---------------------------------------------------------------|
| `config_loader.py`| YAML/OmegaConf config loading, SafeTensors model serialization |
| `async_fn.py`     | Parallel job execution and memory-mapped tensor storage       |
| `math.py`         | Matrix power, Gram-Schmidt orthonormalization                 |
| `misc_utils.py`   | Device inference, memory management, tensor scaling, plotting |

---

## `quantum_diffusion` -- high-level API

Wraps `my_genQC` into a simpler interface for the standard workflow.

| Module                   | Role                                                   |
|--------------------------|--------------------------------------------------------|
| `data/dataset.py`        | `DatasetGenerator` and `DatasetLoader` high-level API  |
| `training/training.py`   | `DiffusionTrainer` with W&B logging and checkpointing  |
| `evaluation/evaluator.py`| `SRVEvaluator` for SRV-based model evaluation          |
| `utils/config.py`        | `ConfigManager` for YAML config I/O                    |
| `utils/logging.py`       | Structured logging for training and evaluation         |
| `utils/model_helper.py`  | Model loading utilities                                |

---

## `quditkit-main_schmidt` -- vendored QuditKit

Higher-dimensional quantum simulator, vendored under `src/quditkit-main_schmidt/src/qudit_sim/`.

Provides `QuantumCircuit`, `Backend`, gate definitions, and tableau-based simulation for qudit systems. Supports NumPy and CuPy backends. Includes a Numba-optimized fast path.

Used by `my_genQC/platform/backends/circuits_quditkit.py` and `my_genQC/platform/backends/circuit_optimizer.py`.
