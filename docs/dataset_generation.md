# Dataset Generation Pipeline

Overview of how `scripts/generate_dataset.py` builds genQC-compatible datasets with Hydra configs.

## Configure
- Choose a dataset preset in `conf/datasets/*.yaml` (default is `conf/datasets/default.yaml`). Key knobs: `gate_set`, `num_qubits`, `num_samples`, `min_gates`/`max_gates`, `condition_type` (`SRV`, `UNITARY`, or `BOTH`), and `output_path`.
- Hydra merges `conf/config.yaml` defaults and any CLI overrides. Use `hydra.run.dir=. hydra.output_subdir=null` to write artifacts into the repo instead of `outputs/`.
- Multi-run sweeps (e.g., `srv_paper_dataset.yaml`) rely on Hydra sweeper params to iterate variants.

## Execution Flow (code reference: `scripts/generate_dataset.py`, `src/quantum_diffusion/data/dataset.py`)
1) **Bootstrap**: CLI kicks off `DatasetGenerator.generate_dataset`, logging the active config and selecting a device (CPU by default).
2) **Toolkit setup**: Builds a vocabulary from `gate_set`, instantiates a `Simulator` with the Qiskit backend, and a `CircuitTokenizer` for instruction encoding.
3) **Condition handling**: `_normalize_condition_types` expands `condition_type` into one or more `CircuitConditionType` targets; each is processed independently (SRV → Schmidt rank vectors, UNITARY → full unitaries).
4) **Circuit synthesis**: Calls `my_genQC.platform.circuits_generation.generate_circuit_dataset` with the tokenizer, backend, gate pool, qubit count, and gate-count range. That routine samples random circuits, computes the requested condition tensors, encodes gates/parameters, filters duplicates, and can parallelize via memory-mapped buffers.
5) **Dataset assembly**:
   - SRV: wraps tensors/labels in `CircuitsConfigDataset`, configured with padding, gate pool, and sample counts.
   - UNITARY: attaches unitary tensors and builds a `MixedCircuitsConfigDataset` (with `model_scale_factor` and padding tuned for compilation tasks).
6) **Persist artifacts**: For each condition, saves under `<output_path>/<condition>/` (or just `<output_path>` if single condition) with `dataset/ds/` for tensors and `config.yaml` describing gates, shapes, padding constants, and storage layout. The generator returns a metadata dict per condition with paths and sample counts.

## Consuming the data
- Training and eval scripts expect a folder containing `dataset/ds/` and `config.yaml`. `DatasetLoader.load_dataset` (same module) reconstructs the dataset and, if requested, a text encoder using the stored `gate_pool`.
- Example generation command:
```bash
python scripts/generate_dataset.py \
  hydra.run.dir=. hydra.output_subdir=null \
  datasets=clifford_3q_unitary \
  datasets.num_samples=2000 \
  datasets.output_path=./datasets/clifford_3q_unitary
```
