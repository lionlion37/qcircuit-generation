# Evaluation Pipeline

How `scripts/evaluate_model.py` scores a diffusion pipeline against a saved genQC dataset.

## Configure
- Pick a preset in `conf/evaluation/*.yaml` (default via `conf/config.yaml`). Key fields: `dataset` (path to folder with `dataset/ds/` + `config.yaml`), `model_dir` (local checkpoint folder) **or** `hf_repo` (Hugging Face repo), `num_samples`, and sampler params under `model_params.*` (`sample_steps`, `guidance_scale`, `auto_batch_size`).
- Keep Hydra outputs local with `hydra.run.dir=. hydra.output_subdir=null`.

## Execution Flow (code: `scripts/evaluate_model.py`)
1) **Bootstrap**: Resolve device via `infer_torch_device`, load `cfg["evaluation"]`.
2) **Dataset load**: `load_dataset` recreates `CircuitsConfigDataset` or `MixedCircuitsConfigDataset` from `config.yaml` + tensor store.
3) **Pipeline load**: `load_pipeline` either pulls from a local `model_dir` (expecting `config.yaml` and weights) or from `hf_repo` via `DiffusionPipeline.from_pretrained`.
4) **Sampling**: Set guidance mode and timesteps, cap `num_samples` by dataset size, infer system size/qubits. Choose path:
   - **Compilation datasets** (have stored unitaries `U`): call `generate_compilation_tensors` with prompts and target unitaries.
   - **SRV datasets**: call `generate_tensors` with prompts.
5) **Decode**: Build a `CircuitTokenizer` from the dataset gate pool, use `decode_tensors_to_backend` with a Qiskit `Simulator` to turn sampled tensors into executable circuits; drop failed decodes.
6) **Metrics**:
   - Compilation: compute target and predicted unitaries, then `UnitaryFrobeniusNorm` and `UnitaryInfidelityNorm` stats.
   - SRV: parse target SRVs from prompt strings, compute exact-match and per-qubit accuracy, plus entanglement-bin histograms.
7) **Reporting**: Prints sample counts, decode failures, and metric summaries to stdout.

## Running it
```bash
python scripts/evaluate_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  evaluation=default \
  evaluation.dataset=./datasets/srv_dataset \
  evaluation.model_dir=./models/quick_test \
  evaluation.num_samples=256 \
  evaluation.model_params.guidance_scale=1.5
```
