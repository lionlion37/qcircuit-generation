# Unitary Curriculum Workflow

Goal: pretrain unitary compilation without `ccx`, then finetune on a smaller full-pool dataset that includes `ccx`.

## Presets
- Dataset stage 1: `conf/datasets/unitary_curriculum_stage1_quditkit.yaml`
- Dataset stage 2: `conf/datasets/unitary_curriculum_stage2_qiskit.yaml`
- Training stage 1: `conf/training/unitary_curriculum_stage1.yaml`
- Training stage 2: `conf/training/unitary_curriculum_stage2.yaml`

## Key Mechanism
- Stage 1 uses:
  - full token `gate_set = ['h', 'cx', 'z', 'x', 'ccx', 'swap']`
  - active `generation_gate_set = ['h', 'cx', 'z', 'x', 'swap']`
- This keeps the model token space fixed across both stages while excluding `ccx` from stage-1 sampled circuits.
- Stage 2 uses the full gate pool for both tokenization and generation.

## Execution
```bash
python scripts/generate_dataset.py \
  hydra.run.dir=. hydra.output_subdir=null \
  datasets=unitary_curriculum_stage1_quditkit

python scripts/train_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  training=unitary_curriculum_stage1

python scripts/generate_dataset.py \
  hydra.run.dir=. hydra.output_subdir=null \
  datasets=unitary_curriculum_stage2_qiskit

python scripts/train_model.py \
  hydra.run.dir=. hydra.output_subdir=null \
  training=unitary_curriculum_stage2
```

## Notes
- Stage-2 training resumes from `general.init_from_pipeline_dir`.
- Resume fails early if the checkpoint gate pool and dataset gate pool are incompatible.
- Adjust `num_samples`, `balance_max`, `num_epochs`, and `learning_rate` via Hydra overrides as needed.
