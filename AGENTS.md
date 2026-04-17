# Project: Quantum Circuit Generation

Diffusion-based generative model for quantum circuits, built for a master's thesis.
Generates circuits conditioned on **Schmidt Rank Vectors (SRV)** or **target unitaries**.

---

## Directory layout

| Directory       | Purpose                                                                 |
|-----------------|-------------------------------------------------------------------------|
| `src/`          | Source packages (`my_genQC`, `quantum_diffusion`, vendored `quditkit`)  |
| `conf/`         | Hydra YAML configs for datasets, training, and evaluation               |
| `experiments/`  | Experiment registry (`registry.yaml`) and audit tooling                 |
| `artifacts/`    | Machine-generated outputs: datasets, models, evaluations, logs          |
| `reports/`      | Thesis-ready curated figures, tables, and summaries                     |
| `notebooks/`    | Analysis notebooks with per-topic helper modules                        |
| `scripts/`      | CLI entry points: `generate_dataset.py`, `train_model.py`, `evaluate_model.py` |

Each of `notebooks/`, `src/`, `experiments/`, and `reports/` has its own `AGENTS.md` with directory-specific conventions.

---

## Global conventions

### Hydra configuration

All CLI scripts use Hydra with configs rooted at `conf/`:

```
conf/
  config.yaml          # root defaults (datasets: default, training: default, evaluation: default)
  datasets/*.yaml      # dataset generation configs
  training/*.yaml      # model training configs
  evaluation/*.yaml    # evaluation configs
```

Override any config group from CLI: `python scripts/train_model.py training=paper_stage_1`.

### Two tasks

| Task     | Model type            | Condition          | Gate set example               |
|----------|-----------------------|--------------------|--------------------------------|
| SRV      | `QC_Cond_UNet`        | Schmidt rank vector| `[h, cx]`                     |
| Unitary  | `QC_Compilation_UNet` | Target unitary     | `[h, cx, z, x, ccx, swap]`   |

### Two backends

| Backend   | Package                    | Notes                                   |
|-----------|----------------------------|-----------------------------------------|
| Qiskit    | `qiskit`                   | Standard, more gates supported          |
| QuditKit  | `quditkit-main_schmidt`    | 3-4x faster generation, vendored in src |

### Multi-stage training

- **Stage 1**: warm-up, learning rate 3e-4, higher guidance
- **Stage 2**: fine-tuning, learning rate 5e-5, loads stage-1 weights via `init_from_pipeline_dir`
- **Curriculum learning**: stage 1 on simplified gate set, stage 2 on full gate set

### Artifact paths

```
artifacts/
  datasets/{experiment-id}/{subset}/     # generated circuit datasets
  models/{experiment-id}/{stage}/        # trained model checkpoints
  evaluations/{experiment-id}/{run}/     # evaluation results (pkl, csv, json, png)
  logs/{experiment}_{timestamp}.log      # training logs
```
