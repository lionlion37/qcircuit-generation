# Experiments

The experiment registry is the **canonical source of truth** for all thesis experiments.

---

## Registry: `registry.yaml`

Every experiment has a single entry with this schema:

```yaml
- id: unique_identifier
  title: Human-readable title
  task: dataset_generation | training_and_evaluation | curriculum_learning
  thesis_goal: Scientific objective (one sentence)
  configs:
    dataset: [conf/datasets/*.yaml, ...]
    training: [conf/training/*.yaml, ...]
    evaluation: [conf/evaluation/*.yaml, ...]
  evidence:
    notebooks: [notebooks/.../*.ipynb, ...]
    docs: [reports/.../*.md, ...]
  expected_artifacts:
    datasets: [artifacts/datasets/..., ...]
    models: [artifacts/models/..., ...]
    evaluations: [artifacts/evaluations/..., ...]
    notes: [free-text context]
  status:
    overall: ready | partial | missing | unknown
    notes: current state description
```

### Status values

| Status    | Meaning                                          |
|-----------|--------------------------------------------------|
| `ready`   | Artifacts present and usable                     |
| `partial` | Some evidence exists but incomplete              |
| `missing` | Expected artifacts not present                   |
| `unknown` | Not yet audited                                  |

---

## Audit tool

```bash
python scripts/audit_experiments.py
```

Compares `expected_artifacts` entries against the filesystem and reports counts, missing paths, and overall status per experiment.

---

## Artifact roots

Defined at the top of `registry.yaml`:

| Category     | Directory               |
|--------------|-------------------------|
| `datasets`   | `artifacts/datasets`    |
| `models`     | `artifacts/models`      |
| `evaluations`| `artifacts/evaluations` |
| `logs`       | `artifacts/logs`        |

---

## Naming conventions

- **Experiment IDs**: kebab-case, prefixed by task (`srv-baseline-reproduction`, `unitary-curriculum-learning`)
- **Config references**: relative paths from project root (`conf/training/paper_stage_1.yaml`)
- **Artifact paths**: `artifacts/{category}/{experiment-id}/{stage-or-variant}/`

---

## Workflow

1. Define experiment in `registry.yaml` with thesis goal, config refs, and expected artifacts
2. Run scripts (`generate_dataset.py`, `train_model.py`, `evaluate_model.py`) with referenced Hydra configs
3. Analyze results in notebooks listed under `evidence.notebooks`
4. Promote thesis-ready outputs to `reports/`
5. Audit status with `audit_experiments.py`
6. Update `status` in registry
