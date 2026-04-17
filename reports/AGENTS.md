# Reports

Thesis-facing curated outputs. This directory contains **publication-ready** figures, tables, and summaries derived from experiment artifacts.

---

## Distinction from `artifacts/`

| `artifacts/`                         | `reports/`                            |
|--------------------------------------|---------------------------------------|
| Machine-generated, raw outputs       | Curated, human-reviewed outputs       |
| Intermediate results, pickles, logs  | Final figures and summary tables      |
| One dir per evaluation run           | One dir per thesis section/experiment |

**Workflow**: run experiments -> review in notebooks -> promote selected outputs here.

---

## Structure

```
reports/
  README.md                              # top-level overview
  summary.csv                            # cross-experiment model comparison
  {experiment-id}/
    README.md                            # experiment findings summary
    *.csv                                # summary tables
    *.png                                # figures (150 dpi, tight bbox)
```

### Naming conventions

- **Subdirectory names** match experiment IDs from `experiments/registry.yaml` (kebab-case, task-prefixed)
- **Figure files**: descriptive snake_case (`srv_generation_time.png`, `accuracy_by_entanglement_bucket.png`)
- **Table files**: descriptive snake_case CSV (`srv_summary.csv`, `all_runs.csv`)
- Each subdirectory has its own `README.md` summarising the key findings

### Current reports

| Directory                         | Content                                              |
|-----------------------------------|------------------------------------------------------|
| `srv-dataset-backend-comparison/` | Quditkit vs Qiskit generation speed and unique ratios |
| `srv-text-encoder-ablation/`      | CLOOB vs CLIP text encoder accuracy comparison       |
