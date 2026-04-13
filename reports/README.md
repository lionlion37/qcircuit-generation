# Reports

This directory is for thesis-facing outputs derived from experiment artifacts.

## Split From `artifacts/`
- `artifacts/` stores raw or machine-generated outputs such as datasets, checkpoints, pickles, JSON summaries, and intermediate figures.
- `reports/` stores curated outputs that are ready to cite, compare, or include in the thesis.

## Layout
- `thesis/notes/`: short markdown summaries per experiment, including what was run, what result is trusted, and which artifact folder backs it.
- `thesis/tables/`: cleaned CSV or markdown tables for thesis use.
- `thesis/figures/`: final exported figures intended for the thesis document.
- `thesis/appendix/`: supporting plots, extra tables, and supplementary material.

## Recommended Workflow
1. Run experiments and evaluations into `artifacts/...`.
2. Review the outputs and decide which results are thesis-worthy.
3. Promote selected figures/tables into `reports/thesis/...`.
4. Add a short note in `reports/thesis/notes/` linking the claim to the supporting artifact directory.
