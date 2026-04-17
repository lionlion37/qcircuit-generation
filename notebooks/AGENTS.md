# Notebooks

Analysis notebooks organised by topic, each with companion `_helper.py` modules that hold reusable logic.

---

## Directory layout

```
notebooks/
  shared/                          # cross-cutting utilities
    bootstrap.py                   # path setup (call first in every notebook)
    evaluation_artifacts.py        # save_figure, save_json, save_dataframe, ...
  datasets/                        # dataset exploration and auditing
    training_dataset_audit_helper.py
    unitary_dataset_exploration_helper.py
  evaluation/
    srv/                           # SRV task model evaluation
    unitary/                       # unitary compilation model evaluation
  training/
    bucket_padding/                # bucket padding strategy analysis
      bucket_deep_dive_helper.py
      bucket_training_alignment_helper.py
  noise_awareness/                 # noise-aware dataset generation
    unitary_noise_common.py
    unitary_noise_analysis_helper.py
    unitary_noise_dataset_exploration_helper.py
    build_unitary_noise_candidate_table.py   # CLI script
    build_unitary_noise_dataset.py           # CLI script
```

---

## Bootstrap pattern

Every notebook that imports from the project must call `setup_notebook_paths()` first.
This finds the project root (walks up until it finds both `src/` and `conf/`), changes CWD there, and adds `project_root`, `project_root/src`, and `project_root/notebooks` to `sys.path`.

```python
from shared.bootstrap import setup_notebook_paths
PROJECT_ROOT = setup_notebook_paths()
```

After this, imports from `my_genQC`, `quantum_diffusion`, and `shared.*` / `notebooks.*` all work.

---

## Shared utilities (`shared/evaluation_artifacts.py`)

| Function                              | What it does                                    |
|---------------------------------------|-------------------------------------------------|
| `make_artifact_dir(root, sub, run)`   | Creates `artifacts/evaluations/{sub}/{run}/`     |
| `save_figure(fig, path, *, dpi=200)`  | Saves matplotlib figure with `bbox_inches="tight"` |
| `save_json(data, path)`               | JSON with numpy/Path serialization               |
| `save_pickle(data, path)`             | Pickle with `HIGHEST_PROTOCOL`                   |
| `save_dataframe(df, path)`            | CSV via pandas                                   |
| `save_text(text, path)`               | Plain UTF-8 text                                 |

All `save_*` functions create parent directories automatically.

---

## Notebook structure conventions

1. **First cell**: markdown title and description
2. **Second cell**: imports + `setup_notebook_paths()` + `make_artifact_dir()`
3. **Config cell**: editable parameters (`SHOTS`, `RUN_NAME`, model paths, etc.)
4. **Analysis cells**: call helper functions, display results
5. **Figure cells**: plot + `save_figure(fig, ARTIFACT_DIR / "name.png")` + `plt.show()`

Always call `save_figure` **before** `plt.show()`.

---

## Helper module conventions

Each notebook directory may have `*_helper.py` modules that:
- Contain all reusable analysis logic (the notebook calls high-level functions)
- Return structured dicts or lists-of-dicts (converted to DataFrames for display)
- Plotting functions return `(fig, axes)` tuples
- Heavy computation is parallelised via `my_genQC.utils.async_fn`

---

## Plot Style Guide

All figures appear in a master's thesis.
They must be consistent, publication-quality, and accessible (legible in greyscale / for colour-blind readers).

### rcParams block

Paste this at the top of every notebook that produces figures, right after imports:

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams.update({
    # figure
    "figure.dpi":          150,
    "figure.facecolor":    "white",
    # axes
    "axes.facecolor":      "white",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.titlesize":      13,
    "axes.titleweight":    "bold",
    "axes.titlepad":       10,
    "axes.labelsize":      11,
    "axes.labelpad":       6,
    # grid
    "axes.grid":           True,
    "grid.linestyle":      "--",
    "grid.linewidth":      0.6,
    "grid.alpha":          0.5,
    "grid.color":          "#AAAAAA",
    "axes.axisbelow":      True,
    # ticks
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    # legend
    "legend.fontsize":     10,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "#CCCCCC",
    # lines
    "lines.linewidth":     2.0,
    "lines.markersize":    6,
    # font
    "font.family":         "sans-serif",
})
```

### Colour palette

#### Qualitative (categorical series)

| Role       | Hex       | Semantic assignment              |
|------------|-----------|----------------------------------|
| Primary    | `#2176AE` | First / main series; Qiskit      |
| Secondary  | `#E05C2A` | Second series; QuditKit          |
| Tertiary   | `#3BAA6E` | Third series                     |
| Quaternary | `#8B5CF6` | Fourth series                    |

Assign colours by **semantic role**, not by iteration order, so the same entity always looks the same across figures.

#### Sequential (ordinal data, e.g. qubit count)

```python
cmap  = plt.get_cmap("Oranges", n)
norm  = mpl.colors.BoundaryNorm(np.arange(q_min - 0.5, q_max + 1.5), cmap.N)
```

#### Continuous 2-D (heatmaps)

Use `cmap="viridis"` (perceptually uniform, greyscale-safe).

#### Optimized vs. unoptimized variants

Pair colour with **hatch** so the distinction survives greyscale print:

```python
HATCH = {True: "",     False: "///"}   # optimized = solid, unoptimized = hatched
```

### Figure sizes

| Use case              | `figsize`   | Layout                    |
|-----------------------|-------------|---------------------------|
| Single panel          | `(6, 3.8)`  | `fig.tight_layout()`      |
| Wide dual-panel (1x2) | `(11, 3.8)` | `fig.tight_layout()`      |
| 2x2 grid              | `(10, 7)`   | `constrained_layout=True` |
| 2x3 grid              | `(14, 8)`   | `constrained_layout=True` |

Always call `fig.tight_layout()` or pass `constrained_layout=True` to `plt.subplots()`.

### Grid and spines

```python
# Spines -- always remove top and right
ax.spines[["top", "right"]].set_visible(False)

# Line / scatter plots -- grid on both axes
ax.grid(True)

# Bar charts / histograms -- horizontal grid only
ax.yaxis.grid(True)
ax.xaxis.grid(False)

# Grid must sit behind data
ax.set_axisbelow(True)
```

### Chart-type rules

**Line plots** (continuous x with >= 3 points):
- `marker="o"`, `markersize=5-7`, `linewidth=2`
- Vary `linestyle` (`"-"`, `"--"`, `":"`) when series share a colour family
- Rates and accuracies: `ax.set_ylim(0, 1.05)`

**Bar charts** (categorical / discrete x, or < 3 x-values):
- Grouped bars for multi-series comparisons
- `alpha=0.90`, `edgecolor="white"`, `linewidth=0.6`
- Horizontal grid only
- Reference baseline: `ax.axhline(ref, color="#d85c41", linestyle="--", linewidth=1.4, label="paper")`

**Histograms**:
- Overlapping series: `alpha=0.65`, shared `bins` edges computed once
- Normalise to percentage: `weights=np.full(len(x), 100.0 / len(x))`
- Horizontal grid only

**Heatmaps** (`imshow`):
- `cmap="viridis"`, `aspect="auto"`
- Always attach a colorbar: `plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)`

**Scatter plots**:
- `alpha=0.55`, `s=18` for dense data
- Third variable via `cmap="viridis"` + colorbar

**Box plots**:
- `showfliers=False`

### Typography

| Element        | Size | Weight |
|----------------|------|--------|
| Figure title   | 13   | bold   |
| Axis label     | 11   | normal |
| Tick label     | 10   | normal |
| Legend text     | 10   | normal |
| Colorbar label | 10   | normal |

### Saving

```python
save_figure(fig, ARTIFACT_DIR / "filename.png")
plt.show()
```

Always call `save_figure` before `plt.show()`.

### Accessibility checklist

- [ ] Colour is **never** the only encoding -- pair with marker shape, line style, or hatch
- [ ] Palette distinguishable under deuteranopia and protanopia
- [ ] Figure readable when printed in greyscale
- [ ] All axes have labels with units where applicable
- [ ] Legend present whenever more than one series is shown
