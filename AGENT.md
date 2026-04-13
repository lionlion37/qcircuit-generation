# Project Agent Guide

## Plot Style Guide

All figures in this repository appear in a master's thesis.
They must be consistent, publication-quality, and accessible (legible in greyscale / for colour-blind readers).

---

### Quick-start: rcParams block

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

---

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

This is established in the evaluation notebooks — do not change it.

#### Continuous 2-D (heatmaps)

Use `cmap="viridis"` (perceptually uniform, greyscale-safe).

#### Optimized vs. unoptimized variants

Pair colour with **hatch** so the distinction survives greyscale print:

```python
HATCH = {True: "",     False: "///"}   # optimized = solid, unoptimized = hatched
```

---

### Figure sizes

| Use case              | `figsize`   | Layout                    |
|-----------------------|-------------|---------------------------|
| Single panel          | `(6, 3.8)`  | `fig.tight_layout()`      |
| Wide dual-panel (1×2) | `(11, 3.8)` | `fig.tight_layout()`      |
| 2×2 grid              | `(10, 7)`   | `constrained_layout=True` |
| 2×3 grid              | `(14, 8)`   | `constrained_layout=True` |

Always call `fig.tight_layout()` or pass `constrained_layout=True` to `plt.subplots()`.
Never rely on default margins.

---

### Grid & spines

```python
# Spines — always remove top and right
ax.spines[["top", "right"]].set_visible(False)

# Line / scatter plots — grid on both axes
ax.grid(True)

# Bar charts / histograms — horizontal grid only
ax.yaxis.grid(True)
ax.xaxis.grid(False)

# Grid must sit behind data
ax.set_axisbelow(True)
```

---

### Chart-type rules

#### Line plots  *(use only when x is continuous with ≥ 3 points)*

- `marker="o"`, `markersize=5–7`, `linewidth=2`
- Add a secondary encoding when series share a colour family: vary `linestyle` (`"-"`, `"--"`, `":"`)
- Rates and accuracies: `ax.set_ylim(0, 1.05)`

#### Bar charts  *(use for categorical / discrete x-axes, or when there are < 3 x-values)*

- Grouped bars for multi-series comparisons
- `alpha=0.90`, `edgecolor="white"`, `linewidth=0.6`
- Horizontal grid only (see above)
- Reference / paper baseline:
  ```python
  ax.axhline(ref, color="#d85c41", linestyle="--", linewidth=1.4, label="paper")
  ```

#### Histograms

- Overlapping series: `alpha=0.65`, shared `bins` edges computed once over all series
- Normalise to percentage when comparing different-sized sets:
  ```python
  weights=np.full(len(x), 100.0 / len(x))
  ```
- Horizontal grid only

#### Heatmaps (`imshow`)

- `cmap="viridis"`, `aspect="auto"`
- Always attach a colorbar:
  ```python
  plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
  ```

#### Scatter plots

- `alpha=0.55`, `s=18` for dense data
- Encode a third variable with `cmap="viridis"` + colorbar

#### Box plots

- `showfliers=False` — outlier dots clutter thesis figures

---

### Typography

| Element      | Size | Weight |
|--------------|------|--------|
| Figure title | 13   | bold   |
| Axis label   | 11   | normal |
| Tick label   | 10   | normal |
| Legend text  | 10   | normal |
| Colorbar label | 10 | normal |

---

### Saving

```python
save_figure(fig, ARTIFACT_DIR / "filename.png")
plt.show()
```

Always call `save_figure` before `plt.show()`.
The helper writes at the DPI set on the figure (150 by default from rcParams).

---

### Accessibility checklist

- [ ] Colour is **never** the only encoding — pair it with marker shape, line style, or hatch
- [ ] Palette distinguishable under deuteranopia and protanopia
- [ ] Figure readable when printed in greyscale
- [ ] All axes have labels with units where applicable
- [ ] Legend present whenever more than one series is shown
