from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from notebooks.noise_awareness.unitary_noise_common import load_saved_dataset, maybe_dataframe, read_json


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib in the qcircuit-generation environment."
        ) from exc
    return plt


def analyze_noise_dataset(dataset_path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    dataset, cfg = load_saved_dataset(dataset_path, device=device, make_contiguous=False)
    if "noise_p" not in dataset.store_dict:
        raise ValueError(f"Dataset at {dataset_path} does not contain noise_p.")

    noise_p = dataset.noise_p.detach().cpu().flatten().numpy()
    rows = []
    for value in sorted(np.unique(noise_p)):
        mask = np.isclose(noise_p, value)
        xs = dataset.x.detach().cpu()[mask]
        gate_lengths = ((xs != 0) & (xs != getattr(dataset.params_config, "pad_constant", len(dataset.gate_pool) + 1))).any(dim=1).sum(dim=1).numpy()
        rows.append(
            {
                "noise_p": float(value),
                "rows": int(mask.sum()),
                "mean_gate_length": float(gate_lengths.mean()),
                "median_gate_length": float(np.median(gate_lengths)),
                "max_gate_length": int(gate_lengths.max()),
            }
        )

    overview = {
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "num_rows": int(dataset.x.shape[0]),
        "num_unique_noise_p": int(len(np.unique(noise_p))),
        "noise_p_values": [float(item) for item in sorted(np.unique(noise_p))],
        "store_dict": dict(dataset.store_dict),
        "config_target": cfg.get("target"),
    }
    return {
        "overview": overview,
        "noise_rows": rows,
    }


def plot_noise_dataset_summary(analysis: dict[str, Any]):
    plt = _require_matplotlib()
    rows = analysis["noise_rows"]
    ps = [row["noise_p"] for row in rows]
    means = [row["mean_gate_length"] for row in rows]
    medians = [row["median_gate_length"] for row in rows]
    counts = [row["rows"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ps, means, "o-", label="mean")
    axes[0].plot(ps, medians, "s--", label="median")
    axes[0].set_title("Circuit Length Across Noise Levels")
    axes[0].set_xlabel("noise_p")
    axes[0].set_ylabel("Gate Length")
    axes[0].legend()

    axes[1].bar([str(p) for p in ps], counts, color="#2a9d8f")
    axes[1].set_title("Rows Per Noise Level")
    axes[1].set_xlabel("noise_p")
    axes[1].set_ylabel("Rows")

    fig.tight_layout()
    return fig
