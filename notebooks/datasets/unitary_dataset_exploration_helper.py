from __future__ import annotations

import ast
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "unitary_dataset_exploration_helper requires the qcircuit-generation "
        "environment with PyTorch installed."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_CFG = PROJECT_ROOT / "conf" / "training" / "paper_unitary.yaml"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from my_genQC.dataset.circuits_dataset import CircuitsConfigDataset, MixedCircuitsConfigDataset


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def default_dataset_path(training_cfg_path: str | Path = DEFAULT_TRAINING_CFG) -> Path:
    cfg = load_yaml(training_cfg_path)
    raw_path = cfg.get("general", {}).get("dataset")
    if not raw_path:
        raise KeyError(f"No general.dataset entry found in {training_cfg_path}")
    return resolve_path(raw_path)


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_dataset_dir(dataset_path: str | Path) -> Path:
    dataset_dir = resolve_path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_dir}")
    if not (dataset_dir / "config.yaml").exists():
        raise FileNotFoundError(f"Missing config.yaml under dataset path: {dataset_dir}")
    if not (dataset_dir / "dataset").exists():
        raise FileNotFoundError(f"Missing dataset/ directory under dataset path: {dataset_dir}")
    return dataset_dir


def load_saved_dataset(
    dataset_dir: str | Path,
    *,
    device: str = "cpu",
    make_contiguous: bool = False,
):
    dataset_dir = resolve_dataset_dir(dataset_dir)
    cfg = load_yaml(dataset_dir / "config.yaml")
    target = cfg.get("target", "")
    dataset_cls = (
        MixedCircuitsConfigDataset
        if str(target).endswith("MixedCircuitsConfigDataset")
        else CircuitsConfigDataset
    )

    dataset = dataset_cls.from_config_file(
        config_path=str(dataset_dir / "config.yaml"),
        device=torch.device(device),
        save_path=str(dataset_dir / "dataset" / "ds"),
        make_contiguous=make_contiguous,
    )
    if "U" not in dataset.store_dict:
        raise ValueError(f"Dataset at {dataset_dir} does not contain unitary conditioning (`U`).")

    return dataset, cfg


def maybe_dataframe(rows: Any):
    if pd is None:
        return rows
    if isinstance(rows, dict):
        return pd.DataFrame([rows])
    return pd.DataFrame(rows)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install it in the current environment "
            "or skip the plotting cells."
        ) from exc
    return plt


def get_pad_constant(dataset: CircuitsConfigDataset | MixedCircuitsConfigDataset) -> int:
    return int(getattr(dataset.params_config, "pad_constant", len(dataset.gate_pool) + 1))


def gate_activity_mask(x: torch.Tensor, pad_constant: int) -> torch.Tensor:
    return (x != 0) & (x != pad_constant)


def count_gate_timesteps(x: torch.Tensor, pad_constant: int) -> torch.Tensor:
    return gate_activity_mask(x, pad_constant).any(dim=1).sum(dim=1).to(torch.int32)


def count_active_qubits(x: torch.Tensor, pad_constant: int) -> torch.Tensor:
    return gate_activity_mask(x, pad_constant).any(dim=2).sum(dim=1).to(torch.int32)


def _histogram_rows(values: torch.Tensor | np.ndarray, key: str) -> list[dict[str, int]]:
    if isinstance(values, np.ndarray):
        unique, counts = np.unique(values, return_counts=True)
        return [{key: int(k), "count": int(v)} for k, v in zip(unique.tolist(), counts.tolist())]

    unique, counts = torch.unique(values.cpu(), return_counts=True)
    return [{key: int(k), "count": int(v)} for k, v in zip(unique.tolist(), counts.tolist())]


def _safe_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    return values.detach().cpu().numpy()


def _summarize_numeric(values: torch.Tensor | np.ndarray, name: str) -> dict[str, float]:
    arr = _safe_numpy(values).astype(np.float64)
    return {
        f"{name}_min": float(arr.min()),
        f"{name}_median": float(np.median(arr)),
        f"{name}_mean": float(arr.mean()),
        f"{name}_max": float(arr.max()),
    }


def parse_compile_prompt(label: Any) -> tuple[str, ...] | None:
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    if isinstance(label, np.ndarray):
        label = label.tolist()
    if not isinstance(label, str):
        return None

    payload = label.split(":", 1)[-1].strip()
    try:
        parsed = ast.literal_eval(payload)
    except Exception:
        return None

    if not isinstance(parsed, (list, tuple)):
        return None
    return tuple(str(item) for item in parsed)


def prompt_to_text(prompt: tuple[str, ...] | None, fallback: Any = None) -> str:
    if prompt is None:
        return str(fallback)
    return "[" + ", ".join(prompt) + "]"


def _normalize_labels(labels: Any) -> list[Any]:
    if isinstance(labels, np.ndarray):
        return labels.tolist()
    if isinstance(labels, list):
        return labels
    if hasattr(labels, "tolist"):
        return labels.tolist()
    return list(labels)


def build_prompt_statistics(
    labels: Any,
    lengths: torch.Tensor,
    gate_pool: list[str],
) -> dict[str, Any]:
    labels_list = _normalize_labels(labels)
    prompt_counter: Counter[str] = Counter()
    prompt_size_counter: Counter[int] = Counter()
    prompt_gate_counter: Counter[str] = Counter()
    prompt_length_sums: defaultdict[str, float] = defaultdict(float)
    prompt_length_values: defaultdict[str, list[int]] = defaultdict(list)
    prompt_size_by_text: dict[str, int | None] = {}
    prompt_size_length_values: defaultdict[int, list[int]] = defaultdict(list)
    invalid_prompt_count = 0

    for label, length in zip(labels_list, lengths.cpu().tolist()):
        prompt = parse_compile_prompt(label)
        if prompt is None:
            invalid_prompt_count += 1
            prompt_text = str(label)
            prompt_size = None
        else:
            prompt_text = prompt_to_text(prompt)
            prompt_size = len(prompt)
            prompt_size_counter[prompt_size] += 1
            prompt_size_length_values[prompt_size].append(int(length))
            for gate in prompt:
                prompt_gate_counter[gate] += 1

        prompt_counter[prompt_text] += 1
        prompt_length_sums[prompt_text] += float(length)
        prompt_length_values[prompt_text].append(int(length))
        prompt_size_by_text[prompt_text] = prompt_size

    prompt_rows = []
    for prompt_text, count in prompt_counter.most_common():
        lengths_for_prompt = np.array(prompt_length_values[prompt_text], dtype=np.float64)
        prompt_rows.append(
            {
                "prompt": prompt_text,
                "count": int(count),
                "fraction": float(count / max(len(labels_list), 1)),
                "prompt_size": prompt_size_by_text[prompt_text],
                "mean_circuit_length": float(lengths_for_prompt.mean()),
                "median_circuit_length": float(np.median(lengths_for_prompt)),
            }
        )

    prompt_size_rows = []
    for size, count in sorted(prompt_size_counter.items()):
        mean_length = float(np.mean(prompt_size_length_values[size]))
        prompt_size_rows.append(
            {
                "prompt_size": int(size),
                "count": int(count),
                "fraction": float(count / max(len(labels_list), 1)),
                "mean_prompt_mean_circuit_length": float(mean_length),
            }
        )

    prompt_gate_rows = []
    for gate in gate_pool:
        count = int(prompt_gate_counter.get(gate, 0))
        prompt_gate_rows.append(
            {
                "gate": gate,
                "count": count,
                "fraction_of_samples": float(count / max(len(labels_list), 1)),
            }
        )

    return {
        "invalid_prompt_count": int(invalid_prompt_count),
        "prompt_unique_count": int(len(prompt_counter)),
        "prompt_rows": prompt_rows,
        "prompt_size_rows": prompt_size_rows,
        "prompt_gate_rows": prompt_gate_rows,
    }


def build_gate_frequency_rows(
    x: torch.Tensor,
    gate_pool: list[str],
    pad_constant: int,
) -> list[dict[str, Any]]:
    gate_ids = x.abs().amax(dim=1)
    gate_ids = gate_ids[(gate_ids != 0) & (gate_ids != pad_constant)].cpu()
    unique, counts = torch.unique(gate_ids, return_counts=True)
    counter = {int(key): int(value) for key, value in zip(unique.tolist(), counts.tolist())}
    total = max(int(sum(counter.values())), 1)

    rows = []
    for token, gate in enumerate(gate_pool, start=1):
        count = counter.get(token, 0)
        rows.append(
            {
                "token": int(token),
                "gate": gate,
                "count": int(count),
                "fraction": float(count / total),
            }
        )
    return rows


def to_complex(unitary_tensor: torch.Tensor) -> torch.Tensor:
    if unitary_tensor.dim() != 4 or unitary_tensor.shape[1] != 2:
        raise ValueError(f"Unexpected unitary tensor shape {tuple(unitary_tensor.shape)}")
    return torch.complex(unitary_tensor[:, 0], unitary_tensor[:, 1])


def _sample_indices(total: int, max_samples: int | None, seed: int) -> torch.Tensor:
    if max_samples is None or total <= max_samples:
        return torch.arange(total, dtype=torch.long)

    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(total, size=max_samples, replace=False))
    return torch.from_numpy(sample.astype(np.int64))


def _pairwise_hs_distance_sample(U: torch.Tensor, num_pairs: int, seed: int) -> np.ndarray:
    n = int(U.shape[0])
    if n < 2 or num_pairs <= 0:
        return np.empty((0,), dtype=np.float32)

    rng = np.random.default_rng(seed)
    ia = rng.integers(0, n, size=num_pairs, endpoint=False)
    ib = rng.integers(0, n, size=num_pairs, endpoint=False)
    equal_mask = ia == ib
    ib[equal_mask] = (ib[equal_mask] + 1) % n

    Ua = U[torch.from_numpy(ia.astype(np.int64))]
    Ub = U[torch.from_numpy(ib.astype(np.int64))]
    dim = U.shape[-1]
    overlaps = torch.einsum("bij,bij->b", Ua.conj(), Ub).abs().real
    distances = torch.sqrt(torch.clamp(2.0 * dim - 2.0 * overlaps, min=0.0))
    distances = distances / math.sqrt(2.0 * dim)
    return distances.cpu().numpy()


def _pca_embedding(features: torch.Tensor) -> np.ndarray:
    if features.shape[0] < 2 or features.shape[1] < 2:
        return np.zeros((features.shape[0], 2), dtype=np.float32)

    centered = features - features.mean(dim=0, keepdim=True)
    q = min(2, centered.shape[0], centered.shape[1])
    try:
        _, _, v = torch.pca_lowrank(centered, q=q)
        embedding = centered @ v[:, :q]
    except RuntimeError:
        embedding = centered[:, :q]

    if embedding.shape[1] == 1:
        zeros = torch.zeros((embedding.shape[0], 1), dtype=embedding.dtype, device=embedding.device)
        embedding = torch.cat([embedding, zeros], dim=1)
    return embedding[:, :2].cpu().numpy()


def analyze_unitary_sample(
    dataset,
    *,
    max_unitary_samples: int = 2048,
    max_eigen_unitaries: int = 512,
    pairwise_pairs: int = 4096,
    seed: int = 0,
) -> dict[str, Any]:
    sample_idx = _sample_indices(len(dataset.x), max_unitary_samples, seed=seed)
    x_sample = dataset.x[sample_idx]
    U_sample = to_complex(dataset.U[sample_idx]).to(torch.complex64)
    pad_constant = get_pad_constant(dataset)
    lengths = count_gate_timesteps(x_sample, pad_constant).cpu().numpy()

    all_labels = _normalize_labels(dataset.y)
    labels_sample = [all_labels[int(idx)] for idx in sample_idx.cpu().tolist()]
    prompt_sizes = np.array(
        [
            len(prompt) if (prompt := parse_compile_prompt(label)) is not None else -1
            for label in labels_sample
        ],
        dtype=np.int32,
    )

    dim = int(U_sample.shape[-1])
    trace = torch.diagonal(U_sample, dim1=-2, dim2=-1).sum(dim=-1)
    diag_energy_frac = (
        torch.diagonal(U_sample, dim1=-2, dim2=-1).abs().pow(2).sum(dim=-1).real / dim
    )
    trace_abs_norm = trace.abs().real / dim
    det_phase = torch.angle(torch.linalg.det(U_sample)).real
    fro_norm_norm = torch.linalg.matrix_norm(U_sample, ord="fro").real / math.sqrt(dim)
    ident = torch.eye(dim, dtype=U_sample.dtype, device=U_sample.device).unsqueeze(0)
    unitarity_error = torch.linalg.matrix_norm(
        U_sample @ U_sample.conj().transpose(-1, -2) - ident, ord="fro"
    ).real
    hs_identity_distance = torch.sqrt(
        torch.clamp(2.0 * dim - 2.0 * trace.abs().real, min=0.0)
    ) / math.sqrt(2.0 * dim)

    eigen_count = min(int(U_sample.shape[0]), int(max_eigen_unitaries))
    eigenphases = np.empty((0,), dtype=np.float32)
    if eigen_count > 0:
        eigvals = torch.linalg.eigvals(U_sample[:eigen_count])
        eigenphases = torch.angle(eigvals).reshape(-1).cpu().numpy()

    features = torch.cat(
        [U_sample.real.reshape(U_sample.shape[0], -1), U_sample.imag.reshape(U_sample.shape[0], -1)],
        dim=1,
    ).float()
    embedding = _pca_embedding(features)
    pairwise_distances = _pairwise_hs_distance_sample(U_sample, num_pairs=pairwise_pairs, seed=seed + 17)

    metric_arrays = {
        "trace_abs_norm": trace_abs_norm.cpu().numpy(),
        "diag_energy_frac": diag_energy_frac.cpu().numpy(),
        "det_phase": det_phase.cpu().numpy(),
        "fro_norm_norm": fro_norm_norm.cpu().numpy(),
        "unitarity_error": unitarity_error.cpu().numpy(),
        "hs_identity_distance": hs_identity_distance.cpu().numpy(),
    }

    summary = {
        "sampled_unitaries": int(U_sample.shape[0]),
        "unitary_dim": dim,
        **_summarize_numeric(metric_arrays["trace_abs_norm"], "trace_abs_norm"),
        **_summarize_numeric(metric_arrays["diag_energy_frac"], "diag_energy_frac"),
        **_summarize_numeric(metric_arrays["hs_identity_distance"], "hs_identity_distance"),
        **_summarize_numeric(metric_arrays["unitarity_error"], "unitarity_error"),
    }

    by_prompt_size = []
    valid_sizes = sorted(size for size in np.unique(prompt_sizes) if size >= 0)
    for size in valid_sizes:
        mask = prompt_sizes == size
        by_prompt_size.append(
            {
                "prompt_size": int(size),
                "count": int(mask.sum()),
                "mean_length": float(lengths[mask].mean()),
                "mean_trace_abs_norm": float(metric_arrays["trace_abs_norm"][mask].mean()),
                "mean_diag_energy_frac": float(metric_arrays["diag_energy_frac"][mask].mean()),
                "mean_hs_identity_distance": float(metric_arrays["hs_identity_distance"][mask].mean()),
            }
        )

    return {
        "sample_indices": sample_idx.cpu().numpy(),
        "sample_lengths": lengths,
        "sample_prompt_sizes": prompt_sizes,
        "labels_sample": labels_sample,
        "metric_arrays": metric_arrays,
        "metric_summary": summary,
        "eigenphases": eigenphases,
        "pairwise_distances": pairwise_distances,
        "embedding": embedding,
        "by_prompt_size_rows": by_prompt_size,
    }


def analyze_unitary_dataset(
    dataset_path: str | Path,
    *,
    device: str = "cpu",
    make_contiguous: bool = False,
    max_unitary_samples: int = 2048,
    max_eigen_unitaries: int = 512,
    pairwise_pairs: int = 4096,
    seed: int = 0,
) -> dict[str, Any]:
    dataset_dir = resolve_dataset_dir(dataset_path)
    dataset, cfg = load_saved_dataset(
        dataset_dir,
        device=device,
        make_contiguous=make_contiguous,
    )

    x = dataset.x
    y = dataset.y
    pad_constant = get_pad_constant(dataset)
    lengths = count_gate_timesteps(x, pad_constant).cpu()
    active_qubits = count_active_qubits(x, pad_constant).cpu()

    prompt_stats = build_prompt_statistics(y, lengths, list(dataset.gate_pool))
    gate_frequency_rows = build_gate_frequency_rows(x, list(dataset.gate_pool), pad_constant)
    unitary_sample = analyze_unitary_sample(
        dataset,
        max_unitary_samples=max_unitary_samples,
        max_eigen_unitaries=max_eigen_unitaries,
        pairwise_pairs=pairwise_pairs,
        seed=seed,
    )

    overview = {
        "dataset_dir": str(dataset_dir),
        "config_target": cfg.get("target"),
        "num_samples": int(x.shape[0]),
        "x_shape": tuple(int(dim) for dim in x.shape),
        "u_shape": tuple(int(dim) for dim in dataset.U.shape),
        "num_qubits_config": int(dataset.params_config.num_of_qubits),
        "min_gates_config": int(dataset.params_config.min_gates),
        "max_gates_config": int(dataset.params_config.max_gates),
        "pad_constant": int(pad_constant),
        "gate_pool": list(dataset.gate_pool),
        "prompt_unique_count": prompt_stats["prompt_unique_count"],
        "invalid_prompt_count": prompt_stats["invalid_prompt_count"],
        **_summarize_numeric(lengths, "circuit_length"),
        **_summarize_numeric(active_qubits, "active_qubits"),
        "sampled_unitaries_for_metrics": int(unitary_sample["metric_summary"]["sampled_unitaries"]),
    }

    return {
        "dataset": dataset,
        "config": cfg,
        "overview": overview,
        "length_rows": _histogram_rows(lengths, "circuit_length"),
        "active_qubit_rows": _histogram_rows(active_qubits, "active_qubits"),
        "prompt_rows": prompt_stats["prompt_rows"],
        "prompt_size_rows": prompt_stats["prompt_size_rows"],
        "prompt_gate_rows": prompt_stats["prompt_gate_rows"],
        "gate_frequency_rows": gate_frequency_rows,
        "unitary_sample": unitary_sample,
    }


def plot_circuit_distributions(analysis: dict[str, Any]):
    plt = _require_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    length_rows = analysis["length_rows"]
    axes[0, 0].bar(
        [row["circuit_length"] for row in length_rows],
        [row["count"] for row in length_rows],
        color="#0f766e",
    )
    axes[0, 0].set_title("Circuit Length Distribution")
    axes[0, 0].set_xlabel("Gate timesteps")
    axes[0, 0].set_ylabel("Samples")

    active_rows = analysis["active_qubit_rows"]
    axes[0, 1].bar(
        [row["active_qubits"] for row in active_rows],
        [row["count"] for row in active_rows],
        color="#2563eb",
    )
    axes[0, 1].set_title("Active-Qubit Distribution")
    axes[0, 1].set_xlabel("Qubits touched at least once")
    axes[0, 1].set_ylabel("Samples")

    gate_rows = analysis["gate_frequency_rows"]
    axes[1, 0].bar(
        [row["gate"] for row in gate_rows],
        [row["count"] for row in gate_rows],
        color="#b45309",
    )
    axes[1, 0].set_title("Gate Occurrence Counts")
    axes[1, 0].set_xlabel("Gate")
    axes[1, 0].set_ylabel("Occurrences")
    axes[1, 0].tick_params(axis="x", rotation=30)

    prompt_size_rows = analysis["prompt_size_rows"]
    axes[1, 1].bar(
        [row["prompt_size"] for row in prompt_size_rows],
        [row["count"] for row in prompt_size_rows],
        color="#7c3aed",
    )
    axes[1, 1].set_title("Prompt-Size Distribution")
    axes[1, 1].set_xlabel("Allowed gates in prompt")
    axes[1, 1].set_ylabel("Samples")

    fig.tight_layout()
    return fig, axes


def plot_prompt_distributions(analysis: dict[str, Any], top_k: int = 15):
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    prompt_rows = analysis["prompt_rows"][:top_k]
    prompt_labels = [row["prompt"] for row in prompt_rows][::-1]
    prompt_counts = [row["count"] for row in prompt_rows][::-1]
    axes[0].barh(prompt_labels, prompt_counts, color="#1d4ed8")
    axes[0].set_title(f"Top {top_k} Prompts")
    axes[0].set_xlabel("Samples")
    axes[0].set_ylabel("Prompt")

    gate_rows = analysis["prompt_gate_rows"]
    axes[1].bar(
        [row["gate"] for row in gate_rows],
        [row["fraction_of_samples"] for row in gate_rows],
        color="#ea580c",
    )
    axes[1].set_title("Gate Presence in Prompts")
    axes[1].set_xlabel("Gate")
    axes[1].set_ylabel("Fraction of samples")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=30)

    prompt_size_rows = analysis["prompt_size_rows"]
    axes[2].plot(
        [row["prompt_size"] for row in prompt_size_rows],
        [row["mean_prompt_mean_circuit_length"] for row in prompt_size_rows],
        marker="o",
        color="#0f766e",
    )
    axes[2].set_title("Mean Circuit Length by Prompt Size")
    axes[2].set_xlabel("Allowed gates in prompt")
    axes[2].set_ylabel("Mean circuit length")

    fig.tight_layout()
    return fig, axes


def plot_unitary_metric_distributions(analysis: dict[str, Any]):
    plt = _require_matplotlib()
    metrics = analysis["unitary_sample"]["metric_arrays"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].hist(metrics["trace_abs_norm"], bins=40, color="#1d4ed8", alpha=0.85)
    axes[0, 0].set_title(r"$|Tr(U)| / N$")
    axes[0, 0].set_xlabel("Normalized trace magnitude")
    axes[0, 0].set_ylabel("Sampled unitaries")

    axes[0, 1].hist(metrics["diag_energy_frac"], bins=40, color="#0f766e", alpha=0.85)
    axes[0, 1].set_title("Diagonal Energy Fraction")
    axes[0, 1].set_xlabel(r"$\sum_i |U_{ii}|^2 / N$")
    axes[0, 1].set_ylabel("Sampled unitaries")

    axes[1, 0].hist(metrics["hs_identity_distance"], bins=40, color="#b45309", alpha=0.85)
    axes[1, 0].set_title("Distance to Identity")
    axes[1, 0].set_xlabel("Phase-invariant Hilbert-Schmidt distance")
    axes[1, 0].set_ylabel("Sampled unitaries")

    log_unitarity_error = np.log10(np.maximum(metrics["unitarity_error"], 1e-12))
    axes[1, 1].hist(log_unitarity_error, bins=40, color="#7c3aed", alpha=0.85)
    axes[1, 1].set_title("Unitarity Error")
    axes[1, 1].set_xlabel(r"$\log_{10} ||UU^\dagger - I||_F$")
    axes[1, 1].set_ylabel("Sampled unitaries")

    fig.tight_layout()
    return fig, axes


def plot_unitary_geometry(analysis: dict[str, Any]):
    plt = _require_matplotlib()
    sample = analysis["unitary_sample"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(sample["eigenphases"], bins=60, color="#0f766e", alpha=0.85)
    axes[0].set_title("Eigenphase Distribution")
    axes[0].set_xlabel("Phase [rad]")
    axes[0].set_ylabel("Eigenvalues")

    axes[1].hist(sample["pairwise_distances"], bins=40, color="#b45309", alpha=0.85)
    axes[1].set_title("Pairwise Unitary Distances")
    axes[1].set_xlabel("Phase-invariant Hilbert-Schmidt distance")
    axes[1].set_ylabel("Random sample pairs")

    scatter = axes[2].scatter(
        sample["embedding"][:, 0],
        sample["embedding"][:, 1],
        c=sample["sample_lengths"],
        cmap="viridis",
        s=16,
        alpha=0.75,
    )
    axes[2].set_title("PCA of Flattened Unitaries")
    axes[2].set_xlabel("PC 1")
    axes[2].set_ylabel("PC 2")
    fig.colorbar(scatter, ax=axes[2], label="Circuit length")

    fig.tight_layout()
    return fig, axes
