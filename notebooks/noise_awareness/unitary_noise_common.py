from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "unitary_noise_common requires the qcircuit-generation environment with "
        "PyTorch installed."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "noise_aware_unitary"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "noise_aware"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from my_genQC.dataset.circuits_dataset import CircuitsConfigDataset, MixedCircuitsConfigDataset
from my_genQC.inference.eval_metrics import UnitaryFrobeniusNorm, UnitaryInfidelityNorm
from my_genQC.inference.sampling import decode_tensors_to_backend, generate_compilation_tensors
from my_genQC.pipeline.diffusion_pipeline import DiffusionPipeline
from my_genQC.platform.simulation import CircuitBackendType, Simulator
from my_genQC.platform.tokenizer.circuits_tokenizer import CircuitTokenizer
from my_genQC.utils.config_loader import load_config


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def maybe_dataframe(rows: Any):
    if pd is None:
        return rows
    if isinstance(rows, dict):
        return pd.DataFrame([rows])
    return pd.DataFrame(rows)


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([])
        return path

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_rows_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


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
    cfg = load_config(dataset_dir / "config.yaml")
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
        raise ValueError(f"Dataset at {dataset_dir} does not contain unitary targets.")
    return dataset, cfg


def load_pipeline(model_dir: str | Path, device: torch.device | str):
    model_dir = resolve_path(model_dir)
    config_path = model_dir if model_dir.is_dir() else model_dir.parent
    cfg_file = config_path / "config.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing pipeline config at {cfg_file}")
    return DiffusionPipeline.from_config_file(
        config_path=str(config_path) + "/",
        device=torch.device(device),
    )


def build_tokenizer(dataset) -> CircuitTokenizer:
    vocabulary = {str(gate): idx for idx, gate in enumerate(dataset.gate_pool)}
    return CircuitTokenizer(vocabulary)


def build_quditkit_simulator() -> Simulator:
    return Simulator(CircuitBackendType.QUDITKIT)


def to_complex(unitary_tensor: torch.Tensor) -> torch.Tensor:
    if unitary_tensor.dim() != 4 or unitary_tensor.shape[1] != 2:
        raise ValueError(f"Unexpected unitary tensor shape {unitary_tensor.shape}")
    return torch.complex(unitary_tensor[:, 0], unitary_tensor[:, 1])


def to_complex_single(unitary_tensor: torch.Tensor) -> torch.Tensor:
    if unitary_tensor.dim() != 3 or unitary_tensor.shape[0] != 2:
        raise ValueError(f"Unexpected single unitary tensor shape {unitary_tensor.shape}")
    return torch.complex(unitary_tensor[0], unitary_tensor[1])


def unitary_metrics(predicted: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> dict[str, float]:
    if isinstance(predicted, np.ndarray):
        predicted = torch.from_numpy(predicted)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    predicted = predicted.to(torch.complex128)
    target = target.to(torch.complex128)
    return {
        "clean_frobenius": float(UnitaryFrobeniusNorm.distance(predicted, target).item()),
        "clean_infidelity": float(UnitaryInfidelityNorm.distance(predicted, target).item()),
    }


def qiskit_prompt_text(label: Any) -> str:
    if isinstance(label, bytes):
        return label.decode("utf-8")
    if isinstance(label, np.ndarray):
        return str(label.tolist())
    return str(label)


def sample_target_indices(total: int, limit: int | None, seed: int) -> list[int]:
    all_indices = np.arange(total)
    if limit is None or limit >= total:
        return all_indices.tolist()
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(all_indices, size=int(limit), replace=False))
    return chosen.tolist()


def get_gate_count_from_tensor(x: torch.Tensor, pad_constant: int) -> int:
    mask = ((x != 0) & (x != pad_constant)).any(dim=0)
    return int(mask.sum().item())


def _extract_ops(qc: Any) -> list[tuple[Any, tuple[int, ...], bool]]:
    if hasattr(qc, "_ops"):
        return list(qc._ops)
    if hasattr(qc, "ops"):
        return list(qc.ops)
    return []


def circuit_structure_metrics(qc: Any) -> dict[str, float]:
    ops = _extract_ops(qc)
    one_qubit = 0
    two_qubit = 0
    gate_names: list[str] = []
    for gate, qudits, _dagger in ops:
        gate_names.append(str(getattr(gate, "name", "unknown")).lower())
        arity = len(qudits)
        if arity <= 1:
            one_qubit += 1
        elif arity == 2:
            two_qubit += 1
    depth_proxy = len(ops)
    if hasattr(qc, "_asap_schedule_packed"):
        try:
            depth_proxy = len(qc._asap_schedule_packed())
        except Exception:
            depth_proxy = len(ops)
    return {
        "gate_count": int(len(ops)),
        "one_qubit_gate_count": int(one_qubit),
        "two_qubit_gate_count": int(two_qubit),
        "depth_proxy": int(depth_proxy),
        "gate_name_histogram_json": json.dumps(dict(sorted(_counter(gate_names).items()))),
    }


def _counter(values: Iterable[Any]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def deterministic_noise_proxy(qc: Any, noise_p: float) -> float:
    metrics = circuit_structure_metrics(qc)
    weighted_gate_count = (
        metrics["one_qubit_gate_count"] + 3.0 * metrics["two_qubit_gate_count"]
    )
    return float(noise_p * weighted_gate_count)


def noisy_unitary_stats(
    qc: Any,
    target_unitary: torch.Tensor,
    noise_p: float,
    simulator: Simulator,
    *,
    realizations: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    if noise_p <= 0:
        predicted = simulator.backend.get_unitary(qc)
        metrics = unitary_metrics(predicted, target_unitary)
        return {
            "noise_p": float(noise_p),
            "noisy_score_mean": metrics["clean_infidelity"],
            "noisy_score_std": 0.0,
            "noisy_score_min": metrics["clean_infidelity"],
            "noisy_score_max": metrics["clean_infidelity"],
        }

    scores: list[float] = []
    for offset in range(int(realizations)):
        qc_noise = qc.copy()
        qc_noise.add_noise_global(float(noise_p))
        realized_qc, _applied = qc_noise.realize_noise(
            seed=int(seed + offset),
            show_identities=False,
            in_place=False,
        )
        predicted = simulator.backend.get_unitary(realized_qc)
        metrics = unitary_metrics(predicted, target_unitary)
        scores.append(metrics["clean_infidelity"])

    score_arr = np.asarray(scores, dtype=np.float64)
    return {
        "noise_p": float(noise_p),
        "noisy_score_mean": float(score_arr.mean()),
        "noisy_score_std": float(score_arr.std(ddof=0)),
        "noisy_score_min": float(score_arr.min()),
        "noisy_score_max": float(score_arr.max()),
    }


def decode_candidate_tensors(
    dataset,
    tensors: torch.Tensor,
    params: torch.Tensor | None = None,
):
    tokenizer = build_tokenizer(dataset)
    simulator = build_quditkit_simulator()
    decoded, _ = decode_tensors_to_backend(
        simulator=simulator,
        tokenizer=tokenizer,
        tensors=tensors,
        params=params,
        silent=True,
        n_jobs=1,
        filter_errs=False,
    )
    return decoded


def sample_model_tensors_for_target(
    *,
    pipeline: Any,
    prompt: str,
    target_unitary: torch.Tensor,
    system_size: int,
    num_qubits: int,
    max_gates: int,
    samples_per_target: int,
    guidance_scale: float,
    auto_batch_size: int,
):
    try:
        out = generate_compilation_tensors(
            pipeline=pipeline,
            prompt=[prompt],
            U=target_unitary.unsqueeze(0),
            samples=int(samples_per_target),
            system_size=system_size,
            num_of_qubits=num_qubits,
            max_gates=max_gates,
            g=float(guidance_scale),
            auto_batch_size=int(auto_batch_size),
            enable_params=True,
            no_bar=True,
        )
        if not isinstance(out, tuple):
            raise RuntimeError(
                "Expected generate_compilation_tensors to return (tensors, params)."
            )
        return out
    except TypeError as err:
        # Older / alternate embedders can expose an `invert(...)` signature without
        # `reduce_spatial`. For the current unitary-compilation gate sets, parameter
        # tensors are not required for decoding, so fall back to parameter-free output.
        if "reduce_spatial" not in str(err):
            raise
        out = generate_compilation_tensors(
            pipeline=pipeline,
            prompt=[prompt],
            U=target_unitary.unsqueeze(0),
            samples=int(samples_per_target),
            system_size=system_size,
            num_of_qubits=num_qubits,
            max_gates=max_gates,
            g=float(guidance_scale),
            auto_batch_size=int(auto_batch_size),
            enable_params=False,
            no_bar=True,
        )
        if isinstance(out, tuple):
            return out[0], None
        return out, None


def timestamp_slug() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_unitary_id(target_index: int) -> str:
    return f"target_{int(target_index):06d}"


def save_torch(path: str | Path, payload: Any) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_torch(path: str | Path) -> Any:
    return torch.load(Path(path).expanduser().resolve(), map_location="cpu")


def parse_float_list(values: str | Iterable[float]) -> list[float]:
    if isinstance(values, str):
        return [float(item.strip()) for item in values.split(",") if item.strip()]
    return [float(item) for item in values]


def clean_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (np.floating,)):
                out[key] = float(value)
            elif isinstance(value, (np.integer,)):
                out[key] = int(value)
            elif isinstance(value, Path):
                out[key] = str(value)
            elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                out[key] = None
            else:
                out[key] = value
        cleaned.append(out)
    return cleaned
