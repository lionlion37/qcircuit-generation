from __future__ import annotations

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "bucket_deep_dive_helper requires the qcircuit-generation training "
        "environment with PyTorch installed."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from my_genQC.dataset.config_dataset import ConfigDataset
from notebooks.bucket_training_alignment_helper import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_TRAINING_CFG,
    count_nonpad_qubits,
    count_nonpad_timesteps,
    get_pad_constant,
    load_prepared_bucket_dataset,
    round_up_to_multiple,
)
from notebooks.training_dataset_audit_helper import entanglement_bucket, parse_srv_label


def to_numpy_strings(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(str)
    if isinstance(values, torch.Tensor):
        return np.array([str(v) for v in values.cpu().tolist()], dtype=object)
    if isinstance(values, list):
        return np.array([str(v) for v in values], dtype=object)
    raise TypeError(f"Unsupported label container: {type(values)}")


def _normalize_qubits(values: torch.Tensor) -> torch.Tensor:
    values = values.clone().to(torch.int32)
    values[values == 0] = 1
    return values


def _normalize_bucket_time(values: torch.Tensor, model_scale_factor: int) -> torch.Tensor:
    values = values.clone().to(torch.int32)
    values[values == 0] = 1
    return round_up_to_multiple(values, model_scale_factor)


def _majority_ratio(values: torch.Tensor) -> float:
    uniques, counts = torch.unique(values.cpu(), return_counts=True)
    if counts.numel() == 0:
        return 0.0
    return float(counts.max().item() / counts.sum().item())


def _tensor_counter(values: torch.Tensor) -> dict[int, int]:
    unique, counts = torch.unique(values.cpu(), return_counts=True)
    return {int(key): int(count) for key, count in zip(unique.tolist(), counts.tolist())}


def _counter_rows(counter: Counter | dict[Any, int], key_name: str) -> list[dict[str, Any]]:
    items = counter.items()
    return [{key_name: key, "count": int(value)} for key, value in sorted(items, key=lambda kv: kv[0])]


def _parse_prompt_grid(y: Any) -> np.ndarray:
    if isinstance(y, np.ndarray):
        return y.astype(str)
    if isinstance(y, torch.Tensor):
        flat = [str(v) for v in y.cpu().tolist()]
        return np.array(flat, dtype=object).reshape(y.shape)
    raise TypeError(f"Unsupported y type: {type(y)}")


def _bucket_prompt_stats(prompt_bucket: np.ndarray) -> dict[str, Any]:
    prompt_counter = Counter(str(v) for v in prompt_bucket.reshape(-1).tolist())
    ent_counter = Counter()
    srv_counter = Counter()
    invalid_srv_count = 0

    for label in prompt_bucket.reshape(-1).tolist():
        parsed = parse_srv_label(label)
        if parsed is None:
            invalid_srv_count += 1
            continue
        srv_counter[tuple(parsed)] += 1
        bucket = entanglement_bucket(label)
        if bucket is not None:
            ent_counter[int(bucket)] += 1

    return {
        "prompt_unique_count": int(len(prompt_counter)),
        "prompt_majority_ratio": 0.0 if not prompt_counter else max(prompt_counter.values()) / sum(prompt_counter.values()),
        "srv_unique_count": int(len(srv_counter)),
        "ent_bucket_unique_count": int(len(ent_counter)),
        "invalid_srv_prompt_count": int(invalid_srv_count),
    }


def get_bucket_components(
    x: torch.Tensor,
    z: torch.Tensor,
    *,
    pad_constant: int,
    model_scale_factor: int = 4,
) -> dict[str, torch.Tensor]:
    x_cpu = x.cpu()
    z_cpu = z.cpu().to(torch.int32)

    actual_qubits = _normalize_qubits(count_nonpad_qubits(x_cpu.reshape(-1, x_cpu.shape[-2], x_cpu.shape[-1]), pad_constant))
    actual_nonpad_time = count_nonpad_timesteps(x_cpu.reshape(-1, x_cpu.shape[-2], x_cpu.shape[-1]), pad_constant)
    actual_bucket_time = _normalize_bucket_time(actual_nonpad_time, model_scale_factor)

    return {
        "flat_x": x_cpu.reshape(-1, x_cpu.shape[-2], x_cpu.shape[-1]),
        "flat_z": z_cpu.reshape(-1, z_cpu.shape[-1]),
        "actual_qubits": actual_qubits,
        "actual_nonpad_time": actual_nonpad_time,
        "actual_bucket_time": actual_bucket_time,
    }


def summarize_bucket_dataset(
    x: torch.Tensor,
    y: Any,
    z: torch.Tensor,
    *,
    pad_constant: int,
    model_scale_factor: int = 4,
) -> dict[str, Any]:
    if x.ndim != 4:
        raise ValueError(f"Expected bucketed x with rank 4, got {x.shape}.")
    if z.ndim != 3:
        raise ValueError(f"Expected bucketed z with rank 3, got {z.shape}.")

    prompt_grid = _parse_prompt_grid(y)
    if prompt_grid.shape[:2] != tuple(x.shape[:2]):
        raise ValueError(f"Expected y shape {x.shape[:2]}, got {prompt_grid.shape}.")

    components = get_bucket_components(x, z, pad_constant=pad_constant, model_scale_factor=model_scale_factor)
    bucket_count, bucket_batch_size = x.shape[0], x.shape[1]

    actual_qubits_bucket = components["actual_qubits"].reshape(bucket_count, bucket_batch_size)
    actual_time_bucket = components["actual_bucket_time"].reshape(bucket_count, bucket_batch_size)
    actual_nonpad_time_bucket = components["actual_nonpad_time"].reshape(bucket_count, bucket_batch_size)
    z_qubits_bucket = z.cpu()[:, :, 0].to(torch.int32)
    z_time_bucket = z.cpu()[:, :, 1].to(torch.int32)

    bucket_rows = []
    bucket_qubit_counter = Counter()
    bucket_time_counter = Counter()

    for bucket_index in range(bucket_count):
        prompt_stats = _bucket_prompt_stats(prompt_grid[bucket_index])
        z_qubits = z_qubits_bucket[bucket_index]
        actual_qubits = actual_qubits_bucket[bucket_index]
        z_time = z_time_bucket[bucket_index]
        actual_time = actual_time_bucket[bucket_index]
        actual_nonpad_time = actual_nonpad_time_bucket[bucket_index]

        row = {
            "bucket_index": int(bucket_index),
            "bucket_batch_size": int(bucket_batch_size),
            "z_max_qubits": int(z_qubits.max().item()),
            "actual_max_qubits": int(actual_qubits.max().item()),
            "z_min_qubits": int(z_qubits.min().item()),
            "actual_min_qubits": int(actual_qubits.min().item()),
            "z_max_time": int(z_time.max().item()),
            "actual_max_time": int(actual_time.max().item()),
            "actual_nonpad_time_max": int(actual_nonpad_time.max().item()),
            "z_min_time": int(z_time.min().item()),
            "actual_min_time": int(actual_time.min().item()),
            "actual_nonpad_time_min": int(actual_nonpad_time.min().item()),
            "z_qubits_unique_count": int(torch.unique(z_qubits).numel()),
            "actual_qubits_unique_count": int(torch.unique(actual_qubits).numel()),
            "z_time_unique_count": int(torch.unique(z_time).numel()),
            "actual_time_unique_count": int(torch.unique(actual_time).numel()),
            "actual_nonpad_time_unique_count": int(torch.unique(actual_nonpad_time).numel()),
            "qubit_majority_ratio": _majority_ratio(actual_qubits),
            "time_majority_ratio": _majority_ratio(actual_time),
            "z_qubit_majority_ratio": _majority_ratio(z_qubits),
            "z_time_majority_ratio": _majority_ratio(z_time),
            "qubit_matches_z_max": bool(int(z_qubits.max().item()) == int(actual_qubits.max().item())),
            "time_matches_z_max": bool(int(z_time.max().item()) == int(actual_time.max().item())),
            **prompt_stats,
        }
        bucket_rows.append(row)
        bucket_qubit_counter.update(torch.unique(actual_qubits).tolist())
        bucket_time_counter.update(torch.unique(actual_time).tolist())

    sample_rows = {
        "z_vs_actual_qubits": _counter_rows(
            Counter((int(zq), int(aq)) for zq, aq in zip(components["flat_z"][:, 0].tolist(), components["actual_qubits"].tolist())),
            "pair",
        ),
        "z_vs_actual_time": _counter_rows(
            Counter((int(zt), int(at)) for zt, at in zip(components["flat_z"][:, 1].tolist(), components["actual_bucket_time"].tolist())),
            "pair",
        ),
    }

    return {
        "bucket_rows": bucket_rows,
        "sample_pair_counts": sample_rows,
        "summary": {
            "bucket_count": int(bucket_count),
            "bucket_batch_size": int(bucket_batch_size),
            "actual_qubits_present": sorted(int(v) for v in bucket_qubit_counter.keys()),
            "actual_times_present": sorted(int(v) for v in bucket_time_counter.keys()),
            "pad_constant": int(pad_constant),
        },
    }


def split_bucket_arrays(
    dataset,
    *,
    split_ratio: float = 0.1,
    seed: int = 1234,
) -> dict[str, Any]:
    x_proc, y_proc, *z_proc = ConfigDataset.x_y_preprocess(
        dataset,
        balance_max=None,
        shuffle=False,
        max_samples=None,
        make_unique=False,
    )
    torch.manual_seed(seed)
    x_train, x_valid, y_train, y_valid, (z_train, z_valid) = dataset.valid_split(
        x_proc,
        y_proc,
        *z_proc,
        p_valid=split_ratio,
        split_sequential=False,
    )
    return {
        "train": {"x": x_train, "y": y_train, "z": z_train[0]},
        "valid": {"x": x_valid, "y": y_valid, "z": z_valid[0]},
    }


def compute_bucket_similarity_rows(
    x: torch.Tensor,
    *,
    pad_constant: int,
    max_buckets: int | None = 300,
    sample_size: int = 24,
) -> list[dict[str, Any]]:
    if x.ndim != 4:
        raise ValueError(f"Expected bucketed x with rank 4, got {x.shape}.")

    bucket_count = x.shape[0] if max_buckets is None else min(int(max_buckets), int(x.shape[0]))
    rows = []

    for bucket_index in range(bucket_count):
        bucket = x[bucket_index].cpu()
        take = min(sample_size, bucket.shape[0])
        bucket = bucket[:take]
        flat = bucket.reshape(take, -1)
        active = (bucket != pad_constant).reshape(take, -1)

        pair_indices = list(combinations(range(take), 2))
        if not pair_indices:
            rows.append(
                {
                    "bucket_index": int(bucket_index),
                    "sample_size": int(take),
                    "pair_count": 0,
                    "token_similarity_mean": float("nan"),
                    "active_jaccard_mean": float("nan"),
                    "duplicate_fraction": 1.0,
                }
            )
            continue

        token_sims = []
        jaccards = []
        for i, j in pair_indices:
            eq = (flat[i] == flat[j]).float().mean().item()
            union = torch.logical_or(active[i], active[j]).sum().item()
            if union == 0:
                jacc = 1.0
            else:
                inter = torch.logical_and(active[i], active[j]).sum().item()
                jacc = inter / union
            token_sims.append(eq)
            jaccards.append(jacc)

        unique_rows = torch.unique(flat, dim=0).shape[0]
        rows.append(
            {
                "bucket_index": int(bucket_index),
                "sample_size": int(take),
                "pair_count": int(len(pair_indices)),
                "token_similarity_mean": float(np.mean(token_sims)),
                "active_jaccard_mean": float(np.mean(jaccards)),
                "duplicate_fraction": float(1.0 - unique_rows / take),
            }
        )

    return rows


def load_and_summarize_bucket_stage(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    training_cfg_path: str | Path = DEFAULT_TRAINING_CFG,
    batch_size: int | None = None,
    split_ratio: float = 0.1,
    seed: int = 1234,
    model_scale_factor: int = 4,
    similarity_max_buckets: int | None = 300,
    similarity_sample_size: int = 24,
    device: str = "cpu",
) -> dict[str, Any]:
    loader, dataset, loader_cfg = load_prepared_bucket_dataset(
        dataset_root,
        training_cfg_path=training_cfg_path,
        batch_size=batch_size,
        device=device,
    )
    del loader

    prepared = summarize_bucket_dataset(
        dataset.x,
        dataset.y,
        dataset.z,
        pad_constant=get_pad_constant(dataset),
        model_scale_factor=model_scale_factor,
    )
    prepared["similarity_rows"] = compute_bucket_similarity_rows(
        dataset.x,
        pad_constant=get_pad_constant(dataset),
        max_buckets=similarity_max_buckets,
        sample_size=similarity_sample_size,
    )

    split = split_bucket_arrays(dataset, split_ratio=split_ratio, seed=seed)
    train = summarize_bucket_dataset(
        split["train"]["x"],
        split["train"]["y"],
        split["train"]["z"],
        pad_constant=get_pad_constant(dataset),
        model_scale_factor=model_scale_factor,
    )
    train["similarity_rows"] = compute_bucket_similarity_rows(
        split["train"]["x"],
        pad_constant=get_pad_constant(dataset),
        max_buckets=similarity_max_buckets,
        sample_size=similarity_sample_size,
    )

    valid = summarize_bucket_dataset(
        split["valid"]["x"],
        split["valid"]["y"],
        split["valid"]["z"],
        pad_constant=get_pad_constant(dataset),
        model_scale_factor=model_scale_factor,
    )
    valid["similarity_rows"] = compute_bucket_similarity_rows(
        split["valid"]["x"],
        pad_constant=get_pad_constant(dataset),
        max_buckets=similarity_max_buckets,
        sample_size=similarity_sample_size,
    )

    return {
        "loader_cfg": loader_cfg,
        "prepared": prepared,
        "train": train,
        "valid": valid,
    }


def maybe_dataframe(rows: list[dict[str, Any]]):
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except Exception:
        return rows
