from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

try:
    import torch
    from torch.utils.data import TensorDataset
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "bucket_training_alignment_helper requires the qcircuit-generation "
        "training environment with PyTorch installed."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_TRAINING_CFG = REPO_ROOT / "conf" / "training" / "paper_srv_bucket.yaml"
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "qc_srv_dataset_3to8qubit"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quantum_diffusion.data import DatasetLoader


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_loader_config(
    training_cfg_path: str | Path = DEFAULT_TRAINING_CFG,
    *,
    batch_size: int | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(training_cfg_path)
    training = cfg.get("training", {})
    return {
        "training": {
            "padding_mode": "bucket",
            "batch_size": int(batch_size or training.get("batch_size", 256)),
        }
    }


def get_pad_constant(dataset) -> int:
    return int(getattr(dataset, "pad_constant", len(dataset.gate_pool) + 1))


def count_nonpad_timesteps(x: torch.Tensor, pad_constant: int) -> torch.Tensor:
    return torch.count_nonzero((x != pad_constant).any(dim=1), dim=1).to(torch.int32)


def count_nonpad_qubits(x: torch.Tensor, pad_constant: int) -> torch.Tensor:
    return torch.count_nonzero((x != pad_constant).any(dim=2), dim=1).to(torch.int32)


def round_up_to_multiple(values: torch.Tensor, multiple: int) -> torch.Tensor:
    rounded = torch.ceil(values.float() / multiple) * multiple
    return rounded.to(torch.int32)


def _normalize_qubits(values: torch.Tensor) -> torch.Tensor:
    values = values.clone().to(torch.int32)
    values[values == 0] = 1
    return values


def _normalize_bucket_times(values: torch.Tensor, model_scale_factor: int) -> torch.Tensor:
    values = values.clone().to(torch.int32)
    values[values == 0] = 1
    return round_up_to_multiple(values, model_scale_factor)


def _tensor_counter(values: torch.Tensor) -> dict[int, int]:
    unique, counts = torch.unique(values.cpu(), return_counts=True)
    return {int(key): int(count) for key, count in zip(unique.tolist(), counts.tolist())}


def _bucket_unique_count_counter(values: torch.Tensor) -> dict[int, int]:
    counter = Counter(int(torch.unique(bucket).numel()) for bucket in values.cpu())
    return {int(key): int(value) for key, value in sorted(counter.items())}


def _extract_z_tensor(tensor_dataset: TensorDataset, dataset) -> torch.Tensor:
    non_xy_keys = [key for key in dataset.store_dict.keys() if key not in {"x", "y"}]
    if "z" not in non_xy_keys:
        raise ValueError("Tensor dataset does not contain z metadata.")
    z_index = non_xy_keys.index("z")
    return tensor_dataset.tensors[2 + z_index]


def _extract_z_from_sample(sample: tuple[Any, ...], dataset) -> torch.Tensor:
    non_xy_keys = [key for key in dataset.store_dict.keys() if key not in {"x", "y"}]
    if "z" not in non_xy_keys:
        raise ValueError("Bucket sample does not contain z metadata.")
    z_index = non_xy_keys.index("z")
    return sample[2 + z_index]


def load_prepared_bucket_dataset(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    training_cfg_path: str | Path = DEFAULT_TRAINING_CFG,
    batch_size: int | None = None,
    device: str = "cpu",
):
    loader_cfg = build_loader_config(training_cfg_path, batch_size=batch_size)
    loader = DatasetLoader(config=loader_cfg, device=device)
    dataset = loader.load_dataset(
        str(Path(dataset_root).expanduser().resolve()),
        load_embedder=False,
    )

    if dataset.x.ndim != 4:
        raise ValueError(
            f"Expected a bucket-prepared dataset with rank-4 x, got {dataset.x.shape}."
        )
    if not hasattr(dataset, "z"):
        raise ValueError("Prepared bucket dataset does not expose z metadata.")

    return loader, dataset, loader_cfg


def summarize_bucket_tensor_alignment(
    x: torch.Tensor,
    z: torch.Tensor,
    *,
    pad_constant: int,
    model_scale_factor: int = 4,
    max_examples: int = 20,
    label: str | None = None,
) -> dict[str, Any]:
    if x.ndim != 4:
        raise ValueError(f"Expected x to have rank 4 [buckets, batch, qubits, time], got {x.shape}.")
    if z.ndim != 3 or z.shape[-1] != 2:
        raise ValueError(f"Expected z to have rank 3 [buckets, batch, 2], got {z.shape}.")

    x = x.cpu()
    z = z.cpu().to(torch.int32)

    bucket_count, bucket_batch_size = x.shape[0], x.shape[1]

    flat_x = x.reshape(-1, x.shape[-2], x.shape[-1])
    flat_z = z.reshape(-1, z.shape[-1])

    actual_qubits = _normalize_qubits(count_nonpad_qubits(flat_x, pad_constant=pad_constant))
    actual_nonpad_time = count_nonpad_timesteps(flat_x, pad_constant=pad_constant)
    expected_bucket_time = _normalize_bucket_times(
        actual_nonpad_time, model_scale_factor=model_scale_factor
    )

    z_qubits = flat_z[:, 0]
    z_time = flat_z[:, 1]

    qubit_mismatch = z_qubits != actual_qubits
    time_mismatch = z_time != expected_bucket_time
    any_mismatch = qubit_mismatch | time_mismatch

    bucket_z_qubits = z_qubits.reshape(bucket_count, bucket_batch_size)
    bucket_actual_qubits = actual_qubits.reshape(bucket_count, bucket_batch_size)
    bucket_z_time = z_time.reshape(bucket_count, bucket_batch_size)
    bucket_actual_time = expected_bucket_time.reshape(bucket_count, bucket_batch_size)

    bucket_z_max_qubits = bucket_z_qubits.max(dim=1).values
    bucket_actual_max_qubits = bucket_actual_qubits.max(dim=1).values
    bucket_z_max_time = bucket_z_time.max(dim=1).values
    bucket_actual_max_time = bucket_actual_time.max(dim=1).values

    bucket_qubit_max_mismatch = bucket_z_max_qubits != bucket_actual_max_qubits
    bucket_time_max_mismatch = bucket_z_max_time != bucket_actual_max_time

    mismatch_examples = []
    mismatch_indices = torch.nonzero(any_mismatch, as_tuple=False).flatten().tolist()
    for flat_index in mismatch_indices[:max_examples]:
        bucket_index = flat_index // bucket_batch_size
        sample_index = flat_index % bucket_batch_size
        mismatch_examples.append(
            {
                "flat_index": int(flat_index),
                "bucket_index": int(bucket_index),
                "sample_index": int(sample_index),
                "z_qubits": int(z_qubits[flat_index].item()),
                "actual_qubits": int(actual_qubits[flat_index].item()),
                "z_time": int(z_time[flat_index].item()),
                "actual_nonpad_time": int(actual_nonpad_time[flat_index].item()),
                "expected_bucket_time": int(expected_bucket_time[flat_index].item()),
            }
        )

    return {
        "label": label,
        "x_shape": tuple(int(dim) for dim in x.shape),
        "z_shape": tuple(int(dim) for dim in z.shape),
        "flattened_samples": int(flat_x.shape[0]),
        "bucket_count": int(bucket_count),
        "bucket_batch_size": int(bucket_batch_size),
        "pad_constant": int(pad_constant),
        "actual_qubits_unique": sorted(int(value) for value in torch.unique(actual_qubits).tolist()),
        "z_qubits_unique": sorted(int(value) for value in torch.unique(z_qubits).tolist()),
        "actual_nonpad_time_unique": sorted(
            int(value) for value in torch.unique(actual_nonpad_time).tolist()
        ),
        "expected_bucket_time_unique": sorted(
            int(value) for value in torch.unique(expected_bucket_time).tolist()
        ),
        "z_time_unique": sorted(int(value) for value in torch.unique(z_time).tolist()),
        "qubit_mismatch_count": int(qubit_mismatch.sum().item()),
        "time_mismatch_count": int(time_mismatch.sum().item()),
        "any_mismatch_count": int(any_mismatch.sum().item()),
        "bucket_z_qubits_unique_count_hist": _bucket_unique_count_counter(bucket_z_qubits),
        "bucket_actual_qubits_unique_count_hist": _bucket_unique_count_counter(
            bucket_actual_qubits
        ),
        "bucket_z_time_unique_count_hist": _bucket_unique_count_counter(bucket_z_time),
        "bucket_actual_time_unique_count_hist": _bucket_unique_count_counter(
            bucket_actual_time
        ),
        "bucket_qubit_max_mismatch_count": int(bucket_qubit_max_mismatch.sum().item()),
        "bucket_time_max_mismatch_count": int(bucket_time_max_mismatch.sum().item()),
        "bucket_z_max_qubits_hist": _tensor_counter(bucket_z_max_qubits),
        "bucket_actual_max_qubits_hist": _tensor_counter(bucket_actual_max_qubits),
        "bucket_z_max_time_hist": _tensor_counter(bucket_z_max_time),
        "bucket_actual_max_time_hist": _tensor_counter(bucket_actual_max_time),
        "mismatch_examples": mismatch_examples,
    }


def summarize_prepared_bucket_dataset(
    dataset,
    *,
    model_scale_factor: int = 4,
    max_examples: int = 20,
) -> dict[str, Any]:
    summary = summarize_bucket_tensor_alignment(
        dataset.x,
        dataset.z,
        pad_constant=get_pad_constant(dataset),
        model_scale_factor=model_scale_factor,
        max_examples=max_examples,
        label="prepared_dataset",
    )
    summary["dataset_type"] = type(dataset).__name__
    summary["collate_fn"] = getattr(dataset, "collate_fn", None)
    return summary


def build_exact_training_tensor_datasets(
    dataset,
    *,
    split_ratio: float = 0.1,
    seed: int = 1234,
    batch_size: int | None = None,
):
    effective_batch_size = int(batch_size or getattr(dataset, "bucket_batch_size", 0))
    if effective_batch_size <= 0:
        raise ValueError("Bucket training dataset must expose a positive bucket_batch_size.")

    torch.manual_seed(seed)
    train_ds, valid_ds = dataset.get_dataloaders(
        batch_size=effective_batch_size,
        text_encoder=None,
        p_valid=split_ratio,
        y_on_cpu=False,
        return_tensor_datasets=True,
        shuffle=True,
        shuffle_cpu_copy=True,
        caching=False,
    )
    return train_ds, valid_ds


def summarize_tensor_dataset_alignment(
    tensor_dataset: TensorDataset,
    dataset,
    *,
    split_name: str,
    model_scale_factor: int = 4,
    max_examples: int = 20,
) -> dict[str, Any]:
    x = tensor_dataset.tensors[0]
    z = _extract_z_tensor(tensor_dataset, dataset)
    summary = summarize_bucket_tensor_alignment(
        x,
        z,
        pad_constant=get_pad_constant(dataset),
        model_scale_factor=model_scale_factor,
        max_examples=max_examples,
        label=split_name,
    )
    summary["tensor_count"] = int(len(tensor_dataset.tensors))
    summary["num_buckets_in_split"] = int(len(tensor_dataset))
    return summary


def summarize_manual_collate(
    tensor_dataset: TensorDataset,
    dataset,
    *,
    split_name: str,
    model_scale_factor: int = 4,
    max_buckets: int = 20,
) -> dict[str, Any]:
    collate_fn = dataset.collate_fn
    if isinstance(collate_fn, str):
        collate_fn = getattr(dataset, collate_fn, None)
    if collate_fn is None:
        raise ValueError("Could not resolve dataset collate_fn.")

    pad_constant = get_pad_constant(dataset)
    checked_bucket_count = min(len(tensor_dataset), max_buckets)

    rows = []
    mismatch_rows = []

    for bucket_index in range(checked_bucket_count):
        sample = tensor_dataset[bucket_index]
        bucket_x = sample[0].cpu()
        bucket_z = _extract_z_from_sample(sample, dataset).cpu().to(torch.int32)

        cut_batch = collate_fn([sample])
        cut_x = cut_batch[0].cpu()

        actual_qubits = _normalize_qubits(
            count_nonpad_qubits(bucket_x, pad_constant=pad_constant)
        )
        actual_time = _normalize_bucket_times(
            count_nonpad_timesteps(bucket_x, pad_constant=pad_constant),
            model_scale_factor=model_scale_factor,
        )

        row = {
            "bucket_index": int(bucket_index),
            "cut_shape": tuple(int(dim) for dim in cut_x.shape),
            "cut_qubits": int(cut_x.shape[1]),
            "cut_time": int(cut_x.shape[2]),
            "z_max_qubits": int(bucket_z[:, 0].max().item()),
            "actual_max_qubits": int(actual_qubits.max().item()),
            "z_max_time": int(bucket_z[:, 1].max().item()),
            "actual_max_time": int(actual_time.max().item()),
        }
        row["qubits_match_z"] = row["cut_qubits"] == row["z_max_qubits"]
        row["qubits_match_actual"] = row["cut_qubits"] == row["actual_max_qubits"]
        row["time_match_z"] = row["cut_time"] == row["z_max_time"]
        row["time_match_actual"] = row["cut_time"] == row["actual_max_time"]

        rows.append(row)

        if not (
            row["qubits_match_z"]
            and row["qubits_match_actual"]
            and row["time_match_z"]
            and row["time_match_actual"]
        ):
            mismatch_rows.append(row)

    return {
        "label": split_name,
        "checked_bucket_count": int(checked_bucket_count),
        "mismatch_bucket_count": int(len(mismatch_rows)),
        "rows": rows,
        "mismatch_rows": mismatch_rows,
    }


def maybe_dataframe(rows: list[dict[str, Any]]):
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except Exception:
        return rows
