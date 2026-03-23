from __future__ import annotations

import ast
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "training_dataset_audit_helper requires the qcircuit-generation training "
        "environment with PyTorch installed."
    ) from exc


TRAINING_REPO = Path("/workspace/qcircuit-generation")
TRAINING_SRC = TRAINING_REPO / "src"
DEFAULT_TRAINING_CFG = TRAINING_REPO / "conf" / "training" / "paper_srv_bucket.yaml"
DEFAULT_DATASET_ROOT = TRAINING_REPO / "datasets" / "qc_srv_dataset_3to8qubit"

if str(TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC))

from my_genQC.dataset.circuits_dataset import CircuitsConfigDataset, MixedCircuitsConfigDataset
from quantum_diffusion.data import DatasetLoader


PAPER_SRV_VARIANTS = {
    3: {"min_gates": 2, "max_gates": 16, "num_samples": 200_000},
    4: {"min_gates": 3, "max_gates": 20, "num_samples": 300_000},
    5: {"min_gates": 4, "max_gates": 28, "num_samples": 500_000},
    6: {"min_gates": 5, "max_gates": 40, "num_samples": 500_000},
    7: {"min_gates": 6, "max_gates": 52, "num_samples": 500_000},
    8: {"min_gates": 7, "max_gates": 52, "num_samples": 600_000},
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_loader_config(
    training_cfg_path: str | Path = DEFAULT_TRAINING_CFG,
    *,
    padding_mode: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(training_cfg_path)
    training = cfg.get("training", {})
    return {
        "training": {
            "padding_mode": padding_mode or training.get("padding_mode", "max"),
            "batch_size": int(batch_size or training.get("batch_size", 256)),
        }
    }


def resolve_dataset_dirs(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if (root / "config.yaml").exists() and (root / "dataset").exists():
        return [root]

    dataset_dirs = sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / "config.yaml").exists() and (child / "dataset").exists()
    )
    if not dataset_dirs:
        raise FileNotFoundError(
            f"No dataset directories with config.yaml + dataset/ found under {root}"
        )
    return dataset_dirs


def load_plain_dataset(dataset_dir: str | Path, device: str = "cpu") -> CircuitsConfigDataset:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    return CircuitsConfigDataset.from_config_file(
        config_path=str(dataset_dir / "config.yaml"),
        device=torch.device(device),
        save_path=str(dataset_dir / "dataset" / "ds"),
    )


def parse_srv_label(label: Any) -> tuple[int, ...] | None:
    if isinstance(label, np.ndarray):
        label = label.tolist()
    if isinstance(label, (list, tuple)):
        try:
            return tuple(int(x) for x in label)
        except Exception:
            return None
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    if isinstance(label, str):
        try:
            parsed = ast.literal_eval(label)
            if isinstance(parsed, (list, tuple)):
                return tuple(int(x) for x in parsed)
        except Exception:
            return None
    return None


def entanglement_bucket(label: Any) -> int | None:
    parsed = parse_srv_label(label)
    if parsed is None:
        return None
    return sum(1 for value in parsed if value == 2)


def count_gate_timesteps(x: torch.Tensor, pad_constant: int) -> torch.Tensor:
    gate_mask = ((x != 0) & (x != pad_constant)).any(dim=1)
    return torch.count_nonzero(gate_mask, dim=1).to(torch.int32)


def count_nonpad_timesteps(x: torch.Tensor, pad_constant: int) -> torch.Tensor:
    return torch.count_nonzero((x != pad_constant).any(dim=1), dim=1).to(torch.int32)


def count_nonpad_qubits(x: torch.Tensor, pad_constant: int) -> torch.Tensor:
    return torch.count_nonzero((x != pad_constant).any(dim=2), dim=1).to(torch.int32)


def round_up_to_multiple(values: torch.Tensor, multiple: int) -> torch.Tensor:
    rounded = torch.ceil(values.float() / multiple) * multiple
    return rounded.to(torch.int32)


def infer_bucket_z_from_raw_x(x: torch.Tensor, pad_constant: int) -> torch.Tensor:
    z = torch.zeros((x.shape[0], 2), device=x.device, dtype=torch.int32)
    z[:, 0] = count_nonpad_qubits(x, pad_constant)
    z[:, 1] = count_nonpad_timesteps(x, pad_constant)
    z[z[:, 0] == 0, 0] = 1
    z[z[:, 1] == 0, 1] = 1
    return z.cpu()


def infer_max_stage_z(dataset: CircuitsConfigDataset, model_scale_factor: int = 4) -> torch.Tensor:
    x = dataset.x.cpu()
    z = torch.zeros((x.shape[0], 2), dtype=torch.int32)
    z[:, 0] = int(dataset.params_config.num_of_qubits)
    z[:, 1] = count_gate_timesteps(x, pad_constant=get_pad_constant(dataset))
    z[z[:, 1] == 0, 1] = 1
    z[:, 1] = round_up_to_multiple(z[:, 1], model_scale_factor)
    return z


def get_pad_constant(dataset: CircuitsConfigDataset | MixedCircuitsConfigDataset) -> int:
    return int(getattr(dataset, "pad_constant", len(dataset.gate_pool) + 1))


def _tensor_counter(values: torch.Tensor) -> dict[int, int]:
    unique, counts = torch.unique(values.cpu(), return_counts=True)
    return {int(key): int(count) for key, count in zip(unique.tolist(), counts.tolist())}


def _counter_to_rows(counter: Counter | dict[Any, int], key_name: str) -> list[dict[str, Any]]:
    items = counter.items() if isinstance(counter, Counter) else counter.items()
    return [{key_name: key, "count": int(value)} for key, value in sorted(items, key=lambda item: item[0])]


def summarize_raw_dataset(
    dataset_dir: str | Path,
    *,
    model_scale_factor: int = 4,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    cfg = load_yaml(dataset_dir / "config.yaml")
    dataset = load_plain_dataset(dataset_dir)
    x = dataset.x.cpu()
    pad_constant = get_pad_constant(dataset)

    bucket_z = infer_bucket_z_from_raw_x(x, pad_constant=pad_constant)
    bucket_z[:, 1] = round_up_to_multiple(bucket_z[:, 1], model_scale_factor)
    max_z = infer_max_stage_z(dataset, model_scale_factor=model_scale_factor)

    gate_lengths = count_gate_timesteps(x, pad_constant=pad_constant)
    nonpad_times = count_nonpad_timesteps(x, pad_constant=pad_constant)
    nonpad_qubits = count_nonpad_qubits(x, pad_constant=pad_constant)

    prompt_counter = Counter(str(label) for label in dataset.y.tolist())
    ent_counter = Counter()
    invalid_prompt_count = 0
    for label in dataset.y.tolist():
        bucket = entanglement_bucket(label)
        if bucket is None:
            invalid_prompt_count += 1
        else:
            ent_counter[bucket] += 1

    expected = PAPER_SRV_VARIANTS.get(int(dataset.params_config.num_of_qubits))

    summary = {
        "dataset_dir": str(dataset_dir),
        "config_target": cfg.get("target"),
        "num_samples_saved": int(x.shape[0]),
        "tensor_shape": tuple(int(dim) for dim in x.shape),
        "num_qubits_config": int(dataset.params_config.num_of_qubits),
        "min_gates_config": int(dataset.params_config.min_gates),
        "max_gates_config": int(dataset.params_config.max_gates),
        "pad_constant": pad_constant,
        "gate_pool": list(dataset.gate_pool),
        "expected_paper_variant": expected,
        "matches_paper_gate_pool": list(dataset.gate_pool) == ["h", "cx"],
        "matches_paper_min_gates": None if expected is None else int(dataset.params_config.min_gates) == expected["min_gates"],
        "matches_paper_max_gates": None if expected is None else int(dataset.params_config.max_gates) == expected["max_gates"],
        "gate_length_min": int(gate_lengths.min().item()),
        "gate_length_median": float(torch.median(gate_lengths.float()).item()),
        "gate_length_max": int(gate_lengths.max().item()),
        "nonpad_time_min": int(nonpad_times.min().item()),
        "nonpad_time_max": int(nonpad_times.max().item()),
        "nonpad_qubits_min": int(nonpad_qubits.min().item()),
        "nonpad_qubits_max": int(nonpad_qubits.max().item()),
        "bucket_z_time_min": int(bucket_z[:, 1].min().item()),
        "bucket_z_time_max": int(bucket_z[:, 1].max().item()),
        "max_stage_z_time_min": int(max_z[:, 1].min().item()),
        "max_stage_z_time_max": int(max_z[:, 1].max().item()),
        "bucket_vs_max_time_mismatch": int((bucket_z[:, 1] != max_z[:, 1]).sum().item()),
        "bucket_vs_max_qubit_mismatch": int((bucket_z[:, 0] != max_z[:, 0]).sum().item()),
        "bucket_time_is_constant": bool(torch.unique(bucket_z[:, 1]).numel() == 1),
        "max_stage_time_is_constant": bool(torch.unique(max_z[:, 1]).numel() == 1),
        "invalid_srv_prompt_count": int(invalid_prompt_count),
        "prompt_unique_count": len(prompt_counter),
        "prompt_count_min": int(min(prompt_counter.values())) if prompt_counter else 0,
        "prompt_count_max": int(max(prompt_counter.values())) if prompt_counter else 0,
        "entanglement_buckets": _counter_to_rows(ent_counter, "bucket"),
        "z_bucket_time_counts": _counter_to_rows(_tensor_counter(bucket_z[:, 1]), "time"),
        "z_max_time_counts": _counter_to_rows(_tensor_counter(max_z[:, 1]), "time"),
        "top_prompts": [
            {"prompt": key, "count": int(value)}
            for key, value in prompt_counter.most_common(10)
        ],
    }
    return summary


def load_prepared_dataset(
    dataset_root: str | Path,
    *,
    training_cfg_path: str | Path = DEFAULT_TRAINING_CFG,
    padding_mode: str = "max",
    batch_size: int | None = None,
    device: str = "cpu",
):
    loader_cfg = build_loader_config(
        training_cfg_path,
        padding_mode=padding_mode,
        batch_size=batch_size,
    )
    loader = DatasetLoader(config=loader_cfg, device=device)
    dataset = loader.load_dataset(str(Path(dataset_root).expanduser().resolve()), load_embedder=False)
    loader.text_encoder = None
    return loader, dataset


def summarize_prepared_dataset(
    dataset: CircuitsConfigDataset | MixedCircuitsConfigDataset,
    *,
    padding_mode: str,
) -> dict[str, Any]:
    pad_constant = get_pad_constant(dataset)
    x = dataset.x.cpu()
    summary = {
        "padding_mode": padding_mode,
        "dataset_type": type(dataset).__name__,
        "x_shape": tuple(int(dim) for dim in x.shape),
        "has_z": hasattr(dataset, "z"),
        "pad_constant": pad_constant,
        "bucket_batch_size": int(getattr(dataset, "bucket_batch_size", -1)),
        "collate_fn": getattr(dataset, "collate_fn", None),
        "params": {
            key: value
            for key, value in asdict(dataset.params_config).items()
            if key in {"num_of_qubits", "max_gates", "min_gates", "random_samples", "dataset_to_gpu"}
        },
    }

    if not hasattr(dataset, "z"):
        return summary

    z = dataset.z.cpu()
    summary["z_shape"] = tuple(int(dim) for dim in z.shape)

    if x.ndim == 3:
        gate_lengths = count_gate_timesteps(x, pad_constant=pad_constant)
        nonpad_times = count_nonpad_timesteps(x, pad_constant=pad_constant)
        nonpad_qubits = count_nonpad_qubits(x, pad_constant=pad_constant)
        summary.update(
            {
                "num_samples": int(x.shape[0]),
                "z_qubits_unique": sorted(int(value) for value in torch.unique(z[:, 0]).tolist()),
                "z_time_unique": sorted(int(value) for value in torch.unique(z[:, 1]).tolist()),
                "z_time_multiple_of_scale_factor": bool(torch.all(z[:, 1] % 4 == 0).item()),
                "nonpad_time_min": int(nonpad_times.min().item()),
                "nonpad_time_max": int(nonpad_times.max().item()),
                "gate_length_min": int(gate_lengths.min().item()),
                "gate_length_max": int(gate_lengths.max().item()),
                "nonpad_qubits_min": int(nonpad_qubits.min().item()),
                "nonpad_qubits_max": int(nonpad_qubits.max().item()),
                "z_vs_nonpad_time_mismatch": int((z[:, 1] != nonpad_times).sum().item()),
                "z_vs_nonpad_qubits_mismatch": int((z[:, 0] != nonpad_qubits).sum().item()),
            }
        )
        return summary

    if x.ndim == 4:
        flat_z = z.reshape(-1, z.shape[-1])
        bucket_uniform_qubits = [int(torch.unique(bucket[:, 0]).numel()) == 1 for bucket in z]
        summary.update(
            {
                "num_buckets": int(x.shape[0]),
                "samples_per_bucket": int(x.shape[1]),
                "flattened_samples": int(x.shape[0] * x.shape[1]),
                "bucket_qubits_all_uniform": bool(all(bucket_uniform_qubits)),
                "bucket_qubits_unique_counts": Counter(int(torch.unique(bucket[:, 0]).numel()) for bucket in z),
                "z_qubits_unique": sorted(int(value) for value in torch.unique(flat_z[:, 0]).tolist()),
                "z_time_unique": sorted(int(value) for value in torch.unique(flat_z[:, 1]).tolist()),
                "z_time_multiple_of_scale_factor": bool(torch.all(flat_z[:, 1] % 4 == 0).item()),
            }
        )
        return summary

    raise ValueError(f"Unsupported prepared tensor rank: {x.ndim}")


def preview_dataloaders(
    loader: DatasetLoader,
    dataset: CircuitsConfigDataset | MixedCircuitsConfigDataset,
    *,
    batch_size: int,
    split_ratio: float = 0.1,
    max_batches: int = 2,
) -> dict[str, Any]:
    torch.manual_seed(0)
    dataloaders = loader.get_dataloaders(
        dataset,
        batch_size=batch_size,
        split_ratio=split_ratio,
        caching=False,
        shuffle=False,
    )

    preview = []
    for name, data_loader in (("train", dataloaders.train), ("valid", dataloaders.valid)):
        batches = []
        for batch_index, batch in enumerate(data_loader):
            if batch_index >= max_batches:
                break
            batch_shapes = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    batch_shapes.append(tuple(int(dim) for dim in item.shape))
                elif isinstance(item, np.ndarray):
                    batch_shapes.append(tuple(int(dim) for dim in item.shape))
                else:
                    batch_shapes.append(type(item).__name__)
            batches.append({"batch_index": batch_index, "shapes": batch_shapes})
        preview.append({"split": name, "loader_batch_size": data_loader.batch_size, "batches": batches})

    return {"preview": preview}


def run_full_audit(
    dataset_root: str | Path,
    *,
    training_cfg_path: str | Path = DEFAULT_TRAINING_CFG,
    batch_size: int | None = None,
    split_ratio: float = 0.1,
) -> dict[str, Any]:
    dataset_dirs = resolve_dataset_dirs(dataset_root)
    raw = [summarize_raw_dataset(dataset_dir) for dataset_dir in dataset_dirs]

    loader_max, max_dataset = load_prepared_dataset(
        dataset_root,
        training_cfg_path=training_cfg_path,
        padding_mode="max",
        batch_size=batch_size,
    )
    loader_bucket, bucket_dataset = load_prepared_dataset(
        dataset_root,
        training_cfg_path=training_cfg_path,
        padding_mode="bucket",
        batch_size=batch_size,
    )

    effective_batch_size = build_loader_config(training_cfg_path, batch_size=batch_size)["training"]["batch_size"]

    return {
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "training_cfg_path": str(Path(training_cfg_path).expanduser().resolve()),
        "raw": raw,
        "prepared_max": summarize_prepared_dataset(max_dataset, padding_mode="max"),
        "prepared_bucket": summarize_prepared_dataset(bucket_dataset, padding_mode="bucket"),
        "preview_max": preview_dataloaders(
            loader_max,
            max_dataset,
            batch_size=effective_batch_size,
            split_ratio=split_ratio,
        ),
        "preview_bucket": preview_dataloaders(
            loader_bucket,
            bucket_dataset,
            batch_size=effective_batch_size,
            split_ratio=split_ratio,
        ),
    }


def maybe_dataframe(rows: list[dict[str, Any]]):
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except Exception:
        return rows
