#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from notebooks.noise_awareness.unitary_noise_common import (
    DEFAULT_DATASET_ROOT,
    load_saved_dataset,
    load_torch,
    read_json,
    read_rows_csv,
    resolve_path,
    save_torch,
    write_json,
)

if str((Path(__file__).resolve().parents[1] / "src")) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from my_genQC.dataset.circuits_dataset import CircuitsConfigDataset


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


def build_noise_dataset(
    *,
    source_dataset_path: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path,
    clean_infidelity_threshold: float,
):
    source_dataset, _cfg = load_saved_dataset(source_dataset_path, device="cpu", make_contiguous=False)
    candidate_dir = resolve_path(candidate_dir)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = read_rows_csv(candidate_dir / "candidates.csv")
    score_rows = read_rows_csv(candidate_dir / "scores.csv")
    payload = load_torch(candidate_dir / "payload.pt")
    overview = read_json(candidate_dir / "overview.json")

    candidate_lookup = {int(row["candidate_id"]): row for row in candidate_rows}
    tensor_lookup = {
        int(cid): payload["candidate_x"][idx]
        for idx, cid in enumerate(payload["candidate_ids"].tolist())
    }
    target_positions = payload["target_indices"].tolist()
    target_U = payload["target_U"]
    target_y = payload["target_y"]

    grouped_scores: dict[tuple[int, float], list[dict[str, str]]] = {}
    for row in score_rows:
        key = (int(row["target_position"]), float(row["noise_p"]))
        grouped_scores.setdefault(key, []).append(row)

    out_x = []
    out_y = []
    out_U = []
    out_noise_p = []
    meta_rows = []

    for target_position, dataset_index in enumerate(target_positions):
        for noise_p in overview["noise_ps"]:
            key = (int(target_position), float(noise_p))
            rows = grouped_scores.get(key, [])
            eligible = []
            fallbacks = []
            for row in rows:
                candidate_id = int(row["candidate_id"])
                candidate_meta = candidate_lookup[candidate_id]
                if not _coerce_bool(candidate_meta.get("is_valid_decode", False)):
                    continue
                clean_infidelity = float(candidate_meta["clean_infidelity"])
                noisy_score = float(row["noisy_score_mean"])
                record = {
                    "candidate_id": candidate_id,
                    "clean_infidelity": clean_infidelity,
                    "noisy_score_mean": noisy_score,
                    "source_model": candidate_meta["source_model"],
                }
                if clean_infidelity <= clean_infidelity_threshold:
                    eligible.append(record)
                fallbacks.append(record)

            if eligible:
                selected = min(
                    eligible,
                    key=lambda item: (item["noisy_score_mean"], item["clean_infidelity"], item["candidate_id"]),
                )
                fallback_flag = False
            elif fallbacks:
                selected = min(
                    fallbacks,
                    key=lambda item: (item["clean_infidelity"], item["noisy_score_mean"], item["candidate_id"]),
                )
                fallback_flag = True
            else:
                raise RuntimeError(
                    f"No decode-valid candidates available for target_position={target_position}, noise_p={noise_p}."
                )

            selected_id = int(selected["candidate_id"])
            if selected_id not in tensor_lookup:
                raise KeyError(f"Missing tensor payload for candidate_id={selected_id}")

            out_x.append(tensor_lookup[selected_id].clone())
            out_y.append(target_y[target_position])
            out_U.append(target_U[target_position].clone())
            out_noise_p.append(float(noise_p))
            meta_rows.append(
                {
                    "target_position": int(target_position),
                    "target_index": int(dataset_index),
                    "noise_p": float(noise_p),
                    "selected_candidate_id": int(selected_id),
                    "selected_source_model": selected["source_model"],
                    "selected_clean_infidelity": float(selected["clean_infidelity"]),
                    "selected_noisy_score_mean": float(selected["noisy_score_mean"]),
                    "fallback_flag": bool(fallback_flag),
                }
            )

    dataset_params = {
        "optimized": bool(getattr(source_dataset.params_config, "optimized", True)),
        "dataset_to_gpu": False,
        "random_samples": int(len(out_x)),
        "num_of_qubits": int(source_dataset.params_config.num_of_qubits),
        "min_gates": int(source_dataset.params_config.min_gates),
        "max_gates": int(source_dataset.params_config.max_gates),
        "gate_pool": list(source_dataset.gate_pool),
        "max_params": int(getattr(source_dataset.params_config, "max_params", 0)),
        "pad_constant": int(getattr(source_dataset.params_config, "pad_constant", len(source_dataset.gate_pool) + 1)),
        "store_dict": {"x": "tensor", "y": "numpy", "U": "tensor", "noise_p": "tensor"},
    }
    dataset = CircuitsConfigDataset(device=torch.device("cpu"), **dataset_params)
    dataset.comment = (
        f"Derived noise-aware dataset from {resolve_path(source_dataset_path)} with "
        f"noise levels {overview['noise_ps']} and clean threshold {clean_infidelity_threshold}."
    )
    dataset.x = torch.stack(out_x, dim=0)
    dataset.y = np.asarray(out_y, dtype=object)
    dataset.U = torch.stack(out_U, dim=0).float()
    dataset.noise_p = torch.tensor(out_noise_p, dtype=torch.float32).unsqueeze(-1)

    dataset.save_dataset(
        config_path=str(output_dir / "config.yaml"),
        save_path=str(output_dir / "dataset" / "ds"),
    )
    save_torch(output_dir / "selection_metadata.pt", meta_rows)
    write_json(
        output_dir / "selection_summary.json",
        {
            "source_dataset_path": str(resolve_path(source_dataset_path)),
            "candidate_dir": str(candidate_dir),
            "clean_infidelity_threshold": float(clean_infidelity_threshold),
            "num_rows": int(len(meta_rows)),
            "num_targets": int(len(target_positions)),
            "noise_ps": overview["noise_ps"],
            "fallback_rate": float(sum(1 for row in meta_rows if row["fallback_flag"]) / max(len(meta_rows), 1)),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Build a new additive noise-aware unitary dataset from candidate tables.")
    parser.add_argument("--source-dataset", required=True, help="Original saved unitary dataset.")
    parser.add_argument("--candidate-dir", required=True, help="Candidate table directory produced by build_unitary_noise_candidate_table.py.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_DATASET_ROOT / "unitary_quditkit_noise_v1"),
        help="Output directory for the derived dataset.",
    )
    parser.add_argument(
        "--clean-infidelity-threshold",
        type=float,
        default=1e-6,
        help="Threshold for eligible relabeling candidates.",
    )
    args = parser.parse_args()
    build_noise_dataset(
        source_dataset_path=args.source_dataset,
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
        clean_infidelity_threshold=args.clean_infidelity_threshold,
    )
    print("Derived dataset saved to:", resolve_path(args.output_dir))


if __name__ == "__main__":
    main()
