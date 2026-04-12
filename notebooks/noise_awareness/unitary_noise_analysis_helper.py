from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from notebooks.noise_awareness.unitary_noise_common import (
    DEFAULT_RESULTS_ROOT,
    build_quditkit_simulator,
    circuit_structure_metrics,
    clean_candidate_rows,
    decode_candidate_tensors,
    deterministic_noise_proxy,
    load_pipeline,
    load_saved_dataset,
    maybe_dataframe,
    noisy_unitary_stats,
    qiskit_prompt_text,
    resolve_path,
    sample_model_tensors_for_target,
    sample_target_indices,
    stable_unitary_id,
    to_complex_single,
    unitary_metrics,
)


DEFAULT_P_GRID = [0.0, 0.01, 0.03, 0.1]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib in the qcircuit-generation environment."
        ) from exc
    return plt


def build_model_specs(model_dirs: list[str | Path]) -> list[dict[str, str]]:
    specs = []
    for raw_path in model_dirs:
        path = resolve_path(raw_path)
        specs.append({"label": path.name, "model_dir": str(path)})
    return specs


def collect_candidate_tables(
    dataset_path: str | Path,
    *,
    model_specs: list[dict[str, str]],
    target_limit: int = 32,
    samples_per_target: int = 32,
    p_grid: list[float] | None = None,
    guidance_scale: float = 4.0,
    auto_batch_size: int = 64,
    clean_infidelity_threshold: float = 1e-6,
    noise_realizations: int = 8,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    p_grid = DEFAULT_P_GRID if p_grid is None else [float(p) for p in p_grid]
    dataset, dataset_cfg = load_saved_dataset(dataset_path, device=device, make_contiguous=False)
    simulator = build_quditkit_simulator()

    target_indices = sample_target_indices(dataset.x.shape[0], target_limit, seed)
    target_rows = []
    candidate_rows = []
    score_rows = []
    candidate_tensors = []
    candidate_target_positions = []
    candidate_ids = []

    pipelines: dict[str, Any] = {}
    for spec in model_specs:
        pipelines[spec["label"]] = load_pipeline(spec["model_dir"], device=torch.device(device))
        pipelines[spec["label"]].guidance_sample_mode = "rescaled"

    candidate_id = 0
    for target_position, dataset_index in enumerate(target_indices):
        target_x = dataset.x[dataset_index].cpu()
        target_u = dataset.U[dataset_index].cpu()
        prompt = qiskit_prompt_text(dataset.y[dataset_index])
        target_complex = to_complex_single(target_u).to(torch.complex128)

        target_rows.append(
            {
                "target_position": int(target_position),
                "target_index": int(dataset_index),
                "target_unitary_id": stable_unitary_id(dataset_index),
                "prompt": prompt,
                "original_gate_count": int(((target_x != 0) & (target_x != getattr(dataset.params_config, "pad_constant", len(dataset.gate_pool) + 1))).any(dim=0).sum().item()),
            }
        )

        original_qc = decode_candidate_tensors(dataset, target_x.unsqueeze(0))[0]
        if original_qc is not None:
            clean_metrics = unitary_metrics(simulator.backend.get_unitary(original_qc), target_complex)
            structure = circuit_structure_metrics(original_qc)
            base_row = {
                "candidate_id": int(candidate_id),
                "target_position": int(target_position),
                "target_index": int(dataset_index),
                "target_unitary_id": stable_unitary_id(dataset_index),
                "source_model": "dataset",
                "source_rank": 0,
                "is_valid_decode": True,
                "passes_clean_threshold": bool(clean_metrics["clean_infidelity"] <= clean_infidelity_threshold),
                "prompt": prompt,
                **clean_metrics,
                **structure,
            }
            candidate_rows.append(base_row)
            candidate_tensors.append(target_x.clone())
            candidate_target_positions.append(target_position)
            candidate_ids.append(candidate_id)
            for noise_p in p_grid:
                noise_stats = noisy_unitary_stats(
                    original_qc,
                    target_complex,
                    float(noise_p),
                    simulator,
                    realizations=noise_realizations,
                    seed=seed + candidate_id * 100,
                )
                score_rows.append(
                    {
                        "candidate_id": int(candidate_id),
                        "target_position": int(target_position),
                        "target_index": int(dataset_index),
                        "target_unitary_id": stable_unitary_id(dataset_index),
                        "source_model": "dataset",
                        "noise_p": float(noise_p),
                        "noise_proxy": float(deterministic_noise_proxy(original_qc, noise_p)),
                        **noise_stats,
                    }
                )
            candidate_id += 1

        for spec in model_specs:
            pipeline = pipelines[spec["label"]]
            tensors_out, params_out = sample_model_tensors_for_target(
                pipeline=pipeline,
                prompt=prompt,
                target_unitary=target_u.to(torch.device(device)),
                system_size=dataset.x.shape[1],
                num_qubits=int(getattr(dataset.params_config, "num_of_qubits", dataset.x.shape[1])),
                max_gates=dataset.x.shape[2],
                samples_per_target=samples_per_target,
                guidance_scale=guidance_scale,
                auto_batch_size=auto_batch_size,
            )
            decoded = decode_candidate_tensors(dataset, tensors_out.cpu(), params_out.cpu())
            for sample_rank, (sample_tensor, qc) in enumerate(zip(tensors_out.cpu(), decoded)):
                if qc is None:
                    candidate_rows.append(
                        {
                            "candidate_id": int(candidate_id),
                            "target_position": int(target_position),
                            "target_index": int(dataset_index),
                            "target_unitary_id": stable_unitary_id(dataset_index),
                            "source_model": spec["label"],
                            "source_rank": int(sample_rank),
                            "is_valid_decode": False,
                            "passes_clean_threshold": False,
                            "prompt": prompt,
                            "clean_frobenius": None,
                            "clean_infidelity": None,
                            "gate_count": None,
                            "one_qubit_gate_count": None,
                            "two_qubit_gate_count": None,
                            "depth_proxy": None,
                            "gate_name_histogram_json": json.dumps({}),
                        }
                    )
                    candidate_id += 1
                    continue

                clean_metrics = unitary_metrics(simulator.backend.get_unitary(qc), target_complex)
                structure = circuit_structure_metrics(qc)
                base_row = {
                    "candidate_id": int(candidate_id),
                    "target_position": int(target_position),
                    "target_index": int(dataset_index),
                    "target_unitary_id": stable_unitary_id(dataset_index),
                    "source_model": spec["label"],
                    "source_rank": int(sample_rank),
                    "is_valid_decode": True,
                    "passes_clean_threshold": bool(clean_metrics["clean_infidelity"] <= clean_infidelity_threshold),
                    "prompt": prompt,
                    **clean_metrics,
                    **structure,
                }
                candidate_rows.append(base_row)
                candidate_tensors.append(sample_tensor.clone())
                candidate_target_positions.append(target_position)
                candidate_ids.append(candidate_id)
                for noise_p in p_grid:
                    noise_stats = noisy_unitary_stats(
                        qc,
                        target_complex,
                        float(noise_p),
                        simulator,
                        realizations=noise_realizations,
                        seed=seed + candidate_id * 100,
                    )
                    score_rows.append(
                        {
                            "candidate_id": int(candidate_id),
                            "target_position": int(target_position),
                            "target_index": int(dataset_index),
                            "target_unitary_id": stable_unitary_id(dataset_index),
                            "source_model": spec["label"],
                            "noise_p": float(noise_p),
                            "noise_proxy": float(deterministic_noise_proxy(qc, noise_p)),
                            **noise_stats,
                        }
                    )
                candidate_id += 1

    overview = {
        "dataset_path": str(resolve_path(dataset_path)),
        "dataset_target_count": int(dataset.x.shape[0]),
        "analyzed_target_count": int(len(target_indices)),
        "model_labels": [spec["label"] for spec in model_specs],
        "samples_per_target": int(samples_per_target),
        "noise_ps": [float(p) for p in p_grid],
        "clean_infidelity_threshold": float(clean_infidelity_threshold),
        "noise_realizations": int(noise_realizations),
    }

    payload = {
        "overview": overview,
        "target_indices": torch.tensor(target_indices, dtype=torch.long),
        "target_x": dataset.x[target_indices].cpu(),
        "target_U": dataset.U[target_indices].cpu(),
        "target_y": [qiskit_prompt_text(dataset.y[idx]) for idx in target_indices],
        "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
        "candidate_target_positions": torch.tensor(candidate_target_positions, dtype=torch.long),
        "candidate_x": torch.stack(candidate_tensors, dim=0) if candidate_tensors else torch.empty((0, *dataset.x.shape[1:]), dtype=dataset.x.dtype),
    }

    return {
        "overview": overview,
        "target_rows": clean_candidate_rows(target_rows),
        "candidate_rows": clean_candidate_rows(candidate_rows),
        "score_rows": clean_candidate_rows(score_rows),
        "payload": payload,
        "dataset_cfg": dataset_cfg,
    }


def summarize_candidate_tables(candidate_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in candidate_rows if row.get("is_valid_decode")]
    threshold_rows = [row for row in valid_rows if row.get("passes_clean_threshold")]
    by_model: dict[str, dict[str, float]] = {}
    for row in valid_rows:
        label = str(row["source_model"])
        bucket = by_model.setdefault(
            label,
            {
                "valid_candidates": 0,
                "mean_clean_infidelity": 0.0,
                "mean_gate_count": 0.0,
            },
        )
        bucket["valid_candidates"] += 1
        bucket["mean_clean_infidelity"] += float(row["clean_infidelity"])
        bucket["mean_gate_count"] += float(row["gate_count"])
    for label, stats in by_model.items():
        denom = max(stats["valid_candidates"], 1)
        stats["mean_clean_infidelity"] /= denom
        stats["mean_gate_count"] /= denom

    by_noise_p: dict[float, dict[str, float]] = {}
    for row in score_rows:
        p = float(row["noise_p"])
        bucket = by_noise_p.setdefault(
            p,
            {"rows": 0, "mean_noisy_score": 0.0, "mean_noise_proxy": 0.0},
        )
        bucket["rows"] += 1
        bucket["mean_noisy_score"] += float(row["noisy_score_mean"])
        bucket["mean_noise_proxy"] += float(row["noise_proxy"])
    for p, stats in by_noise_p.items():
        denom = max(stats["rows"], 1)
        stats["mean_noisy_score"] /= denom
        stats["mean_noise_proxy"] /= denom

    return {
        "total_candidates": int(len(candidate_rows)),
        "valid_candidates": int(len(valid_rows)),
        "passes_clean_threshold": int(len(threshold_rows)),
        "by_model": by_model,
        "by_noise_p": by_noise_p,
    }


def plot_noise_overview(candidate_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]]):
    plt = _require_matplotlib()
    if not candidate_rows or not score_rows:
        raise ValueError("Need non-empty candidate_rows and score_rows for plotting.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    valid_rows = [row for row in candidate_rows if row.get("is_valid_decode")]
    labels = sorted({str(row["source_model"]) for row in valid_rows})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    x_gate = [float(row["gate_count"]) for row in valid_rows]
    y_clean = [float(row["clean_infidelity"]) for row in valid_rows]
    c_idx = [label_to_idx[str(row["source_model"])] for row in valid_rows]
    axes[0, 0].scatter(x_gate, y_clean, c=c_idx, alpha=0.65)
    axes[0, 0].set_title("Clean Infidelity vs Gate Count")
    axes[0, 0].set_xlabel("Gate Count")
    axes[0, 0].set_ylabel("Clean Infidelity")

    p_values = sorted({float(row["noise_p"]) for row in score_rows})
    for noise_p in p_values:
        subset = [row for row in score_rows if float(row["noise_p"]) == noise_p]
        xs = np.arange(len(subset))
        ys = [float(row["noisy_score_mean"]) for row in subset]
        axes[0, 1].plot(xs, ys, ".", alpha=0.5, label=f"p={noise_p:g}")
    axes[0, 1].set_title("Noisy Score Distribution by Noise Level")
    axes[0, 1].set_xlabel("Candidate Row")
    axes[0, 1].set_ylabel("Noisy Score Mean")
    axes[0, 1].legend()

    for noise_p in p_values:
        subset = [row for row in score_rows if float(row["noise_p"]) == noise_p]
        xs = [float(row["noise_proxy"]) for row in subset]
        ys = [float(row["noisy_score_mean"]) for row in subset]
        axes[1, 0].scatter(xs, ys, alpha=0.5, label=f"p={noise_p:g}")
    axes[1, 0].set_title("Noise Proxy vs Noisy Score")
    axes[1, 0].set_xlabel("Deterministic Noise Proxy")
    axes[1, 0].set_ylabel("Noisy Score Mean")
    axes[1, 0].legend()

    grouped = {}
    for row in score_rows:
        key = float(row["noise_p"])
        grouped.setdefault(key, []).append(float(row["noisy_score_mean"]))
    axes[1, 1].boxplot([grouped[p] for p in p_values], tick_labels=[f"{p:g}" for p in p_values])
    axes[1, 1].set_title("Noisy Score Boxplots")
    axes[1, 1].set_xlabel("Noise p")
    axes[1, 1].set_ylabel("Noisy Score Mean")

    fig.tight_layout()
    return fig


def save_analysis_bundle(
    analysis: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> Path:
    output_dir = (
        DEFAULT_RESULTS_ROOT
        / f"analysis_{analysis['overview']['analyzed_target_count']:03d}"
        if output_dir is None
        else resolve_path(output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    from notebooks.noise_awareness.unitary_noise_common import save_torch, write_json, write_rows_csv

    write_json(output_dir / "overview.json", analysis["overview"])
    write_rows_csv(output_dir / "targets.csv", analysis["target_rows"])
    write_rows_csv(output_dir / "candidates.csv", analysis["candidate_rows"])
    write_rows_csv(output_dir / "scores.csv", analysis["score_rows"])
    save_torch(output_dir / "payload.pt", analysis["payload"])
    return output_dir
