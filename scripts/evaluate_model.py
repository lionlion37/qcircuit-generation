#!/usr/bin/env python3
"""Minimal evaluation script that reuses genQC's native utilities."""

from __future__ import annotations

import argparse
import sys
import os
import time
import ast
from collections import Counter
from pathlib import Path
import hydra

import numpy as np
import torch

# Ensure local src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.my_genQC.inference.eval_metrics import UnitaryFrobeniusNorm, UnitaryInfidelityNorm
from src.my_genQC.inference.evaluation_helper import get_unitaries, get_srvs
from src.my_genQC.inference.sampling import generate_compilation_tensors, generate_tensors, decode_tensors_to_backend
from src.my_genQC.pipeline.diffusion_pipeline import DiffusionPipeline
from src.my_genQC.platform.simulation import Simulator, CircuitBackendType
from src.my_genQC.platform.tokenizer.circuits_tokenizer import CircuitTokenizer
from src.my_genQC.utils.misc_utils import infer_torch_device, get_entanglement_bins
from src.my_genQC.dataset import circuits_dataset
from src.my_genQC.models.config_model import ConfigModel
from src.my_genQC.utils.config_loader import load_config


def load_dataset(dataset_path: Path, device: torch.device):
    """Load a saved quantum circuit dataset.

    Args:
        dataset_path: Path to the saved dataset
        **kwargs: Additional loading parameters

    Returns:
        Loaded dataset object
    """

    config_path = os.path.join(dataset_path, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    cfg_data = load_config(config_path)
    target = cfg_data.get("target", "")
    if target.endswith("MixedCircuitsConfigDataset"):
        dataset_cls = circuits_dataset.MixedCircuitsConfigDataset
    else:
        dataset_cls = circuits_dataset.CircuitsConfigDataset

    # Load dataset using genQC
    dataset = dataset_cls.from_config_file(
        config_path=config_path,
        device=device,
        save_path=os.path.join(dataset_path, "dataset", "ds")
    )

    # logger.info(f"Dataset loaded from {dataset_path}")
    return dataset


def parse_srv_targets(labels: np.ndarray) -> torch.Tensor:
    """Extract SRV vectors from stored prompt strings."""
    srv_list = []
    for label in labels:
        text = str(label)
        start = text.find("[")
        end = text.find("]", start)
        if start == -1 or end == -1:
            raise ValueError(f"Could not parse SRV from label: {text}")
        srv = ast.literal_eval(text[start:end+1])
        srv_list.append(srv)
    return torch.tensor(srv_list, dtype=torch.long)


def entanglement_histogram(srvs: torch.Tensor, num_qubits: int) -> tuple[list[float], list[str], float]:
    """Return histogram over entanglement bins defined in genQC."""
    if srvs.numel() == 0:
        return [], [], 0.0

    bins, labels = get_entanglement_bins(num_qubits)
    mapping = {}
    for idx, bucket in enumerate(bins):
        for vector in bucket:
            mapping[tuple(vector)] = idx

    counts = Counter(mapping.get(tuple(vec.tolist()), -1) for vec in srvs)
    total = srvs.shape[0]
    hist = [counts.get(i, 0) / total for i in range(len(labels))]
    other_ratio = counts.get(-1, 0) / total
    return hist, labels, other_ratio


def load_pipeline(model_dir: Path | None, repo_id: str | None, device: torch.device):
    if repo_id:
        return DiffusionPipeline.from_pretrained(repo_id=repo_id, device=device)

    if not model_dir:
        raise ValueError("Provide either --model-dir or --hf-repo.")

    model_dir = model_dir.resolve()
    config_path = model_dir if model_dir.is_dir() else model_dir.parent
    cfg_file = config_path / "config.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing pipeline config at {cfg_file}")

    # DiffusionPipeline expects a directory string ending with '/'
    return DiffusionPipeline.from_config_file(config_path=str(config_path) + "/", device=device)


def to_complex(unitary_tensor: torch.Tensor) -> torch.Tensor:
    """Convert stacked real/imag tensors [B,2,N,N] to complex matrices."""
    if unitary_tensor.dim() != 4 or unitary_tensor.shape[1] != 2:
        raise ValueError(f"Unexpected unitary tensor shape {unitary_tensor.shape}")
    real = unitary_tensor[:, 0]
    imag = unitary_tensor[:, 1]
    return torch.complex(real, imag)


@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg):
    cfg = cfg["evaluation"]

    # parser = argparse.ArgumentParser(description="Evaluate a genQC diffusion pipeline using native helpers.")
    # parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to run evaluation on.")
    # args = parser.parse_args()

    device = torch.device(infer_torch_device())
    dataset = load_dataset(Path(cfg.dataset), device=device)

    pipeline = load_pipeline(
        model_dir=Path(cfg.model_dir) if cfg.model_dir else None,
        repo_id=cfg.hf_repo,
        device=device,
    )

    pipeline.guidance_sample_mode = "rescaled"
    pipeline.scheduler.set_timesteps(cfg.model_params.sample_steps)

    samples = min(cfg.num_samples, dataset.x.shape[0])
    if samples == 0:
        raise ValueError("Dataset is empty – nothing to evaluate.")

    system_size = dataset.x.shape[1]
    max_gates = dataset.x.shape[2]
    num_qubits = getattr(dataset.params_config, "num_of_qubits", system_size)

    is_compilation = hasattr(dataset, "U") and dataset.store_dict.get("U") == "tensor"

    print("Starting tensor generation...")

    if is_compilation:
        unitary_conditions = dataset.U[:samples].to(device)
        prompts = [str(p) for p in dataset.y[:samples]]

        tensors_out = generate_compilation_tensors(
            pipeline=pipeline,
            prompt=prompts,
            U=unitary_conditions,
            samples=samples,
            system_size=system_size,
            num_of_qubits=num_qubits,
            max_gates=max_gates,
            g=cfg.model_params.guidance_scale,
            auto_batch_size=cfg.model_params.auto_batch_size,
            enable_params=True,
            no_bar=False,  # shows diffusion steps
        )
    else:
        prompts = [str(p) for p in dataset.y[:samples]]

        tensors_out = generate_tensors(
            pipeline=pipeline,
            prompt=prompts,
            samples=samples,
            system_size=system_size,
            num_of_qubits=num_qubits,
            max_gates=max_gates,
            g=cfg.model_params.guidance_scale,
            auto_batch_size=cfg.model_params.auto_batch_size,
            enable_params=False,
            no_bar=False,  # shows diffusion steps
        )

    print("Finished tensor generation.")

    if isinstance(tensors_out, tuple):
        tensors, params = tensors_out
    else:
        tensors, params = tensors_out, None

    vocabulary = {gate: idx for idx, gate in enumerate(dataset.gate_pool)}
    tokenizer = CircuitTokenizer(vocabulary)
    simulator = Simulator(CircuitBackendType.QISKIT)

    decoded_circuits, _ = decode_tensors_to_backend(
        simulator=simulator,
        tokenizer=tokenizer,
        tensors=tensors,
        params=params,
        silent=True,
        n_jobs=1,
        filter_errs=False,
    )
    valid = [(idx, qc) for idx, qc in enumerate(decoded_circuits) if qc is not None]
    if not valid:
        raise RuntimeError("Decoding failed for all samples; cannot compute metrics.")

    valid_indices = [idx for idx, _ in valid]
    backend_circuits = [qc for _, qc in valid]
    err_cnt = len(decoded_circuits) - len(backend_circuits)

    print("==== genQC Evaluation ====")
    print(f"Samples requested: {samples}")
    print(f"Decoded circuits : {len(backend_circuits)}")
    print(f"Decode failures  : {err_cnt}")

    if is_compilation:
        idx_tensor = torch.as_tensor(
            valid_indices,
            device=dataset.U.device,
            dtype=torch.long,
        )
        target_complex = to_complex(dataset.U[:samples][idx_tensor].cpu())

        predicted = get_unitaries(simulator, backend_circuits, n_jobs=1)
        predicted = torch.from_numpy(np.stack(predicted)).to(target_complex.dtype)

        frob_metric = UnitaryFrobeniusNorm().distance(predicted, target_complex)
        infid_metric = UnitaryInfidelityNorm().distance(predicted, target_complex)

        print(f"Frobenius  mean/std: {frob_metric.mean().item():.6f} / {frob_metric.std(unbiased=False).item():.6f}")
        print(f"Infidelity mean/std: {infid_metric.mean().item():.6f} / {infid_metric.std(unbiased=False).item():.6f}")
    else:
        print("No target unitaries in dataset; running SRV evaluation.")
        target_srvs = parse_srv_targets(dataset.y[:samples])[valid_indices]
        predicted_srvs = torch.tensor(
            get_srvs(simulator, backend_circuits, n_jobs=1),
            dtype=torch.long,
        )

        if target_srvs.shape != predicted_srvs.shape:
            raise RuntimeError(f"SRV shape mismatch: target {target_srvs.shape} vs predicted {predicted_srvs.shape}")

        exact_match = (predicted_srvs == target_srvs).all(dim=1)
        per_qubit = (predicted_srvs == target_srvs).float().mean(dim=0)

        print(f"SRV exact-match rate : {exact_match.float().mean().item():.4f}")
        print("Per-qubit rank acc   : " + ", ".join(f"q{i}={acc:.3f}" for i, acc in enumerate(per_qubit.tolist())))

        pred_hist, ent_labels, pred_other = entanglement_histogram(predicted_srvs, num_qubits)
        targ_hist, _, targ_other = entanglement_histogram(target_srvs, num_qubits)

        if ent_labels:
            print("Entanglement-bin distribution (target | pred):")
            for label, t_frac, p_frac in zip(ent_labels, targ_hist, pred_hist):
                print(f"  {label:>20}: {t_frac:6.2%} | {p_frac:6.2%}")
            if targ_other > 0 or pred_other > 0:
                print(f"  {'Other/invalid':>20}: {targ_other:6.2%} | {pred_other:6.2%}")

    print("==========================")


if __name__ == "__main__":
    main()
