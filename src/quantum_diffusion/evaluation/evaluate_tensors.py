import argparse
import sys
import os
import time
import datetime
import ast
from collections import Counter
from pathlib import Path
import hydra

import numpy as np
import torch

# Ensure local src/ is importable
sys.path.insert(0, str(Path(os.getcwd()).parent / "src"))

from my_genQC.inference.eval_metrics import UnitaryFrobeniusNorm, UnitaryInfidelityNorm
from my_genQC.inference.evaluation_helper import get_unitaries, get_srvs
from my_genQC.inference.sampling import generate_compilation_tensors, generate_tensors, decode_tensors_to_backend
from my_genQC.pipeline.diffusion_pipeline import DiffusionPipeline
from my_genQC.platform.simulation import Simulator, CircuitBackendType
from my_genQC.platform.tokenizer.circuits_tokenizer import CircuitTokenizer
from my_genQC.utils.misc_utils import infer_torch_device, get_entanglement_bins
from my_genQC.dataset import circuits_dataset
from my_genQC.models.config_model import ConfigModel
from my_genQC.utils.config_loader import load_config, store_tensor, load_tensor


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


def evaluate_tensors(tensor_path: str, dataset_path: str, backend: CircuitBackendType, device: torch.device, n_jobs: int = 1):
    dataset = load_dataset(dataset_path=dataset_path, device=device)
    n_samples = ds.x.shape[0]
    tensor_out = load_tensor("../scripts/inference/8q_599936_samples.pt", device=device)

    vocabulary = {gate: idx for idx, gate in enumerate(dataset.gate_pool)}
    tokenizer = CircuitTokenizer(vocabulary)
    simulator = Simulator(backend)

    decoded_circuits, _ = decode_tensors_to_backend(
        simulator=simulator,
        tokenizer=tokenizer,
        tensors=tensor_out,
        silent=True,
        params=None,
        n_jobs=n_jobs,
        filter_errs=False,
    )

    valid = [(idx, qc) for idx, qc in enumerate(decoded_circuits) if qc is not None]
    valid_indices = [idx for idx, _ in valid]
    backend_circuits = [qc for _, qc in valid]
    err_cnt = len(decoded_circuits) - len(backend_circuits)

    print("==== genQC Evaluation ====")
    print(f"Samples requested: {n_samples}")
    print(f"Decoded circuits : {len(backend_circuits)} / {((len(backend_circuits) / n_samples) * 100):.2f}%")
    print(f"Decode failures  : {err_cnt} / {((err_cnt / n_samples) * 100):.2f}%")