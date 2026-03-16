from pathlib import Path

import numpy as np
import torch
import yaml

from my_genQC.dataset.circuits_dataset import CircuitsConfigDataset
from quantum_diffusion.data import DatasetLoader


def _build_minimal_dataset(dataset_dir: Path) -> Path:
    pad_constant = 3
    dataset = CircuitsConfigDataset(
        device=torch.device("cpu"),
        store_dict={"x": "tensor", "y": "numpy"},
        dataset_to_gpu=False,
        optimized=False,
        random_samples=2,
        num_of_qubits=8,
        min_gates=1,
        max_gates=6,
        max_params=0,
        gate_pool=["h", "cx"],
    )

    x = torch.full((2, 8, 6), fill_value=pad_constant, dtype=torch.int64)
    x[0, :3, :4] = 0
    x[1, :5, :2] = 1

    dataset.x = x
    dataset.y = np.array(["[1, 1, 1]", "[2, 1, 1]"], dtype=object)

    config_path = dataset_dir / "config.yaml"
    save_path = dataset_dir / "dataset" / "ds"
    dataset.save_dataset(config_path=str(config_path), save_path=str(save_path))
    return dataset_dir


def test_bucket_mode_infers_and_persists_z(tmp_path: Path):
    dataset_dir = _build_minimal_dataset(tmp_path / "srv_dataset")
    z_path = dataset_dir / "dataset" / "ds_z.safetensors"
    assert not z_path.exists()

    loader = DatasetLoader(
        config={"training": {"padding_mode": "bucket", "batch_size": 2}},
        device="cpu",
    )
    dataset = loader.load_dataset(str(dataset_dir), load_embedder=False)

    assert "z" in dataset.store_dict
    assert z_path.exists()
    assert torch.equal(
        dataset.z.cpu(),
        torch.tensor([[3, 4], [5, 2]], dtype=torch.int32),
    )

    cfg = yaml.safe_load((dataset_dir / "config.yaml").read_text())
    assert cfg["params"]["store_dict"]["z"] == "tensor"
