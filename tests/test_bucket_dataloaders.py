from pathlib import Path

import numpy as np
import torch

from my_genQC.dataset.circuits_dataset import MixedCircuitsConfigDataset
from quantum_diffusion.data import DatasetLoader


def _build_mixed_dataset(dataset_dir: Path) -> Path:
    pad_constant = 3
    dataset = MixedCircuitsConfigDataset(
        device=torch.device("cpu"),
        store_dict={"x": "tensor", "y": "numpy", "z": "tensor"},
        dataset_to_gpu=False,
        optimized=False,
        random_samples=4,
        num_of_qubits=8,
        min_gates=1,
        max_gates=6,
        max_params=0,
        gate_pool=["h", "cx"],
        pad_constant=pad_constant,
        collate_fn="cut_padding_collate_fn",
        bucket_batch_size=-1,
        model_scale_factor=4,
    )

    x = torch.full((4, 8, 6), fill_value=pad_constant, dtype=torch.int64)
    x[0, :5, :4] = 0
    x[1, :3, :5] = 1
    x[2, :5, :2] = 0
    x[3, :3, :3] = 1

    dataset.x = x
    dataset.y = np.array(["a", "b", "c", "d"], dtype=object)
    dataset.z = torch.tensor([[5, 4], [3, 5], [5, 2], [3, 3]], dtype=torch.int32)

    config_path = dataset_dir / "config.yaml"
    save_path = dataset_dir / "dataset" / "ds"
    dataset.save_dataset(config_path=str(config_path), save_path=str(save_path))
    return dataset_dir


def test_bucket_mode_uses_batch_size_one_and_groups_by_qubit_count(tmp_path: Path):
    dataset_dir = _build_mixed_dataset(tmp_path / "mixed_srv_dataset")

    loader = DatasetLoader(
        config={"training": {"padding_mode": "bucket", "batch_size": 2}},
        device="cpu",
    )
    dataset = loader.load_dataset(str(dataset_dir), load_embedder=False)

    assert dataset.bucket_batch_size == 2
    assert dataset.collate_fn == "cut_padding_Bucket_collate_fn"

    dataloaders = dataset.get_dataloaders(
        batch_size=2,
        text_encoder=None,
        caching=False,
    )

    assert dataloaders.train.batch_size == 1
    assert dataloaders.valid.batch_size == 1

    for tensor_dataset in (dataloaders.train.dataset, dataloaders.valid.dataset):
        for _, _, z_bucket in tensor_dataset:
            assert z_bucket[:, 0].unique().numel() == 1
