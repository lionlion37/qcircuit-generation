from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from my_genQC.dataset.circuits_dataset import MixedCircuitsConfigDataset
from quantum_diffusion.data import DatasetLoader
from quantum_diffusion.training import DiffusionTrainer


class DummyTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 16, seq_len: int = 4, emb_dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.empty_token = torch.zeros((1, seq_len), dtype=torch.long)

    def forward(self, tokens, pool=False):
        return self.embedding(tokens.long())


def _build_bucket_training_dataset(dataset_dir: Path) -> Path:
    pad_constant = 3
    dataset = MixedCircuitsConfigDataset(
        device=torch.device("cpu"),
        store_dict={"x": "tensor", "y": "tensor", "z": "tensor"},
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

    x = torch.full((4, 8, 8), fill_value=pad_constant, dtype=torch.int64)
    x[0, :5, :8] = 0
    x[1, :3, :8] = 1
    x[2, :5, :4] = 0
    x[3, :3, :4] = 1

    dataset.x = x
    dataset.y = torch.tensor(
        [[1, 2, 3, 4], [2, 3, 4, 5], [1, 1, 2, 2], [3, 3, 4, 4]],
        dtype=torch.long,
    )
    dataset.z = torch.tensor([[5, 8], [3, 8], [5, 4], [3, 4]], dtype=torch.int32)

    config_path = dataset_dir / "config.yaml"
    save_path = dataset_dir / "dataset" / "ds"
    dataset.save_dataset(config_path=str(config_path), save_path=str(save_path))
    return dataset_dir


def test_bucket_training_runs_end_to_end_on_cpu(tmp_path: Path):
    dataset_dir = _build_bucket_training_dataset(tmp_path / "bucket_train_dataset")
    text_encoder = DummyTextEncoder()

    loader = DatasetLoader(
        config={"training": {"padding_mode": "bucket", "batch_size": 2}},
        device="cpu",
    )
    dataset = loader.load_dataset(str(dataset_dir), load_embedder=False)
    loader.text_encoder = text_encoder
    dataloaders = loader.get_dataloaders(
        dataset,
        batch_size=2,
        caching=False,
        split_ratio=0.5,
    )

    cfg = OmegaConf.create(
        {
            "model": {
                "type": "QC_Cond_UNet",
                "params": {
                    "model_features": [32, 32, 64],
                    "clr_dim": 4,
                    "num_clrs": 4,
                    "t_emb_size": 32,
                    "cond_emb_size": 8,
                    "num_heads": [1, 1, 1],
                    "num_res_blocks": [1, 1, 1],
                    "transformer_depths": [1, 1, 1],
                },
            },
            "training": {
                "learning_rate": 1e-4,
                "optimizer": "Adam",
                "loss": "MSELoss",
                "num_epochs": 1,
                "batch_size": 2,
                "enable_guidance_train": True,
                "guidance_train_p": 0.1,
                "cached_text_enc": True,
                "ckpt_interval": None,
                "ckpt_path": None,
                "wandb": {"enable": False},
            },
            "scheduler": {
                "type": "DDIMScheduler",
                "params": {
                    "num_train_timesteps": 10,
                    "beta_start": 0.0001,
                    "beta_end": 0.02,
                    "beta_schedule": "linear",
                    "input_perturbation": 0.0,
                    "prediction_type": "epsilon",
                    "eta": 1,
                },
            },
        }
    )

    trainer = DiffusionTrainer(config=cfg, device="cpu")
    trainer.setup_model(dataset=dataset, text_encoder=text_encoder)
    trainer.compile_model()
    trainer.train(dataloaders, save_path=None)

    assert trainer.pipeline is not None
    assert hasattr(trainer.pipeline, "fit_losses")
    assert len(trainer.pipeline.fit_losses) > 0
