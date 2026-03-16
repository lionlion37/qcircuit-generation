from pathlib import Path

import pytest
import torch

from quantum_diffusion.training import DiffusionTrainer


@pytest.mark.integration
def test_load_pipeline_from_local_model_dir():
    if not torch.cuda.is_available():
        pytest.skip("This integration test expects a CUDA-capable runtime.")

    model_dir = Path("models/trained/paper_srv_model")
    if not model_dir.exists():
        pytest.skip(f"Required local model artifact is missing: {model_dir}")

    pipeline = DiffusionTrainer.load_pipeline(
        model_dir=model_dir,
        repo_id=None,
        device="cuda",
    )

    assert pipeline is not None
    assert pipeline.model is not None
    assert pipeline.scheduler is not None
