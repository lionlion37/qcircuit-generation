"""Basic tests for the QuantumDiffusion framework."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all main modules can be imported."""
    try:
        from quantum_diffusion import DatasetGenerator, DatasetLoader
        from quantum_diffusion import DiffusionTrainer, ModelManager
        from quantum_diffusion import Evaluator, MetricsCalculator
        from quantum_diffusion import ConfigManager, Logger
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")

def test_dataset_generator_init():
    """Test DatasetGenerator initialization."""
    from quantum_diffusion.data import DatasetGenerator
    
    generator = DatasetGenerator(device='cpu')
    assert generator.device == 'cpu'
    assert generator.logger is not None

def test_config_manager_init():
    """Test ConfigManager initialization."""
    from quantum_diffusion.utils import ConfigManager
    
    config_manager = ConfigManager()
    assert config_manager.config_dir.name == "configs"

def test_logger_init():
    """Test Logger initialization."""
    from quantum_diffusion.utils import Logger
    
    logger = Logger(__name__)
    assert logger.logger.name == __name__

def test_preset_configs():
    """Test that preset configurations are available."""
    from quantum_diffusion.data import PRESET_CONFIGS
    from quantum_diffusion.models import PRESET_TRAINING_CONFIGS
    
    assert len(PRESET_CONFIGS) > 0
    assert len(PRESET_TRAINING_CONFIGS) > 0
    
    # Check that presets have required keys
    for preset_name, config in PRESET_CONFIGS.items():
        assert "gate_set" in config
        assert "num_qubits" in config
        assert "num_samples" in config

def test_package_version():
    """Test that package version is accessible."""
    from quantum_diffusion import __version__
    assert __version__ == "0.1.0"

def test_quick_start_tutorial():
    """Test that quick start tutorial can be called."""
    from quantum_diffusion import quick_start_tutorial
    
    # Should not raise an error
    try:
        quick_start_tutorial()
    except Exception as e:
        pytest.fail(f"Quick start tutorial failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])