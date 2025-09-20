"""
QuantumDiffusion: A comprehensive framework for generating quantum circuit datasets,
training diffusion models, and evaluating their performance using genQC.

This package provides:
- Dataset generation and management for quantum circuits
- Training diffusion models on quantum circuit data
- Comprehensive evaluation and testing framework
- Configuration management and logging utilities
"""

__version__ = "0.1.0"
__author__ = "Quantum Diffusion Team"

# Core functionality imports
from .data import DatasetGenerator, DatasetLoader, PRESET_CONFIGS
from .models import DiffusionTrainer, ModelManager, PRESET_TRAINING_CONFIGS
from .evaluation import Evaluator, MetricsCalculator
from .utils import ConfigManager, Logger, setup_logging

# Make key classes easily accessible
__all__ = [
    # Data
    'DatasetGenerator',
    'DatasetLoader', 
    'PRESET_CONFIGS',
    
    # Models
    'DiffusionTrainer',
    'ModelManager',
    'PRESET_TRAINING_CONFIGS',
    
    # Evaluation
    'Evaluator',
    'MetricsCalculator',
    
    # Utils
    'ConfigManager',
    'Logger',
    'setup_logging',
    
    # Package info
    '__version__',
    '__author__'
]

# Package-level convenience functions
def quick_start_tutorial():
    """Print a quick start guide for new users."""
    print("""
    QuantumDiffusion Quick Start Guide
    =================================
    
    1. Generate a dataset:
    from quantum_diffusion import DatasetGenerator
    generator = DatasetGenerator()
    generator.generate_dataset(
        gate_set=['h', 'cx', 'cz'],
        num_qubits=3,
        num_samples=1000,
        output_path='./my_dataset'
    )
    
    2. Train a model:
    from quantum_diffusion import DiffusionTrainer, DatasetLoader
    loader = DatasetLoader()
    dataset = loader.load_dataset('./my_dataset')
    dataloaders = loader.get_dataloaders(dataset)
    
    trainer = DiffusionTrainer()
    trainer.setup_model(dataset)
    trainer.compile_model()
    trainer.train(dataloaders, save_path='./my_model')
    
    3. Evaluate the model:
    from quantum_diffusion import Evaluator, ModelManager
    manager = ModelManager()
    model = manager.load_model('./my_model')
    
    evaluator = Evaluator()
    results = evaluator.evaluate_model(model, dataset, output_dir='./evaluation')
    
    For more examples, check the notebooks/ directory!
    """)

def create_example_project(project_path: str = "./quantum_diffusion_example"):
    """Create an example project structure with sample files.
    
    Args:
        project_path: Path where to create the example project
    """
    from pathlib import Path
    import shutil
    
    project_dir = Path(project_path)
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    (project_dir / "datasets").mkdir(exist_ok=True)
    (project_dir / "models").mkdir(exist_ok=True)
    (project_dir / "evaluation").mkdir(exist_ok=True)
    (project_dir / "configs").mkdir(exist_ok=True)
    (project_dir / "logs").mkdir(exist_ok=True)
    
    # Create example script
    example_script = '''#!/usr/bin/env python3
"""Example script for quantum diffusion workflow."""

import sys
sys.path.append('path/to/quantum_diffusion/src')

from quantum_diffusion import (
    DatasetGenerator, DatasetLoader, DiffusionTrainer, 
    Evaluator, setup_logging
)

def main():
    setup_logging(log_level="INFO")
    
    # 1. Generate dataset
    print("Generating dataset...")
    generator = DatasetGenerator()
    generator.generate_dataset(
        gate_set=['h', 'cx', 'cz', 's'],
        num_qubits=3,
        num_samples=500,
        output_path='./datasets/example'
    )
    
    # 2. Load dataset and create dataloaders
    print("Loading dataset...")
    loader = DatasetLoader()
    dataset = loader.load_dataset('./datasets/example')
    dataloaders = loader.get_dataloaders(dataset, batch_size=32)
    
    # 3. Train model
    print("Training model...")
    trainer = DiffusionTrainer()
    trainer.setup_model(dataset)
    trainer.compile_model()
    trainer.train(dataloaders, save_path='./models/example')
    
    # 4. Evaluate model
    print("Evaluating model...")
    evaluator = Evaluator()
    results = evaluator.evaluate_model(
        trainer, dataset, output_dir='./evaluation/example'
    )
    
    print("Workflow completed successfully!")
    print(f"Results saved to: ./evaluation/example")

if __name__ == "__main__":
    main()
'''
    
    with open(project_dir / "example_workflow.py", 'w') as f:
        f.write(example_script)
    
    # Create README
    readme_content = '''# Quantum Diffusion Example Project

This is an example project created by the QuantumDiffusion framework.

## Structure

- `datasets/`: Generated quantum circuit datasets
- `models/`: Trained diffusion models
- `evaluation/`: Evaluation results and plots
- `configs/`: Configuration files
- `logs/`: Log files

## Usage

1. Run the example workflow:
   ```bash
   python example_workflow.py
   ```

2. Or use the command-line tools:
   ```bash
   # Generate dataset
   python -m quantum_diffusion.scripts.generate_dataset --preset clifford_3q_unitary --output ./datasets/my_dataset
   
   # Train model
   python -m quantum_diffusion.scripts.train_model --dataset ./datasets/my_dataset --output ./models/my_model
   
   # Evaluate model
   python -m quantum_diffusion.scripts.evaluate_model --model ./models/my_model --dataset ./datasets/my_dataset --output ./evaluation/my_evaluation
   ```

For more information, see the main QuantumDiffusion documentation.
'''
    
    with open(project_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"Example project created at: {project_dir}")
    print("Run 'python example_workflow.py' to test the full workflow!")