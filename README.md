# QuantumDiffusion

A framework for generating quantum circuit datasets, training diffusion models, and evaluating their performance using [genQC](https://github.com/FlorianFuerrutter/genQC).

## Features

### **Dataset Generation**
- Generate quantum circuit datasets with customizable gate sets
- Support for Clifford gates, universal gate sets, and parameterized gates
- Multiple conditioning types (SRV, Unitary)
- Preset configurations for common scenarios
- Parallel generation for large datasets

### **Model Training** 
- Comprehensive training pipeline for diffusion models
- Support for compilation-aware diffusion models
- Configurable architectures and hyperparameters
- Experiment tracking and logging
- Model management and versioning

### **Evaluation & Testing**
- Comprehensive evaluation metrics (fidelity, circuit properties, diversity)
- Statistical analysis and comparison tools
- Automated plot generation
- Batch evaluation for multiple models
- Export results in multiple formats

### **Configuration Management**
- YAML-based configuration system
- Preset configurations for quick start
- Hierarchical config merging
- Validation and error checking

## Installation

### Prerequisites

1. **Install genQC** (required dependency):
   ```bash
   pip install genQC
   ```

2. **Install other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Generate a Dataset

```python
from quantum_diffusion import DatasetGenerator

generator = DatasetGenerator()
generator.generate_dataset(
    gate_set=['h', 'cx', 'cz', 's'],
    num_qubits=3,
    num_samples=1000,
    condition_type="BOTH",
    output_path="./my_dataset"
)
```

Or use the command line:
```bash
qd-generate --preset clifford_3q_unitary --num-samples 1000 --output ./my_dataset
```

This creates separate folders for SRV and UNITARY conditioning (for example `./my_dataset/srv` and `./my_dataset/unitary`).
```

### 2. Train a Diffusion Model

```python
from quantum_diffusion import DiffusionTrainer, DatasetLoader

# Load dataset
loader = DatasetLoader()
dataset = loader.load_dataset('./my_dataset/unitary')
dataloaders = loader.get_dataloaders(dataset, batch_size=32)

# Train model
trainer = DiffusionTrainer()
trainer.setup_model(dataset)
trainer.compile_model()
trainer.train(dataloaders, save_path='./my_model')
```

Or use the command line:
```bash
qd-train --dataset ./my_dataset/unitary --preset standard --epochs 20 --output ./my_model
```

### 3. Evaluate the Model

```python
from quantum_diffusion import Evaluator, ModelManager

# Load model
manager = ModelManager()
model = manager.load_model('./my_model')

# Evaluate
evaluator = Evaluator()
results = evaluator.evaluate_model(
    model, dataset, output_dir='./evaluation_results'
)
```

Or use the command line:
```bash
qd-evaluate --model ./my_model --dataset ./my_dataset/unitary --output ./evaluation_results
```

## Project Structure

```
quantum-diffusion/
├── src/quantum_diffusion/          # Main package source
│   ├── data/                       # Dataset generation and loading
│   ├── models/                     # Training and model management
│   ├── evaluation/                 # Evaluation and testing framework
│   └── utils/                      # Configuration and logging utilities
├── scripts/                        # Command-line scripts
├── configs/                        # Configuration files
├── notebooks/                      # Tutorial notebooks
├── tests/                          # Unit tests
└── docs/                          # Documentation
```

## Configuration System

The framework uses a hierarchical YAML configuration system:

### Dataset Configuration (`configs/datasets/`)
```yaml
gate_set: ['h', 'cx', 'cz', 's', 'x', 'y', 'z']
num_qubits: 3
num_samples: 1000
min_gates: 2
max_gates: 16
condition_type: "BOTH"
output_path: "./datasets/my_dataset"
```

When `condition_type` is `"BOTH"` (the default), datasets are saved under `output_path/srv` and `output_path/unitary`.

### Training Configuration (`configs/training/`)
```yaml
model:
  type: "QC_Compilation_UNet"
  params:
    model_features: [128, 128, 256]
    t_emb_size: 256
    cond_emb_size: 512

training:
  learning_rate: 0.0001
  num_epochs: 20
  batch_size: 32
  
scheduler:
  type: "DDIMScheduler"
  params:
    num_train_timesteps: 1000
```

### Evaluation Configuration (`configs/evaluation/`)
```yaml
metrics:
  fidelity: true
  circuit_properties: true
  statistical_analysis: true
  diversity_metrics: true

generation:
  num_samples: 100
  num_inference_steps: 50
```

## Available Presets

### Dataset Presets
- `clifford_3q_unitary`: 3-qubit Clifford circuits with unitary conditioning
- `clifford_4q_srv`: 4-qubit Clifford circuits with SRV conditioning
- `universal_3q`: 3-qubit universal gate set

### Training Presets
- `quick_test`: Fast training for testing (5 epochs, small model)
- `standard`: Standard training configuration (20 epochs)
- `large_model`: Large model for better performance (50 epochs)

### Evaluation Presets
- `comprehensive`: All evaluation metrics with plots
- `basic`: Essential metrics only

## Command Line Interface

The package provides three main commands:

### Generate Dataset
```bash
qd-generate [OPTIONS]

Options:
  --preset PRESET           Use preset configuration
  --config PATH            Custom config file
  --gate-set GATE [GATE...] List of gates
  --num-qubits INT         Number of qubits
  --num-samples INT        Number of samples
  --condition-type TYPE    Conditioning type (SRV/UNITARY)
  --output PATH           Output directory
```

### Train Model
```bash
qd-train [OPTIONS]

Options:
  --dataset PATH          Dataset directory (required)
  --preset PRESET         Training preset
  --config PATH          Custom config file
  --epochs INT           Number of epochs
  --batch-size INT       Batch size
  --learning-rate FLOAT  Learning rate
  --output PATH          Model output directory
```

### Evaluate Model
```bash
qd-evaluate [OPTIONS]

Options:
  --model PATH           Model path (required)
  --dataset PATH         Test dataset
  --config PATH          Evaluation config
  --metrics METRIC [METRIC...]  Specific metrics
  --num-samples INT      Number of samples to generate
  --output PATH          Results output directory
```

## API Reference

### Core Classes

#### `DatasetGenerator`
Generate quantum circuit datasets with customizable parameters.

#### `DatasetLoader` 
Load and manage existing datasets, create PyTorch dataloaders.

#### `DiffusionTrainer`
Train diffusion models on quantum circuit data.

#### `ModelManager`
Manage multiple trained models, versioning, and storage.

#### `Evaluator`
Comprehensive evaluation of trained models with multiple metrics.

#### `ConfigManager`
Manage configuration files, presets, and parameter validation.

## Examples

### Create Example Project
```python
import quantum_diffusion
quantum_diffusion.create_example_project("./my_quantum_project")
```

### Batch Processing
```python
from quantum_diffusion import DatasetGenerator

# Generate multiple datasets
configs = [
    {"gate_set": ['h', 'cx'], "num_qubits": 2, "num_samples": 500},
    {"gate_set": ['h', 'cx', 'cz'], "num_qubits": 3, "num_samples": 1000}
]

generator = DatasetGenerator()
results = generator.generate_multiple_datasets(configs)
```

### Model Comparison
```python
from quantum_diffusion.evaluation import evaluate_multiple_models

results = evaluate_multiple_models(
    model_paths=["./model1", "./model2", "./model3"],
    test_dataset=test_data,
    output_base_dir="./comparison_results"
)
```
