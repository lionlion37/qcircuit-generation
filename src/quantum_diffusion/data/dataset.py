"""Dataset generation and loading utilities for quantum circuits."""

import os
import time
import sys
import torch
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional, Union, Tuple, Any
import yaml
from tqdm import tqdm

# Assuming genQC imports (adjust paths as needed)
from src.my_genQC.platform.circuits_generation import generate_circuit_dataset, CircuitConditionType
from src.my_genQC.platform.simulation import Simulator, CircuitBackendType
from src.my_genQC.platform.tokenizer.circuits_tokenizer import CircuitTokenizer
from src.my_genQC.models.config_model import ConfigModel
from src.my_genQC.dataset import circuits_dataset
from src.my_genQC.utils.misc_utils import infer_torch_device

from ..utils.config import ConfigManager
from ..utils.logging import Logger


class DatasetGenerator:
    """Generate quantum circuit datasets with various configurations."""
    
    def __init__(self, config_path: Optional[str] = None, device: Optional[str] = None):
        """Initialize the dataset generator.
        
        Args:
            config_path: Path to configuration file
            device: Device to use for computation ('cpu' or 'cuda')
        """
        self.device = device or infer_torch_device() if 'infer_torch_device' in globals() else 'cpu'
        self.config_manager = ConfigManager()
        self.logger = Logger(__name__)
        
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = {}
            
    def generate_dataset(self, 
                        gate_set: List[str],
                        num_qubits: int,
                        num_samples: int,
                        min_gates: int = 2,
                        max_gates: int = 16,
                        condition_type: str = "UNITARY",
                        output_path: str = "./dataset",
                        **kwargs) -> Dict:
        """Generate a quantum circuit dataset.
        
        Args:
            gate_set: List of quantum gates to include
            num_qubits: Number of qubits in circuits
            num_samples: Number of circuits to generate
            min_gates: Minimum number of gates per circuit
            max_gates: Maximum number of gates per circuit
            condition_type: Type of conditioning ("SRV" or "UNITARY")
            output_path: Path to save the dataset
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing dataset metadata
        """
        self.logger.info(f"Generating dataset with {num_samples} samples, {num_qubits} qubits")
        
        # Create output directory
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Setup vocabulary and tokenizer
        vocabulary = {gate: idx for gate, idx in zip(gate_set, range(len(gate_set)))}
        
        try:
            simulator = Simulator(CircuitBackendType.QISKIT)
            tokenizer = CircuitTokenizer(vocabulary)
            
            # Convert condition type
            condition = getattr(CircuitConditionType, condition_type.upper())
            
            # Generate dataset
            self.logger.info("Starting circuit generation...")
            tensors, ys, Us, params = generate_circuit_dataset(
                backend=simulator.backend,
                tokenizer=tokenizer,
                condition=condition,
                total_samples=num_samples,
                num_of_qubits=num_qubits,
                min_gates=min_gates,
                max_gates=max_gates,
                min_sub_gate_pool_cnt=2,
                fixed_sub_gate_pool=gate_set
            )
            
            # Setup parameters
            dataset_params = {
                "optimized": True,
                "dataset_to_gpu": self.device == "cuda",
                "random_samples": num_samples,
                "num_of_qubits": num_qubits,
                "min_gates": min_gates,
                "max_gates": max_gates,
                "gate_pool": gate_set,
                "max_params": 0,
                "pad_constant": len(vocabulary) + 1
            }
            
            # Set store_dict based on condition type
            if condition == CircuitConditionType.SRV:
                dataset_params["store_dict"] = {'x': 'tensor', 'y': 'numpy'}
            elif condition == CircuitConditionType.UNITARY:
                dataset_params["store_dict"] = {'x': 'tensor', 'y': 'numpy', 'U': 'tensor'}

            dataset = circuits_dataset.CircuitsConfigDataset(device=self.device, **dataset_params)
            dataset.x = tensors
            dataset.y = ys

            if condition == CircuitConditionType.SRV:
                mixed_dataset = dataset

            elif condition == CircuitConditionType.UNITARY:
                dataset.U = Us.float()
                datasets_list = [dataset]

                parameters = asdict(dataset.params_config)
                parameters["model_scale_factor"] = 4

                mixed_dataset, _ = circuits_dataset.MixedCircuitsConfigDataset.from_datasets(
                    datasets_list,
                    balance_maxes=[int(1e8)],
                    pad_constant=dataset_params["pad_constant"],
                    device=self.device,
                    bucket_batch_size=-1,
                    max_samples=[int(1e8)],
                    **parameters
                )

            # Save dataset
            dataset_path = os.path.join(output_path, "dataset", "ds")
            config_path = os.path.join(output_path, "config.yaml")

            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            mixed_dataset.save_dataset(save_path=dataset_path, config_path=config_path)

            self.logger.info(f"Dataset saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating dataset: {e}")
            raise
    
    def generate_multiple_datasets(self, configs: List[Dict]) -> List[Dict]:
        """Generate multiple datasets from a list of configurations.
        
        Args:
            configs: List of dataset configuration dictionaries
            
        Returns:
            List of metadata dictionaries for each dataset
        """
        results = []
        for i, config in enumerate(configs):
            self.logger.info(f"Generating dataset {i+1}/{len(configs)}")
            result = self.generate_dataset(**config)
            results.append(result)
        return results


class DatasetLoader:
    """Load and manage quantum circuit datasets."""
    
    def __init__(self, config: Dict[str, Dict[str, Any]], device: Optional[str] = None):
        """Initialize the dataset loader.
        
        Args:
            config: Configuration for training procedure, text encoder and other settings
            device: Device to use for computation ('cpu' or 'cuda')
        """
        self.device = device or infer_torch_device() if 'infer_torch_device' in globals() else 'cpu'
        self.logger = Logger(__name__)
        self.config = config
    
    def load_dataset(self, dataset_path: str, **kwargs):
        """Load a saved quantum circuit dataset.
        
        Args:
            dataset_path: Path to the saved dataset
            **kwargs: Additional loading parameters
            
        Returns:
            Loaded dataset object
        """
        try:
            config_path = os.path.join(dataset_path, "config.yaml")
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            # Load dataset using genQC
            dataset = circuits_dataset.MixedCircuitsConfigDataset.from_config_file(
                config_path=config_path,
                device=self.device,
                save_path=os.path.join(dataset_path, "dataset", "ds"),
                **kwargs
            )

            self.vocabulary = {gate: idx for gate, idx in zip(dataset.gate_pool, range(len(dataset.gate_pool)))}
            self.tokenizer = CircuitTokenizer(self.vocabulary)

            # Setup text encoder
            time_stamp = time.strftime('%m/%d/%y %H:%M:%S', time.localtime())
            text_encoder_config = self.config["text_encoder"].copy()
            text_encoder_config["save_datetime"] = time_stamp
            text_encoder_config["target"] = f"genQC.models.frozen_open_clip.{text_encoder_config['type']}"

            self.text_encoder = ConfigModel.from_config(text_encoder_config, self.device)

            self.logger.info(f"Dataset loaded from {dataset_path}")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_dataloaders(self, dataset, batch_size: int = 32, split_ratio: float = 0.1, **kwargs):
        """Create data loaders from a dataset.
        
        Args:
            dataset: Dataset object
            batch_size: Batch size for data loaders
            split_ratio: Ratio for validation split
            **kwargs: Additional dataloader parameters
            
        Returns:
            Dictionary containing train and validation data loaders
        """
        try:
            # This would need to be implemented based on genQC's dataloader structure
            dataloaders = dataset.get_dataloaders(
                batch_size=batch_size,
                p_valid=split_ratio,
                text_encoder=self.text_encoder,
                **kwargs,
            )
            
            self.logger.info(f"Created dataloaders with batch size {batch_size}")
            return dataloaders
            
        except Exception as e:
            self.logger.error(f"Error creating dataloaders: {e}")
            raise
    
    def inspect_dataset(self, dataset_path: str) -> Dict:
        """Inspect a dataset and return metadata.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            Dictionary containing dataset metadata
        """
        try:
            config_path = os.path.join(dataset_path, "config.yaml")
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load dataset to get actual statistics
            dataset = self.load_dataset(dataset_path)
            
            metadata = {
                'config': config,
                'num_samples': len(dataset) if hasattr(dataset, '__len__') else 'Unknown',
                'dataset_path': dataset_path,
                'inspection_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error inspecting dataset: {e}")
            raise


# Predefined dataset configurations
PRESET_CONFIGS = {
    "clifford_3q_unitary": {
        "gate_set": ['h', 'cx', 'cz', 's', 'x', 'y', 'z'],
        "num_qubits": 3,
        "num_samples": 1000,
        "min_gates": 2,
        "max_gates": 16,
        "condition_type": "UNITARY"
    },
    "clifford_3q_srv": {
        "gate_set": ['h', 'cx', 'cz', 's', 'x', 'y', 'z'],
        "num_qubits": 3,
        "num_samples": 1000,
        "min_gates": 2,
        "max_gates": 16,
        "condition_type": "SRV"
    },
    "universal_4q": {
        "gate_set": ['h', 'cx', 'ry', 'rz', 'x', 'y', 'z'],
        "num_qubits": 4,
        "num_samples": 2000,
        "min_gates": 3,
        "max_gates": 20,
        "condition_type": "UNITARY"
    }
}
