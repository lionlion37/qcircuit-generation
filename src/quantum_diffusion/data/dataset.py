"""Dataset generation and loading utilities for quantum circuits."""

import os
import time
import sys
import copy
import torch
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional, Union, Tuple, Any, Iterable
import yaml
from tqdm import tqdm

# Assuming genQC imports (adjust paths as needed)
from my_genQC.platform.circuits_generation import generate_circuit_dataset, CircuitConditionType
from my_genQC.platform.simulation import Simulator, CircuitBackendType
from my_genQC.platform.tokenizer.circuits_tokenizer import CircuitTokenizer
from my_genQC.models.config_model import ConfigModel
from my_genQC.dataset import circuits_dataset
from my_genQC.utils.misc_utils import infer_torch_device

from quantum_diffusion.utils.config import ConfigManager
from quantum_diffusion.utils.logging import Logger

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
            
    @staticmethod
    def _normalize_condition_types(condition_type: Union[str, Iterable[str], CircuitConditionType, Iterable[CircuitConditionType], None]) -> List[CircuitConditionType]:
        """Normalize condition input into a list of CircuitConditionType enums."""
        default_conditions = [CircuitConditionType.SRV, CircuitConditionType.UNITARY]
        
        if condition_type is None:
            return default_conditions
        
        if isinstance(condition_type, CircuitConditionType):
            raw_values = [condition_type]
        elif isinstance(condition_type, str):
            raw_values = [condition_type]
        elif isinstance(condition_type, Iterable):
            raw_values = list(condition_type)
        else:
            raise TypeError("condition_type must be a string, enum, iterable, or None")
        
        normalized: List[CircuitConditionType] = []
        for value in raw_values:
            if isinstance(value, CircuitConditionType):
                enum_value = value
                expanded = [enum_value]
            else:
                name = str(value).strip().upper()
                if name == "BOTH":
                    expanded = default_conditions
                else:
                    if not hasattr(CircuitConditionType, name):
                        raise ValueError(f"Unknown condition type '{value}'")
                    expanded = [getattr(CircuitConditionType, name)]
            
            for enum_value in expanded:
                if enum_value not in normalized:
                    normalized.append(enum_value)
        
        if not normalized:
            raise ValueError("No valid condition types provided")
        
        return normalized
    
    def generate_dataset(self,
                        gate_set: List[str],
                        num_qubits: int,
                        num_samples: int,
                        min_gates: int = 2,
                        max_gates: int = 16,
                        backbone: str = "qiskit",
                        optimized: bool = True,
                        n_jobs: int = 1,
                        condition_type: Union[str, List[str], CircuitConditionType, List[CircuitConditionType]] = "SRV",
                        output_path: str = "./datasets",
                        **kwargs) -> Dict[str, Dict[str, Union[str, int]]]:
        """Generate a quantum circuit dataset.
        
        Args:
            gate_set: List of quantum gates to include
            num_qubits: Number of qubits in circuits
            num_samples: Number of circuits to generate
            min_gates: Minimum number of gates per circuit
            max_gates: Maximum number of gates per circuit
            backbone: Simulation backend ("qiskit", etc.)
            optimized: Whether to optimize circuits
            condition_type: Conditioning specification ("SRV", "UNITARY", "BOTH", or a list)
            output_path: Path to save the dataset
            **kwargs: Additional parameters
            
        Returns:
            Dictionary keyed by condition name with dataset metadata
        """
        self.logger.info(f"Generating dataset with {num_samples} samples, {num_qubits} qubits")
        
        vocabulary = {gate: idx for gate, idx in zip(gate_set, range(len(gate_set)))}
        output_root = Path(output_path)
        output_root.mkdir(parents=True, exist_ok=True)

        if backbone == "qiskit":
            backend_type = CircuitBackendType.QISKIT
        elif backbone == "quditkit":
            backend_type = CircuitBackendType.QUDITKIT
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'")
        
        try:
            simulator = Simulator(backend=backend_type)
            tokenizer = CircuitTokenizer(vocabulary)
            target_conditions = self._normalize_condition_types(condition_type)
            multi_condition = len(target_conditions) > 1
            
            results: Dict[str, Dict[str, Union[str, int]]] = {}
            
            for condition in target_conditions:
                condition_name = condition.name
                condition_output = output_root / condition_name.lower() if multi_condition else output_root
                condition_output.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"Starting circuit generation for {condition_name}...")
                tensors, ys, Us, _params = generate_circuit_dataset(
                    backend=simulator.backend,
                    tokenizer=tokenizer,
                    condition=condition,
                    total_samples=num_samples,
                    num_of_qubits=num_qubits,
                    min_gates=min_gates,
                    max_gates=max_gates,
                    min_sub_gate_pool_cnt=2,
                    fixed_sub_gate_pool=gate_set,
                    optimized=optimized,
                    post_randomize_params=False,  # TODO: change when switching to parameterized circuits
                    n_jobs=n_jobs,
                )

                dataset_params = {
                    "optimized": optimized,
                    "dataset_to_gpu": self.device == "cuda",
                    "random_samples": num_samples,
                    "num_of_qubits": num_qubits,
                    "min_gates": min_gates,
                    "max_gates": max_gates,
                    "gate_pool": gate_set,
                    "max_params": 0,
                    "pad_constant": len(vocabulary) + 1,
                    "store_dict": {'x': 'tensor', 'y': 'numpy'} if condition == CircuitConditionType.SRV else {'x': 'tensor', 'y': 'numpy', 'U': 'tensor'}
                }
                
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
                
                dataset_path = condition_output / "dataset" / "ds"
                config_path = condition_output / "config.yaml"

                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.parent.mkdir(parents=True, exist_ok=True)

                mixed_dataset.save_dataset(save_path=str(dataset_path), config_path=str(config_path))

                self.logger.info(f"{condition_name} dataset saved to {condition_output}")
                results[condition_name] = {
                    "condition": condition_name,
                    "output_path": str(condition_output),
                    "config_path": str(config_path),
                    "num_samples": int(tensors.shape[0]),
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating dataset: {e}")
            raise
    
    def generate_multiple_datasets(self, configs: List[Dict]) -> List[Dict]:
        """Generate multiple datasets from a list of configurations.
        
        Args:
            configs: List of dataset configuration dictionaries
            
        Returns:
            List of metadata dictionaries (per condition) for each config
        """
        results = []
        for i, config in enumerate(configs):
            self.logger.info(f"Generating dataset {i+1}/{len(configs)}")
            result = self.generate_dataset(backbone=Quantum, **config)
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
        self.config = dict(config)


    def _load_embedder(self, dataset):
        self.vocabulary = {gate: idx for gate, idx in zip(dataset.gate_pool, range(len(dataset.gate_pool)))}
        self.tokenizer = CircuitTokenizer(self.vocabulary)

        time_stamp = time.strftime('%m/%d/%y %H:%M:%S', time.localtime())
        text_encoder_config = copy.deepcopy(dict(self.config["text_encoder"]))
        text_encoder_config["save_datetime"] = time_stamp

        target = text_encoder_config.get("target")
        if not target:
            module_path = text_encoder_config.pop("module", "my_genQC.models.frozen_open_clip")
            encoder_type = text_encoder_config.get("type")
            if not encoder_type:
                raise ValueError("Text encoder config requires 'type' or explicit 'target'.")
            target = f"{module_path}.{encoder_type}"

        text_encoder_config["target"] = target

        text_encoder = ConfigModel.from_config(text_encoder_config, self.device)

        return text_encoder


    def _load_single_dataset(self, dataset_path: str, load_embedder: bool = True, **kwargs):
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

            with open(config_path, 'r') as cfg_file:
                cfg_data = yaml.safe_load(cfg_file)
            target = cfg_data.get("target", "")
            if target.endswith("MixedCircuitsConfigDataset"):
                dataset_cls = circuits_dataset.MixedCircuitsConfigDataset
            else:
                dataset_cls = circuits_dataset.CircuitsConfigDataset

            dataset = dataset_cls.from_config_file(
                config_path=config_path,
                device=self.device,
                save_path=os.path.join(dataset_path, "dataset", "ds"),
                **kwargs
            )

            if load_embedder:
                self.text_encoder = self._load_embedder(dataset=dataset)

            self.logger.info(f"Dataset loaded from {dataset_path}")
            return dataset

        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise


    def load_dataset(self, dataset_path: str, load_embedder: bool = True, **kwargs):
        # Arguments used only when stitching multiple per-qubit datasets together.
        combine_bucket_batch_size = kwargs.pop("bucket_batch_size", -1)
        combine_model_scale_factor = kwargs.pop("model_scale_factor", 4)
        combine_balance_max = kwargs.pop("balance_max", int(1e8))
        combine_max_samples = kwargs.pop("max_samples", int(1e8))

        if "dataset" in os.listdir(dataset_path):
            self.logger.info("Detected preprocessed dataset. Loading directly...")
            dataset = self._load_single_dataset(dataset_path=dataset_path, load_embedder=load_embedder, **kwargs)

        else:  # combine multiple datasets with different numbers of qubits
            self.logger.info("Detected multiple datasets in dataset_path. Loading all and combining them...")
            datasets = []
            parent_dir = dataset_path

            for dataset_name in sorted(os.listdir(parent_dir)):
                dataset_dir = os.path.join(parent_dir, dataset_name)
                if not os.path.isdir(dataset_dir):
                    continue

                dataset = self._load_single_dataset(dataset_dir, load_embedder)

                load_embedder = False  # only load embedder once

                if self.device == torch.device("cuda"):
                    dataset.dataset_to_gpu = True
                datasets.append(dataset)

            if not datasets:
                raise FileNotFoundError(f"No dataset directories found under {parent_dir}")

            dataset = self.combine_datasets(
                                datasets,
                                model_scale_factor=combine_model_scale_factor,
                                balance_maxes=[combine_balance_max] * len(datasets),
                                pad_constant=len(datasets[0].gate_pool) + 1,
                                device=self.device,
                                bucket_batch_size=combine_bucket_batch_size,
                                max_samples=[combine_max_samples] * len(datasets),
                                )

        return dataset


    def combine_datasets(self, datasets: List[Any], **kwargs):
        """Combine multiple datasets into a mixed dataset.

        Args:
            datasets: List of dataset objects to combine
            **kwargs: Additional parameters for combining

        Returns:
            Combined mixed dataset object
        """
        try:
            dataset_params = asdict(datasets[0].params_config)
            mixed_dataset, _ = circuits_dataset.MixedCircuitsConfigDataset.from_datasets(
                datasets,
                **dataset_params,
                **kwargs
            )
            self.logger.info("Datasets combined into a mixed dataset")
            return mixed_dataset

        except Exception as e:
            self.logger.error(f"Error combining datasets: {e}")
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
        "num_qubits": 5,
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
