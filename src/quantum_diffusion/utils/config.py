"""Configuration management utilities."""

import os
import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
from pathlib import Path
from typing import Dict, Any, Optional, Union
from omegaconf import OmegaConf, DictConfig


class ConfigManager:
    """Manage configuration files and settings."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("conf")
        self.yaml_saver = YAML()
        
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return config
    
    def save_config(self, config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_path: Path where to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            force_flow_style_lists(config)
            self.yaml_saver.dump(config, f)
    
    def merge_configs(self, base_config: Dict, override_config: Dict) -> Dict:
        """Merge two configurations with override taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        base_cfg = OmegaConf.create(base_config)
        override_cfg = OmegaConf.create(override_config)
        
        merged = OmegaConf.merge(base_cfg, override_cfg)
        return OmegaConf.to_container(merged, resolve=True)
    
    def get_preset_config(self, preset_name: str, config_type: str) -> Dict[str, Any]:
        """Get a preset configuration.
        
        Args:
            preset_name: Name of the preset
            config_type: Type of configuration (dataset, training, evaluation)
            
        Returns:
            Preset configuration dictionary
        """
        preset_configs = {
            "training": {
                "quick_test": {
                    "model": {
                        "type": "QC_Compilation_UNet",
                        "params": {
                            "model_features": [64, 64, 128],
                            "t_emb_size": 128,
                            "cond_emb_size": 256
                        }
                    },
                    "training": {
                        "learning_rate": 1e-3,
                        "num_epochs": 5,
                        "batch_size": 16
                    }
                },
                "standard": {
                    "model": {
                        "type": "QC_Compilation_UNet",
                        "params": {
                            "model_features": [128, 128, 256],
                            "t_emb_size": 256,
                            "cond_emb_size": 512
                        }
                    },
                    "training": {
                        "learning_rate": 1e-4,
                        "num_epochs": 20,
                        "batch_size": 32
                    }
                }
            },
            "evaluation": {
                "comprehensive": {
                    "metrics": {
                        "fidelity": True,
                        "circuit_properties": True,
                        "statistical_analysis": True,
                        "diversity_metrics": True
                    },
                    "generation": {
                        "num_samples": 100,
                        "num_inference_steps": 50
                    },
                    "output": {
                        "save_results": True,
                        "create_plots": True
                    }
                },
                "basic": {
                    "metrics": {
                        "circuit_properties": True,
                        "statistical_analysis": True
                    },
                    "generation": {
                        "num_samples": 50
                    },
                    "output": {
                        "save_results": True,
                        "create_plots": False
                    }
                }
            }
        }
        
        if config_type not in preset_configs:
            raise ValueError(f"Unknown config type: {config_type}")
        
        if preset_name not in preset_configs[config_type]:
            raise ValueError(f"Unknown preset '{preset_name}' for config type '{config_type}'")
        
        return preset_configs[config_type][preset_name]
    
    def list_presets(self, config_type: str) -> list:
        """List available presets for a configuration type.
        
        Args:
            config_type: Type of configuration
            
        Returns:
            List of available preset names
        """
        preset_configs = {
            "dataset": ["clifford_3q_unitary", "clifford_4q_srv"],
            "training": ["quick_test", "standard"],
            "evaluation": ["comprehensive", "basic"]
        }
        
        return preset_configs.get(config_type, [])
    
    def validate_config(self, config: Dict[str, Any], config_type: str) -> bool:
        """Validate a configuration dictionary.
        
        Args:
            config: Configuration to validate
            config_type: Type of configuration
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation rules for different config types
        validation_rules = {
            "dataset": {
                "required_keys": ["gate_set", "num_qubits", "num_samples"],
                "optional_keys": ["min_gates", "max_gates", "condition_type"]
            },
            "training": {
                "required_keys": ["model", "training"],
                "optional_keys": ["scheduler", "text_encoder"]
            },
            "evaluation": {
                "required_keys": ["metrics"],
                "optional_keys": ["generation", "comparison", "output"]
            }
        }
        
        if config_type not in validation_rules:
            return True  # Unknown type, assume valid
        
        rules = validation_rules[config_type]
        
        # Check required keys
        for key in rules["required_keys"]:
            if key not in config:
                return False
        
        # Additional validation logic could go here
        return True
    
    def create_experiment_config(self, base_config: Dict, experiment_params: Dict) -> Dict:
        """Create an experiment configuration by modifying a base configuration.
        
        Args:
            base_config: Base configuration
            experiment_params: Parameters to modify or add
            
        Returns:
            New experiment configuration
        """
        return self.merge_configs(base_config, experiment_params)
    
    def setup_config_directory(self, config_dir: Optional[str] = None) -> Path:
        """Setup configuration directory with example conf.
        
        Args:
            config_dir: Directory to setup (uses default if None)
            
        Returns:
            Path to config directory
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        
        self.config_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.config_dir / "datasets").mkdir(exist_ok=True)
        (self.config_dir / "training").mkdir(exist_ok=True)
        (self.config_dir / "evaluation").mkdir(exist_ok=True)
        
        # Create example conf if they don't exist
        self._create_example_configs()
        
        return self.config_dir
    
    def _create_example_configs(self) -> None:
        """Create example configuration files."""
        # Example dataset config
        dataset_config = {
            "gate_set": ['h', 'cx', 'cz', 's', 'x', 'y', 'z'],
            "num_qubits": 3,
            "num_samples": 1000,
            "min_gates": 2,
            "max_gates": 16,
            "condition_type": "BOTH",
            "output_path": "./datasets/example_dataset"
        }
        
        dataset_path = self.config_dir / "datasets" / "example.yaml"
        if not dataset_path.exists():
            self.save_config(dataset_config, dataset_path)
        
        # Example training config
        training_config = {
            "model": {
                "type": "QC_Compilation_UNet",
                "params": {
                    "model_features": [128, 128, 256],
                    "clr_dim": 8,
                    "t_emb_size": 256,
                    "cond_emb_size": 512,
                    "num_heads": [8, 8, 2],
                    "num_res_blocks": [2, 2, 4],
                    "transformer_depths": [1, 2, 1]
                }
            },
            "training": {
                "learning_rate": 1e-4,
                "optimizer": "Adam",
                "loss": "MSELoss",
                "num_epochs": 20,
                "batch_size": 32
            },
            "scheduler": {
                "type": "DDIMScheduler",
                "params": {
                    "num_train_timesteps": 1000,
                    "beta_start": 0.0001,
                    "beta_end": 0.02,
                    "beta_schedule": "cos_alpha"
                }
            }
        }
        
        training_path = self.config_dir / "training" / "example.yaml"
        if not training_path.exists():
            self.save_config(training_config, training_path)
        
        # Example evaluation config
        evaluation_config = {
            "metrics": {
                "fidelity": True,
                "circuit_properties": True,
                "statistical_analysis": True,
                "diversity_metrics": True
            },
            "generation": {
                "num_samples": 100,
                "num_inference_steps": 50,
                "guidance_scale": 1.0
            },
            "comparison": {
                "compare_to_training": True,
                "compare_to_random": True
            },
            "output": {
                "save_results": True,
                "create_plots": True,
                "verbose": True
            }
        }
        
        evaluation_path = self.config_dir / "evaluation" / "example.yaml"
        if not evaluation_path.exists():
            self.save_config(evaluation_config, evaluation_path)


def force_flow_style_lists(d):
    for key, value in d.items():
        if isinstance(value, list):
            d[key] = CommentedSeq(value)
            d[key].fa.set_flow_style()  # sets flow style / inline list
        elif isinstance(value, dict):
            force_flow_style_lists(value)
