"""Training and model management for diffusion training on quantum circuits."""

import os
import time
import torch
import copy
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml
from tqdm import tqdm
from omegaconf import  OmegaConf

from my_genQC.pipeline.compilation_diffusion_pipeline import DiffusionPipeline_Compilation
from my_genQC.pipeline.diffusion_pipeline import DiffusionPipeline
from my_genQC.scheduler.scheduler_ddim import DDIMScheduler
from my_genQC.models.unet_qc import QC_Compilation_UNet, QC_Cond_UNet
from my_genQC.models.config_model import ConfigModel
from my_genQC.models.unitary_encoder import Unitary_encoder_config
from my_genQC.platform.tokenizer.circuits_tokenizer import CircuitTokenizer
from my_genQC.utils.misc_utils import infer_torch_device
from my_genQC.pipeline.callbacks import Callback

from ..utils.config import ConfigManager
from ..utils.logging import Logger


class WandbLoggingCallback(Callback):
    """Log training progress to Weights & Biases."""
    order = 100

    def __init__(self, run):
        self.run = run

    def after_epoch(self, pipeline):
        if not self.run or not hasattr(pipeline, "out_metric_dict"):
            return
        metrics = {k: float(v) for k, v in pipeline.out_metric_dict.items()}
        metrics["epoch"] = pipeline.epoch + 1
        self.run.log(metrics, step=pipeline.epoch + 1)


class DiffusionTrainer:
    """Train diffusion training for quantum circuit generation."""
    
    def __init__(self, config: Dict[str, Dict[str, Any]] = None, device: Optional[str] = None):
        """Initialize the diffusion trainer.
        
        Args:
            config: training configuration
            device: Device to use for training ('cpu' or 'cuda')
        """
        self.device = device or infer_torch_device() if 'infer_torch_device' in globals() else 'cpu'
        self.config_manager = ConfigManager()
        self.logger = Logger(__name__)
        self.config = config

        self.pipeline = None
        self.model = None
        self.scheduler = None
        self.wandb_run = None


    def setup_model(self, dataset, text_encoder, tokenizer: Optional = None) -> None:
        """Setup the diffusion model and related components.
        
        Args:
            dataset: Training dataset
            tokenizer: Circuit tokenizer (will be created if None)
        """
        try:
            self.logger.info("Setting up diffusion model...")
            
            # Create tokenizer if not provided
            if tokenizer is None:
                # Extract vocabulary from dataset config or create default
                gate_pool = getattr(dataset, 'gate_pool', ['h', 'cx', 'cz', 's', 'x', 'y', 'z'])
                vocabulary = {gate: idx for gate, idx in zip(gate_pool, range(len(gate_pool)))}
                tokenizer = CircuitTokenizer(vocabulary)
            
            # Setup model configuration
            model_config = copy.deepcopy(dict(self.config["model"]))
            model_params = model_config["params"]
            model_type = model_config.get("type", "QC_Compilation_UNet")
            self.uses_unitary_conditioning = model_type == "QC_Compilation_UNet"
            
            # Update num_clrs based on tokenizer
            model_params["num_clrs"] = len(tokenizer.vocabulary) + 2  # +1 for background, +1 for padding
            
            # Create unitary encoder config if needed
            if "unitary_encoder_config" in model_params:
                unitary_config = model_params["unitary_encoder_config"]
                if isinstance(unitary_config, dict):
                    model_params["unitary_encoder_config"] = Unitary_encoder_config(**unitary_config)
            
            # Setup timestamp
            time_stamp = time.strftime('%m/%d/%y %H:%M:%S', time.localtime())
            model_config["save_datetime"] = time_stamp
            model_config["target"] = f"genQC.models.unet_qc.{model_config['type']}"
            
            # Create model
            if self.uses_unitary_conditioning:
                self.model = QC_Compilation_UNet.from_config(model_config, self.device, freeze=False)
            else:
                self.model = QC_Cond_UNet.from_config(model_config, self.device, freeze=False)
            
            # Setup scheduler
            scheduler_config = copy.deepcopy(dict(self.config["scheduler"]))
            scheduler_config["params"] = dict(scheduler_config["params"])
            scheduler_config["params"]["device"] = self.device
            scheduler_config["target"] = f"genQC.scheduler.scheduler_ddim.{scheduler_config['type']}"
            
            self.scheduler = DDIMScheduler.from_config(scheduler_config, self.device)
            
            # Create embedder (same as model for compilation pipeline)
            embedder = copy.deepcopy(self.model)  # TODO: add option to use different model
            
            # Setup pipeline
            training_config = self.config["training"]
            
            pipeline_cls = DiffusionPipeline_Compilation if self.uses_unitary_conditioning else DiffusionPipeline
            self.pipeline = pipeline_cls(
                scheduler=self.scheduler,
                model=self.model,
                text_encoder=text_encoder,
                embedder=embedder,
                device=self.device,
                enable_guidance_train=training_config.get("enable_guidance_train", True),
                guidance_train_p=training_config.get("guidance_train_p", 0.1),
                cached_text_enc=training_config.get("cached_text_enc", True)
            )
            
            self.logger.info("Model setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up model: {e}")
            raise
    
    def compile_model(self) -> None:
        """Compile the model for training."""
        try:
            training_config = self.config["training"]
            
            # Get optimizer and loss functions
            optimizer_name = training_config.get("optimizer", "Adam")
            loss_name = training_config.get("loss", "MSELoss")
            
            optimizer_fn = getattr(torch.optim, optimizer_name)
            loss_fn = getattr(torch.nn, loss_name)
            
            self.pipeline.compile(
                optim_fn=optimizer_fn,
                loss_fn=loss_fn,
                lr=training_config.get("learning_rate", 1e-4)
            )
            
            self.logger.info("Model compiled for training")
            
        except Exception as e:
            self.logger.error(f"Error compiling model: {e}")
            raise

    def _setup_wandb(self):
        training_config = self.config.get("training", {}) if self.config else {}
        wandb_cfg = training_config.get("wandb", {})
        enabled = wandb_cfg.get("enable", False) or wandb_cfg.get("enabled", False)
        if not enabled:
            return None
        try:
            import wandb
        except ImportError:
            self.logger.warning("wandb logging requested but the package is not installed.")
            return None

        general_cfg = self.config.get("general", {}) if self.config else {}
        project = wandb_cfg.get("project", "qcircuit-generation")
        run_name = wandb_cfg.get("run_name", general_cfg.get("experiment_name"))
        return wandb.init(project=project, name=run_name, config=dict(self.config))
    
    def train(self, dataloaders, save_path: Optional[str] = None) -> Dict:
        """Train the diffusion model.
        
        Args:
            dataloaders: Training and validation data loaders
            save_path: Path to save the trained model
            
        Returns:
            Training history and metrics
        """
        try:
            training_config = dict(self.config["training"])
            num_epochs = training_config.get("num_epochs", 10)
            
            self.logger.info(f"Starting training for {num_epochs} epochs...")
            
            self.wandb_run = self._setup_wandb()
            if self.wandb_run:
                cbs = list(self.pipeline.cbs) if getattr(self.pipeline, "cbs", None) else []
                cbs.append(WandbLoggingCallback(self.wandb_run))
                self.pipeline.cbs = cbs

            # Train the model
            history = self.pipeline.fit(
                num_epochs=num_epochs,
                data_loaders=dataloaders,
                ckpt_interval=self.config.training.ckpt_interval,
                ckpt_path=self.config.training.ckpt_path,
            )
            
            # Save model if path provided
            if save_path:
                self.save_model(save_path)
            
            self.logger.info("Training completed successfully")
            return history
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
        finally:
            if self.wandb_run:
                self.wandb_run.finish()
    
    def save_model(self, save_path: str) -> None:
        """Save the trained model and configuration.
        
        Args:
            save_path: Path to save the model
        """
        try:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            # Save model and config to the same directory
            self.pipeline.text_encoder.params_config.version = "cloob"
            self.pipeline.store_pipeline(config_path=save_path + "/", save_path=save_path + "/")
            
            # Save configuration
            config_path = os.path.join(save_path, "training_config.yaml")
            # self.config_manager.save_config(self.config, config_path)
            OmegaConf.save(self.config, config_path)
            
            # Save metadata
            metadata = {
                'model_type': self.config["model"]["type"],
                'training_epochs': self.config["training"]["num_epochs"],
                'save_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device_used': self.device
            }
            
            metadata_path = os.path.join(save_path, "metadata.yaml")
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            
            self.logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str) -> None:
        """Load a previously trained model.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            # Load configuration
            model_folder = Path(model_path).parent
            config_path = os.path.join(model_folder, "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            
            # Load model weights
            if os.path.exists(model_path):
                if self.model is None:
                    self.logger.warning("Model not initialized. Call setup_model first.")
                    return
                
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            
            self.logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise


class ModelManager:
    """Manage multiple diffusion training and their configurations."""
    
    def __init__(self, models_dir: str = "./training"):
        """Initialize the model manager.
        
        Args:
            models_dir: Directory to store training
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(__name__)
        
        self.models = {}  # Store loaded training
    
    def register_model(self, name: str, trainer: DiffusionTrainer) -> None:
        """Register a trained model.
        
        Args:
            name: Model name/identifier
            trainer: Trained diffusion trainer
        """
        self.models[name] = trainer
        self.logger.info(f"Model '{name}' registered")
    
    def save_model(self, name: str, trainer: DiffusionTrainer) -> str:
        """Save a model with a given name.
        
        Args:
            name: Model name/identifier
            trainer: Diffusion trainer to save
            
        Returns:
            Path where the model was saved
        """
        model_path = self.models_dir / name
        trainer.save_model(str(model_path))
        self.register_model(name, trainer)
        return str(model_path)
    
    def load_model(self, model_path: str, device: Optional[str] = None) -> DiffusionTrainer:
        """Load a previously saved model.
        
        Args:
            model_path: path to model
            device: Device to load model on
            
        Returns:
            Loaded diffusion trainer
        """
        model_path = Path(model_path)
        name = model_path.name

        if not model_path.exists():
            raise FileNotFoundError(f"Model '{name}' not found at {model_path}")
        
        trainer = DiffusionTrainer(device=device)
        trainer.load_model(str(model_path))
        self.register_model(name, trainer)
        
        return trainer
    
    def list_models(self) -> List[str]:
        """List available training.
        
        Returns:
            List of model names
        """
        models = []
        
        # From memory
        models.extend(self.models.keys())
        
        # From disk
        if self.models_dir.exists():
            for path in self.models_dir.iterdir():
                if path.is_dir() and path.name not in models:
                    models.append(path.name)
        
        return sorted(models)
    
    def get_model_info(self, name: str) -> Dict:
        """Get information about a model.
        
        Args:
            name: Model name/identifier
            
        Returns:
            Dictionary containing model information
        """
        model_path = self.models_dir / name
        info = {'name': name, 'path': str(model_path)}
        
        # Try to load metadata
        metadata_path = model_path / "metadata.yaml"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            info.update(metadata)
        
        return info
    
    def delete_model(self, name: str) -> None:
        """Delete a model.
        
        Args:
            name: Model name/identifier
        """
        # Remove from memory
        if name in self.models:
            del self.models[name]
        
        # Remove from disk
        model_path = self.models_dir / name
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path)
            self.logger.info(f"Model '{name}' deleted")
