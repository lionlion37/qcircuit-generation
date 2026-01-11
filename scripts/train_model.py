#!/usr/bin/env python3
"""Script for training diffusion training on quantum circuits."""

import argparse
import sys
import os
from pathlib import Path
import time
import hydra
import torch

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent /"src"))

from quantum_diffusion.data import DatasetLoader
from quantum_diffusion.training import DiffusionTrainer, ModelManager
from quantum_diffusion.utils import Logger, ExperimentLogger, setup_logging

from my_genQC.utils.misc_utils import infer_torch_device
import my_genQC


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    print("my_genQC:", my_genQC.__file__)

    cfg = cfg["training"]

    if cfg.general.device == "auto":
        device = infer_torch_device()
    else:
        device = cfg.general.device
    
    # Setup logging
    setup_logging(
        log_level="DEBUG" if cfg.general.verbose else "INFO",
        console_output=True
    )
    logger = Logger(__name__)
    
    # Setup experiment logging
    experiment_name = cfg.general.experiment_name or f"training_{int(time.time())}"
    exp_logger = ExperimentLogger(experiment_name)

    try:
        # Load dataset
        logger.info(f"Loading dataset from {cfg.general.dataset}")

        dataset_loader = DatasetLoader(device=device, config=cfg)
        dataset = dataset_loader.load_dataset(cfg.general.dataset)

        # Create data loaders
        batch_size = cfg.training.batch_size or 32
        dataloaders = dataset_loader.get_dataloaders(dataset, batch_size=batch_size, text_encoder_njobs=cfg.general.njobs)
        
        # Initialize trainer
        trainer = DiffusionTrainer(config=cfg, device=device)

        logger.info(f"Training configuration: {trainer.config}")
        
        # Start experiment logging
        exp_logger.start_experiment(trainer.config)
        
        # Setup model
        logger.info("Setting up diffusion model...")
        exp_logger.log_step("setup", "Initializing model architecture")
        trainer.setup_model(dataset=dataset, text_encoder=dataset_loader.text_encoder)
        
        # Load checkpoint if resuming TODO: check implementation and test, not correct!
        if cfg.general.resume:
            logger.info(f"Resuming training from {cfg.general.resume}")
            trainer.load_model(cfg.general.resume)
        
        # Compile model
        logger.info("Compiling model for training...")
        exp_logger.log_step("setup", "Compiling model with optimizer and loss function")
        trainer.compile_model()
        
        # Train model
        logger.info("Starting model training...")
        exp_logger.log_step("training", "Beginning training loop")
        
        # Custom training loop with experiment logging
        training_config = trainer.config["training"]
        num_epochs = training_config.get("num_epochs", 10)

        history = trainer.train(dataloaders, save_path=cfg.general.output_path)
        
        # Save model
        output_path = cfg.general.output_path
        model_name = cfg.general.model_name
        if cfg.general.model_name:
            output_path = str(Path(output_path) / model_name)
        
        logger.info(f"Saving trained model to {output_path}")
        exp_logger.log_step("saving", f"Saving model to {output_path}")
        trainer.save_model(output_path)
        
        # Register model with model manager  TODO: check this model manager
        model_manager = ModelManager()
        model_name = cfg.general.model_name or f"model_{int(time.time())}"
        model_manager.register_model(model_name, trainer)
        
        # End experiment logging
        duration = exp_logger.end_experiment(success=True)
        
        logger.info("Training completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Model saved to: {output_path}")
        print(f"Model type: {trainer.config['model']['type']}")
        print(f"Training epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {trainer.config['training']['learning_rate']}")
        print(f"Device used: {trainer.device}")
        print(f"Training duration: {duration:.2f} seconds")
        print("="*50)
        
        # Print metrics summary
        metrics_summary = exp_logger.get_metrics_summary()
        if metrics_summary:
            print("\nMETRICS SUMMARY:")
            for metric_name, stats in metrics_summary.items():
                print(f"  {metric_name}:")
                print(f"    Final value: {stats['latest']:.6f}")
                print(f"    Best value: {stats['min']:.6f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        exp_logger.end_experiment(success=False)
        raise e


if __name__ == "__main__":
    main()
