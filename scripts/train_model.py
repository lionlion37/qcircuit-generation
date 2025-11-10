#!/usr/bin/env python3
"""Script for training diffusion models on quantum circuits."""

import argparse
import sys
from pathlib import Path
import time
import hydra

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantum_diffusion.data import DatasetLoader
from quantum_diffusion.models import DiffusionTrainer, ModelManager, PRESET_TRAINING_CONFIGS
from quantum_diffusion.utils import ConfigManager, Logger, ExperimentLogger, setup_logging

def main():
    parser = argparse.ArgumentParser(description="Train diffusion models on quantum circuits")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--preset", "-p",
        type=str,
        choices=list(PRESET_TRAINING_CONFIGS.keys()),
        help="Use a preset training configuration"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Path to the condition-specific dataset directory (e.g., ./datasets/my_dataset/unitary)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./models/trained",
        help="Output directory for the trained model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name for the trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to model checkpoint to resume training"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for the experiment (for logging)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level="DEBUG" if args.verbose else "INFO",
        console_output=True
    )
    logger = Logger(__name__)
    
    # Setup experiment logging
    experiment_name = args.experiment_name or f"training_{int(time.time())}"
    exp_logger = ExperimentLogger(experiment_name)
    
    try:
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            config_manager = ConfigManager()
            config = config_manager.load_config(args.config)
        elif args.preset:
            logger.info(f"Using preset configuration: {args.preset}")
            config = PRESET_TRAINING_CONFIGS[args.preset].copy()
        else:  # TODO: add default config
            raise  ValueError("Either --config or --preset must be specified")

        # Load dataset
        logger.info(f"Loading dataset from {args.dataset}")
        dataset_loader = DatasetLoader(device=args.device, config=config)
        dataset = dataset_loader.load_dataset(args.dataset)
        
        # Create data loaders
        batch_size = args.batch_size or 32
        dataloaders = dataset_loader.get_dataloaders(dataset, batch_size=batch_size)
        
        # Initialize trainer
        trainer = DiffusionTrainer(config=config, device=args.device)
        
        # Override config with command line arguments
        if args.epochs:
            trainer.config["training"]["num_epochs"] = args.epochs
        if args.learning_rate:
            trainer.config["training"]["learning_rate"] = args.learning_rate
        if args.batch_size:
            trainer.config["training"]["batch_size"] = args.batch_size
        
        logger.info(f"Training configuration: {trainer.config}")
        
        # Start experiment logging
        exp_logger.start_experiment(trainer.config)
        
        # Setup model
        logger.info("Setting up diffusion model...")
        exp_logger.log_step("setup", "Initializing model architecture")
        trainer.setup_model(dataset=dataset, text_encoder=dataset_loader.text_encoder)
        
        # Load checkpoint if resuming
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            trainer.load_model(args.resume)
        
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

        history = trainer.train(dataloaders, save_path=args.output)
        
        # Save model
        output_path = args.output
        if args.model_name:
            output_path = str(Path(args.output) / args.model_name)
        
        logger.info(f"Saving trained model to {output_path}")
        exp_logger.log_step("saving", f"Saving model to {output_path}")
        trainer.save_model(output_path)
        
        # Register model with model manager
        model_manager = ModelManager()
        model_name = args.model_name or f"model_{int(time.time())}"
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
        sys.exit(1)


if __name__ == "__main__":
    main()
