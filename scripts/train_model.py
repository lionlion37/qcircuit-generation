#!/usr/bin/env python3
"""Script for training diffusion training on quantum circuits."""

import sys
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


def build_one_cycle_scheduler(training_cfg, num_epochs: int, steps_per_epoch: int):
    """Build optional OneCycle LR scheduler factory from config."""
    one_cycle_cfg = training_cfg.get("one_cycle", {})
    if not one_cycle_cfg or not one_cycle_cfg.get("enable", False):
        return None

    max_lr = float(one_cycle_cfg.get("max_lr", training_cfg.get("learning_rate", 3e-4)))
    pct_start = float(one_cycle_cfg.get("pct_start", 0.1))
    anneal_strategy = str(one_cycle_cfg.get("anneal_strategy", "cos"))
    div_factor = float(one_cycle_cfg.get("div_factor", 25.0))
    final_div_factor = float(one_cycle_cfg.get("final_div_factor", 1e4))

    def _factory(optimizer):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )

    return _factory


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

        staged_cfg = cfg.get("staged_training", {})
        staged_enabled = bool(staged_cfg and staged_cfg.get("enable", False))
        stage1_cfg = staged_cfg.get("stage1", {}) if staged_enabled else {}
        stage1_batch_size = stage1_cfg.get("batch_size", cfg.training.batch_size or 32)
        stage1_bucket_batch_size = stage1_cfg.get("bucket_batch_size", -1)

        dataset = dataset_loader.load_dataset(
            cfg.general.dataset,
            bucket_batch_size=stage1_bucket_batch_size,
        )
        dataloaders = dataset_loader.get_dataloaders(
            dataset,
            batch_size=stage1_batch_size,
            text_encoder_njobs=cfg.general.njobs,
        )
        
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
        total_epochs = 0

        if staged_enabled:
            stage1_epochs = int(stage1_cfg.get("num_epochs", training_config.get("num_epochs", 10)))
            stage1_sched = build_one_cycle_scheduler(training_config, stage1_epochs, len(dataloaders.train))

            trainer.train(
                dataloaders,
                num_epochs=stage1_epochs,
                lr_sched=stage1_sched,
                setup_wandb=True,
                finish_wandb=False,
                stage_name="stage1-max-padding",
            )
            total_epochs += stage1_epochs

            stage2_cfg = staged_cfg.get("stage2", {})
            stage2_epochs = int(stage2_cfg.get("num_epochs", 0))
            if stage2_epochs > 0:
                stage2_batch_size = int(stage2_cfg.get("batch_size", 1))
                stage2_bucket_batch_size = int(stage2_cfg.get("bucket_batch_size", 256))

                logger.info(
                    "Preparing stage 2 dataloaders with "
                    f"{stage2_batch_size=} and {stage2_bucket_batch_size=}"
                )
                stage2_dataset = dataset_loader.load_dataset(
                    cfg.general.dataset,
                    load_embedder=False,
                    bucket_batch_size=stage2_bucket_batch_size,
                )
                dataset_loader.text_encoder = trainer.pipeline.text_encoder
                stage2_dataloaders = dataset_loader.get_dataloaders(
                    stage2_dataset,
                    batch_size=stage2_batch_size,
                    text_encoder_njobs=cfg.general.njobs,
                )
                stage2_sched = build_one_cycle_scheduler(training_config, stage2_epochs, len(stage2_dataloaders.train))

                trainer.train(
                    stage2_dataloaders,
                    num_epochs=stage2_epochs,
                    lr_sched=stage2_sched,
                    setup_wandb=False,
                    finish_wandb=True,
                    stage_name="stage2-bucket-padding",
                )
                total_epochs += stage2_epochs
            else:
                # Close logging if stage 2 is disabled.
                if trainer.wandb_run:
                    trainer.wandb_run.finish()
                    trainer.wandb_run = None
        else:
            num_epochs = int(training_config.get("num_epochs", 10))
            lr_sched = build_one_cycle_scheduler(training_config, num_epochs, len(dataloaders.train))
            trainer.train(
                dataloaders,
                num_epochs=num_epochs,
                lr_sched=lr_sched,
                stage_name="single-stage",
            )
            total_epochs = num_epochs
        
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
        print(f"Training epochs: {total_epochs}")
        print(f"Batch size (stage 1): {stage1_batch_size}")
        if staged_enabled:
            stage2_cfg = staged_cfg.get("stage2", {})
            if int(stage2_cfg.get("num_epochs", 0)) > 0:
                print(
                    "Batch size (stage 2): "
                    f"{int(stage2_cfg.get('batch_size', 1))} "
                    f"(bucket_batch_size={int(stage2_cfg.get('bucket_batch_size', 256))})"
                )
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
