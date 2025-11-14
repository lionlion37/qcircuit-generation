#!/usr/bin/env python3
"""Script for generating quantum circuit datasets."""
# TODO: inspect created datasets, create different datasets to test out
# TODO: improve config structures, presets, etc

import argparse
import sys
from pathlib import Path
import yaml
import hydra
import logging

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantum_diffusion.data.dataset import DatasetGenerator, PRESET_CONFIGS
from quantum_diffusion.utils import ConfigManager, Logger, setup_logging


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    cfg = cfg["datasets"]

    ###### Argument and config management ##############################################################################

    """
    parser = argparse.ArgumentParser(description="Generate quantum circuit datasets")

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to use for generation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    """
    
    # Setup logging
    setup_logging(
        log_level="DEBUG", # if args.verbose else "INFO",
        console_output=True
    )
    logger = Logger(__name__)
    logging.getLogger("qiskit").setLevel(logging.WARNING)

    logger.info(f"Dataset configuration: {cfg}")

    # Initialize generator, generate dataset
    generator = DatasetGenerator(device="cpu") # args.device if args.device else "cpu")
    generation_results = generator.generate_dataset(**cfg)

    logger.info(f"Dataset generation completed successfully!")
    for condition_name, metadata in generation_results.items():
        logger.info(f"{condition_name}: saved to {metadata['output_path']}")

    # Print summary  TODO: better summary, more details
    print("\n" + "="*50)
    print("DATASET GENERATION SUMMARY")
    print("="*50)
    print("Output paths:")
    for condition_name, metadata in generation_results.items():
        print(f"  {condition_name}: {metadata['output_path']}")
    print(f"Gate set: {cfg['gate_set']}")
    print(f"Number of qubits: {cfg['num_qubits']}")
    print(f"Number of samples: {cfg['num_samples']}")
    print(f"Gate range: {cfg['min_gates']}-{cfg['max_gates']}")
    print(f"Condition type(s): {cfg['condition_type']}")
    print(f"Device used: {generator.device}")
    print("="*50)


if __name__ == "__main__":
    main()
