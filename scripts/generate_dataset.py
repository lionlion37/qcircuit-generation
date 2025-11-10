#!/usr/bin/env python3
"""Script for generating quantum circuit datasets."""

import argparse
import sys
from pathlib import Path
import yaml
import logging

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantum_diffusion.data.dataset import DatasetGenerator, PRESET_CONFIGS
from quantum_diffusion.utils import ConfigManager, Logger, setup_logging


def main():
    ###### Argument and config management ##############################################################################

    parser = argparse.ArgumentParser(description="Generate quantum circuit datasets")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to dataset configuration file"
    )
    parser.add_argument(
        "--preset", "-p",
        type=str,
        choices=list(PRESET_CONFIGS.keys()),
        help="Use a preset configuration"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for the dataset"
    )

    # Override config with command line arguments
    parser.add_argument(
        "--gate-set",
        nargs="+",
        help="List of gates to include (overrides config)"
    )
    parser.add_argument(
        "--num-qubits",
        type=int,
        help="Number of qubits (overrides config)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to generate (overrides config)"
    )
    parser.add_argument(
        "--min-gates",
        type=int,
        help="Minimum gates per circuit (overrides config)"
    )
    parser.add_argument(
        "--max-gates",
        type=int,
        help="Maximum gates per circuit (overrides config)"
    )
    parser.add_argument(
        "--condition-type",
        choices=["SRV", "UNITARY", "BOTH"],
        action="append",
        help="Conditioning types to generate (repeat flag to specify multiple)"
    )
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
    
    # Setup logging
    setup_logging(
        log_level="DEBUG" if args.verbose else "INFO",
        console_output=True
    )
    logger = Logger(__name__)
    logging.getLogger("qiskit").setLevel(logging.WARNING)

    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
    elif args.preset:
        logger.info(f"Using preset configuration: {args.preset}")
        config = PRESET_CONFIGS[args.preset].copy()
    else:
        # Use default configuration
        logger.info("Using default configuration")
        config = {
            "gate_set": ['h', 'cx', 'cz', 's', 'x', 'y', 'z'],
            "num_qubits": 3,
            "num_samples": 10,  # 1000
            "min_gates": 2,
            "max_gates": 16,
            "condition_type": "SRV",
        }

    # Override config with command line arguments
    if args.gate_set:
        config["gate_set"] = args.gate_set
    if args.num_qubits:
        config["num_qubits"] = args.num_qubits
    if args.num_samples:
        config["num_samples"] = args.num_samples
    if args.min_gates:
        config["min_gates"] = args.min_gates
    if args.max_gates:
        config["max_gates"] = args.max_gates
    if args.condition_type:
        config["condition_type"] = args.condition_type if len(args.condition_type) > 1 else args.condition_type[0]
    if args.output:
        config["output_path"] = args.output

    logger.info(f"Dataset configuration: {config}")

    try:
        # Initialize generator
        generator = DatasetGenerator(device=args.device)

        # Generate dataset
        generation_results = generator.generate_dataset(**config)  # TODO: change name of training_config.yaml to config.yaml
        
        logger.info(f"Dataset generation completed successfully!")
        for condition_name, metadata in generation_results.items():
            logger.info(f"{condition_name}: saved to {metadata['output_path']}")
        
        # Print summary
        print("\n" + "="*50)
        print("DATASET GENERATION SUMMARY")
        print("="*50)
        print("Output paths:")
        for condition_name, metadata in generation_results.items():
            print(f"  {condition_name}: {metadata['output_path']}")
        print(f"Gate set: {config['gate_set']}")
        print(f"Number of qubits: {config['num_qubits']}")
        print(f"Number of samples: {config['num_samples']}")
        print(f"Gate range: {config['min_gates']}-{config['max_gates']}")
        print(f"Condition type(s): {config['condition_type']}")
        print(f"Device used: {generator.device}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
