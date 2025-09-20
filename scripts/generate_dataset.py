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
        choices=["SRV", "UNITARY"],
        help="Conditioning type (overrides config)"
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

    try:
        # Initialize generator
        generator = DatasetGenerator(device=args.device)
        
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
                "condition_type": "SRV",  # "UNITARY"
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
            config["condition_type"] = args.condition_type
        if args.output:
            config["output_path"] = args.output
        
        logger.info(f"Dataset configuration: {config}")
        
        # Generate dataset
        result = generator.generate_dataset(**config)
        
        logger.info(f"Dataset generation completed successfully!")
        logger.info(f"Dataset saved to: {args.output}")
        
        # Print summary
        print("\n" + "="*50)
        print("DATASET GENERATION SUMMARY")
        print("="*50)
        print(f"Output path: {args.output}")
        print(f"Gate set: {config['gate_set']}")
        print(f"Number of qubits: {config['num_qubits']}")
        print(f"Number of samples: {config['num_samples']}")
        print(f"Gate range: {config['min_gates']}-{config['max_gates']}")
        print(f"Condition type: {config['condition_type']}")
        print(f"Device used: {generator.device}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()