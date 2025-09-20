#!/usr/bin/env python3
"""Script for evaluating trained diffusion models."""

import argparse
import sys
from pathlib import Path
import time

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantum_diffusion.models import ModelManager
from quantum_diffusion.data import DatasetLoader
from quantum_diffusion.evaluation import Evaluator
from quantum_diffusion.utils import ConfigManager, Logger, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained diffusion models")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to the trained model or model name"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to evaluation configuration file"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Path to test dataset"
    )
    parser.add_argument(
        "--reference-circuits",
        type=str,
        help="Path to reference circuits for comparison"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to generate for evaluation (overrides config)"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["fidelity", "circuit_properties", "statistical_analysis", "diversity_metrics"],
        help="Specific metrics to evaluate (overrides config)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Don't create evaluation plots"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to use for evaluation"
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
    
    try:
        # Initialize evaluator
        evaluator = Evaluator(config_path=args.config, device=args.device)
        
        # Override config with command line arguments
        if args.num_samples:
            evaluator.config["generation"]["num_samples"] = args.num_samples
        if args.metrics:
            # Reset all metrics to False
            for metric in evaluator.config["metrics"]:
                evaluator.config["metrics"][metric] = False
            # Enable specified metrics
            for metric in args.metrics:
                evaluator.config["metrics"][metric] = True
        if args.no_plots:
            evaluator.config["output"]["create_plots"] = False
        
        logger.info(f"Evaluation configuration: {evaluator.config}")
        
        # Load model
        logger.info(f"Loading model from {args.model}")
        model_manager = ModelManager()
        
        # Check if it's a model name or path
        if Path(args.model).exists():
            # It's a path
            model_trainer = model_manager.load_model(args.model, device=args.device)
        else:
            # It's a model name
            model_trainer = model_manager.load_model(args.model, device=args.device)
        
        # Load test dataset
        test_dataset = None
        if args.dataset:
            logger.info(f"Loading test dataset from {args.dataset}")
            dataset_loader = DatasetLoader(device=args.device)
            test_dataset = dataset_loader.load_dataset(args.dataset)
        
        # Load reference circuits
        reference_circuits = None
        if args.reference_circuits:
            logger.info(f"Loading reference circuits from {args.reference_circuits}")
            # This would need to be implemented based on the file format
            # For now, we'll pass None
            logger.warning("Reference circuit loading not implemented yet")
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate model
        logger.info("Starting model evaluation...")
        start_time = time.time()
        
        results = evaluator.evaluate_model(
            model_trainer=model_trainer,
            test_dataset=test_dataset,
            reference_circuits=reference_circuits,
            output_dir=str(output_dir)
        )
        
        evaluation_time = time.time() - start_time
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        logger.info(f"Results saved to: {output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {args.model}")
        print(f"Output directory: {output_dir}")
        print(f"Evaluation time: {evaluation_time:.2f} seconds")
        
        # Print key metrics
        if "generated_circuits_info" in results:
            num_generated = results["generated_circuits_info"]["num_generated"]
            print(f"Generated circuits: {num_generated}")
        
        if "fidelity_metrics" in results:
            fidelity = results["fidelity_metrics"]
            if "avg_circuit_fidelity" in fidelity:
                print(f"Average circuit fidelity: {fidelity['avg_circuit_fidelity']:.4f}")
            if "avg_state_fidelity" in fidelity:
                print(f"Average state fidelity: {fidelity['avg_state_fidelity']:.4f}")
        
        if "circuit_properties" in results and "generated" in results["circuit_properties"]:
            props = results["circuit_properties"]["generated"]
            if "depth_stats" in props:
                avg_depth = props["depth_stats"]["mean"]
                print(f"Average circuit depth: {avg_depth:.2f}")
            if "total_gates_stats" in props:
                avg_gates = props["total_gates_stats"]["mean"]
                print(f"Average gate count: {avg_gates:.2f}")
        
        if "diversity_metrics" in results:
            diversity = results["diversity_metrics"]["diversity_metrics"]
            if "unique_circuits" in diversity:
                unique_circuits = diversity["unique_circuits"]
                total_circuits = results.get("generated_circuits_info", {}).get("num_generated", 0)
                if total_circuits > 0:
                    uniqueness_ratio = unique_circuits / total_circuits
                    print(f"Circuit uniqueness: {uniqueness_ratio:.2%} ({unique_circuits}/{total_circuits})")
        
        # Print evaluation metrics enabled
        enabled_metrics = [metric for metric, enabled in evaluator.config["metrics"].items() if enabled]
        print(f"Evaluated metrics: {', '.join(enabled_metrics)}")
        
        print("="*60)
        
        # Print file locations
        print("\nGenerated files:")
        if (output_dir / "evaluation_results.yaml").exists():
            print(f"  - Full results: {output_dir / 'evaluation_results.yaml'}")
        if (output_dir / "evaluation_summary.csv").exists():
            print(f"  - Summary: {output_dir / 'evaluation_summary.csv'}")
        
        plots_dir = output_dir / "plots"
        if plots_dir.exists() and list(plots_dir.glob("*.png")):
            print(f"  - Plots: {plots_dir}")
            for plot_file in plots_dir.glob("*.png"):
                print(f"    - {plot_file.name}")
        
        # Detailed results
        if args.verbose:
            print("\nDetailed Results:")
            for section, data in results.items():
                if section in ["evaluation_timestamp", "config", "model_info"]:
                    continue
                print(f"\n{section.upper().replace('_', ' ')}:")
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value}")
                        elif isinstance(value, dict):
                            print(f"  {key}:")
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, (int, float)):
                                    print(f"    {subkey}: {subvalue}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()