"""Comprehensive evaluation and testing framework for quantum diffusion training."""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import yaml
import pandas as pd
from scipy import stats
from sklearn.metrics import pairwise_distances
import warnings

# Quantum computing imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, process_fidelity, state_fidelity

# genQC imports
from my_genQC.platform.simulation import Simulator, CircuitBackendType
from my_genQC.platform.tokenizer.circuits_tokenizer import CircuitTokenizer
from my_genQC.inference.sampling import generate_compilation_tensors, decode_tensors_to_backend
from my_genQC.inference.evaluation_helper import get_unitaries
from my_genQC.inference.eval_metrics import UnitaryFrobeniusNorm, UnitaryInfidelityNorm
from my_genQC.utils.misc_utils import infer_torch_device

from ..utils.logging import Logger
from ..utils.config import ConfigManager


class MetricsCalculator:
    """Calculate various metrics for quantum circuit evaluation."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize metrics calculator.
        
        Args:
            device: Device to use for computation
        """
        self.device = device or infer_torch_device() if 'infer_torch_device' in globals() else 'cpu'
        self.logger = Logger(__name__)
        
    def circuit_fidelity(self, circuit1: QuantumCircuit, circuit2: QuantumCircuit) -> float:
        """Calculate fidelity between two quantum circuits.
        
        Args:
            circuit1: First quantum circuit
            circuit2: Second quantum circuit
            
        Returns:
            Fidelity value between 0 and 1
        """
        try:
            # Convert circuits to operators
            op1 = Operator(circuit1)
            op2 = Operator(circuit2)
            
            # Calculate process fidelity
            fidelity = process_fidelity(op1, op2)
            return float(fidelity)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate fidelity: {e}")
            return 0.0
    
    def state_vector_fidelity(self, circuit1: QuantumCircuit, circuit2: QuantumCircuit) -> float:
        """Calculate state vector fidelity between circuits.
        
        Args:
            circuit1: First quantum circuit  
            circuit2: Second quantum circuit
            
        Returns:
            State vector fidelity
        """
        try:
            # Get final state vectors
            state1 = Statevector.from_instruction(circuit1)
            state2 = Statevector.from_instruction(circuit2)
            
            # Calculate fidelity
            fidelity = state_fidelity(state1, state2)
            return float(fidelity)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate state fidelity: {e}")
            return 0.0
    
    def circuit_depth(self, circuit: QuantumCircuit) -> int:
        """Calculate circuit depth.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Circuit depth
        """
        return circuit.depth()
    
    def gate_count(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """Count gates in a circuit.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Dictionary with gate counts
        """
        gate_counts = {}
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        return gate_counts
    
    def total_gate_count(self, circuit: QuantumCircuit) -> int:
        """Get total number of gates in circuit.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Total gate count
        """
        return len(circuit.data)
    
    def two_qubit_gate_count(self, circuit: QuantumCircuit) -> int:
        """Count two-qubit gates in circuit.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Two-qubit gate count
        """
        count = 0
        for instruction in circuit.data:
            if instruction.operation.num_qubits == 2:
                count += 1
        return count
    
    def circuit_statistics(self, circuits: List[QuantumCircuit]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a list of circuits.
        
        Args:
            circuits: List of quantum circuits
            
        Returns:
            Dictionary containing various statistics
        """
        if not circuits:
            return {}
        
        depths = [self.circuit_depth(c) for c in circuits]
        total_gates = [self.total_gate_count(c) for c in circuits]
        two_qubit_gates = [self.two_qubit_gate_count(c) for c in circuits]
        
        # Collect all gate types
        all_gate_counts = {}
        for circuit in circuits:
            gate_counts = self.gate_count(circuit)
            for gate, count in gate_counts.items():
                if gate not in all_gate_counts:
                    all_gate_counts[gate] = []
                all_gate_counts[gate].append(count)
        
        # Fill missing gate counts with 0
        for gate in all_gate_counts:
            while len(all_gate_counts[gate]) < len(circuits):
                all_gate_counts[gate].append(0)
        
        stats_dict = {
            'num_circuits': len(circuits),
            'depth_stats': {
                'mean': np.mean(depths),
                'std': np.std(depths),
                'min': np.min(depths),
                'max': np.max(depths),
                'median': np.median(depths)
            },
            'total_gates_stats': {
                'mean': np.mean(total_gates),
                'std': np.std(total_gates),
                'min': np.min(total_gates),
                'max': np.max(total_gates),
                'median': np.median(total_gates)
            },
            'two_qubit_gates_stats': {
                'mean': np.mean(two_qubit_gates),
                'std': np.std(two_qubit_gates),
                'min': np.min(two_qubit_gates),
                'max': np.max(two_qubit_gates),
                'median': np.median(two_qubit_gates)
            },
            'gate_type_stats': {}
        }
        
        # Gate type statistics
        for gate, counts in all_gate_counts.items():
            stats_dict['gate_type_stats'][gate] = {
                'mean': np.mean(counts),
                'std': np.std(counts),
                'total': np.sum(counts),
                'frequency': np.sum(np.array(counts) > 0) / len(circuits)
            }
        
        return stats_dict


class Evaluator:
    """Comprehensive evaluator for quantum diffusion training."""
    
    def __init__(self, config_path: Optional[str] = None, device: Optional[str] = None):
        """Initialize the evaluator.
        
        Args:
            config_path: Path to evaluation configuration
            device: Device to use for evaluation
        """
        self.device = device or infer_torch_device() if 'infer_torch_device' in globals() else 'cpu'
        self.config_manager = ConfigManager()
        self.logger = Logger(__name__)
        self.metrics_calculator = MetricsCalculator(device)
        self.simulator = Simulator(CircuitBackendType.QISKIT)
        self.frobenius_metric = UnitaryFrobeniusNorm()
        self.infidelity_metric = UnitaryInfidelityNorm()
        
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default evaluation configuration."""
        return {
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
    
    def evaluate_model(self, 
                      model_trainer, 
                      test_dataset,
                      reference_circuits: Optional[List[QuantumCircuit]] = None,
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive evaluation of a diffusion model.
        
        Args:
            model_trainer: Trained diffusion model trainer
            test_dataset: Test dataset for comparison
            reference_circuits: Reference circuits for comparison
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing all evaluation results
        """
        self.logger.info("Starting comprehensive model evaluation...")
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'model_info': self._get_model_info(model_trainer)
        }
        
        try:
            if test_dataset is None:
                raise ValueError("Evaluation requires a dataset that provides unitary conditions.")
            
            generation_ctx = self._prepare_generation_context(test_dataset)
            generation_output = self._generate_circuits(model_trainer, generation_ctx)
            generated_circuits = generation_output['circuits']

            results['generated_circuits_info'] = {
                'num_requested': generation_output['requested'],
                'num_generated': len(generated_circuits),
                'decode_failures': generation_output['decode_failures'],
                'generation_config': self.config['generation']
            }

            reference_set = reference_circuits
            if (not reference_set and 
                self.config.get('comparison', {}).get('compare_to_training', False) and
                test_dataset is not None):
                reference_set = self._load_reference_circuits_from_dataset(
                    test_dataset,
                    generation_ctx['tokenizer'],
                    generation_output['requested']
                )
            
            # Circuit properties evaluation
            if self.config['metrics']['circuit_properties']:
                results['circuit_properties'] = self._evaluate_circuit_properties(
                    generated_circuits, reference_set
                )
            
            # Fidelity evaluation
            if self.config['metrics']['fidelity']:
                fidelity_metrics = self._evaluate_fidelity(
                    generated_circuits=generated_circuits,
                    unitary_targets=generation_ctx.get('unitary_conditions'),
                    valid_indices=generation_output['valid_indices'],
                    reference_circuits=reference_set
                )
                if fidelity_metrics:
                    results['fidelity_metrics'] = fidelity_metrics
            
            # Statistical analysis
            if self.config['metrics']['statistical_analysis']:
                results['statistical_analysis'] = self._statistical_analysis(
                    generated_circuits, test_dataset, reference_set
                )
            
            # Diversity metrics
            if self.config['metrics']['diversity_metrics']:
                results['diversity_metrics'] = self._evaluate_diversity(generated_circuits)
            
            # Generate plots
            if self.config['output']['create_plots'] and output_dir:
                self._create_evaluation_plots(results, output_dir)
            
            # Save results
            if self.config['output']['save_results'] and output_dir:
                self._save_results(results, output_dir)
            
            self.logger.info("Model evaluation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}")
            raise
    
    def _get_model_info(self, model_trainer) -> Dict:
        """Extract model information."""
        return {
            'model_type': model_trainer.config.get('model', {}).get('type', 'Unknown'),
            'device': model_trainer.device,
            'training_epochs': model_trainer.config.get('training', {}).get('num_epochs', 'Unknown')
        }
    
    def _generate_circuits(self, model_trainer, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Generate circuits using my_genQC sampling utilities."""
        pipeline = getattr(model_trainer, "pipeline", None)
        if pipeline is None:
            raise ValueError("Model trainer pipeline is not initialized. Call setup_model before evaluation.")

        unitary_conditions = ctx.get("unitary_conditions")
        if unitary_conditions is None:
            raise ValueError("Dataset does not provide unitary conditions required for compilation evaluation.")

        num_samples = unitary_conditions.shape[0]
        prompt = self.config["generation"].get("prompt", "quantum circuit compilation")
        guidance_scale = self.config["generation"].get("guidance_scale", 1.0)
        auto_batch_size = self.config["generation"].get("auto_batch_size", 512)

        tensors_out = generate_compilation_tensors(
            pipeline=pipeline,
            prompt=prompt,
            U=unitary_conditions.to(pipeline.device),
            samples=num_samples,
            system_size=ctx["system_size"],
            num_of_qubits=ctx["num_qubits"],
            max_gates=ctx["max_gates"],
            g=guidance_scale,
            auto_batch_size=auto_batch_size,
            enable_params=True,
            no_bar=not self.config["output"].get("verbose", True)
        )

        if isinstance(tensors_out, tuple):
            generated_tensors, params = tensors_out
        else:
            generated_tensors, params = tensors_out, None

        generated_tensors = generated_tensors.detach().cpu()
        if params is not None:
            params = params.detach().cpu()

        backend_objs, _ = decode_tensors_to_backend(
            simulator=self.simulator,
            tokenizer=ctx["tokenizer"],
            tensors=generated_tensors,
            params=params,
            silent=not self.config["output"].get("verbose", True),
            n_jobs=self.config["generation"].get("decode_workers", 1),
            filter_errs=False
        )

        valid_indices = [idx for idx, qc in enumerate(backend_objs) if qc is not None]
        valid_circuits = [backend_objs[idx] for idx in valid_indices]
        decode_failures = len(backend_objs) - len(valid_circuits)

        return {
            "circuits": valid_circuits,
            "valid_indices": valid_indices,
            "decode_failures": decode_failures,
            "requested": num_samples
        }
    
    def _prepare_generation_context(self, dataset) -> Dict[str, Any]:
        """Collect shapes, tokenizers, and conditions from the dataset."""
        if not hasattr(dataset, "x"):
            raise ValueError("Dataset missing tensor encodings (x).")
        
        num_samples = min(
            self.config["generation"].get("num_samples", 100),
            dataset.x.shape[0]
        )

        if num_samples <= 0:
            raise ValueError("Dataset does not contain any samples for evaluation.")
        
        system_size = dataset.x.shape[1]
        max_gates = dataset.x.shape[2]
        num_qubits = getattr(dataset.params_config, "num_of_qubits", system_size)

        gate_pool = getattr(dataset, "gate_pool", None)
        if not gate_pool:
            raise ValueError("Dataset does not expose a gate pool required for decoding.")
        
        vocabulary = {gate: idx for idx, gate in enumerate(gate_pool)}
        tokenizer = CircuitTokenizer(vocabulary)

        if not hasattr(dataset, "U"):
            raise ValueError("Dataset does not contain unitary tensors (attribute 'U'); required for compilation evaluation.")
        unitary_conditions = dataset.U[:num_samples].to(self.device)
        
        return {
            "num_samples": num_samples,
            "system_size": system_size,
            "max_gates": max_gates,
            "num_qubits": num_qubits,
            "tokenizer": tokenizer,
            "unitary_conditions": unitary_conditions
        }
    
    def _load_reference_circuits_from_dataset(self, dataset, tokenizer: CircuitTokenizer, max_cnt: int) -> List[QuantumCircuit]:
        """Decode reference circuits from the dataset tensors."""
        max_cnt = min(max_cnt, dataset.x.shape[0])
        tensors = dataset.x[:max_cnt].detach().cpu()
        params = None
        if hasattr(dataset, "store_dict") and "params" in dataset.store_dict:
            params = getattr(dataset, "params")[:max_cnt].detach().cpu()
        
        backend_objs, _ = decode_tensors_to_backend(
            simulator=self.simulator,
            tokenizer=tokenizer,
            tensors=tensors,
            params=params,
            silent=True,
            n_jobs=self.config["generation"].get("decode_workers", 1),
            filter_errs=False
        )
        
        return [qc for qc in backend_objs if qc is not None]
    
    def _unitary_tensor_to_complex(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert stacked real/imag tensors into complex matrices."""
        if tensor.dim() == 4:
            real, imag = tensor[:, 0], tensor[:, 1]
        elif tensor.dim() == 3:
            real, imag = tensor[0], tensor[1]
        else:
            raise ValueError(f"Unexpected unitary tensor shape: {tensor.shape}")
        return torch.complex(real, imag)
    
    def _compute_predicted_unitaries(self, circuits: List[QuantumCircuit]) -> torch.Tensor:
        """Simulate circuits to retrieve their unitary matrices."""
        if not circuits:
            return torch.empty(0, dtype=torch.complex64, device=self.device)
        
        matrices = get_unitaries(
            simulator=self.simulator,
            backend_obj_list=circuits,
            n_jobs=self.config["generation"].get("unitary_jobs", 1)
        )
        matrices_np = np.stack(matrices).astype(np.complex64)
        return torch.from_numpy(matrices_np).to(self.device)
    
    def _evaluate_circuit_properties(self, 
                                   generated_circuits: List[QuantumCircuit],
                                   reference_circuits: Optional[List[QuantumCircuit]]) -> Dict:
        """Evaluate basic circuit properties."""
        results = {}
        
        # Generated circuit statistics
        if generated_circuits:
            results['generated'] = self.metrics_calculator.circuit_statistics(generated_circuits)
        
        # Reference circuit statistics (if available)
        if reference_circuits:
            results['reference'] = self.metrics_calculator.circuit_statistics(reference_circuits)
        
        # Comparison metrics
        if generated_circuits and reference_circuits:
            results['comparison'] = self._compare_circuit_properties(
                generated_circuits, reference_circuits
            )
        
        return results
    
    def _compare_circuit_properties(self, 
                                  circuits1: List[QuantumCircuit], 
                                  circuits2: List[QuantumCircuit]) -> Dict:
        """Compare properties between two sets of circuits."""
        stats1 = self.metrics_calculator.circuit_statistics(circuits1)
        stats2 = self.metrics_calculator.circuit_statistics(circuits2)
        
        comparison = {}
        
        # Compare depth statistics
        if 'depth_stats' in stats1 and 'depth_stats' in stats2:
            depths1 = [self.metrics_calculator.circuit_depth(c) for c in circuits1]
            depths2 = [self.metrics_calculator.circuit_depth(c) for c in circuits2]
            
            # Statistical tests
            ks_stat, ks_pval = stats.ks_2samp(depths1, depths2)
            comparison['depth_comparison'] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'mean_diff': stats1['depth_stats']['mean'] - stats2['depth_stats']['mean']
            }
        
        return comparison
    
    def _evaluate_fidelity(self, 
                          generated_circuits: List[QuantumCircuit],
                          unitary_targets: Optional[torch.Tensor],
                          valid_indices: List[int],
                          reference_circuits: Optional[List[QuantumCircuit]]) -> Dict:
        """Evaluate fidelity using my_genQC unitary metrics, with circuit fallback."""
        if unitary_targets is not None and generated_circuits:
            target_subset = unitary_targets[valid_indices].to(self.device)
            target_complex = self._unitary_tensor_to_complex(target_subset)
            predicted = self._compute_predicted_unitaries(generated_circuits)
            if predicted.shape[0] == 0:
                return {}
            
            frob = self.frobenius_metric.distance(predicted, target_complex)
            infidelity = self.infidelity_metric.distance(predicted, target_complex)
            
            return {
                'frobenius_mean': float(frob.mean().item()),
                'frobenius_std': float(frob.std(unbiased=False).item()) if frob.numel() > 1 else 0.0,
                'infidelity_mean': float(infidelity.mean().item()),
                'infidelity_std': float(infidelity.std(unbiased=False).item()) if infidelity.numel() > 1 else 0.0,
                'avg_circuit_fidelity': float(1 - infidelity.mean().item()),
                'avg_state_fidelity': float(1 - infidelity.mean().item())
            }
        
        if reference_circuits:
            results = {
                'circuit_fidelities': [],
                'state_fidelities': [],
                'avg_circuit_fidelity': 0.0,
                'avg_state_fidelity': 0.0
            }
            
            num_comparisons = min(len(generated_circuits), len(reference_circuits))
            for i in range(num_comparisons):
                circ_fidelity = self.metrics_calculator.circuit_fidelity(
                    generated_circuits[i], reference_circuits[i]
                )
                results['circuit_fidelities'].append(circ_fidelity)
                
                state_fidelity = self.metrics_calculator.state_vector_fidelity(
                    generated_circuits[i], reference_circuits[i]
                )
                results['state_fidelities'].append(state_fidelity)
            
            if results['circuit_fidelities']:
                results['avg_circuit_fidelity'] = float(np.mean(results['circuit_fidelities']))
                results['circuit_fidelity_std'] = float(np.std(results['circuit_fidelities']))
            
            if results['state_fidelities']:
                results['avg_state_fidelity'] = float(np.mean(results['state_fidelities']))
                results['state_fidelity_std'] = float(np.std(results['state_fidelities']))
            return results
        
        return {}
    
    def _statistical_analysis(self,
                            generated_circuits: List[QuantumCircuit],
                            test_dataset,
                            reference_circuits: Optional[List[QuantumCircuit]]) -> Dict:
        """Perform statistical analysis of generated circuits."""
        results = {}
        
        if not generated_circuits:
            return results
        
        # Basic statistics
        depths = [self.metrics_calculator.circuit_depth(c) for c in generated_circuits]
        gate_counts = [self.metrics_calculator.total_gate_count(c) for c in generated_circuits]
        
        results['basic_stats'] = {
            'depth_distribution': {
                'mean': np.mean(depths),
                'std': np.std(depths),
                'skewness': stats.skew(depths),
                'kurtosis': stats.kurtosis(depths)
            },
            'gate_count_distribution': {
                'mean': np.mean(gate_counts),
                'std': np.std(gate_counts),
                'skewness': stats.skew(gate_counts),
                'kurtosis': stats.kurtosis(gate_counts)
            }
        }
        
        # Normality tests
        _, depth_normality_p = stats.normaltest(depths)
        _, gates_normality_p = stats.normaltest(gate_counts)
        
        results['normality_tests'] = {
            'depth_normality_pvalue': depth_normality_p,
            'gates_normality_pvalue': gates_normality_p
        }
        
        return results
    
    def _evaluate_diversity(self, circuits: List[QuantumCircuit]) -> Dict:
        """Evaluate diversity of generated circuits."""
        results = {}
        
        if len(circuits) < 2:
            return results
        
        # Convert circuits to feature vectors (simplified)
        features = []
        for circuit in circuits:
            feature_vector = [
                self.metrics_calculator.circuit_depth(circuit),
                self.metrics_calculator.total_gate_count(circuit),
                self.metrics_calculator.two_qubit_gate_count(circuit)
            ]
            
            # Add gate type counts
            gate_counts = self.metrics_calculator.gate_count(circuit)
            all_gates = set()
            for c in circuits:
                all_gates.update(self.metrics_calculator.gate_count(c).keys())
            
            for gate in sorted(all_gates):
                feature_vector.append(gate_counts.get(gate, 0))
            
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Calculate pairwise distances
        distances = pairwise_distances(features)
        
        results['diversity_metrics'] = {
            'mean_pairwise_distance': np.mean(distances),
            'std_pairwise_distance': np.std(distances),
            'min_distance': np.min(distances[distances > 0]),  # Exclude diagonal
            'max_distance': np.max(distances),
            'unique_circuits': len(np.unique(features, axis=0))
        }
        
        return results
    
    def _create_evaluation_plots(self, results: Dict, output_dir: str) -> None:
        """Create evaluation plots and save them."""
        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plt.style.use('default')
        
        # Circuit properties plots
        if 'circuit_properties' in results:
            self._plot_circuit_properties(results['circuit_properties'], plots_dir)
        
        # Fidelity plots
        if 'fidelity_metrics' in results:
            self._plot_fidelity_metrics(results['fidelity_metrics'], plots_dir)
        
        # Statistical analysis plots
        if 'statistical_analysis' in results:
            self._plot_statistical_analysis(results['statistical_analysis'], plots_dir)
    
    def _plot_circuit_properties(self, properties: Dict, plots_dir: Path) -> None:
        """Plot circuit properties comparisons."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Circuit Properties Comparison', fontsize=16)
        
        # Plot depth statistics
        if 'generated' in properties and 'reference' in properties:
            gen_depth = properties['generated']['depth_stats']
            ref_depth = properties['reference']['depth_stats']
            
            categories = ['Generated', 'Reference']
            means = [gen_depth['mean'], ref_depth['mean']]
            stds = [gen_depth['std'], ref_depth['std']]
            
            axes[0, 0].bar(categories, means, yerr=stds, capsize=5)
            axes[0, 0].set_title('Mean Circuit Depth')
            axes[0, 0].set_ylabel('Depth')
        
        # Similar plots for other metrics...
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'circuit_properties.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_fidelity_metrics(self, fidelity: Dict, plots_dir: Path) -> None:
        """Plot fidelity metrics."""
        if not fidelity.get('circuit_fidelities'):
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Circuit fidelity histogram
        axes[0].hist(fidelity['circuit_fidelities'], bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_title('Circuit Fidelity Distribution')
        axes[0].set_xlabel('Fidelity')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(fidelity['avg_circuit_fidelity'], color='red', 
                       linestyle='--', label='Mean')
        axes[0].legend()
        
        # State fidelity histogram
        if fidelity.get('state_fidelities'):
            axes[1].hist(fidelity['state_fidelities'], bins=20, alpha=0.7, edgecolor='black')
            axes[1].set_title('State Fidelity Distribution')
            axes[1].set_xlabel('Fidelity')
            axes[1].set_ylabel('Frequency')
            axes[1].axvline(fidelity['avg_state_fidelity'], color='red',
                           linestyle='--', label='Mean')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'fidelity_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_analysis(self, stats: Dict, plots_dir: Path) -> None:
        """Plot statistical analysis results."""
        # This would create various statistical plots
        pass
    
    def _save_results(self, results: Dict, output_dir: str) -> None:
        """Save evaluation results to files."""
        # Save full results as YAML
        results_path = Path(output_dir) / "evaluation_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        # Save summary as CSV for easy analysis
        summary_data = self._extract_summary_data(results)
        if summary_data:
            summary_path = Path(output_dir) / "evaluation_summary.csv"
            pd.DataFrame([summary_data]).to_csv(summary_path, index=False)
        
        self.logger.info(f"Evaluation results saved to {output_dir}")
    
    def _extract_summary_data(self, results: Dict) -> Dict:
        """Extract key metrics for summary."""
        summary = {
            'evaluation_date': results.get('evaluation_timestamp'),
            'model_type': results.get('model_info', {}).get('model_type'),
            'num_generated': results.get('generated_circuits_info', {}).get('num_generated', 0)
        }
        
        # Add key metrics
        if 'fidelity_metrics' in results:
            fidelity = results['fidelity_metrics']
            summary['avg_circuit_fidelity'] = fidelity.get('avg_circuit_fidelity', 0)
            summary['avg_state_fidelity'] = fidelity.get('avg_state_fidelity', 0)
        
        if 'circuit_properties' in results and 'generated' in results['circuit_properties']:
            props = results['circuit_properties']['generated']
            if 'depth_stats' in props:
                summary['avg_depth'] = props['depth_stats'].get('mean', 0)
            if 'total_gates_stats' in props:
                summary['avg_gate_count'] = props['total_gates_stats'].get('mean', 0)
        
        return summary


# Utility functions for batch evaluation
def evaluate_multiple_models(model_paths: List[str], 
                           test_dataset,
                           output_base_dir: str,
                           config_path: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate multiple training and compare results.
    
    Args:
        model_paths: List of paths to trained training
        test_dataset: Test dataset for evaluation
        output_base_dir: Base directory for outputs
        config_path: Evaluation configuration path
        
    Returns:
        Comparison results across all training
    """
    evaluator = Evaluator(config_path)
    results = {}
    
    for i, model_path in enumerate(model_paths):
        model_name = f"model_{i}" if isinstance(model_path, str) else model_path
        
        # Load model (this would need to be implemented)
        # model_trainer = load_model(model_path)
        
        # Evaluate model
        output_dir = os.path.join(output_base_dir, str(model_name))
        # model_results = evaluator.evaluate_model(model_trainer, test_dataset, output_dir=output_dir)
        # results[model_name] = model_results
    
    return results
