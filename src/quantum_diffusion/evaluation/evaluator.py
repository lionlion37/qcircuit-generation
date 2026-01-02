import argparse
import sys
import os
import time
import datetime
import ast
from collections import Counter
from pathlib import Path
import hydra

import numpy as np
import torch

# Ensure local src/ is importable
sys.path.insert(0, str(Path(os.getcwd()).parent / "src"))

from my_genQC.inference.eval_metrics import UnitaryFrobeniusNorm, UnitaryInfidelityNorm
from my_genQC.inference.evaluation_helper import get_unitaries, get_srvs
from my_genQC.inference.sampling import generate_compilation_tensors, generate_tensors, decode_tensors_to_backend
from my_genQC.pipeline.diffusion_pipeline import DiffusionPipeline
from my_genQC.platform.simulation import Simulator, CircuitBackendType
from my_genQC.platform.tokenizer.circuits_tokenizer import CircuitTokenizer
from my_genQC.utils.misc_utils import infer_torch_device, get_entanglement_bins
from my_genQC.dataset import circuits_dataset
from my_genQC.models.config_model import ConfigModel
from my_genQC.utils.config_loader import load_config, store_tensor, load_tensor

from quantum_diffusion.data.dataset import DatasetLoader
from quantum_diffusion.utils import Logger


class SRVEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = infer_torch_device()

        self.logger = Logger(__name__)
        self.wandb_run = self._setup_wandb()

        self.dataset_loader = DatasetLoader(self.config, device=self.device)
        self.dataset = self.dataset_loader.load_dataset(self.config.dataset, load_embedder=False)
        self.samples = min(self.config.num_samples, self.dataset.x.shape[0])
        if self.samples == 0:
            raise ValueError("Dataset is empty - nothing to evaluate.")
        self.system_size = self.dataset.x.shape[1]
        self.max_gates = self.dataset.x.shape[2]
        self.num_qubits = getattr(self.dataset.params_config, "num_of_qubits", self.system_size)

        self.pipeline = self._load_pipeline(model_dir=Path(self.config.model_dir) if self.config.model_dir else None,
                                            repo_id=self.config.hf_repo)

        self.vocabulary = {gate: idx for idx, gate in enumerate(self.dataset.gate_pool)}
        self.tokenizer = CircuitTokenizer(self.vocabulary)
        self.simulator = Simulator(CircuitBackendType.QUDITKIT)  # TODO: add to config or something

        if self.wandb_run:
            self.wandb_run.config.update({
                "eval/samples": self.samples,
                "data/system_size": self.system_size,
                "data/max_gates": self.max_gates,
                "data/num_qubits": self.num_qubits,
                "pipeline/guidance_sample_mode": self.pipeline.guidance_sample_mode,
                "pipeline/sample_steps": self.config.model_params.sample_steps,
                "device": str(self.device),
            }, allow_val_change=True)


    def _setup_wandb(self):
        wandb_cfg = self.config.get("wandb", {})
        enabled = wandb_cfg.get("enable", False) or wandb_cfg.get("enabled", False)
        if not enabled:
            self.logger.info("Running w/o wandb")
            return None
        try:
            import wandb
        except ImportError:
            print("wandb logging requested but the package is not installed.")
            return None

        self.logger.info("Setting up wandb...")
        project = wandb_cfg.get("project", "qcircuit-generation")
        run_name = wandb_cfg.get("run_name", wandb_cfg.get("experiment_name"))
        return wandb.init(project=project, name=run_name, config=dict(self.config))

    @staticmethod
    def _parse_srv_targets(labels: np.ndarray) -> torch.Tensor:
        """Extract SRV vectors from stored prompt strings."""
        srv_list = []
        for label in labels:
            text = str(label)
            start = text.find("[")
            end = text.find("]", start)
            if start == -1 or end == -1:
                raise ValueError(f"Could not parse SRV from label: {text}")
            srv = ast.literal_eval(text[start:end + 1])
            srv_list.append(srv)
        return torch.tensor(srv_list, dtype=torch.long)

    def _accuracy_per_entangled(self, target_srvs, predicted_srvs):
        n_samples_per_entangled = {n: 0 for n in range(self.num_qubits + 1) if n != 1}
        correct_per_entangled = {n: 0 for n in range(self.num_qubits + 1) if n != 1}
        acc_per_entangled = {n: 0 for n in range(self.num_qubits + 1) if n != 1}

        for target, predicted in zip(target_srvs, predicted_srvs):
            n_entangled = int((target == 2).sum())  # qubit is entangled if value in SRV = 2
            n_samples_per_entangled[n_entangled] += 1

            if (target != predicted).sum() == 0:
                correct_per_entangled[n_entangled] += 1

        for n_entangled, n_correct, n_samples in zip(correct_per_entangled.keys(), correct_per_entangled.values(),
                                                     n_samples_per_entangled.values()):
            if n_samples > 0:
                acc_per_entangled[n_entangled] = n_correct / n_samples

        return acc_per_entangled

    @staticmethod
    def _get_exact_match_rate(target_srvs, predicted_srvs):
        exact_match = (predicted_srvs == target_srvs).all(dim=1)
        srv_exact_match_rate = exact_match.float().mean().item()

        return srv_exact_match_rate


    def _load_pipeline(self, model_dir: Path | None, repo_id: str | None):
        if repo_id:
            pipeline = DiffusionPipeline.from_pretrained(repo_id=repo_id, device=self.device)
            pipeline.guidance_sample_mode = "rescaled"
            pipeline.scheduler.set_timesteps(self.config.model_params.sample_steps)
            return pipeline

        if not model_dir:
            raise ValueError("Provide either model-dir or hf-repo in config.")

        model_dir = model_dir.resolve()
        config_path = model_dir if model_dir.is_dir() else model_dir.parent
        cfg_file = config_path / "config.yaml"
        if not cfg_file.exists():
            raise FileNotFoundError(f"Missing pipeline config at {cfg_file}")

        # DiffusionPipeline expects a directory string ending with '/'
        pipeline = DiffusionPipeline.from_config_file(config_path=str(config_path) + "/", device=self.device)
        pipeline.guidance_sample_mode = "rescaled"
        pipeline.scheduler.set_timesteps(self.config.model_params.sample_steps)

        return pipeline


    def generate_tensors(self, save_output: bool = True, save_path: str | None = None):
        self.logger.info("Starting tensor generation...")

        prompts = [str(p) for p in self.dataset.y[:self.samples]]

        start_time = time.time()
        tensors_out = generate_tensors(
            pipeline=self.pipeline,
            prompt=prompts,
            samples=self.samples,
            system_size=self.system_size,
            num_of_qubits=self.num_qubits,
            max_gates=self.max_gates,
            g=self.config.model_params.guidance_scale,
            auto_batch_size=self.config.model_params.auto_batch_size,
            enable_params=False,
            no_bar=False,  # shows diffusion steps
        )
        self.logger.info(f"Finished tensor generation. Took {(time.time() - start_time):.2f} seconds.")

        if save_output and save_path:
            self.logger.info(f"Saving generated tensors to {save_path}...")
            save_path = os.path.join(save_path, f"{self.num_qubits}q_{self.samples}_samples.pt")  # _{timestamp}.pt")
            store_tensor(tensors_out, save_path)
            self.logger.info("Saving successful.")

        return tensors_out


    def decode_tensors(self, tensors_out: torch.Tensor):
        self.logger.info("Decoding tensors...")

        start_time = time.time()
        decoded_circuits, _ = decode_tensors_to_backend(
            simulator=self.simulator,
            tokenizer=self.tokenizer,
            tensors=tensors_out,
            params=None,
            silent=True,
            n_jobs=1,
            filter_errs=False,
        )
        self.logger.info(f"Finished tensor decoding. Took {(time.time() - start_time):.2f} seconds.")

        return decoded_circuits


    def validate_and_calculate_srvs(self, decoded_circuits, save_output: bool = True, save_path: str | None = None):

        valid = [(idx, qc) for idx, qc in enumerate(decoded_circuits) if qc is not None]

        valid_indices = [idx for idx, _ in valid]
        backend_circuits = [qc for _, qc in valid]
        err_cnt = len(decoded_circuits) - len(backend_circuits)

        self.logger.info("==== genQC Evaluation ====")
        self.logger.info(f"Samples requested: {self.samples}")
        self.logger.info(f"Decoded circuits : {len(backend_circuits)}")
        self.logger.info(f"Decode failures  : {err_cnt}")

        if self.wandb_run:
            self.wandb_run.summary["eval/decoded_circuits"] = len(backend_circuits)
            self.wandb_run.summary["eval/decode_failures"] = err_cnt

        target_srvs = self._parse_srv_targets(self.dataset.y[:self.samples])[valid_indices]

        self.logger.info("Calculating SRVs...")
        predicted_srvs = torch.tensor(
            get_srvs(self.simulator, backend_circuits, n_jobs=1),
            dtype=torch.long,
        )

        if save_output and save_path:
            self.logger.info(f"Saving generated tensors to {save_path}...")

            save_path_srvs = os.path.join(save_path, f"{self.num_qubits}q_predicted_srvs.pt")
            store_tensor(predicted_srvs, save_path_srvs)

            save_path_ids = os.path.join(save_path, f"{self.num_qubits}q_valid_indices.pt")
            store_tensor(torch.tensor(valid_indices), save_path_ids)

            self.logger.info("Saving successful.")

        return valid_indices, target_srvs, predicted_srvs


    def calculate_metrics(self, target_srvs, predicted_srvs):
        self.logger.info("Calculating metrics...")

        acc_per_entangled = self._accuracy_per_entangled(target_srvs, predicted_srvs)
        srv_exact_match_rate = self._get_exact_match_rate(target_srvs, predicted_srvs)

        self.logger.info(f"Exact SRV match rate: {srv_exact_match_rate:.4f}")

        for n_entangled, acc in acc_per_entangled.items():
            self.logger.info(f"{n_entangled} entangled qubits: {acc:.4f} acc")

        if self.wandb_run:
            self.wandb_run.summary["eval/srv_exact_match_rate"] = srv_exact_match_rate
            for i, acc in acc_per_entangled.items():
                self.wandb_run.summary[f"eval/n_entangled_acc/{i}"] = acc

        return srv_exact_match_rate, acc_per_entangled


    def evaluate(self):
        tensors_out = self.generate_tensors(save_output=self.config.save_output, save_path=self.config.save_folder)
        decoded_circuits = self.decode_tensors(tensors_out)
        valid_indices, target_srvs, predicted_srvs = self.validate_and_calculate_srvs(decoded_circuits,
                                                                                      save_output=self.config.save_output,
                                                                                      save_path=self.config.save_folder)
        srv_exact_match_rate, acc_per_entanglement = self.calculate_metrics(target_srvs, predicted_srvs)

        return srv_exact_match_rate, acc_per_entanglement
