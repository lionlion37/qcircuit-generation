"""Logging utilities for the quantum diffusion framework."""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


class Logger:
    """Custom logger for the quantum diffusion framework."""
    
    def __init__(self, 
                 name: str,
                 level: str = "INFO",
                 log_file: Optional[Union[str, Path]] = None,
                 console_output: bool = True):
        """Initialize logger.
        
        Args:
            name: Logger name (usually __name__)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            console_output: Whether to output to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
    
    def log_experiment_start(self, experiment_name: str, config: dict) -> None:
        """Log the start of an experiment."""
        self.info(f"Starting experiment: {experiment_name}")
        self.info(f"Configuration: {config}")
    
    def log_experiment_end(self, experiment_name: str, duration: float) -> None:
        """Log the end of an experiment."""
        self.info(f"Experiment '{experiment_name}' completed in {duration:.2f} seconds")
    
    def log_training_progress(self, epoch: int, total_epochs: int, loss: float) -> None:
        """Log training progress."""
        self.info(f"Epoch {epoch}/{total_epochs}, Loss: {loss:.6f}")
    
    def log_evaluation_results(self, results: dict) -> None:
        """Log evaluation results."""
        self.info("Evaluation Results:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                self.info(f"  {key}: {value}")
            elif isinstance(value, dict):
                self.info(f"  {key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        self.info(f"    {subkey}: {subvalue}")


class ExperimentLogger:
    """Logger specifically for tracking experiments."""
    
    def __init__(self, experiment_name: str, log_dir: str = "./logs"):
        """Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        self.logger = Logger(
            name=f"experiment.{experiment_name}",
            log_file=log_file
        )
        
        self.start_time = None
        self.metrics = {}
    
    def start_experiment(self, config: dict) -> None:
        """Start logging an experiment."""
        self.start_time = time.time()
        self.logger.log_experiment_start(self.experiment_name, config)
    
    def log_step(self, step: str, message: str) -> None:
        """Log an experiment step."""
        self.logger.info(f"[{step}] {message}")
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        metric_entry = {"value": value, "timestamp": time.time()}
        if step is not None:
            metric_entry["step"] = step
        
        self.metrics[metric_name].append(metric_entry)
        
        step_info = f" (step {step})" if step is not None else ""
        self.logger.info(f"Metric {metric_name}: {value}{step_info}")
    
    def log_training_epoch(self, epoch: int, total_epochs: int, train_loss: float, val_loss: Optional[float] = None) -> None:
        """Log training epoch information."""
        val_info = f", Val Loss: {val_loss:.6f}" if val_loss is not None else ""
        self.logger.log_training_progress(epoch, total_epochs, train_loss)
        if val_loss is not None:
            self.log_metric("val_loss", val_loss, epoch)
        self.log_metric("train_loss", train_loss, epoch)
    
    def log_evaluation(self, results: dict) -> None:
        """Log evaluation results."""
        self.logger.log_evaluation_results(results)
        
        # Store evaluation metrics
        for key, value in results.items():
            if isinstance(value, (int, float)):
                self.log_metric(f"eval_{key}", value)
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        self.log_metric(f"eval_{key}_{subkey}", subvalue)
    
    def end_experiment(self, success: bool = True) -> float:
        """End the experiment and return duration."""
        if self.start_time is None:
            self.logger.warning("Experiment was not properly started")
            return 0.0
        
        duration = time.time() - self.start_time
        status = "successfully" if success else "with errors"
        self.logger.log_experiment_end(self.experiment_name, duration)
        
        return duration
    
    def get_metrics_summary(self) -> dict:
        """Get summary of all logged metrics."""
        summary = {}
        for metric_name, entries in self.metrics.items():
            values = [entry["value"] for entry in entries]
            summary[metric_name] = {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": sum(values) / len(values) if values else None,
                "latest": values[-1] if values else None
            }
        return summary


def setup_logging(log_level: str = "INFO", 
                  log_dir: str = "./logs",
                  console_output: bool = True) -> None:
    """Setup global logging configuration.
    
    Args:
        log_level: Global logging level
        log_dir: Directory for log files
        console_output: Whether to show logs in console
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup basic logging configuration
    log_file = log_dir / "quantum_diffusion.log"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) if console_output else logging.NullHandler()
        ]
    )
    
    # Suppress some verbose libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


# Convenience function for getting loggers
def get_logger(name: str, **kwargs) -> Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for Logger
        
    Returns:
        Logger instance
    """
    return Logger(name, **kwargs)