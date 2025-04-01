import os
import json
from pathlib import Path
from datetime import datetime

from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    set_seed,
    Trainer,
    TrainingArguments,
    PatchTSTForPretraining
)
import numpy as np
import pandas as pd
from src.data.pred_dataset import TUH_Dataset_Test
import torch

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

# Environment variables for Neptune logging
os.environ["NEPTUNE_API_TOKEN"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjNzI3OWFjMy02ZWQ5LTQzMTctOGYxMC1iMDliZTk3N2Y1YzMifQ=="
os.environ["NEPTUNE_PROJECT"] = "gayalkuruppu/tuh-ssl-test"

# NCCL settings for distributed training
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

@dataclass
class ExperimentConfig:
    """Configuration for the experiment"""
    # Experiment metadata
    experiment_name: str = "tuhab"  # Base name for the experiment
    description: str = ""  # Optional description of the experiment
    
    # Model parameters
    context_length: int = 1000
    forecast_horizon: int = 96
    patch_length: int = 100
    patch_stride: int = 100
    num_workers: int = 16
    batch_size: int = 128
    random_mask_ratio: float = 0.4
    
    # Training parameters
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    max_steps: int = 1000000
    eval_steps: int = 20000
    logging_steps: int = 10000
    save_steps: float = 0.05
    
    # Data parameters
    dataset_name: str = "tuhab"  # Name of the dataset used
    features: str = 'M'
    scale: bool = False
    use_time_features: bool = False
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Machine specific paths
    machine_paths: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        if self.machine_paths is None:
            self.machine_paths = {
                'cs-u-bach': {
                    'base_dir': '/home/gayal/ssl-analyses-repos/PatchTST',
                    'data_dir': '/home/gayal/ssl-analyses-repos/PatchTST/tuhab_records/tuhab_records_cropped',
                },
                'cs-u-vivaldi': {
                    'base_dir': '/home/gayal/ssl-project/PatchTST',
                    'data_dir': '/home/gayal/ssl-project/PatchTST/tuhab_records/tuhab_records_cropped',
                }
            }
            
            # Add csv_path derived from data_dir for each machine
            for machine in self.machine_paths:
                data_dir = self.machine_paths[machine]['data_dir']
                self.machine_paths[machine]['csv_path'] = os.path.join(data_dir, 'file_lengths_map.csv')

def get_machine_paths(config: ExperimentConfig) -> Dict[str, str]:
    """Get paths based on current machine"""
    import socket
    hostname = socket.gethostname()
    return config.machine_paths.get(hostname, config.machine_paths['cs-u-bach'])

def create_experiment_dirs(base_dir: str, config: ExperimentConfig) -> Dict[str, str]:
    """Create experiment directories and return paths"""
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment name with key parameters
    exp_name = f"{config.experiment_name}_{config.dataset_name}_{config.max_steps//1000}k_{timestamp}"
    
    # Create nested directory structure
    exp_dir = Path(base_dir) / 'patchtst_pretrained_models' / config.dataset_name / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs and outputs directories
    logs_dir = exp_dir / 'logs'
    outputs_dir = exp_dir / 'outputs'
    logs_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    return {
        'exp_dir': str(exp_dir),
        'logs_dir': str(logs_dir),
        'outputs_dir': str(outputs_dir)
    }

def save_config(config: ExperimentConfig, paths: Dict[str, str]):
    """Save experiment configuration"""
    config_path = Path(paths['exp_dir']) / 'config.json'
    
    # Get current machine paths
    current_paths = get_machine_paths(config)
    
    # Create config dictionary with all parameters including paths
    config_dict = {
        k: v for k, v in config.__dict__.items() 
        if not k.startswith('_') and k != 'machine_paths'
    }
    
    # Add current machine paths
    config_dict['data_paths'] = current_paths
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def main():
    # Initialize experiment configuration
    config = ExperimentConfig(
        experiment_name="tuhab_ssl",  # Custom experiment name
        description="Self-supervised learning on TUHAB dataset",
        dataset_name="tuhab",
        context_length=1000,
        patch_length=100,
        max_steps=1000000,
        random_mask_ratio=0.4,
        lr_scheduler_type="cosine",
        random_seed=42  # Set random seed
    )
    
    # Get machine-specific paths
    paths = get_machine_paths(config)
    
    # Create experiment directories
    dirs = create_experiment_dirs(paths['base_dir'], config)
    
    # Save configuration
    save_config(config, dirs)
    
    # Set random seeds for reproducibility
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    set_seed(config.random_seed)
    
    # Initialize datasets
    tuh_data = TUH_Dataset_Test(
        root_path=paths['data_dir'],
        data_path='',
        csv_path=paths['csv_path'],
        features=config.features,
        scale=config.scale,
        size=[config.context_length, 0, config.patch_length],
        use_time_features=config.use_time_features
    )

    tuh_eval_data = TUH_Dataset_Test(
        root_path=paths['data_dir'],
        data_path='',
        csv_path=paths['csv_path'],
        features=config.features,
        scale=config.scale,
        size=[config.context_length, 0, config.patch_length],
        use_time_features=config.use_time_features,
        split='val'
    )
    
    # Initialize model config
    model_config = PatchTSTConfig(
        num_input_channels=19,
        context_length=config.context_length,
        patch_length=config.patch_length,
        patch_stride=config.patch_stride,
        mask_type='random',
        random_mask_ratio=config.random_mask_ratio,
        use_cls_token=True,
    )
    
    # Initialize model
    model = PatchTSTForPretraining(model_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=dirs['outputs_dir'],
        overwrite_output_dir=True,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        max_steps=config.max_steps,
        do_eval=True,
        eval_strategy="steps",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        dataloader_num_workers=config.num_workers,
        save_strategy="steps",
        logging_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        greater_is_better=False,
        logging_dir=dirs['logs_dir'],
        report_to="neptune"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tuh_data,
        eval_dataset=tuh_eval_data,
    )
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()

# NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python IMU_SSL.py