"""
Configuration management utilities for RFSinger.
Handles loading, saving, and merging configurations from YAML files.
"""

import os
import yaml
import shutil
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the YAML file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Recursively merge two configuration dictionaries.
    Values in override_config take precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
            
    return result


class Config:
    """
    Configuration class that provides attribute-style access to config values.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with a default fallback."""
        return getattr(self, key, default)
    
    def __repr__(self):
        return f"Config({self.to_dict()})"


def get_experiment_dir(config: Dict[str, Any], base_dir: Optional[str] = None) -> str:
    """
    Get the experiment directory path based on config.
    
    Args:
        config: Configuration dictionary
        base_dir: Override base directory (default: from config)
        
    Returns:
        Path to the experiment directory
    """
    if base_dir is None:
        base_dir = config['checkpoint']['output_dir']
    
    exp_name = config['experiment']['name']
    return os.path.join(base_dir, exp_name)


def setup_experiment_dir(config: Dict[str, Any], copy_config: bool = True) -> str:
    """
    Set up the experiment directory structure.
    Creates the directory and optionally copies the config file.
    
    Args:
        config: Configuration dictionary
        copy_config: Whether to copy config.yaml to experiment dir
        
    Returns:
        Path to the experiment directory
    """
    exp_dir = get_experiment_dir(config)
    
    # Create experiment directory
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config to experiment directory
    if copy_config:
        config_save_path = os.path.join(exp_dir, 'config.yaml')
        save_config(config, config_save_path)
        print(f"Config saved to {config_save_path}")
    
    return exp_dir


def load_config_from_checkpoint(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Load configuration from a checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory containing config.yaml
        
    Returns:
        Configuration dictionary
    """
    # Try to find config.yaml in checkpoint directory or parent
    config_paths = [
        os.path.join(checkpoint_dir, 'config.yaml'),
        os.path.join(os.path.dirname(checkpoint_dir), 'config.yaml'),
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            return load_config(config_path)
    
    raise FileNotFoundError(f"Config file not found in {checkpoint_dir}")


def get_hparams_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract preprocessing hyperparameters from config in the format
    expected by preprocess.py
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        HPARAMS dictionary for preprocessing
    """
    return {
        "sample_rate": config['audio']['sample_rate'],
        "n_fft": config['audio']['n_fft'],
        "hop_length": config['audio']['hop_length'],
        "win_length": config['audio']['win_length'],
        "n_mels": config['audio']['n_mels'],
        "fmin": config['audio']['fmin'],
        "fmax": config['audio']['fmax'],
        "data_path": config['data']['raw_data_path'],
        "csv_path": config['data']['csv_path'],
        "output_dir": config['data']['processed_dir'],
        "use_log_f0": config['f0']['use_log_f0'],
        "f0_floor": config['f0']['f0_floor'],
        "f0_ceil": config['f0']['f0_ceil'],
        "f0_method": config['f0'].get('method', 'harvest'),
        "rmvpe_ckpt": config['f0'].get('rmvpe_ckpt', './ckpts/rmvpe.pt'),
        # Augmentation parameters
        "augmentation": config.get('augmentation', {}),
        # Multi-speaker parameters
        "speakers": config['data'].get('speakers', None),
    }


def get_model_args_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model arguments from config.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Model arguments dictionary
    """
    model_cfg = config['model']
    return {
        'd_model': model_cfg['d_model'],
        'n_encoder_layers': model_cfg['n_encoder_layers'],
        'n_head': model_cfg['n_head'],
        'mel_channels': model_cfg['mel_channels'],
        'flow_hidden': model_cfg['flow_hidden'],
        'n_flow_layers': model_cfg['n_flow_layers'],
        # Coarse Mel Decoder parameters (CNN-Transformer-CNN architecture)
        'coarse_n_layers': model_cfg.get('coarse_n_layers', 2),
        'coarse_n_head': model_cfg.get('coarse_n_head', 2),
        'coarse_conv_channels': model_cfg.get('coarse_conv_channels', 512),
        'coarse_kernel_size': model_cfg.get('coarse_kernel_size', 5),
    }


def get_training_args_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training arguments from config.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Training arguments dictionary
    """
    return {
        'epochs': config['training']['epochs'],
        'batch_size': config['training']['batch_size'],
        'num_workers': config['training']['num_workers'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'betas': tuple(config['training']['betas']),
        'grad_clip': config['training']['grad_clip'],
        'loss_weights': config['training']['loss_weights'],
        'scheduler': config['training']['scheduler'],
        'lr_min_ratio': config['training']['lr_min_ratio'],
    }


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """
    Pretty print configuration.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")
