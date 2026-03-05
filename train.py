import os
import json
import math
import argparse
import glob
import shutil
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing import Optional

from dataset import SingingDataset, collate_fn
from fs2encoder import FastSpeech2Encoder, VarianceAdaptor
from reflow import RectifiedFlowDecoder
from utils.config import (
    load_config, save_config, Config, 
    setup_experiment_dir, get_experiment_dir,
    get_model_args_from_config, get_training_args_from_config,
    print_config
)


class EMA:
    """
    Exponential Moving Average for model parameters.
    Maintains a shadow copy of model parameters that is updated with exponential decay.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        """
        Args:
            model: The model to track
            decay: EMA decay rate (higher = slower update, more stable)
            device: Device to store shadow parameters
        """
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}
        self.num_updates = 0
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(device if device else param.device)
    
    def update(self, model: nn.Module):
        """
        Update shadow parameters with current model parameters.
        Uses warmup: decay increases from 0 to target decay over first updates.
        """
        self.num_updates += 1
        # Warmup: use smaller decay at the beginning
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def apply_shadow(self, model: nn.Module):
        """
        Apply shadow parameters to model (for evaluation/inference).
        Backs up current parameters first.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module):
        """
        Restore original parameters from backup (after evaluation).
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'shadow': {k: v.cpu() for k, v in self.shadow.items()}
        }
    
    def load_state_dict(self, state_dict, device=None):
        """Load state dict from checkpoint."""
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow = {}
        for k, v in state_dict['shadow'].items():
            self.shadow[k] = v.to(device if device else self.device)


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing learning rate scheduler."""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.param_groups[0]['lr']
        self.min_lr = self.base_lr * min_lr_ratio
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]
    
    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr,
        }
    
    def load_state_dict(self, state_dict):
        self.current_step = state_dict.get('current_step', 0)
        if 'warmup_steps' in state_dict:
            self.warmup_steps = state_dict['warmup_steps']
        if 'total_steps' in state_dict:
            self.total_steps = state_dict['total_steps']
        if 'base_lr' in state_dict:
            self.base_lr = state_dict['base_lr']
        if 'min_lr' in state_dict:
            self.min_lr = state_dict['min_lr']


class RFSingerModel(nn.Module):
    """Complete RFSinger Model combining FastSpeech2 Encoder and Rectified Flow Decoder"""
    
    def __init__(self, vocab_size, d_model=256, n_encoder_layers=4, n_head=2, 
                 mel_channels=128, flow_hidden=256, n_flow_layers=20, dropout=0.1,
                 coarse_n_layers=2, coarse_n_head=2, coarse_conv_channels=512, coarse_kernel_size=5,
                 n_speakers=1, speaker_embedding_dim=256):
        super().__init__()
        self.mel_channels = mel_channels
        self.n_speakers = n_speakers
        
        # FastSpeech2 Encoder
        self.encoder = FastSpeech2Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_head=n_head,
            dropout=dropout,
            n_speakers=n_speakers,
            speaker_embedding_dim=speaker_embedding_dim
        )
        
        # Variance Adaptor (Duration, Pitch) with enhanced Coarse Mel Decoder
        self.variance_adaptor = VarianceAdaptor(
            d_model=d_model,
            mel_channels=mel_channels,
            dropout=dropout,
            coarse_n_layers=coarse_n_layers,
            coarse_n_head=coarse_n_head,
            coarse_conv_channels=coarse_conv_channels,
            coarse_kernel_size=coarse_kernel_size
        )
        
        # Rectified Flow Decoder
        self.flow_decoder = RectifiedFlowDecoder(
            in_channels=mel_channels,
            hidden_channels=flow_hidden,
            n_layers=n_flow_layers,
            dropout=dropout,
            cond_channels=d_model
        )
        
    def forward(self, text, src_mask, mel_mask, duration, f0, mel_target=None, t=None, uv=None, speaker_id=None):
        """
        Args:
            text: (B, T_text) phoneme ids
            src_mask: (B, T_text) text mask
            mel_mask: (B, T_mel) mel mask
            duration: (B, T_text) duration for each phoneme
            f0: (B, T_mel) pitch contour (required, normalized)
            mel_target: (B, T_mel, 128) target mel for training
            t: (B,) time step for flow matching (0 to 1)
            uv: (B, T_mel) unvoiced mask (1=unvoiced, 0=voiced), optional
            speaker_id: (B,) speaker index, optional (for multi-speaker)
        """
        # 1. Encode phonemes
        encoder_out = self.encoder(text, src_mask, speaker_id=speaker_id)  # (B, T_text, d_model)
        
        # 2. Variance adaptation (expand to mel length, add pitch with UV handling)
        adapted, coarse_mel = self.variance_adaptor(
            encoder_out, src_mask, mel_mask, duration, f0, uv=uv
        )  # adapted: (B, T_mel, d_model)
        
        if mel_target is not None:
            # Training mode: compute flow matching loss
            # Rectified Flow: x_t = (1-t) * x_0 + t * x_1
            # where x_0 is noise (coarse_mel), x_1 is target (mel_target)
            
            batch_size = mel_target.size(0)
            
            # Sample random time for each batch item
            if t is None:
                t = torch.rand(batch_size, device=mel_target.device)
            
            # Create noisy interpolation
            t_expand = t[:, None, None]  # (B, 1, 1)
            x_t = (1 - t_expand) * coarse_mel + t_expand * mel_target
            
            # Predict velocity (target - noise direction)
            v_pred = self.flow_decoder(x_t, t, adapted)  # (B, T_mel, 128)
            
            # True velocity
            v_target = mel_target - coarse_mel
            
            return {
                'v_pred': v_pred,
                'v_target': v_target,
                'coarse_mel': coarse_mel,
                'mel_mask': mel_mask
            }
        else:
            # Inference mode
            return {
                'coarse_mel': coarse_mel,
                'adapted': adapted,
            }
    
    @torch.no_grad()
    def inference(self, text, src_mask, duration, f0, n_steps=10, uv=None, speaker_id=None):
        """
        Generate mel spectrogram from text using ODE sampling.
        
        Args:
            text: (B, T_text) phoneme ids
            src_mask: (B, T_text) text mask
            duration: (B, T_text) duration
            f0: (B, T_mel) pitch contour (required, normalized)
            n_steps: number of ODE steps
            uv: (B, T_mel) unvoiced mask (1=unvoiced, 0=voiced), optional
            speaker_id: (B,) speaker index, optional (for multi-speaker)
        
        Returns:
            mel: (B, T_mel, mel_channels) generated mel spectrogram
        """
        # Compute mel length from duration
        mel_len = duration.sum(dim=1).max().item()
        mel_mask = torch.zeros(text.size(0), mel_len, dtype=torch.bool, device=text.device)
        for i in range(text.size(0)):
            mel_mask[i, :duration[i].sum()] = True
        
        # Encode
        encoder_out = self.encoder(text, src_mask, speaker_id=speaker_id)
        
        # Variance adaptation (F0 is required, UV is optional)
        adapted, coarse_mel = self.variance_adaptor(
            encoder_out, src_mask, mel_mask, duration, f0, uv=uv
        )
        
        # ODE sampling: integrate from t=0 (coarse) to t=1 (refined)
        x = coarse_mel
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.full((x.size(0),), step * dt, device=x.device)
            v = self.flow_decoder(x, t, adapted)
            x = x + v * dt
        
        return x


def compute_loss(outputs, mel_mask, loss_weights=None):
    """
    Compute training losses.
    
    Args:
        outputs: model outputs dict
        mel_mask: mel spectrogram mask
        loss_weights: optional loss weight dict
    
    Returns:
        total_loss, loss_dict
    """
    if loss_weights is None:
        loss_weights = {
            'flow': 1.0,
            'coarse': 0.2,  # Weight for coarse mel loss
        }
    
    v_pred = outputs['v_pred']
    v_target = outputs['v_target']
    coarse_mel = outputs['coarse_mel']
    mel_mask = outputs['mel_mask']
    
    # Flow matching loss (MSE on velocity)
    flow_loss = nn.functional.mse_loss(v_pred, v_target, reduction='none')
    flow_loss = (flow_loss * mel_mask.unsqueeze(-1)).sum() / mel_mask.sum() / v_pred.size(-1)
    
    # Coarse mel loss (MSE between coarse mel and target)
    # This helps the variance adaptor learn better initial estimates
    mel_target = coarse_mel + v_target  # Reconstruct target from coarse + velocity
    coarse_loss = nn.functional.mse_loss(coarse_mel, mel_target, reduction='none')
    coarse_loss = (coarse_loss * mel_mask.unsqueeze(-1)).sum() / mel_mask.sum() / coarse_mel.size(-1)
    
    total_loss = loss_weights['flow'] * flow_loss + loss_weights.get('coarse', 0.1) * coarse_loss
    
    return total_loss, {
        'total': total_loss.item(),
        'flow': flow_loss.item(),
        'coarse': coarse_loss.item(),
    }


def validate(model, val_loader, device, loss_weights, use_amp=False):
    """
    Run validation and return average losses.
    
    Args:
        model: RFSingerModel
        val_loader: validation DataLoader
        device: torch device
        loss_weights: loss weight dict
        use_amp: whether to use automatic mixed precision
    
    Returns:
        dict of average losses
    """
    model.eval()
    total_losses = {'total': 0.0, 'flow': 0.0, 'coarse': 0.0}
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            text, duration, f0, uv, mel, src_mask, mel_mask, speaker_ids = batch
            text = text.to(device)
            duration = duration.to(device)
            f0 = f0.to(device)
            uv = uv.to(device)
            mel = mel.to(device)
            src_mask = src_mask.to(device)
            mel_mask = mel_mask.to(device)
            speaker_ids = speaker_ids.to(device)
            
            t = torch.rand(text.size(0), device=device)
            
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
                outputs = model(text, src_mask, mel_mask, duration, f0, mel, t, uv=uv, speaker_id=speaker_ids)
                _, loss_dict = compute_loss(outputs, mel_mask, loss_weights)
            
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            n_batches += 1
    
    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    model.train()
    return avg_losses


class CheckpointManager:
    """Manages checkpoint saving, loading, and cleanup."""
    
    def __init__(self, exp_dir: str, keep_last_n: int = 5, save_best: bool = True):
        """
        Args:
            exp_dir: Experiment directory (checkpoints saved directly here)
            keep_last_n: Keep only last N checkpoints (0 = keep all)
            save_best: Whether to save best model separately
        """
        self.ckpt_dir = exp_dir  # Save directly in experiment directory
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.best_loss = float('inf')
        
        os.makedirs(self.ckpt_dir, exist_ok=True)
    
    def save(self, state: dict, epoch: int, loss: float, is_latest: bool = True) -> str:
        """
        Save a checkpoint.
        
        Args:
            state: Checkpoint state dict
            epoch: Current epoch
            loss: Current loss for best model tracking
            is_latest: Whether to also save as latest
            
        Returns:
            Path to saved checkpoint
        """
        # Save epoch checkpoint
        ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save(state, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")
        
        # Save latest
        if is_latest:
            latest_path = os.path.join(self.ckpt_dir, "checkpoint_latest.pt")
            torch.save(state, latest_path)
        
        # Save best
        if self.save_best and loss < self.best_loss:
            self.best_loss = loss
            best_path = os.path.join(self.ckpt_dir, "checkpoint_best.pt")
            torch.save(state, best_path)
            print(f"New best model saved (loss: {loss:.4f})")
        
        # Cleanup old checkpoints
        self._cleanup()
        
        return ckpt_path
    
    def _cleanup(self):
        """Remove old checkpoints, keeping only the last N."""
        if self.keep_last_n <= 0:
            return
        
        # Find all epoch checkpoints (not latest/best)
        pattern = os.path.join(self.ckpt_dir, "checkpoint_epoch*.pt")
        ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
        
        # Remove old ones
        while len(ckpts) > self.keep_last_n:
            old_ckpt = ckpts.pop(0)
            os.remove(old_ckpt)
            print(f"Removed old checkpoint: {old_ckpt}")
    
    def load_latest(self, device='cpu'):
        """Load the latest checkpoint if exists."""
        latest_path = os.path.join(self.ckpt_dir, "checkpoint_latest.pt")
        if os.path.exists(latest_path):
            return torch.load(latest_path, map_location=device)
        return None
    
    def get_latest_path(self) -> str:
        """Get path to latest checkpoint."""
        return os.path.join(self.ckpt_dir, "checkpoint_latest.pt")
    
    def get_best_path(self) -> str:
        """Get path to best checkpoint."""
        return os.path.join(self.ckpt_dir, "checkpoint_best.pt")


def train(config: dict):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
    """
    # Set random seed
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup experiment directory
    exp_dir = setup_experiment_dir(config, copy_config=True)
    print(f"Experiment directory: {exp_dir}")
    
    # Extract config sections
    data_config = config['data']
    model_config = config['model']
    train_config = config['training']
    ckpt_config = config['checkpoint']
    log_config = config['logging']
    
    # Load phone map
    data_dir = data_config['processed_dir']
    with open(os.path.join(data_dir, 'phone_map.json'), 'r') as f:
        phone_map = json.load(f)
    vocab_size = len(phone_map)
    print(f"Vocabulary size: {vocab_size}")
    
    # Load speaker map (multi-speaker support)
    speaker_map_path = os.path.join(data_dir, 'speaker_map.json')
    n_speakers = 1
    if os.path.exists(speaker_map_path):
        with open(speaker_map_path, 'r') as f:
            speaker_map = json.load(f)
        n_speakers = len(speaker_map)
        print(f"Multi-speaker mode: {n_speakers} speakers: {list(speaker_map.keys())}")
    else:
        print("Single-speaker mode (no speaker_map.json found)")
    
    # Dataset and DataLoader
    reflow_dir = data_config.get('reflow_dir', None)
    
    # Online augmentation config (applied per-sample during training)
    aug_config = config.get('augmentation', {})
    online_aug = aug_config.get('online', {}) if aug_config.get('enabled', False) else {}
    
    train_dataset = SingingDataset(
        meta_file=os.path.join(data_dir, 'train.txt'),
        data_dir=data_dir,
        phone_map=phone_map,
        reflow_dir=reflow_dir,
        augmentation=online_aug
    )
    if reflow_dir:
        print(f"Using reflow targets from {reflow_dir}")
    
    # Validation split
    val_ratio = train_config.get('val_ratio', 0.0)
    val_loader = None
    if val_ratio > 0:
        n_val = max(1, int(len(train_dataset) * val_ratio))
        n_train = len(train_dataset) - n_val
        train_subset, val_subset = random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=train_config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=train_config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        print(f"Training samples: {n_train}, Validation samples: {n_val}")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=train_config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        print(f"Training samples: {len(train_dataset)}")
    
    # Model
    use_spk_emb = model_config.get('use_speaker_embedding', True) and n_speakers > 1
    model = RFSingerModel(
        vocab_size=vocab_size,
        d_model=model_config['d_model'],
        n_encoder_layers=model_config['n_encoder_layers'],
        n_head=model_config['n_head'],
        mel_channels=model_config['mel_channels'],
        flow_hidden=model_config['flow_hidden'],
        n_flow_layers=model_config['n_flow_layers'],
        dropout=model_config.get('dropout', 0.1),
        coarse_n_layers=model_config.get('coarse_n_layers', 2),
        coarse_n_head=model_config.get('coarse_n_head', 2),
        coarse_conv_channels=model_config.get('coarse_conv_channels', 512),
        coarse_kernel_size=model_config.get('coarse_kernel_size', 5),
        n_speakers=n_speakers if use_spk_emb else 1,
        speaker_embedding_dim=model_config.get('speaker_embedding_dim', 256),
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Freeze encoder for reflow training
    if train_config.get('freeze_encoder', False):
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.variance_adaptor.parameters():
            param.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Froze encoder and variance adaptor for reflow training")
        print(f"Trainable parameters (after freeze): {trainable_params:,}")
    
    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        betas=tuple(train_config['betas']),
        weight_decay=train_config['weight_decay']
    )
    
    # Learning rate scheduler with warmup
    warmup_steps = train_config.get('warmup_steps', 0)
    total_steps = train_config['epochs'] * len(train_loader)
    scheduler_type = train_config.get('scheduler', 'cosine')
    if scheduler_type == 'cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=train_config.get('lr_min_ratio', 0.1)
        )
        print(f"Using WarmupCosine scheduler (warmup={warmup_steps} steps, total={total_steps} steps)")
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_config.get('step_size', 30) * len(train_loader),
            gamma=train_config.get('gamma', 0.5)
        )
    else:
        scheduler = None
    
    # AMP (Automatic Mixed Precision)
    use_amp = train_config.get('use_amp', False) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("AMP (Automatic Mixed Precision) enabled")
    
    # Checkpoint manager
    ckpt_manager = CheckpointManager(
        exp_dir=exp_dir,
        keep_last_n=ckpt_config.get('keep_last_n', 5),
        save_best=ckpt_config.get('save_best', True)
    )
    
    # EMA (Exponential Moving Average)
    use_ema = train_config.get('use_ema', True)
    ema = None
    if use_ema:
        ema_decay = train_config.get('ema_decay', 0.9999)
        ema = EMA(model, decay=ema_decay, device=device)
        print(f"EMA enabled with decay={ema_decay}")
    
    # Resume training if checkpoint exists
    start_epoch = 0
    global_step = 0
    resume_path = ckpt_manager.get_latest_path()
    if os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except (KeyError, TypeError, AttributeError):
                # Fallback for format mismatch (e.g., switching scheduler type)
                if hasattr(scheduler, 'current_step'):
                    scheduler.current_step = global_step
                print("Warning: Scheduler state format mismatch, using global_step for position")
        start_epoch = checkpoint['epoch'] + 1
        if 'best_loss' in checkpoint:
            ckpt_manager.best_loss = checkpoint['best_loss']
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        # Restore EMA state
        if ema and 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'], device=device)
            print(f"Restored EMA state (num_updates={ema.num_updates})")
        # Restore AMP scaler state
        if use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("Restored AMP scaler state")
    
    # Loss weights
    loss_weights = train_config.get('loss_weights', {'flow': 1.0, 'coarse': 0.2})
    
    # Training loop
    model.train()
    # Use loaded global_step if available, otherwise calculate from epoch
    if global_step == 0:
        global_step = start_epoch * len(train_loader)
    
    for epoch in range(start_epoch, train_config['epochs']):
        epoch_losses = {'total': 0, 'flow': 0, 'coarse': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch (now includes UV mask and speaker IDs)
            text, duration, f0, uv, mel, src_mask, mel_mask, speaker_ids = batch
            text = text.to(device)
            duration = duration.to(device)
            f0 = f0.to(device)
            uv = uv.to(device)
            mel = mel.to(device)
            src_mask = src_mask.to(device)
            mel_mask = mel_mask.to(device)
            speaker_ids = speaker_ids.to(device)
            
            # Sample random time
            t = torch.rand(text.size(0), device=device)
            
            # Forward pass with AMP autocast
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(text, src_mask, mel_mask, duration, f0, mel, t, uv=uv, speaker_id=speaker_ids)
                total_loss, loss_dict = compute_loss(outputs, mel_mask, loss_weights)
            
            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()
            
            # Update EMA
            if ema:
                ema.update(model)
            
            # Update epoch losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0] if scheduler else train_config['learning_rate']
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'flow': f"{loss_dict['flow']:.4f}",
                'coarse': f"{loss_dict['coarse']:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            global_step += 1
            
            # Log to console periodically
            if global_step % log_config['log_interval'] == 0:
                avg_losses = {k: v / (batch_idx + 1) for k, v in epoch_losses.items()}
                print(f"\n[Step {global_step}] Avg Loss: {avg_losses['total']:.4f}, "
                    f"Flow: {avg_losses['flow']:.4f}, Coarse: {avg_losses['coarse']:.4f}")
        
        # Epoch summary
        num_batches = len(train_loader)
        avg_epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_epoch_losses['total']:.4f}")
        print(f"  Flow Loss: {avg_epoch_losses['flow']:.4f}")
        print(f"  Coarse Loss: {avg_epoch_losses['coarse']:.4f}")
        
        # Validation
        val_losses = None
        val_interval = train_config.get('val_interval', 5)
        is_save_epoch = (epoch + 1) % ckpt_config['save_interval'] == 0 or epoch == train_config['epochs'] - 1
        run_val = val_loader is not None and (
            (epoch + 1) % val_interval == 0 or is_save_epoch
        )
        if run_val:
            if ema:
                ema.apply_shadow(model)
            val_losses = validate(model, val_loader, device, loss_weights, use_amp=use_amp)
            if ema:
                ema.restore(model)
            print(f"  Validation Loss: {val_losses['total']:.4f}")
            print(f"  Val Flow Loss: {val_losses['flow']:.4f}")
            print(f"  Val Coarse Loss: {val_losses['coarse']:.4f}")
        
        # Save checkpoint
        if is_save_epoch:
            checkpoint_state = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if use_amp else None,
                'ema_state_dict': ema.state_dict() if ema else None,
                'loss': avg_epoch_losses,
                'val_loss': val_losses,
                'best_loss': ckpt_manager.best_loss,
                'config': config,
                'model_config': model_config,  # For easy loading during inference
                'n_speakers': n_speakers,       # For multi-speaker model reconstruction
            }
            save_loss = val_losses['total'] if val_losses is not None else avg_epoch_losses['total']
            ckpt_manager.save(checkpoint_state, epoch, save_loss)
            
            # Also save EMA model separately for inference
            if ema:
                ema.apply_shadow(model)
                ema_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'model_config': model_config,
                }
                ema_path = os.path.join(exp_dir, "checkpoint_ema.pt")
                torch.save(ema_state, ema_path)
                ema.restore(model)
                print(f"Saved EMA model to {ema_path}")
    
    print("Training complete!")
    print(f"Best model saved at: {ckpt_manager.get_best_path()}")
    print(f"Latest model saved at: {ckpt_manager.get_latest_path()}")
    if ema:
        print(f"EMA model saved at: {os.path.join(exp_dir, 'checkpoint_ema.pt')}")


def generate_reflow_data(config, checkpoint_path, device, n_steps=10):
    """
    Generate reflow training targets using a trained model.
    
    For iterative rectified flow, this generates mel targets by running
    the current model's ODE solver, then saves them for retraining with
    straighter trajectories.
    
    Args:
        config: configuration dict
        checkpoint_path: path to trained model checkpoint
        device: torch device
        n_steps: ODE integration steps for generation
    
    Returns:
        path to reflow data directory
    """
    from inference import load_model
    
    data_dir = config['data']['processed_dir']
    reflow_dir = os.path.join(data_dir, 'reflow')
    os.makedirs(reflow_dir, exist_ok=True)
    
    # Load phone map
    with open(os.path.join(data_dir, 'phone_map.json'), 'r') as f:
        phone_map = json.load(f)
    
    # Load model
    model, _, _, _ = load_model(checkpoint_path, device, config=config)
    model.eval()
    
    # Load dataset (batch_size=1 for per-sample generation)
    dataset = SingingDataset(
        meta_file=os.path.join(data_dir, 'train.txt'),
        data_dir=data_dir,
        phone_map=phone_map
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    print(f"Generating reflow targets for {len(dataset)} samples with {n_steps} ODE steps...")
    
    for idx, batch in enumerate(tqdm(loader, desc="Generating reflow pairs")):
        text, duration, f0, uv, mel, src_mask, mel_mask, speaker_ids = batch
        text = text.to(device)
        duration = duration.to(device)
        f0 = f0.to(device)
        uv = uv.to(device)
        src_mask = src_mask.to(device)
        mel_mask = mel_mask.to(device)
        speaker_ids = speaker_ids.to(device)
        
        with torch.no_grad():
            mel_gen = model.inference(text, src_mask, duration, f0, n_steps=n_steps, uv=uv, speaker_id=speaker_ids)
        
        # Get file ID from dataset
        fid = dataset.lines[idx].strip().split('|')[0]
        mel_gen_np = mel_gen.squeeze(0).cpu().numpy()
        
        # Trim to actual length (remove padding)
        actual_len = int(mel_mask.sum().item())
        mel_gen_np = mel_gen_np[:actual_len]
        
        out_path = os.path.join(reflow_dir, f'{fid}_mel.npy')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, mel_gen_np)
    
    print(f"Reflow data saved to {reflow_dir}")
    return reflow_dir


def main():
    parser = argparse.ArgumentParser(description="Train RFSinger Model")
    
    # Config file argument
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help='Path to configuration YAML file')
    
    # Optional overrides
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Override experiment name')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to experiment dir to resume from (overrides config)')
    
    # Iterative Rectified Flow (ReFlow) arguments
    parser.add_argument('--reflow', action='store_true',
                        help='Generate reflow training targets from a trained model')
    parser.add_argument('--reflow_checkpoint', type=str, default=None,
                        help='Checkpoint path for reflow data generation')
    parser.add_argument('--reflow_n_steps', type=int, default=None,
                        help='ODE steps for reflow generation (default: from config or 10)')
    parser.add_argument('--reflow_train', action='store_true',
                        help='Train on previously generated reflow targets')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Apply command-line overrides
    if args.exp_name:
        config['experiment']['name'] = args.exp_name
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    # Handle resume
    if args.resume:
        # Load config from the experiment to resume
        resume_config_path = os.path.join(args.resume, 'config.yaml')
        if os.path.exists(resume_config_path):
            config = load_config(resume_config_path)
            print(f"Loaded config from resume experiment: {resume_config_path}")
    
    # Handle reflow data generation
    if args.reflow:
        reflow_ckpt = args.reflow_checkpoint
        if reflow_ckpt is None:
            exp_dir = get_experiment_dir(config)
            for candidate in ['checkpoint_best.pt', 'checkpoint_ema.pt', 'checkpoint_latest.pt']:
                path = os.path.join(exp_dir, candidate)
                if os.path.exists(path):
                    reflow_ckpt = path
                    break
        if reflow_ckpt is None or not os.path.exists(reflow_ckpt):
            print("Error: No checkpoint found for reflow generation. Use --reflow_checkpoint.")
            return
        
        reflow_n_steps = args.reflow_n_steps or config['training'].get('reflow', {}).get('n_steps', 10)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nGenerating reflow data from {reflow_ckpt} with {reflow_n_steps} ODE steps")
        generate_reflow_data(config, reflow_ckpt, device, n_steps=reflow_n_steps)
        return
    
    # Handle reflow training
    if args.reflow_train:
        reflow_dir = os.path.join(config['data']['processed_dir'], 'reflow')
        if not os.path.isdir(reflow_dir):
            print(f"Error: Reflow data not found at {reflow_dir}. Run --reflow first.")
            return
        config['data']['reflow_dir'] = reflow_dir
        reflow_config = config['training'].get('reflow', {})
        if reflow_config.get('freeze_encoder', True):
            config['training']['freeze_encoder'] = True
        # Override loss weights for reflow: disable coarse loss since encoder is frozen
        reflow_loss_weights = reflow_config.get('loss_weights', None)
        if reflow_loss_weights:
            config['training']['loss_weights'] = reflow_loss_weights
        else:
            config['training']['loss_weights'] = {'flow': 1.0, 'coarse': 0.0}
        # Append '-reflow' to experiment name to separate from base training
        config['experiment']['name'] = config['experiment']['name'] + '-reflow'
        print(f"Reflow training mode: targets from {reflow_dir}")
    
    # Print config
    print("\n" + "="*50)
    print("Configuration:")
    print("="*50)
    print_config(config)
    print("="*50 + "\n")
    
    train(config)


if __name__ == '__main__':
    main()
