import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SingingDataset(Dataset):
    def __init__(self, meta_file, data_dir, phone_map, reflow_dir=None, augmentation=None):
        self.data_dir = data_dir
        self.reflow_dir = reflow_dir  # Optional: directory with reflow mel targets
        self.phone_map = phone_map  # Dict: phoneme str -> int ID
        with open(meta_file, 'r') as f:
            self.lines = f.readlines()

        self.f0_stats = np.load(f"{data_dir}/f0_stats.npy")
        
        # Online augmentation config (applied on-the-fly during training)
        self.aug = augmentation if augmentation else {}
        self.aug_enabled = self.aug.get('enabled', False)

    def __len__(self):
        return len(self.lines)

    def _feature_path(self, base_dir, fid, suffix):
        """Build feature path from metadata id (supports nested speaker/id)."""
        norm_fid = fid.replace('\\', '/').strip('/')
        return os.path.join(base_dir, f"{norm_fid}_{suffix}.npy")

    def _apply_online_augmentation(self, mel, f0, uv):
        """
        Apply lightweight online augmentation to features during training.
        These are small random perturbations that don't require re-extracting
        features from raw audio.
        
        Args:
            mel: Mel spectrogram (time, n_mels)
            f0: F0 contour (time,)
            uv: Unvoiced mask (time,)
            
        Returns:
            Augmented (mel, f0)
        """
        # F0 jitter: small random perturbation to pitch contour
        f0_jitter = self.aug.get('f0_jitter', 0.0)
        if f0_jitter > 0:
            # Only perturb voiced regions (where uv == 0)
            voiced_mask = (uv < 0.5)
            noise = np.random.normal(0, f0_jitter, size=f0.shape).astype(np.float32)
            f0 = f0 + noise * voiced_mask * np.abs(f0)
        
        # Mel noise: small Gaussian noise added to mel spectrogram
        mel_noise_std = self.aug.get('mel_noise_std', 0.0)
        if mel_noise_std > 0:
            noise = np.random.normal(0, mel_noise_std, size=mel.shape).astype(np.float32)
            mel = mel + noise
        
        # Time masking: zero out a random contiguous block of frames
        time_mask_max = self.aug.get('time_mask_max_len', 0)
        if time_mask_max > 0 and mel.shape[0] > time_mask_max:
            mask_len = np.random.randint(1, time_mask_max + 1)
            start = np.random.randint(0, mel.shape[0] - mask_len)
            mel[start:start + mask_len, :] = 0.0
        
        # Frequency masking: zero out a random contiguous block of mel bins
        freq_mask_max = self.aug.get('freq_mask_max_bins', 0)
        if freq_mask_max > 0 and mel.shape[1] > freq_mask_max:
            mask_width = np.random.randint(1, freq_mask_max + 1)
            start = np.random.randint(0, mel.shape[1] - mask_width)
            mel[:, start:start + mask_width] = 0.0
        
        return mel, f0

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split('|')
        fid = line[0]
        phones = [self.phone_map.get(p, 0) for p in line[1].split()]
        durations = [int(d) for d in line[2].split()]
        # Speaker ID: 4th field if present, else 0 (single-speaker fallback)
        speaker_id = int(line[3]) if len(line) > 3 else 0

        mel_dir = self.reflow_dir if self.reflow_dir is not None else self.data_dir
        mel = np.load(self._feature_path(mel_dir, fid, 'mel'))
        f0 = np.load(self._feature_path(self.data_dir, fid, 'f0'))
        
        # Load UV mask (unvoiced mask: 1=unvoiced, 0=voiced)
        uv_path = self._feature_path(self.data_dir, fid, 'uv')
        if os.path.exists(uv_path):
            uv = np.load(uv_path)
        else:
            uv = np.zeros_like(f0)

        # Apply online augmentation (before normalization for f0 jitter to be meaningful)
        if self.aug_enabled:
            mel, f0 = self._apply_online_augmentation(mel, f0, uv)

        # F0 Z-score Normalization
        f0 = (f0 - self.f0_stats[0]) / (self.f0_stats[1] + 1e-6)

        return {
            "id": fid,
            "text": torch.LongTensor(phones),
            "duration": torch.LongTensor(durations),
            "mel": torch.FloatTensor(mel),
            "f0": torch.FloatTensor(f0),
            "uv": torch.FloatTensor(uv),  # UV mask for voiced/unvoiced handling
            "speaker_id": speaker_id,
        }


def collate_fn(batch):
    # Dynamic padding
    # Get max length in batch
    max_text_len = max([len(x['text']) for x in batch])
    max_mel_len = max([len(x['mel']) for x in batch])
    mel_dim = batch[0]['mel'].shape[1]

    text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    mel_padded = torch.zeros(len(batch), max_mel_len, mel_dim)
    f0_padded = torch.zeros(len(batch), max_mel_len)
    uv_padded = torch.ones(len(batch), max_mel_len)  # Default to unvoiced (1) for padding
    dur_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    speaker_ids = torch.LongTensor([x['speaker_id'] for x in batch])

    # Masks
    src_mask = torch.zeros(len(batch), max_text_len, dtype=torch.bool)
    mel_mask = torch.zeros(len(batch), max_mel_len, dtype=torch.bool)

    for i, x in enumerate(batch):
        t_l = len(x['text'])
        m_l = len(x['mel'])

        text_padded[i, :t_l] = x['text']
        dur_padded[i, :t_l] = x['duration']
        mel_padded[i, :m_l] = x['mel']
        f0_padded[i, :m_l] = x['f0']
        uv_padded[i, :m_l] = x['uv']

        src_mask[i, :t_l] = True
        mel_mask[i, :m_l] = True

    return text_padded, dur_padded, f0_padded, uv_padded, mel_padded, src_mask, mel_mask, speaker_ids