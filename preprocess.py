import os
import argparse
import json
import numpy as np
import pandas as pd
import librosa
import pyworld as pw
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d
import random

from utils.config import load_config, get_hparams_from_config

# Populated from config.yaml in main(). All preprocessing functions read from this dict.
HPARAMS: dict = {}

# ── RMVPE lazy singleton ─────────────────────────────────────────────────────
_RMVPE_INSTANCE = None


def _get_rmvpe():
    """Lazy-load the RMVPE model so the checkpoint is read only once."""
    global _RMVPE_INSTANCE
    if _RMVPE_INSTANCE is None:
        from utils.rmvpe import RMVPE
        ckpt = HPARAMS.get("rmvpe_ckpt", "./ckpts/rmvpe.pt")
        print(f"[INFO] Loading RMVPE from {ckpt}")
        _RMVPE_INSTANCE = RMVPE(ckpt)
    return _RMVPE_INSTANCE


def get_num_frames(wav_length):
    """Calculate the expected number of frames for consistent alignment."""
    # Use the same frame calculation as librosa with center=True (default)
    # This ensures mel and f0 have the same target frame count
    return int(np.ceil(wav_length / HPARAMS["hop_length"]))


def get_mel(wav, target_frames=None):
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=HPARAMS["sample_rate"],
        n_fft=HPARAMS["n_fft"],
        hop_length=HPARAMS["hop_length"],
        win_length=HPARAMS["win_length"],
        n_mels=HPARAMS["n_mels"],
        fmin=HPARAMS["fmin"],
        fmax=HPARAMS["fmax"],
        power=1.0,
        center=True  # Explicitly set for clarity
    )
    # Use log-mel
    mel = np.log(mel + 1e-5)
    mel = mel.T  # (time, n_mels)
    
    # Align to target frames if specified
    if target_frames is not None:
        if mel.shape[0] > target_frames:
            mel = mel[:target_frames]
        elif mel.shape[0] < target_frames:
            # Pad with the last frame value
            pad_len = target_frames - mel.shape[0]
            mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='edge')
    
    return mel


def extract_f0_harvest(wav, target_frames=None):
    """Extract F0 with the Harvest algorithm (PyWorld)."""
    # frame_period converted from hop_length
    frame_period = HPARAMS["hop_length"] / HPARAMS["sample_rate"] * 1000

    # Convert to float64
    wav = wav.astype(np.float64)
    f0, t = pw.harvest(
        wav, 
        HPARAMS["sample_rate"], 
        frame_period=frame_period, 
        f0_floor=HPARAMS.get("f0_floor", 50.0), 
        f0_ceil=HPARAMS.get("f0_ceil", 1100.0)
    )

    # Align to target frames if specified
    if target_frames is not None:
        if len(f0) > target_frames:
            f0 = f0[:target_frames]
        elif len(f0) < target_frames:
            f0 = np.pad(f0, (0, target_frames - len(f0)), mode='constant', constant_values=0)
    
    return f0


def extract_f0_rmvpe(wav, target_frames=None):
    """Extract F0 with the RMVPE neural pitch estimator.

    RMVPE always operates at 16 kHz with a 160-sample hop (10 ms per frame),
    regardless of the input audio's sample rate.  The project's mel uses
    sample_rate / hop_length seconds per frame (e.g. 44100/512 ≈ 11.6 ms).
    We resample the F0 contour from RMVPE's time grid to the project's time
    grid so that frame indices align with the mel spectrogram.
    """
    rmvpe = _get_rmvpe()
    f0 = rmvpe.infer_from_audio(
        wav.astype(np.float32),
        sample_rate=HPARAMS["sample_rate"],
    )

    # ── Resample F0 from RMVPE time-base to project time-base ────────────
    # RMVPE frame period: 160 / 16000 = 0.01 s (10 ms)
    rmvpe_hop_sec = 160.0 / 16000.0
    # Project frame period
    project_hop_sec = HPARAMS["hop_length"] / HPARAMS["sample_rate"]

    # Number of frames the project expects
    if target_frames is not None:
        n_target = target_frames
    else:
        n_target = int(np.ceil(len(wav) / HPARAMS["hop_length"]))

    n_rmvpe = len(f0)

    if n_rmvpe == 0:
        return np.zeros(n_target, dtype=np.float64)

    # Time axes (frame-centre times)
    times_rmvpe = np.arange(n_rmvpe) * rmvpe_hop_sec
    times_project = np.arange(n_target) * project_hop_sec

    # Separate voiced (>0) and unvoiced (==0) before interpolation so we
    # don't smear zeros into the F0 contour.
    voiced_mask = f0 > 0

    if voiced_mask.sum() >= 2:
        # Interpolate only voiced values
        voiced_times = times_rmvpe[voiced_mask]
        voiced_values = f0[voiced_mask]

        interp_func = interp1d(
            voiced_times, voiced_values,
            kind='linear', bounds_error=False,
            fill_value=(voiced_values[0], voiced_values[-1])
        )
        f0_resampled = interp_func(times_project)

        # Rebuild unvoiced mask: a project frame is unvoiced if the nearest
        # RMVPE frame was unvoiced.
        nearest_idx = np.clip(
            np.round(times_project / rmvpe_hop_sec).astype(int),
            0, n_rmvpe - 1
        )
        f0_resampled[~voiced_mask[nearest_idx]] = 0.0
    elif voiced_mask.sum() == 1:
        val = float(f0[voiced_mask][0])
        f0_resampled = np.full(n_target, val, dtype=np.float64)
        nearest_idx = np.clip(
            np.round(times_project / rmvpe_hop_sec).astype(int),
            0, n_rmvpe - 1
        )
        f0_resampled[~voiced_mask[nearest_idx]] = 0.0
    else:
        f0_resampled = np.zeros(n_target, dtype=np.float64)

    return f0_resampled


_F0_EXTRACTORS = {
    "harvest": extract_f0_harvest,
    "rmvpe":   extract_f0_rmvpe,
}


def extract_f0(wav, target_frames=None):
    """Extract F0 using the method specified by HPARAMS['f0_method']."""
    method = HPARAMS.get("f0_method", "harvest").lower()
    if method not in _F0_EXTRACTORS:
        raise ValueError(
            f"Unknown f0_method '{method}'. "
            f"Choose from: {list(_F0_EXTRACTORS.keys())}"
        )
    return _F0_EXTRACTORS[method](wav, target_frames)


def interpolate_f0(f0):
    # Add uv mask
    uv = (f0 == 0).astype(np.float32)

    # Interpolate F0 to fill unvoiced
    f0_interp = f0.copy()
    nonzero_ids = np.where(f0 > 0)[0]  # Get the actual indices (1D array)
    if len(nonzero_ids) > 1:  # Need at least 2 points for interpolation
        interp_func = interp1d(
            nonzero_ids,
            f0[nonzero_ids],
            kind='linear',
            bounds_error=False,
            fill_value=(f0[nonzero_ids[0]], f0[nonzero_ids[-1]])  # Extrapolate with boundary values
        )
        full_ids = np.arange(len(f0))
        f0_interp = interp_func(full_ids)
    elif len(nonzero_ids) == 1:
        # Only one voiced frame, fill all with that value
        f0_interp = np.full_like(f0, f0[nonzero_ids[0]])
    elif len(nonzero_ids) == 0:
        # All unvoiced, use a default low frequency
        f0_interp = np.full_like(f0, HPARAMS.get("f0_floor", 50.0))

    return f0_interp, uv


def align_durations_to_length(durations, target_length):
    """
    Align phoneme durations to match the target feature length.
    Uses proportional scaling to preserve relative timing.
    """
    durations = np.array(durations, dtype=np.float64)
    total_dur = np.sum(durations)
    
    if total_dur == 0:
        # Edge case: all zero durations, distribute evenly
        n_phones = len(durations)
        durations = np.full(n_phones, target_length / n_phones)
    elif total_dur != target_length:
        # Proportionally scale durations
        scale_factor = target_length / total_dur
        durations = durations * scale_factor
    
    # Convert to integers using a cumulative rounding approach
    # This ensures the sum exactly equals target_length
    cumsum = np.cumsum(durations)
    cumsum_rounded = np.round(cumsum).astype(np.int32)
    # Ensure last element matches target
    cumsum_rounded[-1] = target_length
    
    # Convert back to individual durations
    aligned_durations = np.diff(np.concatenate([[0], cumsum_rounded]))
    
    # Ensure all durations are at least 1
    for i in range(len(aligned_durations)):
        if aligned_durations[i] < 1:
            aligned_durations[i] = 1
            # Steal from the longest duration to compensate
            excess = np.sum(aligned_durations) - target_length
            if excess > 0:
                max_idx = np.argmax(aligned_durations)
                aligned_durations[max_idx] = max(1, aligned_durations[max_idx] - excess)
    
    # Final check and adjustment
    final_sum = np.sum(aligned_durations)
    if final_sum != target_length:
        diff = target_length - final_sum
        # Adjust the longest duration
        max_idx = np.argmax(aligned_durations)
        aligned_durations[max_idx] += diff
    
    return aligned_durations.astype(np.int32)


# =============================================================================
# Data Augmentation Functions
# =============================================================================

def augment_time_stretch(wav, sr, rate):
    """
    Time-stretch audio without changing pitch.
    
    Args:
        wav: Audio waveform (numpy array)
        sr: Sample rate
        rate: Stretch rate (>1 = faster/shorter, <1 = slower/longer)
        
    Returns:
        Time-stretched waveform
    """
    return librosa.effects.time_stretch(wav, rate=rate)


def augment_pitch_shift(wav, sr, n_steps):
    """
    Shift pitch without changing tempo.
    
    Args:
        wav: Audio waveform (numpy array)
        sr: Sample rate
        n_steps: Number of semitones to shift (positive = higher, negative = lower)
        
    Returns:
        Pitch-shifted waveform
    """
    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=n_steps)


def generate_augmentation_params(aug_config, rng):
    """
    Sample random augmentation parameters from the configured ranges.
    
    Args:
        aug_config: Augmentation config dict from HPARAMS
        rng: numpy RandomState for reproducibility
        
    Returns:
        Dict with sampled augmentation parameters
    """
    params = {"time_stretch_rate": 1.0, "pitch_shift_semitones": 0.0}
    
    ts_cfg = aug_config.get("time_stretch", {})
    if ts_cfg.get("enabled", False):
        params["time_stretch_rate"] = rng.uniform(
            ts_cfg.get("min_rate", 0.8),
            ts_cfg.get("max_rate", 1.2)
        )
    
    ps_cfg = aug_config.get("pitch_shift", {})
    if ps_cfg.get("enabled", False):
        params["pitch_shift_semitones"] = rng.uniform(
            ps_cfg.get("min_semitones", -3),
            ps_cfg.get("max_semitones", 3)
        )
    
    return params


def apply_augmentation(wav, sr, params):
    """
    Apply augmentation transforms to a waveform.
    
    Args:
        wav: Audio waveform
        sr: Sample rate
        params: Dict with augmentation parameters
        
    Returns:
        Augmented waveform
    """
    aug_wav = wav.copy()
    
    # Apply time stretching
    rate = params.get("time_stretch_rate", 1.0)
    if abs(rate - 1.0) > 1e-3:
        aug_wav = augment_time_stretch(aug_wav, sr, rate)
    
    # Apply pitch shifting
    n_steps = params.get("pitch_shift_semitones", 0.0)
    if abs(n_steps) > 1e-3:
        aug_wav = augment_pitch_shift(aug_wav, sr, n_steps)
    
    return aug_wav


def process_single_utterance(wav, dur_seconds, file_id, output_dir, target_frames=None):
    """
    Extract features from a waveform and save to disk.
    Shared logic for both original and augmented samples.
    
    Args:
        wav: Audio waveform (numpy array)
        dur_seconds: Phoneme durations in seconds (numpy array)
        file_id: Identifier string for saving files
        output_dir: Directory to save features
        target_frames: Expected number of frames (computed from wav length if None)
        
    Returns:
        Tuple of (f0_voiced_values, metadata_line) or (None, None) on failure
    """
    if target_frames is None:
        target_frames = get_num_frames(len(wav))
    
    mel = get_mel(wav, target_frames=target_frames)
    f0_raw = extract_f0(wav, target_frames=target_frames)
    
    # Verify alignment
    if mel.shape[0] != len(f0_raw) or mel.shape[0] != target_frames:
        print(f"Warning: Frame mismatch for {file_id}: mel={mel.shape[0]}, "
              f"f0={len(f0_raw)}, target={target_frames}. Skipping.")
        return None, None
    
    # Duration alignment
    durations_float = dur_seconds * HPARAMS["sample_rate"] / HPARAMS["hop_length"]
    durations = align_durations_to_length(durations_float, target_frames)
    
    if np.sum(durations) != target_frames:
        print(f"Warning: Duration sum mismatch for {file_id}. Skipping.")
        return None, None
    
    # F0 post-processing
    f0_interp, uv = interpolate_f0(f0_raw)
    if HPARAMS["use_log_f0"]:
        f0_interp = np.log(f0_interp + 1e-5)
    
    # Save features (supports nested ids like "speaker_name/sample_id")
    out_prefix = os.path.join(output_dir, str(file_id))
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_prefix}_mel.npy", mel)
    np.save(f"{out_prefix}_f0.npy", f0_interp)
    np.save(f"{out_prefix}_uv.npy", uv)
    
    voiced_f0 = f0_interp[uv < 1]
    phones_str = None  # Will be set by caller
    dur_str = ' '.join(map(str, durations))
    
    return voiced_f0, dur_str


def preprocess():
    os.makedirs(HPARAMS["output_dir"], exist_ok=True)

    f0_all = []  # Initialize empty list
    metadata = []  # Initialize empty list
    
    # RNG for reproducible augmentation
    aug_seed = HPARAMS.get("augmentation", {}).get("seed", 42)
    aug_rng = np.random.RandomState(aug_seed)
    
    # =========================================================================
    # Determine speaker list: multi-speaker or single-speaker mode
    # =========================================================================
    speakers_config = HPARAMS.get("speakers", None)
    if speakers_config and isinstance(speakers_config, dict) and len(speakers_config) > 0:
        # Multi-speaker mode
        speaker_list = sorted(speakers_config.keys())
        speaker_datasets = []
        for spk_name in speaker_list:
            spk_cfg = speakers_config[spk_name]
            spk_data_path = spk_cfg.get("raw_data_path", "")
            spk_csv_path = spk_cfg.get("csv_path", "")
            if not spk_data_path or not spk_csv_path:
                print(f"Warning: Speaker '{spk_name}' missing data paths, skipping.")
                continue
            speaker_datasets.append((spk_name, spk_data_path, spk_csv_path))
        print(f"Multi-speaker mode: {len(speaker_datasets)} speakers: "
              f"{[s[0] for s in speaker_datasets]}")
    else:
        # Single-speaker fallback
        speaker_datasets = [("default", HPARAMS["data_path"], HPARAMS["csv_path"])]
        speaker_list = ["default"]
        print("Single-speaker mode")
    
    # Build speaker map: speaker_name -> int ID
    speaker_map = {spk: idx for idx, spk in enumerate(sorted(
        set(s[0] for s in speaker_datasets)
    ))}
    
    # Save speaker map
    with open(f"{HPARAMS['output_dir']}/speaker_map.json", 'w') as f:
        json.dump(speaker_map, f, ensure_ascii=False, indent=2)
    print(f"Speaker map: {speaker_map}")
    
    # =========================================================================
    # Build unified phoneme vocabulary across all speakers
    # =========================================================================
    phone_set = set()
    all_dataframes = []
    for spk_name, spk_data_path, spk_csv_path in speaker_datasets:
        df = pd.read_csv(spk_csv_path)
        df['_speaker'] = spk_name
        df['_data_path'] = spk_data_path
        all_dataframes.append(df)
        for _, row in df.iterrows():
            phones = row['ph_seq'].split()
            phone_set.update(phones)
    
    df_all = pd.concat(all_dataframes, ignore_index=True)
    
    phone_list = sorted(list(phone_set))
    phone_map = {p: i + 1 for i, p in enumerate(phone_list)}  # 0 reserved for padding
    phone_map['<pad>'] = 0
    
    # Save phone map
    with open(f"{HPARAMS['output_dir']}/phone_map.json", 'w') as f:
        json.dump(phone_map, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # Process all utterances across all speakers
    # =========================================================================
    for _, row in tqdm(df_all.iterrows(), total=len(df_all)):
        spk_name = row['_speaker']
        spk_data_path = row['_data_path']
        spk_id = speaker_map[spk_name]
        file_id = row['name']
        
        # Use per-speaker subfolders in processed output for multi-speaker data
        if len(speaker_datasets) > 1:
            prefixed_id = f"{spk_name}/{file_id}"
        else:
            prefixed_id = file_id
        
        wav_path = os.path.join(spk_data_path, f"{file_id}.wav")

        if not os.path.exists(wav_path):
            print(f"Warning: {wav_path} not found.")
            continue

        # 1. Load waveform
        wav, _ = librosa.load(wav_path, sr=HPARAMS["sample_rate"])

        # 2. Calculate target frame count for consistent alignment
        target_frames = get_num_frames(len(wav))
        
        # 3. Parse duration info
        dur_seconds = np.array([float(x) for x in row['ph_dur'].split()])
        phones_str = row['ph_seq']

        # 4. Process original sample
        voiced_f0, dur_str = process_single_utterance(
            wav, dur_seconds, prefixed_id, HPARAMS["output_dir"], target_frames
        )
        if voiced_f0 is not None:
            f0_all.extend(voiced_f0)
            # Format: file_id|phones|durations|speaker_id
            metadata.append(f"{prefixed_id}|{phones_str}|{dur_str}|{spk_id}")

        # 5. Generate augmented copies (offline augmentation)
        aug_config = HPARAMS.get("augmentation", {})
        if aug_config.get("enabled", False):
            n_copies = aug_config.get("num_augmented_copies", 2)
            for aug_idx in range(n_copies):
                aug_params = generate_augmentation_params(aug_config, aug_rng)
                try:
                    aug_wav = apply_augmentation(
                        wav, HPARAMS["sample_rate"], aug_params
                    )
                    aug_file_id = f"{prefixed_id}_aug{aug_idx:02d}"
                    aug_target_frames = get_num_frames(len(aug_wav))
                    
                    # For time-stretched audio, scale durations proportionally
                    ts_rate = aug_params.get("time_stretch_rate", 1.0)
                    # Time stretch rate > 1 means shorter audio, so durations shrink
                    aug_dur_seconds = dur_seconds / ts_rate if abs(ts_rate - 1.0) > 1e-3 else dur_seconds
                    
                    aug_voiced_f0, aug_dur_str = process_single_utterance(
                        aug_wav, aug_dur_seconds, aug_file_id,
                        HPARAMS["output_dir"], aug_target_frames
                    )
                    if aug_voiced_f0 is not None:
                        f0_all.extend(aug_voiced_f0)
                        metadata.append(f"{aug_file_id}|{phones_str}|{aug_dur_str}|{spk_id}")
                except Exception as e:
                    print(f"Warning: Augmentation failed for {prefixed_id} "
                          f"(copy {aug_idx}, params={aug_params}): {e}")
                    continue

    # Calculate global F0 statistics for normalization
    f0_mean = np.mean(f0_all)
    f0_std = np.std(f0_all)

    with open(f"{HPARAMS['output_dir']}/train.txt", "w") as f:
        f.write("\n".join(metadata))

    np.save(f"{HPARAMS['output_dir']}/f0_stats.npy", [f0_mean, f0_std])
    print(f"Preprocessing Done. {len(metadata)} utterances from {len(speaker_datasets)} speaker(s).")
    print(f"F0 Mean: {f0_mean}, Std: {f0_std}")


def main():
    global HPARAMS

    parser = argparse.ArgumentParser(description="Preprocess audio data for RFSinger")
    parser.add_argument('--config', type=str, default="config.yaml",
                        help='Path to configuration YAML file')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override raw_data_path from config')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Override csv_path from config')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override processed_dir from config')

    args = parser.parse_args()

    config = load_config(args.config)
    HPARAMS = get_hparams_from_config(config)
    print(f"Loaded config from {args.config}")

    if args.data_path:
        HPARAMS['data_path'] = args.data_path
    if args.csv_path:
        HPARAMS['csv_path'] = args.csv_path
    if args.output_dir:
        HPARAMS['output_dir'] = args.output_dir

    print("\nPreprocessing Configuration:")
    print("-" * 40)
    for key, value in HPARAMS.items():
        print(f"  {key}: {value}")
    print("-" * 40 + "\n")

    preprocess()


if __name__ == "__main__":
    main()