"""
RFSinger Inference from DiffSinger (.ds) Project Files

This script enables inference using .ds project files, extracting phonemes,
durations, and F0 contours directly from the project data.
"""

import os
import json
import argparse
import numpy as np
import torch
import soundfile as sf
from typing import List, Dict, Tuple, Optional

from inference import (
    load_model, 
    load_nsf_hifigan,
    mel_to_audio_nsf_hifigan,
    mel_to_audio_pyworld,
    mel_to_audio,
    text_to_sequence,
    resolve_checkpoint_path
)
from utils.config import load_config, load_config_from_checkpoint


def parse_ds_file(ds_path: str) -> List[Dict]:
    """
    Parse a .ds (DiffSinger) project file.
    
    Args:
        ds_path: Path to .ds file
    
    Returns:
        List of segment dictionaries
    """
    with open(ds_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("DS file must contain a list of segments")
    
    return data


def interpolate_f0_with_uv(f0_hz: np.ndarray, uv_threshold: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate F0 over unvoiced regions and return UV mask.

    Args:
        f0_hz: F0 contour in Hz
        uv_threshold: threshold below which frames are treated as unvoiced

    Returns:
        (f0_interp_hz, uv_mask) where uv_mask is 1=unvoiced, 0=voiced
    """
    uv = (f0_hz <= uv_threshold).astype(np.float32)
    f0_interp = f0_hz.copy()

    voiced_ids = np.where(f0_hz > uv_threshold)[0]
    if len(voiced_ids) > 1:
        import scipy.interpolate
        interp_func = scipy.interpolate.interp1d(
            voiced_ids,
            f0_hz[voiced_ids],
            kind='linear',
            bounds_error=False,
            fill_value=(f0_hz[voiced_ids[0]], f0_hz[voiced_ids[-1]])
        )
        full_ids = np.arange(len(f0_hz))
        f0_interp = interp_func(full_ids).astype(np.float32)
    elif len(voiced_ids) == 1:
        f0_interp = np.full_like(f0_hz, f0_hz[voiced_ids[0]], dtype=np.float32)
    else:
        # No voiced frames: set to small constant to avoid NaNs
        f0_interp = np.full_like(f0_hz, uv_threshold, dtype=np.float32)

    return f0_interp, uv


def extract_synthesis_params(segment: Dict, sample_rate: int = 44100,
                            hop_length: int = 512, f0_timestep: float = 0.01) -> Tuple:
    """
    Extract synthesis parameters from a DS segment.
    
    Args:
        segment: DS segment dictionary
        sample_rate: Audio sample rate
        hop_length: STFT hop length (frames)
        f0_timestep: F0 timestep in seconds
    
    Returns:
        (phonemes, durations, f0_contour, segment_name)
    """
    # Extract phoneme sequence
    ph_seq = segment.get('ph_seq', '')
    if not ph_seq:
        raise ValueError("Segment missing 'ph_seq' field")
    
    # Extract phoneme durations (in seconds)
    ph_dur_str = segment.get('ph_dur', '')
    if not ph_dur_str:
        raise ValueError("Segment missing 'ph_dur' field")
    
    ph_durs = [float(d) for d in ph_dur_str.strip().split()]
    
    # Convert durations from seconds to frames
    # Formula: frames = seconds * sample_rate / hop_length
    ph_dur_frames = np.round(np.array(ph_durs) * sample_rate / hop_length).astype(int).tolist()
    # Ensure at least 1 frame per phoneme
    ph_dur_frames = [max(1, d) for d in ph_dur_frames]
    
    # Extract F0 sequence (in Hz)
    f0_seq_str = segment.get('f0_seq', '')
    if not f0_seq_str:
        raise ValueError("Segment missing 'f0_seq' field")
    
    f0_hz = np.array([float(f) for f in f0_seq_str.strip().split()], dtype=np.float32)
    
    # F0 timestep (from DS file, usually 0.01s = 10ms)
    ds_f0_timestep = float(segment.get('f0_timestep', 0.01))
    
    # Calculate expected frame count from phoneme durations (seconds-based)
    total_time = sum(ph_durs)
    total_frames = int(np.round(total_time * sample_rate / hop_length))
    
    # Resample F0 to match frame count if needed
    # DS F0 is sampled at f0_timestep intervals, we need to match hop_length
    frame_timestep = hop_length / sample_rate  # Target timestep (usually 512/44100 = ~0.0116s)
    
    # Calculate target F0 length
    target_f0_len = max(1, total_frames)
    
    # Interpolate F0 to match target length
    if len(f0_hz) != target_f0_len:
        import scipy.interpolate
        x_old = np.arange(len(f0_hz)) * ds_f0_timestep
        x_new = np.arange(target_f0_len) * frame_timestep
        f_interp = scipy.interpolate.interp1d(x_old, f0_hz, kind='linear', fill_value='extrapolate')
        f0_hz = f_interp(x_new).astype(np.float32)

    # Adjust durations to match target frames
    total_dur = sum(ph_dur_frames)
    if total_dur > target_f0_len:
        diff = total_dur - target_f0_len
        idx = len(ph_dur_frames) - 1
        while diff > 0 and idx >= 0:
            reducible = max(0, ph_dur_frames[idx] - 1)
            if reducible > 0:
                reduce_by = min(diff, reducible)
                ph_dur_frames[idx] -= reduce_by
                diff -= reduce_by
            idx -= 1
        if diff > 0:
            ph_dur_frames[-1] = max(1, ph_dur_frames[-1] - diff)
    elif total_dur < target_f0_len:
        ph_dur_frames[-1] += (target_f0_len - total_dur)
    
    # Interpolate F0 and compute UV mask (1=unvoiced)
    f0_interp, uv = interpolate_f0_with_uv(f0_hz, uv_threshold=50.0)

    # Convert F0 from Hz to log scale (as expected by model)
    f0_log = np.log(f0_interp + 1e-5)
    
    # Generate segment name from text or offset
    text = segment.get('text', '')
    offset = segment.get('offset', 0)
    if text:
        # Use first few words as name
        name_parts = text.strip().split()[:3]
        segment_name = '_'.join(name_parts).replace('SP', '').strip('_')
        if not segment_name:
            segment_name = f"seg_{offset:.2f}"
    else:
        segment_name = f"seg_{offset:.2f}"
    
    # Clean up segment name
    segment_name = segment_name.replace(' ', '_')[:50]  # Limit length
    
    return ph_seq.strip(), ' '.join(map(str, ph_dur_frames)), f0_log, f0_interp, uv, segment_name


def synthesize_from_ds_segment(model, phone_map, segment: Dict, device, n_steps: int = 10,
                               sample_rate: int = 44100, hop_length: int = 512,
                               f0_stats: Optional[Tuple[float, float]] = None,
                               speaker_id: Optional[int] = None):
    """
    Synthesize audio from a single DS segment.
    
    Args:
        model: RFSinger model
        phone_map: Phoneme to ID mapping
        segment: DS segment dictionary
        device: torch device
        n_steps: ODE integration steps
        sample_rate: Audio sample rate
        hop_length: STFT hop length
        f0_stats: (mean, std) for F0 normalization, or None to skip normalization
        speaker_id: int speaker index, optional (for multi-speaker)
    
    Returns:
        (mel, f0_log, uv, segment_name, offset_seconds)
    """
    # Extract synthesis parameters
    phonemes, durations, f0_log, f0_hz, uv, segment_name = extract_synthesis_params(
        segment, sample_rate, hop_length
    )
    offset_seconds = float(segment.get('offset', 0.0))
    
    print(f"\nProcessing segment: {segment_name}")
    print(f"  Phonemes: {phonemes}")
    print(f"  Duration frames: {durations}")
    print(f"  F0 length: {len(f0_log)}")
    
    # Normalize F0 if stats provided
    f0_normalized = f0_log
    if f0_stats is not None:
        f0_mean, f0_std = f0_stats
        f0_normalized = (f0_log - f0_mean) / (f0_std + 1e-6)
    
    # Convert to tensors
    text = text_to_sequence(phonemes, phone_map).to(device)
    dur_list = [int(d) for d in durations.strip().split()]
    duration = torch.LongTensor(dur_list).unsqueeze(0).to(device)
    src_mask = torch.ones(1, text.size(1), dtype=torch.bool, device=device)
    f0_tensor = torch.FloatTensor(f0_normalized).unsqueeze(0).to(device)
    uv_tensor = torch.FloatTensor(uv).unsqueeze(0).to(device)
    
    # Speaker ID (optional)
    spk_tensor = None
    if speaker_id is not None:
        spk_tensor = torch.LongTensor([speaker_id]).to(device)
    
    # Synthesize
    with torch.no_grad():
        mel = model.inference(text, src_mask, duration, f0_tensor, n_steps=n_steps, uv=uv_tensor, speaker_id=spk_tensor)
    
    mel_np = mel.squeeze(0).cpu().numpy()
    
    return mel_np, f0_hz, uv, segment_name, offset_seconds


def main():
    parser = argparse.ArgumentParser(
        description="RFSinger Inference from DiffSinger (.ds) Project Files"
    )
    
    # Required arguments
    parser.add_argument('ds_file', type=str, help='Path to .ds project file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (or experiment directory)')
    
    # Optional config arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml (optional)')
    parser.add_argument('--phone_map', type=str, default=None,
                       help='Path to phone_map.json (optional)')
    parser.add_argument('--f0_stats', type=str, default=None,
                       help='Path to F0 statistics for normalization (optional)')
    
    # Segment selection
    parser.add_argument('--segment_index', type=int, default=None,
                       help='Select specific segment index (0-based, None=all)')
    parser.add_argument('--max_segments', type=int, default=None,
                       help='Maximum number of segments to synthesize')
    
    # Speaker selection (multi-speaker models)
    parser.add_argument('--speaker', type=str, default=None,
                       help='Speaker name or ID for multi-speaker models')
    
    # Output arguments
    parser.add_argument('--output', '--output_path', dest='output_path', type=str, default=None,
                       help='Output path for merged .wav (includes filename)')
    
    # Synthesis arguments
    parser.add_argument('--n_steps', type=int, default=None,
                       help='Number of ODE steps (default: from config)')
    parser.add_argument('--sample_rate', type=int, default=None,
                       help='Audio sample rate (default: from config)')
    parser.add_argument('--hop_length', type=int, default=None,
                       help='STFT hop length (default: from config)')

    # Vocoder arguments
    parser.add_argument('--vocoder', type=str,
                       choices=['griffin_lim', 'pyworld', 'nsf_hifigan', 'none'],
                       default=None,
                       help='Vocoder for mel-to-audio conversion (default: from config)')
    parser.add_argument('--vocoder_config', type=str, default=None,
                       help='Path to NSF-HiFiGAN config.json (default: from config)')
    parser.add_argument('--vocoder_ckpt', type=str, default=None,
                       help='Path to NSF-HiFiGAN model checkpoint (default: from config)')
    
    # Additional options
    parser.add_argument('--save_mel', action='store_true',
                       help='Save mel spectrograms as .npy files')
    parser.add_argument('--save_f0', action='store_true',
                       help='Save F0 contours as .npy files')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse DS file
    print(f"\nLoading DS file: {args.ds_file}")
    segments = parse_ds_file(args.ds_file)
    print(f"Found {len(segments)} segments")
    
    # Filter segments
    if args.segment_index is not None:
        if args.segment_index < 0 or args.segment_index >= len(segments):
            print(f"Error: segment_index {args.segment_index} out of range [0, {len(segments)-1}]")
            return
        segments = [segments[args.segment_index]]
        print(f"Selected segment index {args.segment_index}")
    elif args.max_segments is not None:
        segments = segments[:args.max_segments]
        print(f"Limited to first {args.max_segments} segments")
    
    # Resolve checkpoint path
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    
    # Try to load config
    config = None
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        try:
            config = load_config_from_checkpoint(os.path.dirname(checkpoint_path))
            print(f"Loaded config from checkpoint directory")
        except FileNotFoundError:
            print("Warning: Could not load config, using defaults")
    
    # Determine paths from config or arguments
    if config:
        data_dir = config['data']['processed_dir']
        default_phone_map = os.path.join(data_dir, 'phone_map.json')
        default_f0_stats = os.path.join(data_dir, 'f0_stats.npy')
        default_vocoder_config = config['inference']['vocoder_config']
        default_vocoder_ckpt = config['inference']['vocoder_ckpt']
        default_vocoder = config['inference']['vocoder']
        default_n_steps = config['inference']['n_steps']
        default_output_dir = config['inference']['output_dir']
        sample_rate = config['audio']['sample_rate']
        hop_length = config['audio']['hop_length']
    else:
        default_phone_map = './processed_data/phone_map.json'
        default_f0_stats = './processed_data/f0_stats.npy'
        default_vocoder_config = './ckpts/nsf_hifigan/config.json'
        default_vocoder_ckpt = './ckpts/nsf_hifigan/model.ckpt'
        default_vocoder = 'nsf_hifigan'
        default_n_steps = 10
        default_output_dir = './outputs'
        sample_rate = 44100
        hop_length = 512

    # Apply overrides: CLI args take precedence when explicitly provided
    phone_map_path = args.phone_map or default_phone_map
    f0_stats_path = args.f0_stats or default_f0_stats
    vocoder_type = args.vocoder if args.vocoder is not None else default_vocoder
    vocoder_config_path = args.vocoder_config if args.vocoder_config is not None else default_vocoder_config
    vocoder_ckpt_path = args.vocoder_ckpt if args.vocoder_ckpt is not None else default_vocoder_ckpt
    n_steps = args.n_steps if args.n_steps is not None else default_n_steps
    if args.sample_rate is not None:
        sample_rate = args.sample_rate
    if args.hop_length is not None:
        hop_length = args.hop_length
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}")
    model, phone_map, loaded_config, model_config = load_model(
        checkpoint_path, device, phone_map_path, config
    )
    
    # Load speaker map and resolve speaker_id
    speaker_id = None
    _data_dir = config['data']['processed_dir'] if config else './processed_data'
    _spk_map_path = os.path.join(_data_dir, 'speaker_map.json')
    _speaker_map = {}
    if os.path.exists(_spk_map_path):
        with open(_spk_map_path, 'r') as _f:
            _speaker_map = json.load(_f)
        if len(_speaker_map) > 1:
            print(f"Multi-speaker model. Available speakers: {list(_speaker_map.keys())}")
    
    if args.speaker is not None:
        if args.speaker in _speaker_map:
            speaker_id = _speaker_map[args.speaker]
        elif args.speaker.isdigit():
            speaker_id = int(args.speaker)
        else:
            print(f"Warning: Speaker '{args.speaker}' not found. "
                  f"Available: {list(_speaker_map.keys())}. Defaulting to 0.")
            speaker_id = 0
        print(f"Using speaker_id={speaker_id}")
    elif len(_speaker_map) > 1:
        speaker_id = 0
        print(f"No --speaker specified, defaulting to speaker_id=0")
    
    # Load F0 stats if available
    f0_stats = None
    if os.path.exists(f0_stats_path):
        f0_stats_arr = np.load(f0_stats_path)
        f0_stats = (f0_stats_arr[0], f0_stats_arr[1])
        print(f"Loaded F0 stats from {f0_stats_path}")
        print(f"  F0 mean: {f0_stats[0]:.3f}, std: {f0_stats[1]:.3f}")
    else:
        print("Warning: F0 stats not found, will use unnormalized F0")
    
    # Load vocoder if needed
    vocoder = None
    vocoder_h = None
    if vocoder_type == 'nsf_hifigan':
        if not os.path.exists(vocoder_config_path):
            print(f"Error: Vocoder config not found at {vocoder_config_path}")
            print("Falling back to pyworld vocoder")
            vocoder_type = 'pyworld'
        elif not os.path.exists(vocoder_ckpt_path):
            print(f"Error: Vocoder checkpoint not found at {vocoder_ckpt_path}")
            print("Falling back to pyworld vocoder")
            vocoder_type = 'pyworld'
        else:
            # Check mel parameter compatibility
            with open(vocoder_config_path, 'r') as f:
                voc_config = json.load(f)
            model_n_mels = model_config.get('mel_channels', 128)
            voc_n_mels = voc_config.get('num_mels', 128)
            if model_n_mels != voc_n_mels:
                print(f"Warning: Mel spectrogram mismatch!")
                print(f"  Model outputs {model_n_mels} mel channels, vocoder expects {voc_n_mels}")
                print("Falling back to pyworld vocoder for compatibility")
                vocoder_type = 'pyworld'
            else:
                vocoder, vocoder_h = load_nsf_hifigan(
                    vocoder_config_path, vocoder_ckpt_path, device
                )
    
    # Determine merged output path and auxiliary output directory
    if args.output_path:
        merged_output_path = args.output_path
    else:
        ds_base = os.path.splitext(os.path.basename(args.ds_file))[0]
        merged_output_path = os.path.join('./outputs', f"{ds_base}.wav")
    aux_output_dir = os.path.dirname(merged_output_path) or '.'
    os.makedirs(aux_output_dir, exist_ok=True)
    output_stem = os.path.splitext(os.path.basename(merged_output_path))[0]
    
    # Process segments
    print(f"\n{'='*60}")
    print(f"Synthesizing {len(segments)} segment(s)...")
    print(f"{'='*60}")
    
    merged_audio_segments = []

    for idx, segment in enumerate(segments):
        try:
            # Synthesize
            mel, f0_hz, uv, segment_name, offset_seconds = synthesize_from_ds_segment(
                model, phone_map, segment, device, n_steps,
                sample_rate, hop_length, f0_stats, speaker_id=speaker_id
            )
            
            # Generate auxiliary output filename base
            if args.segment_index is not None:
                output_base = f"{output_stem}_{args.segment_index}_{segment_name}"
            else:
                output_base = f"{output_stem}_{idx:03d}_{segment_name}"
            
            # Save mel spectrogram
            if args.save_mel:
                mel_path = os.path.join(aux_output_dir, f"{output_base}_mel.npy")
                np.save(mel_path, mel)
                print(f"  Saved mel: {mel_path}")
            
            # Save F0 (Hz, before normalization) for reference
            if args.save_f0:
                f0_path = os.path.join(aux_output_dir, f"{output_base}_f0.npy")
                np.save(f0_path, f0_hz)
                print(f"  Saved F0: {f0_path}")
            
            # Convert to audio and collect for merged output
            if vocoder_type == 'none':
                print("  Skipping audio generation (vocoder=none)")
                audio = None
            elif vocoder_type == 'griffin_lim':
                print("  Converting to audio (Griffin-Lim)...")
                audio = mel_to_audio(mel, sample_rate=sample_rate, hop_length=hop_length)
            elif vocoder_type == 'pyworld':
                print("  Converting to audio (PyWorld)...")
                # PyWorld needs log-scale F0; convert Hz to log, zero unvoiced
                f0_log_for_vocoder = np.log(f0_hz + 1e-5)
                if uv is not None:
                    f0_log_for_vocoder[uv > 0.5] = np.log(1e-5)
                audio = mel_to_audio_pyworld(
                    mel, f0_log_for_vocoder, sample_rate=sample_rate,
                    hop_length=hop_length, f0_is_log=True
                )
            elif vocoder_type == 'nsf_hifigan':
                print("  Converting to audio (NSF-HiFiGAN)...")
                # NSF-HiFiGAN needs F0 in linear Hz; zero unvoiced
                f0_linear = f0_hz.copy()
                if uv is not None:
                    f0_linear[uv > 0.5] = 0.0

                audio = mel_to_audio_nsf_hifigan(
                    mel, f0_linear, vocoder, vocoder_h, device
                )

            if audio is not None:
                merged_audio_segments.append((audio, offset_seconds))
            
            print(f"  ✓ Segment {idx+1}/{len(segments)} complete")
            
        except Exception as e:
            print(f"  ✗ Error processing segment {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if merged_audio_segments:
        # Place segments based on offset (seconds). Overlaps are summed.
        max_end = 0
        for audio, offset_seconds in merged_audio_segments:
            start = max(0, int(round(offset_seconds * sample_rate)))
            end = start + len(audio)
            if end > max_end:
                max_end = end

        merged_audio = np.zeros(max_end, dtype=np.float32)
        for audio, offset_seconds in merged_audio_segments:
            start = max(0, int(round(offset_seconds * sample_rate)))
            end = start + len(audio)
            merged_audio[start:end] += audio

        sf.write(merged_output_path, merged_audio, sample_rate)
        print(f"\nMerged audio saved to: {merged_output_path}")
    else:
        print("\nNo audio was generated to merge (check vocoder settings).")

    print(f"\n{'='*60}")
    print(f"Synthesis complete! Output file: {merged_output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
