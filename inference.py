import os
import json
import argparse
import numpy as np
import torch
import soundfile as sf

from train import RFSingerModel
from utils.config import load_config, load_config_from_checkpoint


def resolve_checkpoint_path(checkpoint_path: str) -> str:
    """
    Resolve a checkpoint path from a file or experiment directory.
    Supports both current layout (checkpoint_*.pt in exp dir)
    and legacy layout (checkpoints/ subdir).
    """
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    exp_dir = checkpoint_path
    candidates = [
        os.path.join(exp_dir, "checkpoint_best.pt"),
        os.path.join(exp_dir, "checkpoint_latest.pt"),
        os.path.join(exp_dir, "checkpoint_ema.pt"),
        os.path.join(exp_dir, "checkpoints", "checkpoint_best.pt"),
        os.path.join(exp_dir, "checkpoints", "checkpoint_latest.pt"),
        os.path.join(exp_dir, "checkpoints", "checkpoint_ema.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"No checkpoint file found in directory: {checkpoint_path}")


def load_model(checkpoint_path, device, phone_map_path=None, config=None):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device
        phone_map_path: Optional path to phone_map.json (will try to infer from config)
        config: Optional config dict (will try to load from checkpoint)
    
    Returns:
        model, phone_map, config, model_config
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to get config from checkpoint
    if config is None:
        config = checkpoint.get('config', None)
    
    # Determine phone_map path
    if phone_map_path is None and config is not None:
        phone_map_path = os.path.join(config['data']['processed_dir'], 'phone_map.json')
    elif phone_map_path is None:
        # Try default path
        phone_map_path = './processed_data/phone_map.json'
    
    # Load phone map
    with open(phone_map_path, 'r') as f:
        phone_map = json.load(f)
    vocab_size = len(phone_map)
    
    # Get model config - prefer from checkpoint
    model_config = checkpoint.get('model_config', None)
    if model_config is None and config is not None:
        model_config = config.get('model', {})
    if model_config is None:
        # Fall back to old args format
        args = checkpoint.get('args', {})
        model_config = {
            'd_model': args.get('d_model', 256),
            'n_encoder_layers': args.get('n_encoder_layers', 4),
            'n_head': args.get('n_head', 2),
            'mel_channels': args.get('mel_channels', 80),
            'flow_hidden': args.get('flow_hidden', 256),
            'n_flow_layers': args.get('n_flow_layers', 20),
            'dropout': args.get('dropout', 0.1),
            # Coarse Mel Decoder parameters (defaults for backward compatibility)
            'coarse_n_layers': args.get('coarse_n_layers', 2),
            'coarse_n_head': args.get('coarse_n_head', 2),
            'coarse_conv_channels': args.get('coarse_conv_channels', 512),
            'coarse_kernel_size': args.get('coarse_kernel_size', 5),
        }

    # If checkpoint model_config is incomplete, fill missing fields from config.yaml
    if config is not None and isinstance(config.get('model', None), dict):
        for key, value in config['model'].items():
            if key not in model_config:
                model_config[key] = value
    
    # Align coarse decoder settings to checkpoint weights when config mismatches.
    state_dict = checkpoint.get('model_state_dict', {})
    if isinstance(state_dict, dict) and state_dict:
        overrides = {}
        conv_key = "variance_adaptor.coarse_decoder.input_conv.0.conv.0.weight"
        if conv_key in state_dict:
            conv_weight = state_dict[conv_key]
            if hasattr(conv_weight, "shape") and len(conv_weight.shape) == 3:
                overrides["coarse_conv_channels"] = int(conv_weight.shape[0])
                overrides["coarse_kernel_size"] = int(conv_weight.shape[-1])

        layer_prefix = "variance_adaptor.coarse_decoder.transformer.layers."
        layer_ids = set()
        for key in state_dict.keys():
            if key.startswith(layer_prefix):
                remainder = key[len(layer_prefix):]
                idx_str = remainder.split('.', 1)[0]
                if idx_str.isdigit():
                    layer_ids.add(int(idx_str))
        if layer_ids:
            overrides["coarse_n_layers"] = max(layer_ids) + 1

        if overrides:
            for key, value in overrides.items():
                if model_config.get(key) != value:
                    model_config[key] = value
            print("Adjusted coarse decoder config from checkpoint state_dict:", overrides)

    # Create model
    # Determine number of speakers from checkpoint or speaker_map
    n_speakers = checkpoint.get('n_speakers', 1)
    if n_speakers <= 1 and config is not None:
        # Try loading speaker map from data dir
        spk_map_path = os.path.join(config['data']['processed_dir'], 'speaker_map.json')
        if os.path.exists(spk_map_path):
            with open(spk_map_path, 'r') as f:
                spk_map = json.load(f)
            n_speakers = len(spk_map)

    # Final fallback: detect speaker embedding from state_dict itself
    spk_emb_key = "encoder.speaker_embedding.weight"
    if n_speakers <= 1 and spk_emb_key in state_dict:
        n_speakers = int(state_dict[spk_emb_key].shape[0])
        print(f"Detected n_speakers={n_speakers} from checkpoint state_dict")

    use_spk_emb = model_config.get('use_speaker_embedding', True) and n_speakers > 1
    
    model = RFSingerModel(
        vocab_size=vocab_size,
        d_model=model_config.get('d_model', 256),
        n_encoder_layers=model_config.get('n_encoder_layers', 4),
        n_head=model_config.get('n_head', 2),
        mel_channels=model_config.get('mel_channels', 80),
        flow_hidden=model_config.get('flow_hidden', 256),
        n_flow_layers=model_config.get('n_flow_layers', 20),
        dropout=model_config.get('dropout', 0.1),
        # Coarse Mel Decoder parameters
        coarse_n_layers=model_config.get('coarse_n_layers', 2),
        coarse_n_head=model_config.get('coarse_n_head', 2),
        coarse_conv_channels=model_config.get('coarse_conv_channels', 512),
        coarse_kernel_size=model_config.get('coarse_kernel_size', 5),
        # Multi-speaker parameters
        n_speakers=n_speakers if use_spk_emb else 1,
        speaker_embedding_dim=model_config.get('speaker_embedding_dim', 256),
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Trained for {checkpoint.get('epoch', 'unknown')+1} epochs")
    
    return model, phone_map, config, model_config


def load_nsf_hifigan(config_path, checkpoint_path, device):
    """
    Load NSF-HiFiGAN vocoder.
    
    Args:
        config_path: path to config.json
        checkpoint_path: path to model.ckpt
        device: torch device
    
    Returns:
        vocoder model, h (config)
    """
    from utils.nsf_hifigan.models import load_model as load_hifigan_model
    import pathlib
    
    # Load model (load_model automatically loads config from same directory)
    vocoder, h = load_hifigan_model(pathlib.Path(checkpoint_path))
    vocoder = vocoder.to(device)
    vocoder.eval()
    
    print(f"Loaded NSF-HiFiGAN vocoder from {checkpoint_path}")
    
    return vocoder, h


def mel_to_audio_nsf_hifigan(mel, f0, vocoder, h, device):
    """
    Convert mel spectrogram to audio using NSF-HiFiGAN vocoder.
    
    Args:
        mel: (T, 80) mel spectrogram (log10 scale)
        f0: (T,) fundamental frequency in Hz (linear scale, not log)
        vocoder: NSF-HiFiGAN model
        h: vocoder config
        device: torch device
    
    Returns:
        audio: (N,) synthesized audio waveform
    """
    with torch.no_grad():
        # Prepare mel: (T, 128) -> (1, 128, T)
        mel_tensor = torch.FloatTensor(mel).T.unsqueeze(0).to(device) 
        
        # Prepare f0: (T,) -> (1, T)
        f0_tensor = torch.FloatTensor(f0).unsqueeze(0).to(device)
        
        # Generate audio
        audio = vocoder(mel_tensor, f0_tensor)
        
        # (1, 1, N) -> (N,)
        audio = audio.squeeze().cpu().numpy()
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95
    
    return audio.astype(np.float32)


def text_to_sequence(text, phone_map):
    """Convert phoneme string to tensor."""
    phones = text.strip().split()
    sequence = [phone_map.get(p, 0) for p in phones]
    return torch.LongTensor(sequence).unsqueeze(0)


def mel_to_audio(mel, sample_rate=44100, hop_length=256):
    """
    Convert mel spectrogram to audio using Griffin-Lim algorithm.
    For better quality, consider using a neural vocoder like HiFi-GAN.
    
    Args:
        mel: (T, 80) mel spectrogram (log scale)
        sample_rate: audio sample rate
        hop_length: STFT hop length
    
    Returns:
        audio: (T * hop_length,) audio waveform
    """
    import librosa
    
    # Convert from log scale
    mel = np.power(10, mel)
    
    # Mel to linear spectrogram (approximate inverse)
    n_fft = 1024
    n_mels = mel.shape[-1]
    
    # Create mel filterbank
    mel_basis = librosa.filters.mel(
        sr=sample_rate, 
        n_fft=n_fft, 
        n_mels=n_mels,
        fmin=20,
        fmax=8000
    )
    
    # Pseudo-inverse to convert mel to linear
    mel_basis_pinv = np.linalg.pinv(mel_basis)
    
    # mel: (T, 80) -> (80, T)
    mel_T = mel.T
    
    # Convert to linear spectrogram
    linear = np.maximum(1e-10, np.dot(mel_basis_pinv, mel_T))
    
    # Griffin-Lim
    audio = librosa.griffinlim(
        linear,
        n_iter=60,
        hop_length=hop_length,
        win_length=n_fft
    )
    
    return audio


def mel_to_audio_pyworld(mel, f0, sample_rate=44100, hop_length=256, f0_is_log=True):
    """
    Convert mel spectrogram to audio using PyWorld vocoder.
    
    Args:
        mel: (T, 80) mel spectrogram (log10 scale)
        f0: (T,) fundamental frequency contour (log scale if f0_is_log=True)
        sample_rate: audio sample rate
        hop_length: STFT hop length
        f0_is_log: whether f0 is in log scale
    
    Returns:
        audio: synthesized audio waveform
    """
    import pyworld as pw
    import librosa
    
    # Convert F0 from log scale to linear Hz
    if f0_is_log:
        f0_linear = np.exp(f0)
        # Handle very small values (unvoiced regions were logged with +1e-5)
        f0_linear = np.where(f0_linear < 1, 0, f0_linear)
    else:
        f0_linear = f0.copy()
    
    # Ensure F0 is float64 for PyWorld
    f0_linear = f0_linear.astype(np.float64)
    
    # Frame period in ms
    frame_period = hop_length / sample_rate * 1000
    
    # Convert mel (log10) to linear power spectrum
    mel_linear = np.power(10, mel)  # (T, 80)
    
    # Create mel filterbank for inversion
    n_fft = 1024
    n_mels = mel.shape[-1]
    fft_size = n_fft
    
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=20,
        fmax=8000
    )
    
    # Pseudo-inverse to get approximate linear spectrogram
    mel_basis_pinv = np.linalg.pinv(mel_basis)
    
    # mel_linear: (T, 80) -> (80, T)
    linear_spec = np.maximum(1e-10, np.dot(mel_basis_pinv, mel_linear.T))  # (n_fft/2+1, T)
    
    # PyWorld needs spectral envelope (sp) with shape (T, fft_size/2 + 1)
    # Convert power spectrum to spectral envelope
    sp = linear_spec.T  # (T, n_fft/2+1)
    
    # Ensure correct size for PyWorld (needs fft_size // 2 + 1 = 513 for fft_size=1024)
    expected_sp_size = fft_size // 2 + 1
    if sp.shape[1] != expected_sp_size:
        # Resize using interpolation
        from scipy.ndimage import zoom
        zoom_factor = expected_sp_size / sp.shape[1]
        sp = zoom(sp, (1, zoom_factor), order=1)
    
    # Ensure sp is positive and float64
    sp = np.maximum(sp, 1e-10).astype(np.float64)
    
    # Generate aperiodicity (ap) - use default band aperiodicity
    # For singing voice, we can estimate or use a simple model
    n_frames = len(f0_linear)
    ap = pw.d4c(np.zeros(int(n_frames * hop_length + hop_length), dtype=np.float64), 
                f0_linear, 
                np.arange(n_frames) * frame_period / 1000,  # temporal positions in seconds
                sample_rate,
                fft_size=fft_size)
    
    # If ap shape doesn't match, create default aperiodicity
    if ap.shape[0] != n_frames:
        # Create default aperiodicity (0.0 = fully periodic, 1.0 = fully aperiodic)
        ap = np.ones((n_frames, expected_sp_size), dtype=np.float64) * 0.5
        # Lower frequencies are more periodic for voiced sounds
        for i in range(n_frames):
            if f0_linear[i] > 0:  # Voiced
                ap[i, :expected_sp_size//4] = 0.1  # More periodic at low frequencies
            else:  # Unvoiced
                ap[i, :] = 0.9  # More aperiodic
    
    # Ensure shapes match
    min_frames = min(len(f0_linear), sp.shape[0], ap.shape[0])
    f0_linear = np.ascontiguousarray(f0_linear[:min_frames])
    sp = np.ascontiguousarray(sp[:min_frames])
    ap = np.ascontiguousarray(ap[:min_frames])
    
    # Synthesize audio using PyWorld
    audio = pw.synthesize(
        f0_linear,
        sp,
        ap,
        sample_rate,
        frame_period=frame_period
    )
    
    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95
    
    return audio.astype(np.float32)


def synthesize(model, phone_map, phonemes, durations, f0, device=None, n_steps=10, uv=None, speaker_id=None):
    """
    Synthesize mel spectrogram from phonemes.
    
    Args:
        model: trained RFSinger model
        phone_map: phoneme to id mapping
        phonemes: space-separated phoneme string
        durations: space-separated duration string (in frames)
        f0: F0 contour (numpy array, normalized) - REQUIRED
        device: torch device
        n_steps: ODE integration steps
        uv: UV mask (numpy array, 1=unvoiced, 0=voiced), optional
        speaker_id: int speaker index, optional (for multi-speaker)
    
    Returns:
        mel: (T, 80) synthesized mel spectrogram
    """
    if device is None:
        device = torch.device('cpu')
    
    # Convert phonemes to sequence
    text = text_to_sequence(phonemes, phone_map).to(device)
    
    # Parse durations
    dur_list = [int(d) for d in durations.strip().split()]
    duration = torch.LongTensor(dur_list).unsqueeze(0).to(device)
    
    # Create source mask
    src_mask = torch.ones(1, text.size(1), dtype=torch.bool, device=device)
    
    # F0 (required)
    f0_tensor = torch.FloatTensor(f0).unsqueeze(0).to(device)
    
    # UV mask (optional)
    uv_tensor = None
    if uv is not None:
        uv_tensor = torch.FloatTensor(uv).unsqueeze(0).to(device)
    
    # Speaker ID (optional)
    spk_tensor = None
    if speaker_id is not None:
        spk_tensor = torch.LongTensor([speaker_id]).to(device)
    
    # Inference
    with torch.no_grad():
        mel = model.inference(
            text, src_mask, duration, f0_tensor, n_steps=n_steps, uv=uv_tensor, speaker_id=spk_tensor
        )
    
    return mel.squeeze(0).cpu().numpy()


def synthesize_from_file(model, phone_map, input_file, device=None, n_steps=10, f0_file=None, f0_stats=None, uv_file=None):
    """
    Synthesize from a metadata file line.
    
    Args:
        input_file: path to file with format: id|phonemes|durations
        uv_file: path to UV mask file (.npy), optional
    """
    if device is None:
        device = torch.device('cpu')
    with open(input_file, 'r') as f:
        line = f.readline().strip()
    
    parts = line.split('|')
    file_id = parts[0]
    phonemes = parts[1]
    durations = parts[2]
    
    print(f"Synthesizing: {file_id}")
    print(f"Phonemes: {phonemes}")
    print(f"Durations: {durations}")
    
    if f0_file is None:
        raise ValueError("f0_file is required to synthesize from file input")

    f0_raw = np.load(f0_file)
    if f0_stats is not None:
        f0_raw = (f0_raw - f0_stats[0]) / (f0_stats[1] + 1e-6)
    
    # Load UV mask if provided
    uv = None
    if uv_file is not None and os.path.exists(uv_file):
        uv = np.load(uv_file)
        print(f"Using UV mask from {uv_file}")

    mel = synthesize(
        model, phone_map, phonemes, durations,
        f0=f0_raw, device=device, n_steps=n_steps, uv=uv
    )
    
    return mel, f0_raw, file_id


def main():
    parser = argparse.ArgumentParser(description="RFSinger Inference")
    
    # Config and model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (or experiment directory)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (optional, will try to load from checkpoint dir)')
    parser.add_argument('--phone_map', type=str, default=None,
                        help='Path to phone map JSON (optional, inferred from config)')
    parser.add_argument('--f0_stats', type=str, default=None,
                        help='Path to F0 statistics (optional, inferred from config)')
    
    # Input arguments (select from meta_file or input_file)
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input file with format: id|phonemes|durations')
    parser.add_argument('--f0_file', type=str, default=None,
                        help='F0 file (.npy) - required when using --input_file')
    
    # 新增：从训练集选择特定条目
    parser.add_argument('--meta_file', type=str, default='./processed_data/train.txt',
                        help='Path to metadata file (train.txt)')
    parser.add_argument('--index', type=int, default=None,
                        help='Select specific line index from meta_file (0-based, uses GT F0)')
    parser.add_argument('--sample_id', type=str, default=None,
                        help='Select specific sample by ID from meta_file (uses GT F0)')
    parser.add_argument('--data_dir', type=str, default='./processed_data',
                        help='Data directory for loading ground truth F0')
    
    # Speaker selection (multi-speaker models)
    parser.add_argument('--speaker', type=str, default=None,
                        help='Speaker name or ID for multi-speaker models')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--output_name', type=str, default='output',
                        help='Output filename (without extension)')
    
    # Synthesis arguments
    parser.add_argument('--n_steps', type=int, default=10,
                        help='Number of ODE steps')
    parser.add_argument('--vocoder', type=str, choices=['griffin_lim', 'pyworld', 'nsf_hifigan', 'none'], 
                        default='nsf_hifigan',
                        help='Vocoder for mel-to-audio conversion')
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help='Audio sample rate')
    
    # NSF-HiFiGAN vocoder arguments
    parser.add_argument('--vocoder_config', type=str, default='./ckpts/nsf_hifigan/config.json',
                        help='Path to NSF-HiFiGAN config.json')
    parser.add_argument('--vocoder_ckpt', type=str, default='./ckpts/nsf_hifigan/model.ckpt',
                        help='Path to NSF-HiFiGAN model checkpoint')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Resolve checkpoint path
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    config = None
    
    # Try to load config from checkpoint directory or explicit config
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        try:
            config = load_config_from_checkpoint(os.path.dirname(checkpoint_path))
            print(f"Loaded config from checkpoint directory")
        except FileNotFoundError:
            config = None
    
    # Determine data paths from config or defaults
    if config:
        data_dir = config['data']['processed_dir']
        default_phone_map = os.path.join(data_dir, 'phone_map.json')
        default_f0_stats = os.path.join(data_dir, 'f0_stats.npy')
        default_meta_file = os.path.join(data_dir, 'train.txt')
        default_vocoder_config = config['inference']['vocoder_config']
        default_vocoder_ckpt = config['inference']['vocoder_ckpt']
        default_vocoder = config['inference']['vocoder']
        default_n_steps = config['inference']['n_steps']
        default_output_dir = config['inference']['output_dir']
        sample_rate = config['audio']['sample_rate']
    else:
        data_dir = './processed_data'
        default_phone_map = './processed_data/phone_map.json'
        default_f0_stats = './processed_data/f0_stats.npy'
        default_meta_file = './processed_data/train.txt'
        default_vocoder_config = './ckpts/nsf_hifigan/config.json'
        default_vocoder_ckpt = './ckpts/nsf_hifigan/model.ckpt'
        default_vocoder = 'nsf_hifigan'
        default_n_steps = 10
        default_output_dir = './outputs'
        sample_rate = 44100
    
    # Apply argument overrides
    phone_map_path = args.phone_map or default_phone_map
    f0_stats_path = args.f0_stats or default_f0_stats
    meta_file = args.meta_file if args.meta_file != './processed_data/train.txt' else default_meta_file
    vocoder_type = args.vocoder if args.vocoder != 'nsf_hifigan' else default_vocoder
    vocoder_config_path = args.vocoder_config if args.vocoder_config != './ckpts/nsf_hifigan/config.json' else default_vocoder_config
    vocoder_ckpt_path = args.vocoder_ckpt if args.vocoder_ckpt != './ckpts/nsf_hifigan/model.ckpt' else default_vocoder_ckpt
    n_steps = args.n_steps if args.n_steps != 10 else default_n_steps
    output_dir = args.output_dir if args.output_dir != './outputs' else default_output_dir
    if args.sample_rate != 44100:
        sample_rate = args.sample_rate
    if args.data_dir != './processed_data':
        data_dir = args.data_dir
    
    # Load model
    model, phone_map, loaded_config, model_config = load_model(checkpoint_path, device, phone_map_path, config)
    
    # Load speaker map and resolve speaker_id
    speaker_id = None
    speaker_map_path = os.path.join(data_dir, 'speaker_map.json')
    speaker_map = {}
    if os.path.exists(speaker_map_path):
        with open(speaker_map_path, 'r') as f:
            speaker_map = json.load(f)
        if len(speaker_map) > 1:
            print(f"Multi-speaker model. Available speakers: {list(speaker_map.keys())}")
    
    if args.speaker is not None:
        if args.speaker in speaker_map:
            speaker_id = speaker_map[args.speaker]
        elif args.speaker.isdigit():
            speaker_id = int(args.speaker)
        else:
            print(f"Warning: Speaker '{args.speaker}' not found in speaker map. "
                  f"Available: {list(speaker_map.keys())}. Using speaker_id=0.")
            speaker_id = 0
        print(f"Using speaker_id={speaker_id}")
    elif len(speaker_map) > 1:
        # Default to first speaker
        speaker_id = 0
        print(f"No --speaker specified, defaulting to speaker_id=0")
    
    # Load F0 stats for denormalization
    f0_stats = np.load(f0_stats_path)
    f0_mean, f0_std = f0_stats[0], f0_stats[1]
    
    # Load NSF-HiFiGAN vocoder if needed
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine input method
    phonemes = None
    durations = None
    output_name = args.output_name
    f0_input = None  # F0 contour (normalized)
    uv_input = None  # UV mask (1=unvoiced, 0=voiced)
    
    # Method 1: Select from training set by index
    if args.index is not None:
        with open(meta_file, 'r') as f:
            lines = f.readlines()
        
        if args.index < 0 or args.index >= len(lines):
            print(f"Error: Index {args.index} out of range. Meta file has {len(lines)} entries.")
            return
        
        line = lines[args.index].strip()
        parts = line.split('|')
        output_name = parts[0]
        phonemes = parts[1]
        durations = parts[2]
        
        print(f"Selected index {args.index}: {output_name}")
        
        # Load GT F0 (required for this mode)
        f0_path = os.path.join(data_dir, f"{output_name}_f0.npy")
        if os.path.exists(f0_path):
            f0_raw = np.load(f0_path)
            f0_input = (f0_raw - f0_mean) / (f0_std + 1e-6)
            print(f"Using ground truth F0 from {f0_path}")
        else:
            print(f"Error: GT F0 file not found at {f0_path}")
            return
        
        # Load UV mask if available
        uv_path = os.path.join(data_dir, f"{output_name}_uv.npy")
        uv_input = None
        if os.path.exists(uv_path):
            uv_input = np.load(uv_path)
            print(f"Using UV mask from {uv_path}")
        
    # Method 2: Select from training set by sample ID
    elif args.sample_id is not None:
        with open(meta_file, 'r') as f:
            lines = f.readlines()
        
        found = False
        for line in lines:
            parts = line.strip().split('|')
            if parts[0] == args.sample_id:
                output_name = parts[0]
                phonemes = parts[1]
                durations = parts[2]
                found = True
                break
        
        if not found:
            print(f"Error: Sample ID '{args.sample_id}' not found in {meta_file}")
            return
        
        print(f"Selected sample: {output_name}")
        
        # Load GT F0 (required for this mode)
        f0_path = os.path.join(data_dir, f"{output_name}_f0.npy")
        if os.path.exists(f0_path):
            f0_raw = np.load(f0_path)
            f0_input = (f0_raw - f0_mean) / (f0_std + 1e-6)
            print(f"Using ground truth F0 from {f0_path}")
        else:
            print(f"Error: GT F0 file not found at {f0_path}")
            return
        
        # Load UV mask if available
        uv_path = os.path.join(data_dir, f"{output_name}_uv.npy")
        uv_input = None
        if os.path.exists(uv_path):
            uv_input = np.load(uv_path)
            print(f"Using UV mask from {uv_path}")
    
    # Method 3: From single-line input file (requires F0 file)
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            line = f.readline().strip()
        parts = line.split('|')
        output_name = parts[0]
        phonemes = parts[1]
        durations = parts[2]
        
        # F0 is required for this mode
        if args.f0_file:
            f0_raw = np.load(args.f0_file)
            f0_input = (f0_raw - f0_mean) / (f0_std + 1e-6)
            print(f"Using F0 from {args.f0_file}")
            
            # Try to load UV mask from same directory as F0 file
            uv_path = args.f0_file.replace('_f0.npy', '_uv.npy')
            if os.path.exists(uv_path):
                uv_input = np.load(uv_path)
                print(f"Using UV mask from {uv_path}")
        else:
            print("Error: --f0_file is required when using --input_file")
            return
        
    else:
        print("Error: Please provide one of the following:")
        print("  --index N              : Select Nth line from meta_file (uses GT F0)")
        print("  --sample_id ID         : Select sample by ID from meta_file (uses GT F0)")
        print("  --input_file FILE --f0_file F0.npy : Read from file with external F0")
        return
    
    print(f"Phonemes: {phonemes}")
    print(f"Durations: {durations}")
    
    # Synthesize (f0_input is required, uv_input is optional)
    mel = synthesize(
        model, phone_map, phonemes, durations, 
        f0=f0_input, device=device, n_steps=n_steps, uv=uv_input, speaker_id=speaker_id
    )
    
    # Save mel spectrogram
    mel_path = os.path.join(output_dir, f"{output_name}_mel.npy")
    np.save(mel_path, mel)
    print(f"Saved mel spectrogram to {mel_path}")
    
    # Denormalize F0 for saving and vocoder
    f0_denorm = f0_input * (f0_std + 1e-6) + f0_mean
    f0_path = os.path.join(output_dir, f"{output_name}_f0_used.npy")
    np.save(f0_path, f0_denorm)
    print(f"Saved F0 (used for synthesis) to {f0_path}")
    
    # Convert to audio
    if vocoder_type == 'griffin_lim':
        print("Converting mel to audio using Griffin-Lim...")
        audio = mel_to_audio(mel, sample_rate=sample_rate, hop_length=256)
        
        audio_path = os.path.join(output_dir, f"{output_name}.wav")
        sf.write(audio_path, audio, sample_rate)
        print(f"Saved audio to {audio_path}")
        
    elif vocoder_type == 'pyworld':
        print("Converting mel to audio using PyWorld vocoder...")
        audio = mel_to_audio_pyworld(
            mel, 
            f0_denorm,  # Use the actual F0 used in synthesis
            sample_rate=sample_rate, 
            hop_length=256,
            f0_is_log=True
        )
        
        audio_path = os.path.join(output_dir, f"{output_name}.wav")
        sf.write(audio_path, audio, sample_rate)
        print(f"Saved audio to {audio_path}")
        
    elif vocoder_type == 'nsf_hifigan':
        print("Converting mel to audio using NSF-HiFiGAN vocoder...")
        
        # NSF-HiFiGAN needs F0 in Hz (linear scale)
        # f0_denorm is in log scale, convert to linear
        f0_linear = np.exp(f0_denorm)
        # Handle unvoiced regions (very small values after exp)
        f0_linear = np.where(f0_linear < 1, 0, f0_linear)
        
        audio = mel_to_audio_nsf_hifigan(
            mel, f0_linear, vocoder, vocoder_h, device
        )
        
        audio_path = os.path.join(output_dir, f"{output_name}.wav")
        sf.write(audio_path, audio, sample_rate)
        print(f"Saved audio to {audio_path}")
    
    print("Synthesis complete!")


def batch_inference(checkpoint_path, phone_map_path, meta_file, output_dir, 
                    n_steps=10, device='cpu', vocoder_type='nsf_hifigan',
                    vocoder_config_path='./ckpts/nsf_hifigan/config.json',
                    vocoder_ckpt_path='./ckpts/nsf_hifigan/model.ckpt',
                    f0_stats_path='./processed_data/f0_stats.npy',
                    sample_rate=44100):
    """
    Batch inference for all entries in a metadata file.
    
    Args:
        checkpoint_path: path to model checkpoint
        phone_map_path: path to phone map JSON
        meta_file: path to metadata file (train.txt format)
        output_dir: output directory
        n_steps: ODE steps
        device: torch device
        vocoder_type: 'nsf_hifigan', 'pyworld', 'griffin_lim', or 'none'
        vocoder_config_path: path to NSF-HiFiGAN config
        vocoder_ckpt_path: path to NSF-HiFiGAN checkpoint
        f0_stats_path: path to F0 statistics
        sample_rate: audio sample rate
    """
    from tqdm import tqdm
    
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)

    # Load model
    model, phone_map, _, model_config = load_model(checkpoint_path, device, phone_map_path)
    
    # Load F0 stats
    f0_stats = np.load(f0_stats_path)
    f0_mean, f0_std = f0_stats[0], f0_stats[1]
    data_dir = os.path.dirname(f0_stats_path)
    
    # Load vocoder
    vocoder = None
    vocoder_config = None
    if vocoder_type == 'nsf_hifigan':
        vocoder, vocoder_config = load_nsf_hifigan(
            vocoder_config_path, vocoder_ckpt_path, device
        )
    
    # Load metadata
    with open(meta_file, 'r') as f:
        lines = f.readlines()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for line in tqdm(lines, desc="Synthesizing"):
        parts = line.strip().split('|')
        file_id = parts[0]
        phonemes = parts[1]
        durations = parts[2]
        
        try:
            f0_path = os.path.join(data_dir, f"{file_id}_f0.npy")
            if not os.path.exists(f0_path):
                print(f"Warning: F0 file missing for {file_id}, skipping")
                continue

            f0_raw = np.load(f0_path)
            f0_norm = (f0_raw - f0_mean) / (f0_std + 1e-6)
            
            # Load UV mask if available
            uv_path = os.path.join(data_dir, f"{file_id}_uv.npy")
            uv = None
            if os.path.exists(uv_path):
                uv = np.load(uv_path)

            mel = synthesize(
                model, phone_map, phonemes, durations,
                f0=f0_norm, device=device, n_steps=n_steps, uv=uv
            )
            
            # Save mel
            np.save(os.path.join(output_dir, f"{file_id}_mel.npy"), mel)
            
            # Denormalize pitch
            pitch_denorm = f0_raw
            
            # Convert to audio
            if vocoder_type == 'nsf_hifigan' and vocoder is not None:
                f0_linear = np.exp(pitch_denorm)
                f0_linear = np.where(f0_linear < 1, 0, f0_linear)
                audio = mel_to_audio_nsf_hifigan(mel, f0_linear, vocoder, vocoder_config, device)
                sf.write(os.path.join(output_dir, f"{file_id}.wav"), audio, sample_rate)
            elif vocoder_type == 'pyworld':
                audio = mel_to_audio_pyworld(mel, pitch_denorm, sample_rate=sample_rate)
                sf.write(os.path.join(output_dir, f"{file_id}.wav"), audio, sample_rate)
            elif vocoder_type == 'griffin_lim':
                audio = mel_to_audio(mel, sample_rate=sample_rate)
                sf.write(os.path.join(output_dir, f"{file_id}.wav"), audio, sample_rate)
            
        except Exception as e:
            print(f"Error synthesizing {file_id}: {e}")
            continue
    
    print(f"Batch inference complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()