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
        mel: (T, n_mels) mel spectrogram (log scale)
        f0: (T,) fundamental frequency in Hz (linear scale)
        vocoder: NSF-HiFiGAN model
        h: vocoder config
        device: torch device

    Returns:
        audio: (N,) synthesized waveform
    """
    with torch.no_grad():
        mel_tensor = torch.FloatTensor(mel).T.unsqueeze(0).to(device)
        f0_tensor = torch.FloatTensor(f0).unsqueeze(0).to(device)
        audio = vocoder(mel_tensor, f0_tensor)
        audio = audio.squeeze().cpu().numpy()

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95

    return audio.astype(np.float32)


def text_to_sequence(text, phone_map):
    """Convert phoneme string to tensor."""
    phones = text.strip().split()
    sequence = [phone_map.get(p, 0) for p in phones]
    return torch.LongTensor(sequence).unsqueeze(0)


def mel_to_audio(mel, sample_rate=44100, hop_length=512, n_fft=2048,
                 fmin=40, fmax=16000):
    """
    Convert mel spectrogram to audio using Griffin-Lim. Lowest-quality fallback.

    Args:
        mel: (T, n_mels) log-scale mel spectrogram
        sample_rate, hop_length, n_fft, fmin, fmax: STFT parameters; pass the
            same values used during preprocessing.

    Returns:
        audio: (N,) waveform
    """
    import librosa

    mel_linear = np.power(10, mel)
    n_mels = mel_linear.shape[-1]

    mel_basis = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
    )
    mel_basis_pinv = np.linalg.pinv(mel_basis)
    linear = np.maximum(1e-10, np.dot(mel_basis_pinv, mel_linear.T))

    audio = librosa.griffinlim(
        linear, n_iter=60, hop_length=hop_length, win_length=n_fft,
    )
    return audio


def mel_to_audio_pyworld(mel, f0, sample_rate=44100, hop_length=512,
                         n_fft=2048, fmin=40, fmax=16000, f0_is_log=True):
    """
    Convert mel spectrogram to audio using PyWorld synthesis.

    Args:
        mel: (T, n_mels) log-scale mel spectrogram
        f0: (T,) F0 contour, log scale if `f0_is_log` else linear Hz
        sample_rate, hop_length, n_fft, fmin, fmax: STFT parameters; pass the
            same values used during preprocessing.
        f0_is_log: whether f0 is in log scale

    Returns:
        audio: (N,) waveform
    """
    import pyworld as pw
    import librosa
    from scipy.ndimage import zoom

    if f0_is_log:
        f0_linear = np.exp(f0)
        f0_linear = np.where(f0_linear < 1, 0, f0_linear)
    else:
        f0_linear = f0.copy()
    f0_linear = f0_linear.astype(np.float64)

    frame_period = hop_length / sample_rate * 1000
    mel_linear = np.power(10, mel)
    n_mels = mel.shape[-1]

    mel_basis = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
    )
    mel_basis_pinv = np.linalg.pinv(mel_basis)
    linear_spec = np.maximum(1e-10, np.dot(mel_basis_pinv, mel_linear.T))

    sp = linear_spec.T  # (T, n_fft/2+1)
    expected_sp_size = n_fft // 2 + 1
    if sp.shape[1] != expected_sp_size:
        sp = zoom(sp, (1, expected_sp_size / sp.shape[1]), order=1)
    sp = np.maximum(sp, 1e-10).astype(np.float64)

    n_frames = len(f0_linear)
    try:
        ap = pw.d4c(
            np.zeros(int(n_frames * hop_length + hop_length), dtype=np.float64),
            f0_linear,
            np.arange(n_frames) * frame_period / 1000,
            sample_rate,
            fft_size=n_fft,
        )
    except Exception:
        ap = None

    if ap is None or ap.shape[0] != n_frames:
        # Fallback: synthesize a plausible aperiodicity profile from F0/UV.
        ap = np.ones((n_frames, expected_sp_size), dtype=np.float64) * 0.5
        for i in range(n_frames):
            if f0_linear[i] > 0:
                ap[i, :expected_sp_size // 4] = 0.1
            else:
                ap[i, :] = 0.9

    min_frames = min(len(f0_linear), sp.shape[0], ap.shape[0])
    f0_linear = np.ascontiguousarray(f0_linear[:min_frames])
    sp = np.ascontiguousarray(sp[:min_frames])
    ap = np.ascontiguousarray(ap[:min_frames])

    audio = pw.synthesize(f0_linear, sp, ap, sample_rate, frame_period=frame_period)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95
    return audio.astype(np.float32)


def synthesize(model, phone_map, phonemes, durations, f0, device=None, n_steps=10, uv=None, speaker_id=None):
    """
    Synthesize mel spectrogram from phonemes.

    Args:
        model: trained RFSinger model
        phone_map: phoneme-to-id mapping
        phonemes: space-separated phoneme string
        durations: space-separated duration string (in frames)
        f0: F0 contour (numpy array, normalized) — REQUIRED
        device: torch device
        n_steps: ODE integration steps
        uv: UV mask (numpy array, 1=unvoiced, 0=voiced), optional
        speaker_id: int speaker index, optional (for multi-speaker models)

    Returns:
        mel: (T, n_mels) synthesized mel spectrogram
    """
    if device is None:
        device = torch.device('cpu')

    text = text_to_sequence(phonemes, phone_map).to(device)
    dur_list = [int(d) for d in durations.strip().split()]
    duration = torch.LongTensor(dur_list).unsqueeze(0).to(device)
    src_mask = torch.ones(1, text.size(1), dtype=torch.bool, device=device)
    f0_tensor = torch.FloatTensor(f0).unsqueeze(0).to(device)

    uv_tensor = torch.FloatTensor(uv).unsqueeze(0).to(device) if uv is not None else None
    spk_tensor = torch.LongTensor([speaker_id]).to(device) if speaker_id is not None else None

    with torch.no_grad():
        mel = model.inference(
            text, src_mask, duration, f0_tensor, n_steps=n_steps,
            uv=uv_tensor, speaker_id=spk_tensor,
        )

    return mel.squeeze(0).cpu().numpy()


def _resolve_speaker_id(speaker_arg, speaker_map):
    """Map a --speaker CLI value (name or numeric id) to an int, or None."""
    if speaker_arg is None:
        return 0 if len(speaker_map) > 1 else None
    if speaker_arg in speaker_map:
        return speaker_map[speaker_arg]
    if speaker_arg.isdigit():
        return int(speaker_arg)
    print(f"Warning: speaker '{speaker_arg}' not in {list(speaker_map.keys())}, using 0")
    return 0


def _load_meta_entry(meta_file, *, index=None, sample_id=None):
    """Return (file_id, phonemes, durations) for an entry in train.txt."""
    with open(meta_file, 'r') as f:
        lines = f.readlines()

    if index is not None:
        if not 0 <= index < len(lines):
            raise IndexError(f"index {index} out of range for {len(lines)}-entry meta file")
        parts = lines[index].strip().split('|')
    else:
        parts = next(
            (line.strip().split('|') for line in lines if line.strip().split('|')[0] == sample_id),
            None,
        )
        if parts is None:
            raise KeyError(f"sample_id '{sample_id}' not found in {meta_file}")

    return parts[0], parts[1], parts[2]


def main():
    parser = argparse.ArgumentParser(description="RFSinger Inference")

    # Model and config
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint or experiment directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (auto-loaded from checkpoint dir when omitted)')
    parser.add_argument('--phone_map', type=str, default=None,
                        help='Override phone_map.json path (default: <processed_dir>/phone_map.json)')
    parser.add_argument('--f0_stats', type=str, default=None,
                        help='Override f0_stats.npy path (default: <processed_dir>/f0_stats.npy)')

    # Sample selection (one of --index or --sample_id is required)
    parser.add_argument('--meta_file', type=str, default=None,
                        help='Override metadata file path (default: <processed_dir>/train.txt)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override processed data directory for loading GT F0/UV')
    parser.add_argument('--index', type=int, default=None,
                        help='Pick the Nth line from meta_file (uses GT F0/UV)')
    parser.add_argument('--sample_id', type=str, default=None,
                        help='Pick a sample by id from meta_file (uses GT F0/UV)')

    # Multi-speaker
    parser.add_argument('--speaker', type=str, default=None,
                        help='Speaker name or numeric id (multi-speaker models only)')

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override inference.output_dir from config')

    # Synthesis
    parser.add_argument('--n_steps', type=int, default=None,
                        help='Override inference.n_steps from config')
    parser.add_argument('--vocoder', type=str, default=None,
                        choices=['griffin_lim', 'pyworld', 'nsf_hifigan', 'none'],
                        help='Override inference.vocoder from config')
    parser.add_argument('--vocoder_config', type=str, default=None,
                        help='Override inference.vocoder_config from config')
    parser.add_argument('--vocoder_ckpt', type=str, default=None,
                        help='Override inference.vocoder_ckpt from config')

    args = parser.parse_args()

    if args.index is None and args.sample_id is None:
        parser.error("provide either --index or --sample_id")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Resolve config (from --config or from the experiment dir of the checkpoint)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    config = None
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        try:
            config = load_config_from_checkpoint(os.path.dirname(checkpoint_path))
            print("Loaded config from checkpoint directory")
        except FileNotFoundError:
            parser.error("no config found next to checkpoint; pass --config explicitly")

    # Pull defaults exclusively from config; CLI flags only override when set.
    audio_cfg = config['audio']
    inf_cfg = config['inference']
    data_dir = args.data_dir or config['data']['processed_dir']
    phone_map_path = args.phone_map or os.path.join(data_dir, 'phone_map.json')
    f0_stats_path = args.f0_stats or os.path.join(data_dir, 'f0_stats.npy')
    meta_file = args.meta_file or os.path.join(data_dir, 'train.txt')
    output_dir = args.output_dir or inf_cfg['output_dir']
    n_steps = args.n_steps if args.n_steps is not None else inf_cfg['n_steps']
    vocoder_type = args.vocoder or inf_cfg['vocoder']
    vocoder_config_path = args.vocoder_config or inf_cfg['vocoder_config']
    vocoder_ckpt_path = args.vocoder_ckpt or inf_cfg['vocoder_ckpt']
    sample_rate = audio_cfg['sample_rate']
    hop_length = audio_cfg['hop_length']
    n_fft = audio_cfg['n_fft']
    fmin = audio_cfg['fmin']
    fmax = audio_cfg['fmax']

    # Model + phone map
    model, phone_map, _, model_config = load_model(
        checkpoint_path, device, phone_map_path, config,
    )

    # Speaker map
    speaker_map = {}
    speaker_map_path = os.path.join(data_dir, 'speaker_map.json')
    if os.path.exists(speaker_map_path):
        with open(speaker_map_path, 'r') as f:
            speaker_map = json.load(f)
        if len(speaker_map) > 1:
            print(f"Available speakers: {list(speaker_map.keys())}")
    speaker_id = _resolve_speaker_id(args.speaker, speaker_map)
    if speaker_id is not None:
        print(f"Using speaker_id={speaker_id}")

    # F0 stats
    f0_stats = np.load(f0_stats_path)
    f0_mean, f0_std = float(f0_stats[0]), float(f0_stats[1])

    # Vocoder (with mel-channel sanity check for NSF-HiFiGAN)
    vocoder, vocoder_h = None, None
    if vocoder_type == 'nsf_hifigan':
        if not (os.path.exists(vocoder_config_path) and os.path.exists(vocoder_ckpt_path)):
            print(f"NSF-HiFiGAN files missing at {vocoder_config_path} / {vocoder_ckpt_path}")
            print("Falling back to pyworld vocoder")
            vocoder_type = 'pyworld'
        else:
            with open(vocoder_config_path, 'r') as f:
                voc_config = json.load(f)
            model_n_mels = model_config.get('mel_channels', audio_cfg['n_mels'])
            voc_n_mels = voc_config.get('num_mels', audio_cfg['n_mels'])
            if model_n_mels != voc_n_mels:
                print(f"Mel channel mismatch: model={model_n_mels}, vocoder={voc_n_mels}; using pyworld")
                vocoder_type = 'pyworld'
            else:
                vocoder, vocoder_h = load_nsf_hifigan(vocoder_config_path, vocoder_ckpt_path, device)

    os.makedirs(output_dir, exist_ok=True)

    # Resolve sample (--index or --sample_id)
    try:
        output_name, phonemes, durations = _load_meta_entry(
            meta_file, index=args.index, sample_id=args.sample_id,
        )
    except (IndexError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Selected sample: {output_name}")
    print(f"Phonemes: {phonemes}")
    print(f"Durations: {durations}")

    # Load GT F0 and UV
    f0_path = os.path.join(data_dir, f"{output_name}_f0.npy")
    if not os.path.exists(f0_path):
        parser.error(f"GT F0 file not found at {f0_path}")
    f0_raw = np.load(f0_path)
    f0_input = (f0_raw - f0_mean) / (f0_std + 1e-6)

    uv_path = os.path.join(data_dir, f"{output_name}_uv.npy")
    uv_input = np.load(uv_path) if os.path.exists(uv_path) else None
    if uv_input is not None:
        print(f"Using UV mask from {uv_path}")

    # Synthesize mel
    mel = synthesize(
        model, phone_map, phonemes, durations,
        f0=f0_input, device=device, n_steps=n_steps,
        uv=uv_input, speaker_id=speaker_id,
    )
    mel_path = os.path.join(output_dir, f"{output_name}_mel.npy")
    np.save(mel_path, mel)
    print(f"Saved mel to {mel_path}")

    # Save the F0 used for synthesis (denormalized, log-Hz)
    f0_used = f0_input * (f0_std + 1e-6) + f0_mean
    f0_used_path = os.path.join(output_dir, f"{output_name}_f0_used.npy")
    np.save(f0_used_path, f0_used)

    # Vocoding
    if vocoder_type == 'none':
        print("Synthesis complete (vocoder=none, skipping audio).")
        return

    audio_path = os.path.join(output_dir, f"{output_name}.wav")
    if vocoder_type == 'griffin_lim':
        print("Vocoding with Griffin-Lim...")
        audio = mel_to_audio(
            mel, sample_rate=sample_rate, hop_length=hop_length,
            n_fft=n_fft, fmin=fmin, fmax=fmax,
        )
    elif vocoder_type == 'pyworld':
        print("Vocoding with PyWorld...")
        audio = mel_to_audio_pyworld(
            mel, f0_used, sample_rate=sample_rate, hop_length=hop_length,
            n_fft=n_fft, fmin=fmin, fmax=fmax, f0_is_log=True,
        )
    else:  # nsf_hifigan
        print("Vocoding with NSF-HiFiGAN...")
        f0_linear = np.exp(f0_used)
        f0_linear = np.where(f0_linear < 1, 0, f0_linear)
        audio = mel_to_audio_nsf_hifigan(mel, f0_linear, vocoder, vocoder_h, device)

    sf.write(audio_path, audio, sample_rate)
    print(f"Saved audio to {audio_path}")
    print("Synthesis complete.")


if __name__ == '__main__':
    main()
