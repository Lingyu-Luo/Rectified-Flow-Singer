"""
Self-contained RMVPE pitch estimator.

No external dependencies beyond PyTorch, NumPy, torchaudio and the local
rmvpe sub-package.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from .constants import *
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_f0, to_viterbi_f0


class RMVPE:
    """Robust MIDI-based Vocal Pitch Estimation (RMVPE).

    Parameters
    ----------
    model_path : str
        Path to the RMVPE checkpoint (``rmvpe.pt``).
    hop_length : int
        Hop length in samples at 16 kHz (default 160 → 10 ms).
    device : str or None
        ``'cuda'``, ``'cpu'``, or *None* for auto-detect.
    """

    def __init__(self, model_path: str, hop_length: int = 160, device: str | None = None):
        self.resample_kernel: dict[str, Resample] = {}
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = E2E0(4, 1, (2, 2)).eval().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        ).to(self.device)
        # Internal hop (seconds) – RMVPE always operates at 16 kHz / 160 hop → 10 ms
        self._hop_sec = hop_length / SAMPLE_RATE

    # ── Internal helpers ─────────────────────────────────────────────────
    @torch.no_grad()
    def mel2hidden(self, mel: torch.Tensor) -> torch.Tensor:
        n_frames = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode='constant')
        hidden = self.model(mel)
        return hidden[:, :n_frames]

    def decode(self, hidden: torch.Tensor, thred: float = 0.03,
               use_viterbi: bool = False) -> np.ndarray:
        if use_viterbi:
            return to_viterbi_f0(hidden, thred=thred)
        return to_local_average_f0(hidden, thred=thred)

    # ── Public API ───────────────────────────────────────────────────────
    @torch.no_grad()
    def infer_from_audio(self, audio: np.ndarray, sample_rate: int = 16000,
                         thred: float = 0.03, use_viterbi: bool = False) -> np.ndarray:
        """Return F0 in Hz (unvoiced → 0) for a 1-D waveform array."""
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        if sample_rate != SAMPLE_RATE:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, SAMPLE_RATE,
                                                         lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.device)
            audio_t = self.resample_kernel[key_str](audio_t)
        mel = self.mel_extractor(audio_t, center=True)
        hidden = self.mel2hidden(mel)
        return self.decode(hidden, thred=thred, use_viterbi=use_viterbi)

    def infer_from_wav(self, wav_path: str, *, sample_rate: int | None = None,
                       thred: float = 0.03, use_viterbi: bool = False):
        """Convenience wrapper that reads a wav file and returns (f0_hz, times).

        *times* is an array of frame centre times (seconds) matching the
        internal 10 ms hop of RMVPE.
        """
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        f0 = self.infer_from_audio(audio, sample_rate=sr, thred=thred,
                                   use_viterbi=use_viterbi)
        times = np.arange(len(f0)) * self._hop_sec
        return f0.astype(np.float32), times
