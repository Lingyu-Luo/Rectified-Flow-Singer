"""
RFSinger Streamlit GUI
Upload a .ds (DiffSinger) project file, configure parameters, and synthesize audio.
"""

import os
import io
import json
import tempfile
import numpy as np
import torch
import soundfile as sf
import streamlit as st

from inference import (
    load_model,
    load_nsf_hifigan,
    mel_to_audio_nsf_hifigan,
    mel_to_audio_pyworld,
    mel_to_audio,
    resolve_checkpoint_path,
)
from infer_ds import (
    parse_ds_file,
    synthesize_from_ds_segment,
)
from utils.config import load_config, load_config_from_checkpoint


# ---------------------------------------------------------------------------
# Cached loaders – avoid reloading on every rerun
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading RFSinger model …")
def cached_load_model(checkpoint_path: str, phone_map_path: str, config_path: str | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = None
    if config_path and os.path.isfile(config_path):
        config = load_config(config_path)
    else:
        try:
            config = load_config_from_checkpoint(os.path.dirname(checkpoint_path))
        except FileNotFoundError:
            pass
    model, phone_map, loaded_config, model_config = load_model(
        checkpoint_path, device, phone_map_path, config
    )
    return model, phone_map, loaded_config, model_config, device


@st.cache_resource(show_spinner="Loading NSF-HiFiGAN vocoder …")
def cached_load_vocoder(vocoder_config_path: str, vocoder_ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocoder, vocoder_h = load_nsf_hifigan(vocoder_config_path, vocoder_ckpt_path, device)
    return vocoder, vocoder_h, device


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_speaker_map(data_dir: str) -> dict:
    path = os.path.join(data_dir, "speaker_map.json")
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def load_f0_stats(path: str):
    if os.path.isfile(path):
        arr = np.load(path)
        return (arr[0], arr[1])
    return None


def merge_audio_segments(segments, sample_rate: int) -> np.ndarray:
    """Merge (audio, offset_seconds) pairs into a single waveform."""
    if not segments:
        return np.array([], dtype=np.float32)
    max_end = 0
    for audio, offset in segments:
        start = max(0, int(round(offset * sample_rate)))
        end = start + len(audio)
        if end > max_end:
            max_end = end
    merged = np.zeros(max_end, dtype=np.float32)
    for audio, offset in segments:
        start = max(0, int(round(offset * sample_rate)))
        end = start + len(audio)
        merged[start:end] += audio
    return merged


# ---------------------------------------------------------------------------
# Main Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="RFSinger", page_icon="🎤", layout="wide")
    st.title("🎤 RFSinger – Singing Voice Synthesis")
    st.caption("Upload a `.ds` project file and generate audio.")

    # ---- Sidebar: global settings ----
    with st.sidebar:
        st.header("⚙️ Settings")

        config_path = st.text_input(
            "Config YAML", value="./ckpts/opencpop/config.yaml",
            help="Path to config.yaml (leave default if unsure)."
        )

        # Try to load config for defaults
        config = None
        if os.path.isfile(config_path):
            config = load_config(config_path)

        if config:
            default_ckpt = os.path.join(
                config["checkpoint"]["output_dir"],
                config["experiment"]["name"],
            )
            default_processed = config["data"]["processed_dir"]
            default_voc_cfg = config["inference"]["vocoder_config"]
            default_voc_ckpt = config["inference"]["vocoder_ckpt"]
            default_vocoder = config["inference"]["vocoder"]
            default_n_steps = config["inference"]["n_steps"]
            default_sr = config["audio"]["sample_rate"]
            default_hop = config["audio"]["hop_length"]
        else:
            default_ckpt = "./ckpts/opencpop"
            default_processed = "./processed_data"
            default_voc_cfg = "./ckpts/nsf_hifigan/config.json"
            default_voc_ckpt = "./ckpts/nsf_hifigan/model.ckpt"
            default_vocoder = "nsf_hifigan"
            default_n_steps = 50
            default_sr = 44100
            default_hop = 512

        checkpoint_path = st.text_input("Model checkpoint", value=default_ckpt)
        phone_map_path = st.text_input(
            "Phone map", value=os.path.join(default_processed, "phone_map.json")
        )
        f0_stats_path = st.text_input(
            "F0 stats (.npy)", value=os.path.join(default_processed, "f0_stats.npy")
        )

        st.divider()
        st.subheader("Vocoder")
        vocoder_type = st.selectbox(
            "Vocoder", ["nsf_hifigan", "pyworld", "griffin_lim", "none"],
            index=["nsf_hifigan", "pyworld", "griffin_lim", "none"].index(default_vocoder),
        )
        vocoder_config_path = st.text_input("Vocoder config", value=default_voc_cfg)
        vocoder_ckpt_path = st.text_input("Vocoder checkpoint", value=default_voc_ckpt)

        st.divider()
        st.subheader("Synthesis")
        n_steps = st.slider("ODE steps", min_value=1, max_value=200, value=default_n_steps)
        sample_rate = st.number_input("Sample rate", value=default_sr, step=100)
        hop_length = st.number_input("Hop length", value=default_hop, step=64)

        # Speaker
        speaker_map = load_speaker_map(default_processed)
        speaker_names = list(speaker_map.keys())
        speaker_id: int | None = None
        if len(speaker_names) > 1:
            chosen_speaker = st.selectbox("Speaker", speaker_names)
            speaker_id = speaker_map.get(chosen_speaker, 0)
        elif len(speaker_names) == 1:
            speaker_id = 0

    # ---- Main area: upload & synthesise ----
    uploaded_file = st.file_uploader(
        "Upload a `.ds` project file", type=["ds", "json"],
        help="DiffSinger project files are JSON arrays of segments."
    )

    if uploaded_file is None:
        st.info("Upload a `.ds` file to get started.")
        return

    # Parse uploaded .ds content
    try:
        raw = uploaded_file.read().decode("utf-8")
        segments = json.loads(raw)
        if not isinstance(segments, list):
            st.error("The uploaded file must contain a JSON array of segments.")
            return
    except Exception as exc:
        st.error(f"Failed to parse `.ds` file: {exc}")
        return

    st.success(f"Loaded **{len(segments)}** segment(s) from `{uploaded_file.name}`")

    # Show segment overview
    with st.expander("📋 Segment overview", expanded=False):
        for i, seg in enumerate(segments):
            ph = seg.get("ph_seq", "")[:80]
            text = seg.get("text", "")
            offset = seg.get("offset", 0)
            st.markdown(f"**Seg {i}** — offset {offset:.2f}s | `{text}` | phones: `{ph}`…")

    # Segment selection
    seg_options = ["All segments"] + [
        "Seg {}: {}".format(i, seg.get("text", "")[:30] or "offset {:.2f}s".format(seg.get("offset", 0)))
        for i, seg in enumerate(segments)
    ]
    selected = st.selectbox("Segments to synthesise", seg_options)

    if selected == "All segments":
        target_segments = list(enumerate(segments))
    else:
        idx = seg_options.index(selected) - 1
        target_segments = [(idx, segments[idx])]

    # ----- Synthesise button -----
    if not st.button("🎵 Synthesise", type="primary", use_container_width=True):
        return

    # Resolve checkpoint
    try:
        ckpt_resolved = resolve_checkpoint_path(checkpoint_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    # Load model
    try:
        model, phone_map, loaded_config, model_config, device = cached_load_model(
            ckpt_resolved, phone_map_path, config_path
        )
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    # Load F0 stats
    f0_stats = load_f0_stats(f0_stats_path)
    if f0_stats:
        st.sidebar.caption(f"F0 mean={f0_stats[0]:.2f}, std={f0_stats[1]:.2f}")

    # Load vocoder
    vocoder = None
    vocoder_h = None
    if vocoder_type == "nsf_hifigan":
        if not os.path.isfile(vocoder_config_path) or not os.path.isfile(vocoder_ckpt_path):
            st.warning("NSF-HiFiGAN files not found – falling back to pyworld.")
            vocoder_type = "pyworld"
        else:
            # Mel-channel compatibility check
            with open(vocoder_config_path, "r") as f:
                voc_cfg = json.load(f)
            model_n_mels = model_config.get("mel_channels", 128)
            voc_n_mels = voc_cfg.get("num_mels", 128)
            if model_n_mels != voc_n_mels:
                st.warning(
                    f"Mel channel mismatch (model={model_n_mels}, vocoder={voc_n_mels}). "
                    "Falling back to pyworld."
                )
                vocoder_type = "pyworld"
            else:
                try:
                    vocoder, vocoder_h, _ = cached_load_vocoder(
                        vocoder_config_path, vocoder_ckpt_path
                    )
                except Exception as exc:
                    st.warning(f"Vocoder load error: {exc}. Falling back to pyworld.")
                    vocoder_type = "pyworld"

    # Process each segment
    progress = st.progress(0, text="Synthesising …")
    audio_segments: list[tuple[np.ndarray, float]] = []
    total = len(target_segments)

    for step, (seg_idx, segment) in enumerate(target_segments):
        progress.progress((step) / total, text=f"Segment {seg_idx + 1}/{len(segments)} …")
        try:
            mel, f0_hz, uv, seg_name, offset_sec = synthesize_from_ds_segment(
                model, phone_map, segment, device, n_steps,
                sample_rate, hop_length, f0_stats, speaker_id=speaker_id,
            )

            # Mel → audio
            audio: np.ndarray | None = None
            if vocoder_type == "nsf_hifigan" and vocoder is not None:
                f0_lin = f0_hz.copy()
                if uv is not None:
                    f0_lin[uv > 0.5] = 0.0
                audio = mel_to_audio_nsf_hifigan(mel, f0_lin, vocoder, vocoder_h, device)
            elif vocoder_type == "pyworld":
                f0_log = np.log(f0_hz + 1e-5)
                if uv is not None:
                    f0_log[uv > 0.5] = np.log(1e-5)
                audio = mel_to_audio_pyworld(
                    mel, f0_log, sample_rate=sample_rate,
                    hop_length=hop_length, f0_is_log=True,
                )
            elif vocoder_type == "griffin_lim":
                audio = mel_to_audio(mel, sample_rate=sample_rate, hop_length=hop_length)

            if audio is not None:
                audio_segments.append((audio, offset_sec))

        except Exception as exc:
            st.error(f"Segment {seg_idx} failed: {exc}")

    progress.progress(1.0, text="Done!")

    if not audio_segments:
        st.warning("No audio was generated. Check your settings.")
        return

    # Merge and present
    merged = merge_audio_segments(audio_segments, sample_rate)

    # Write to an in-memory WAV buffer
    wav_buf = io.BytesIO()
    sf.write(wav_buf, merged, sample_rate, format="WAV")
    wav_buf.seek(0)
    wav_bytes = wav_buf.getvalue()

    st.subheader("🔊 Result")
    st.audio(wav_bytes, format="audio/wav", sample_rate=sample_rate)

    # Download button
    ds_stem = os.path.splitext(uploaded_file.name)[0]
    st.download_button(
        label="⬇️ Download WAV",
        data=wav_bytes,
        file_name=f"{ds_stem}.wav",
        mime="audio/wav",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
