# RFSinger

> **Notice:** The RMVPE F0 extractor integration has known bugs. The iterative rectified flow (reflow) stage also has performance issues. Prefer `f0.method: harvest` and base flow training when running the pipeline.

A singing voice synthesis (SVS) system based on **Rectified Flow** and **FastSpeech2** architecture. RFSinger combines the efficient encoder-decoder structure of FastSpeech2 with flow matching for high-quality mel-spectrogram generation.

## Features

- **FastSpeech2 Encoder**: Transformer-based phoneme encoder with positional encoding
- **Variance Adaptor**: Duration regulation, pitch conditioning with UV gating, and coarse mel decoder
- **Rectified Flow Decoder**: Flow matching mel-spectrogram generation with ODE sampling
- **Multi-speaker support**: Speaker embeddings for training on multiple datasets
- **Multiple vocoders**: NSF-HiFiGAN, PyWorld, and Griffin-Lim
- **DiffSinger (.ds) inference**: Synthesize from DiffSinger project files
- **Data augmentation**: Offline time-stretch and pitch-shift augmentation
- **Training features**: AMP, EMA, cosine LR schedule, gradient clipping, iterative reflow

## Architecture

```
Phonemes → FS2Encoder → VarianceAdaptor(LR + Pitch/UV Gate + CoarseMelDecoder)
                                │                        │
                          adapted (B,T,D)          coarse_mel (B,T,n_mels)
                                │                        │
                                ▼                        │
                       RectifiedFlowDecoder ─────────────┘
                                │
                          mel (B,T,n_mels)  →  Vocoder  →  Audio
```

The flow decoder uses `coarse_mel` as `x_0` (not Gaussian noise): `x_t = (1-t)*coarse_mel + t*mel_target`. This is why `coarse_loss` is part of the training objective.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Preprocess
python preprocess.py --config config.yaml

# 2. Train
python train.py --config config.yaml

# 3. Inference
python inference.py --checkpoint ./ckpts/rfsinger --index 0
```

## Configuration

All hyperparameters live in `config.yaml`. Edit it directly — CLI flags only override a small subset. The config is copied into the experiment directory at training start, so inference auto-loads it from the checkpoint folder.

### Key Audio Parameters

| Parameter | Value |
|-----------|-------|
| Sample Rate | 44100 Hz |
| FFT Size | 2048 |
| Hop Length | 512 |
| Win Length | 2048 |
| Mel Bands | 128 |
| F0 Range | 50–1100 Hz |

These must match the vocoder's expectations (NSF-HiFiGAN `num_mels`, `fmax`, etc.).

## Data Preparation

### Directory Structure

```
data/<speaker>/
├── wavs/
│   ├── sample1.wav
│   └── ...
└── transcriptions.csv
```

### CSV Format

| Column | Description |
|--------|-------------|
| `name` | Audio file name (without extension) |
| `ph_seq` | Space-separated phoneme sequence |
| `ph_dur` | Space-separated phoneme durations (seconds) |
| `ph_num` | Number of phonemes |
| `note_seq` | Note sequence (optional, used by .ds inference) |
| `note_dur` | Note durations (optional) |

### Multi-speaker Setup

Add entries under `data.speakers` in `config.yaml`:

```yaml
data:
  speakers:
    speaker_a:
      raw_data_path: "./data/speaker_a/wavs"
      csv_path: "./data/speaker_a/transcriptions.csv"
    speaker_b:
      raw_data_path: "./data/speaker_b/wavs"
      csv_path: "./data/speaker_b/transcriptions.csv"
```

When `speakers` is non-empty, preprocessing writes `speaker_map.json` and `train.txt` gains a speaker ID field. The model enables speaker embeddings automatically when `n_speakers > 1`.

### Preprocessing

```bash
python preprocess.py --config config.yaml
```

Output in `processed_data/`:
- `{speaker}/{id}_mel.npy` — Log-mel spectrogram (128-band)
- `{speaker}/{id}_f0.npy` — Interpolated F0 (log-scale if `use_log_f0: true`)
- `{speaker}/{id}_uv.npy` — Unvoiced mask (1=unvoiced, 0=voiced)
- `phone_map.json` — Phoneme → integer ID mapping
- `speaker_map.json` — Speaker → integer ID (multi-speaker only)
- `f0_stats.npy` — `[mean, std]` for z-score normalization
- `train.txt` — `file_id|ph_seq|ph_dur|speaker_id` per line

## Training

```bash
# Basic
python train.py --config config.yaml

# With overrides
python train.py --config config.yaml --exp_name myexp --epochs 200 --batch_size 16

# Resume from experiment directory (auto-loads saved config)
python train.py --config config.yaml --resume ./ckpts/myexp
```

### Iterative Reflow (two-step)

```bash
# Step 1: Generate reflow targets (writes processed_data/reflow/*_mel.npy)
python train.py --config config.yaml --reflow

# Step 2: Train on reflow targets (appends '-reflow' to experiment name)
python train.py --config config.yaml --reflow_train
```

Reflow freezes the encoder and trains only the flow decoder on straightened ODE trajectories.

### Checkpoint Structure

```
ckpts/<experiment_name>/
├── config.yaml              # Snapshot of training config
├── checkpoint_epoch10.pt
├── checkpoint_latest.pt
├── checkpoint_best.pt       # Best validation loss
└── checkpoint_ema.pt        # EMA weights (preferred for inference)
```

## Inference

```bash
# From metadata by index
python inference.py --checkpoint ./ckpts/rfsinger --index 0

# By sample ID
python inference.py --checkpoint ./ckpts/rfsinger --sample_id "sample_0001"

# Direct phoneme input
python inference.py --checkpoint ./ckpts/rfsinger \
    --phonemes "sh i zh ong g uo" \
    --durations "5 10 8 12 6 15"

# Choose vocoder and step count
python inference.py --checkpoint ./ckpts/rfsinger --index 0 \
    --vocoder pyworld --n_steps 100
```

### DiffSinger (.ds) File Inference

```bash
python infer_ds.py path/to/song.ds --checkpoint ./ckpts/rfsinger --output out.wav

# Multi-speaker
python infer_ds.py song.ds --checkpoint ./ckpts/rfsinger --speaker speaker_a

# Save intermediate mel/F0
python infer_ds.py song.ds --checkpoint ./ckpts/rfsinger --save_mel --save_f0
```

### Vocoders

| Vocoder | Quality | Notes |
|---------|---------|-------|
| `nsf_hifigan` | Best | Requires `ckpts/nsf_hifigan/{config.json, model.ckpt}` |
| `pyworld` | Medium | Traditional WORLD synthesis, good for debugging |
| `griffin_lim` | Low | No extra model needed |

## License

MIT License

## Acknowledgments

- FastSpeech2: [FastSpeech 2](https://arxiv.org/abs/2006.04558)
- Rectified Flow: [Flow Matching](https://arxiv.org/abs/2210.02747)
- NSF-HiFiGAN: [SingingVocoders](https://github.com/openvpi/SingingVocoders)
- RMVPE: [RMVPE](https://github.com/yxlllc/RMVPE)
