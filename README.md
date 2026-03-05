# RFSinger

A singing voice synthesis (SVS) system based on **Rectified Flow** and **FastSpeech2** architecture. RFSinger combines the efficient encoder-decoder structure of FastSpeech2 with the powerful generative capabilities of flow matching for high-quality mel-spectrogram generation.

## Features

- **FastSpeech2 Encoder**: Transformer-based phoneme encoder with positional encoding
- **Variance Adaptor**: Duration regulation and pitch conditioning
- **Rectified Flow Decoder**: Flow matching-based mel-spectrogram generation with ODE sampling
- **Multiple Vocoders**: Support for NSF-HiFiGAN, PyWorld, and Griffin-Lim
- **Flexible Inference**: Multiple input modes (direct phonemes, metadata file, sample ID)

## Architecture

```
Phonemes → FastSpeech2 Encoder → Variance Adaptor → Rectified Flow Decoder → Mel Spectrogram → Vocoder → Audio
                                      ↑
                                 Duration + F0
```

## Configuration

RFSinger uses a centralized `config.yaml` file to control all hyperparameters. This makes it easy to:
- Reproduce experiments with the same settings
- Share configurations between training and inference
- Track experiment parameters alongside checkpoints

### Checkpoint Organization

Checkpoints are saved in a structured folder hierarchy:
```
ckpts/
└── {experiment_name}/
    ├── config.yaml              # Copy of training config
    ├── checkpoint_epoch10.pt
    ├── checkpoint_epoch20.pt
    ├── checkpoint_best.pt       # Best model by loss
    ├── checkpoint_latest.pt     # Most recent
    └── checkpoint_ema.pt        # EMA model for inference
```

### Checkpoint Contents

Each checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state for resuming training
- `scheduler_state_dict`: Learning rate scheduler state
- `ema_state_dict`: EMA shadow weights and state
- `global_step`: Total training steps completed
- `epoch`: Current epoch number
- `config`: Full training configuration

## Data Preparation

### Directory Structure

```
data/
└── YourDataset/
    ├── wavs/
    │   ├── sample1.wav
    │   ├── sample2.wav
    │   └── ...
    └── transcriptions.csv
```

### CSV Format

The `transcriptions.csv` should have the following columns:

| Column | Description |
|--------|-------------|
| `name` | Audio file name (without extension) |
| `ph_seq` | Space-separated phoneme sequence |
| `ph_dur` | Space-separated phoneme durations (in seconds) |
| `ph_num` | Number of phonemes |
| `note_seq` | Note sequence (optional) |
| `note_dur` | Note durations (optional) |

### Preprocessing

```bash
# Using config file
python preprocess.py --config config.yaml

# Or with command-line overrides
python preprocess.py --config config.yaml --output_dir ./my_data
```

This will:
1. Extract mel-spectrograms (80-band, log-scale)
2. Extract F0 using configurable method (`harvest` or `rmvpe`)
3. Interpolate F0 for unvoiced regions
4. Build phoneme vocabulary
5. Save processed data to `./processed_data/`

**F0 extractor configuration (`config.yaml`):**
```yaml
f0:
    method: "harvest"        # "harvest" or "rmvpe"
    f0_floor: 50.0
    f0_ceil: 1100.0
    rmvpe_ckpt: "./ckpts/rmvpe.pt"
```

- `harvest`: PyWorld Harvest extractor (default)
- `rmvpe`: neural pitch extractor loaded from `utils/rmvpe`

**Output files:**
- `{id}_mel.npy` - Mel spectrogram for each sample
- `{id}_f0.npy` - F0 contour for each sample
- `phone_map.json` - Phoneme to ID mapping
- `f0_stats.npy` - F0 mean and std for normalization
- `train.txt` - Metadata file

## Training

```bash
# Basic training with config file
python train.py --config config.yaml

# With command-line overrides
python train.py --config config.yaml --exp_name my_experiment --epochs 200

# Resume training from experiment directory
python train.py --config config.yaml --resume ./ckpts/rfsinger_exp01
```

### Training Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Path to configuration YAML file (default: `./config.yaml`) |
| `--exp_name` | Override experiment name from config |
| `--epochs` | Override number of epochs |
| `--batch_size` | Override batch size |
| `--learning_rate` | Override learning rate |
| `--resume` | Path to experiment directory to resume from |

### Configuration Options

All hyperparameters can be set in `config.yaml`. Key options include:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `experiment` | `name` | `rfsinger_exp01` | Experiment name for checkpoint folder |
| `experiment` | `seed` | 42 | Random seed |
| `model` | `d_model` | 256 | Model hidden dimension |
| `model` | `n_encoder_layers` | 4 | Number of transformer encoder layers |
| `model` | `n_head` | 2 | Number of attention heads |
| `model` | `flow_hidden` | 256 | Flow decoder hidden dimension |
| `model` | `n_flow_layers` | 20 | Number of flow decoder layers |
| `f0` | `method` | `harvest` | F0 extractor: `harvest` or `rmvpe` |
| `f0` | `rmvpe_ckpt` | `./ckpts/rmvpe.pt` | RMVPE checkpoint path (used when `method=rmvpe`) |
| `training` | `epochs` | 100 | Training epochs |
| `training` | `batch_size` | 16 | Batch size |
| `training` | `learning_rate` | 1e-4 | Learning rate |
| `training` | `grad_clip` | 1.0 | Gradient clipping |
| `training` | `use_ema` | true | Enable EMA for model weights |
| `training` | `ema_decay` | 0.9999 | EMA decay rate |
| `checkpoint` | `save_interval` | 10 | Save every N epochs |
| `checkpoint` | `keep_last_n` | 5 | Keep only last N checkpoints |
| `checkpoint` | `save_best` | true | Save best model |

## Inference

### Basic Usage

```bash
# Using experiment directory (recommended) - automatically loads config
python inference.py \
    --checkpoint ./ckpts/rfsinger_exp01 \
    --index 0

# Using specific checkpoint file with config
python inference.py \
    --checkpoint ./ckpts/rfsinger_exp01/checkpoints/checkpoint_best.pt \
    --config ./ckpts/rfsinger_exp01/config.yaml \
    --index 0

# Using sample ID
python inference.py \
    --checkpoint ./ckpts/rfsinger_exp01 \
    --sample_id "1_slice_0001" \
    --vocoder nsf_hifigan

# Direct phoneme input
python inference.py \
    --checkpoint ./ckpts/rfsinger_exp01 \
    --phonemes "sh i zh ong g uo" \
    --durations "5 10 8 12 6 15" \
    --output_name "test_output"
```

### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to model checkpoint or experiment directory |
| `--config` | - | Path to config.yaml (auto-detected from experiment dir) |
| `--phone_map` | (from config) | Phone map path |
| `--f0_stats` | (from config) | F0 statistics path |
| `--index` | - | Select sample by index from meta_file |
| `--sample_id` | - | Select sample by ID |
| `--phonemes` | - | Direct phoneme input |
| `--durations` | - | Direct duration input (frames) |
| `--use_gt_f0` | - | Use ground truth F0 |
| `--n_steps` | (from config) | ODE integration steps |
| `--vocoder` | (from config) | Vocoder: `nsf_hifigan`, `pyworld`, `griffin_lim`, `none` |
| `--output_dir` | (from config) | Output directory |

### Vocoders

1. **NSF-HiFiGAN** (recommended): Neural source-filter HiFi-GAN for high-quality synthesis
   - Requires `config.json` and `model.ckpt` in `./ckpts/nsf_hifigan/`
   
2. **PyWorld**: Traditional vocoder using WORLD synthesis
   - Good for debugging, lower quality than neural vocoders
   
3. **Griffin-Lim**: Phase reconstruction algorithm
   - No additional model required, lowest quality

## Model Architecture Details

### FastSpeech2 Encoder
- Embedding layer for phonemes
- Sinusoidal positional encoding
- Multi-layer transformer encoder

### Variance Adaptor
- **Length Regulator**: Expands encoder output based on phoneme durations
- **Pitch Predictor**: Conv1D stack predicting F0 contour
- **Coarse Decoder**: Lightweight network generating initial mel estimate

### Rectified Flow Decoder
- Sinusoidal time embedding
- Dilated convolutional residual blocks
- Gated activation mechanism
- ODE-based sampling from noise to mel-spectrogram

### Loss Functions
- **Flow Loss**: MSE between predicted and target velocity fields
- **Pitch Loss**: MSE between predicted and ground truth F0

## Hyperparameters

### Audio Processing
| Parameter | Value |
|-----------|-------|
| Sample Rate | 44100 Hz |
| FFT Size | 1024 |
| Hop Length | 256 |
| Win Length | 1024 |
| Mel Bands | 80 |
| F0 Range | 50-1100 Hz |

## License

MIT License

## Acknowledgments

- FastSpeech2 architecture inspired by [FastSpeech 2](https://arxiv.org/abs/2006.04558)
- Rectified Flow based on [Flow Matching](https://arxiv.org/abs/2210.02747)
- NSF-HiFiGAN from [NSF-HiFiGAN](https://github.com/openvpi/SingingVocoders)
- RMVPE checkpoint from [RMVPE](https://github.com/yxlllc/RMVPE)