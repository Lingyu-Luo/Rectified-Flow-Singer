# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RFSinger is a singing voice synthesis system: phoneme + duration + F0 → mel-spectrogram → audio. The mel generator combines a FastSpeech2-style encoder/variance adaptor (deterministic) with a Rectified Flow decoder that refines a coarse mel via ODE integration. The README's notice is current: the iterative-reflow stage has known issues — prefer the base flow training when verifying changes.

## Common commands

All scripts read `config.yaml` by default; CLI flags only override a small set of fields. Edit `config.yaml` for anything else.

```bash
# 1. Preprocess (extract mel/F0, build phone_map.json, f0_stats.npy, train.txt)
python preprocess.py --config config.yaml

# 2. Train
python train.py --config config.yaml
python train.py --config config.yaml --exp_name myexp --epochs 200 --batch_size 16
python train.py --config config.yaml --resume ./ckpts/<exp_name>   # auto-loads exp's saved config

# 3. Reflow (two-step: generate targets, then train on them)
python train.py --config config.yaml --reflow                       # writes processed_data/reflow/*_mel.npy
python train.py --config config.yaml --reflow_train                 # appends '-reflow' to exp name

# 4. Inference
python inference.py --checkpoint ./ckpts/<exp_name> --index 0
python inference.py --checkpoint ./ckpts/<exp_name> --sample_id <id> --vocoder nsf_hifigan
python inference.py --checkpoint ./ckpts/<exp_name> --phonemes "sh i" --durations "5 10"

# 5. .ds (DiffSinger project) inference
python infer_ds.py path/to/song.ds --checkpoint ./ckpts/<exp_name> --output out.wav
```

Dependencies: `pip install -r requirements.txt`. There is no test suite, linter, or formatter configured — verification is by running the pipeline end-to-end on a small sample.

## Data layout the code assumes

```
data/<speaker>/wavs/*.wav
data/<speaker>/transcriptions.csv     # columns: name, ph_seq, ph_dur, ph_num, [note_seq, note_dur]
processed_data/                        # output of preprocess.py
  <speaker>/<id>_mel.npy
  <speaker>/<id>_f0.npy
  <speaker>/<id>_uv.npy                # unvoiced mask (1=unvoiced)
  phone_map.json                       # phoneme str -> int id (shared across speakers)
  speaker_map.json                     # only present in multi-speaker mode
  f0_stats.npy                         # [mean, std] used for z-score normalization
  train.txt                            # 'fid|ph_seq|ph_dur|speaker_id' per line
  reflow/<speaker>/<id>_mel.npy        # only after `train.py --reflow`
ckpts/<exp_name>/                      # one dir per experiment
  config.yaml                          # snapshot copied at training start
  checkpoint_{epochN,latest,best}.pt
ckpts/nsf_hifigan/{config.json,model.ckpt}   # required for nsf_hifigan vocoder
ckpts/rmvpe.pt                         # only if f0.method == "rmvpe"
```

`SingingDataset` in `dataset.py` z-score-normalizes F0 using `f0_stats.npy` before returning it; the model trains on normalized F0 and inference must apply the same normalization (see `infer_ds.py:synthesize_from_ds_segment` and `inference.py`). UV padding defaults to **1** (unvoiced) so padded frames don't contribute pitch to the encoder.

## Architecture (cross-file, not obvious from any single file)

```
text → FS2Encoder → VarianceAdaptor(LR + pitch+UV gate + CoarseMelDecoder)
                              │                     │
                         adapted (B,T_mel,D)   coarse_mel (B,T_mel,n_mels)
                              │                     │
                              ▼                     │
                     RectifiedFlowDecoder ──────────┘
                              │
                          mel (B,T_mel,n_mels)  →  vocoder  →  audio
```

- `fs2encoder.py` — `FastSpeech2Encoder`, `LengthRegulator`, `VarianceAdaptor`, `CoarseMelDecoder` (CNN-Transformer-CNN). The variance adaptor's `uv_gate` reduces F0 influence in unvoiced regions instead of zeroing it.
- `reflow.py` — `RectifiedFlowDecoder`: dilated 1D-conv residual blocks with sinusoidal time embedding. Conditions on `adapted` from the variance adaptor.
- `train.py` — `RFSingerModel` glues the three modules together. Flow matching uses `x_t = (1-t)*coarse_mel + t*mel_target` (so `coarse_mel` plays the role of `x_0` instead of pure Gaussian noise — this is intentional and is why `coarse_loss` is part of the objective). Inference samples by Euler integration from `coarse_mel` for `n_steps`.
- `dataset.py` — owns padding/masking conventions (`src_mask`/`mel_mask` are `True=valid`; transformer modules invert them).
- `preprocess.py` — extracts log-mel via librosa, F0 via Harvest or RMVPE; aligns CSV durations to the mel frame grid in `align_durations_to_length`, applies optional offline augmentation, builds `phone_map.json` and per-utterance `_mel/_f0/_uv.npy`.
- `inference.py` — model loading (`load_model` auto-detects `n_speakers` and `mel_channels` from checkpoint), three vocoder backends (`nsf_hifigan`, `pyworld`, `griffin_lim`), and the `synthesize` entrypoint.
- `infer_ds.py` — DiffSinger `.ds` project parser; converts segment-level note/phoneme/timing data into the model's input format and concatenates per-segment audio with offsets.
- `utils/config.py` — single source of truth for config I/O. `setup_experiment_dir` copies `config.yaml` into the experiment dir; `load_config_from_checkpoint` reads it back so inference doesn't need explicit `--config`.

## Conventions to preserve when editing

- Configuration is centralized in `config.yaml`. New hyperparameters go there and flow through `utils/config.py:get_*_args_from_config`. Don't hard-code values in `train.py`/`inference.py`.
- `audio.n_mels` and `model.mel_channels` must match each other AND the vocoder's `num_mels`. `infer_ds.py` falls back to pyworld on mismatch.
- Multi-speaker is opt-in: when `data.speakers` is set in config, `preprocess.py` writes `speaker_map.json` and `train.txt` gains a 4th `speaker_id` field. `RFSingerModel` enables the speaker embedding only when `n_speakers > 1` AND `model.use_speaker_embedding` is true.
- Reflow training reuses the main training loop with `freeze_encoder=True` and a different `reflow_dir`; the experiment name gets `-reflow` appended so checkpoints don't collide.
- Checkpoint format includes `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `ema_state_dict`, `global_step`, `epoch`, `best_loss`, `scaler_state_dict`, `config`. Adding fields is fine; renaming requires backward-compat handling in `train.py`'s resume block.
- EMA (`train.py:EMA`) keeps a shadow copy with warmup decay; for inference prefer `checkpoint_ema.pt` when present.
