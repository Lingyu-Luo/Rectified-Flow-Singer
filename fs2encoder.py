import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""
    pe: torch.Tensor
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class SamePadConv1d(nn.Module):
    """Conv1d with explicit same padding for odd and even kernels."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        # Keep sequence length consistent for even/odd kernels.
        effective_kernel = self.dilation * (self.kernel_size - 1) + 1
        pad_total = max(effective_kernel - 1, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        if pad_total > 0:
            x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)


class ConvBlock(nn.Module):
    """1D Convolutional Block with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            SamePadConv1d(in_channels, out_channels, kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            SamePadConv1d(out_channels, out_channels, kernel_size),
            nn.BatchNorm1d(out_channels),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        residual = self.residual(x)
        out = self.conv(x)
        out = self.relu(out + residual)
        out = self.dropout(out)
        return out


class CoarseMelDecoder(nn.Module):
    """
    Enhanced Coarse Mel Decoder with CNN-Transformer-CNN architecture.
    Structure: CNN input -> Transformer layers -> CNN-Norm -> Linear output
    """
    def __init__(self, d_model=256, mel_channels=128, n_layers=2, n_head=2, 
                 conv_channels=512, kernel_size=5, dropout=0.1):
        super().__init__()
        
        # 1. CNN Input Block - project and extract local features
        self.input_conv = nn.Sequential(
            ConvBlock(d_model, conv_channels, kernel_size, dropout),
            ConvBlock(conv_channels, d_model, kernel_size, dropout),
        )
        
        # 2. Positional encoding for transformer
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        
        # 3. Transformer layers for global context modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. CNN-Norm output block - refine and smooth predictions
        self.output_conv = nn.Sequential(
            SamePadConv1d(d_model, conv_channels, kernel_size),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            SamePadConv1d(conv_channels, d_model, kernel_size),
            nn.LayerNorm([d_model]),  # Will be applied after transpose
        )
        self.output_norm = nn.LayerNorm(d_model)
        
        # 5. Linear output projection
        self.output_linear = nn.Linear(d_model, mel_channels)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, d_model) input features
            mask: (B, T) boolean mask (True = valid, False = padding)
        Returns:
            mel: (B, T, mel_channels) coarse mel prediction
        """
        # 1. CNN input: (B, T, D) -> (B, D, T) -> conv -> (B, D, T) -> (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.input_conv(x)
        x = x.transpose(1, 2)  # (B, T, D)
        
        # 2. Add positional encoding
        x = self.pos_enc(x)
        
        # 3. Transformer with optional padding mask
        if mask is not None:
            # TransformerEncoder expects src_key_padding_mask where True = ignore
            x = self.transformer(x, src_key_padding_mask=~mask)
        else:
            x = self.transformer(x)
        
        # 4. CNN-Norm output: (B, T, D) -> (B, D, T) -> conv -> (B, D, T)
        x_conv = x.transpose(1, 2)  # (B, D, T)
        x_conv = self.output_conv[0](x_conv)  # Conv1d
        x_conv = self.output_conv[1](x_conv)  # BatchNorm1d
        x_conv = self.output_conv[2](x_conv)  # GELU
        x_conv = self.output_conv[3](x_conv)  # Dropout
        x_conv = self.output_conv[4](x_conv)  # Conv1d
        x_conv = x_conv.transpose(1, 2)  # (B, T, D)
        x = self.output_norm(x_conv)  # LayerNorm
        
        # 5. Linear output projection
        mel = self.output_linear(x)
        
        return mel


class FastSpeech2Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_head=2, dropout=0.1,
                 n_speakers=1, speaker_embedding_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = SinusoidalPositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Speaker conditioning
        self.n_speakers = n_speakers
        self.use_speaker_emb = n_speakers > 1
        if self.use_speaker_emb:
            self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)
            self.speaker_proj = nn.Linear(speaker_embedding_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, src_mask, speaker_id=None):
        # x: (batch, seq_len)
        x = self.embedding(x)
        x = self.pos_emb(x)  # Add positional embedding
        x = self.dropout(x)  # Apply dropout after embedding + positional encoding
        
        # Add speaker embedding (broadcast across sequence length)
        if self.use_speaker_emb and speaker_id is not None:
            spk_emb = self.speaker_embedding(speaker_id)  # (B, spk_dim)
            spk_emb = self.speaker_proj(spk_emb)  # (B, d_model)
            x = x + spk_emb.unsqueeze(1)  # (B, T, d_model)

        # Transformer inference (padding mask needs to be inverted)
        out = self.transformer(x, src_key_padding_mask=~src_mask)
        return out



class LengthRegulator(nn.Module):
    """Expand hidden vectors according to Duration"""

    def forward(self, x, duration, max_len):
        output = []
        for i in range(x.size(0)):
            # Repeat_interleave
            expanded = torch.repeat_interleave(x[i], duration[i], dim=0)
            # Pad or Trim
            if expanded.size(0) < max_len:
                pad = torch.zeros(max_len - expanded.size(0), x.size(2)).to(x.device)
                expanded = torch.cat([expanded, pad], dim=0)
            else:
                expanded = expanded[:max_len]
            output.append(expanded)
        return torch.stack(output)


class VarianceAdaptor(nn.Module):
    def __init__(self, d_model=256, mel_channels=128, dropout=0.1,
                 coarse_n_layers=2, coarse_n_head=2, coarse_conv_channels=512, coarse_kernel_size=5):
        super().__init__()
        self.lr = LengthRegulator()
        self.dropout = nn.Dropout(dropout)
        self.mel_channels = mel_channels
        
        # Pitch embedding (Continuous F0 -> Dense Vector)
        self.pitch_emb = nn.Linear(1, d_model)
        
        # UV (Unvoiced) embedding - learn voiced/unvoiced distinction
        # UV mask: 1=unvoiced, 0=voiced
        self.uv_emb = nn.Linear(1, d_model)
        
        # Optional: UV gating mechanism - control F0 influence on unvoiced regions
        self.uv_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # Enhanced Coarse Mel Decoder (CNN-Transformer-CNN architecture)
        # Used to predict conditions for shallow diffusion
        self.coarse_decoder = CoarseMelDecoder(
            d_model=d_model,
            mel_channels=mel_channels,
            n_layers=coarse_n_layers,
            n_head=coarse_n_head,
            conv_channels=coarse_conv_channels,
            kernel_size=coarse_kernel_size,
            dropout=dropout
        )

    def forward(self, x, src_mask, mel_mask, duration, f0, uv=None):
        """
        Args:
            x: encoder output (B, T_text, d_model)
            src_mask: (B, T_text) source mask
            mel_mask: (B, T_mel) mel mask
            duration: (B, T_text) duration for each phoneme
            f0: (B, T_mel) pitch contour (required, normalized)
            uv: (B, T_mel) unvoiced mask (1=unvoiced, 0=voiced), optional
        """
        # 1. Length Regulation
        max_len = mel_mask.size(1)
        x_expanded = self.lr(x, duration, max_len)
        x_expanded = self.dropout(x_expanded)  # Apply dropout after expansion

        # 2. Pitch injection with UV mask handling
        pitch_cond = f0.unsqueeze(-1)  # (B, T_mel, 1)
        pitch_emb = self.pitch_emb(pitch_cond)  # (B, T_mel, d_model)
        
        if uv is not None:
            # UV-aware pitch conditioning
            uv_cond = uv.unsqueeze(-1)  # (B, T_mel, 1)
            uv_emb = self.uv_emb(uv_cond)  # (B, T_mel, d_model)
            
            # Gating mechanism: reduce F0 influence in unvoiced regions
            # voiced (uv=0): gate close to 1, retain full pitch information
            # unvoiced (uv=1): gate close to 0, suppress pitch information
            gate = self.uv_gate(x_expanded)  # (B, T_mel, d_model)
            voiced_mask = (1.0 - uv_cond)  # 0 for unvoiced, 1 for voiced
            
            # Combine: pitch embedding weighted by voiced degree + UV embedding provides unvoiced information
            x_adapted = x_expanded + pitch_emb * voiced_mask * gate + uv_emb
        else:
            # Fallback: use pitch embedding directly if no UV mask (backward compatibility)
            x_adapted = x_expanded + pitch_emb

        # 3. Predict coarse Mel (used for shallow diffusion initialization) with mel_mask for proper padding handling
        m_coarse = self.coarse_decoder(x_adapted, mask=mel_mask)

        return x_adapted, m_coarse