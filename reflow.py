import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        self.dilated_conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.cond_conv = nn.Conv1d(channels, channels, 1)  # Condition projection
        self.time_conv = nn.Linear(channels, channels)  # Time embedding projection

        self.output_conv = nn.Conv1d(channels, channels, 1)
        self.gate_conv = nn.Conv1d(channels, channels, 1)  # Gating mechanism
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c, t):
        # x: (Input tensor)
        # c: (Condition from Encoder)
        # t: (Time embedding)

        res = x
        x = self.dilated_conv(x) + self.cond_conv(c) + self.time_conv(t).unsqueeze(-1)

        gate = torch.sigmoid(self.gate_conv(x))
        filter = torch.tanh(self.output_conv(x))
        x = gate * filter
        x = self.dropout(x)  # Apply dropout before residual connection

        return x + res


class RectifiedFlowDecoder(nn.Module):
    def __init__(self, in_channels=80, hidden_channels=256, n_layers=20, dropout=0.1, cond_channels=256):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, hidden_channels, 1)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.cond_adapt = nn.Conv1d(cond_channels, hidden_channels, 1)  # FS2 hidden to Flow hidden
        self.dropout = nn.Dropout(dropout)

        # Initialize residual blocks with increasing dilation
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, dilation=2 ** (i % 4), dropout=dropout)
            for i in range(n_layers)
        ])

        self.out_conv = nn.Conv1d(hidden_channels, in_channels, 1)

    def forward(self, x, t, cond):
        # x: ->
        x = x.transpose(1, 2)
        cond = cond.transpose(1, 2)

        x = self.in_conv(x)
        x = self.dropout(x)  # Apply dropout after input conv
        c = self.cond_adapt(cond)
        t_emb = self.time_emb(t)

        for block in self.blocks:
            x = block(x, c, t_emb)

        x = self.out_conv(x)
        return x.transpose(1, 2)  #