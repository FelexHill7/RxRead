"""
model.py — CRNN + Attention architecture for handwriting recognition.

The network has four stages:
  1. CNN backbone      — extracts visual feature maps from a grayscale word image.
  2. BiLSTM layers     — reads the feature sequence bidirectionally to capture
                         context in both directions.
  3. Attention layer   — learns to focus on the most relevant timesteps,
                         improving character-level predictions.
  4. Linear head       — projects each timestep to a distribution over the
                         character vocabulary (+ CTC blank).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Single-head additive attention over LSTM output timesteps.

    For each timestep, computes an attention-weighted context vector
    over all timesteps, then concatenates it with the original hidden
    state. This helps the model focus on relevant spatial positions.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, rnn_out):
        # rnn_out: (B, T, H)
        energy = torch.tanh(self.attn(rnn_out))     # (B, T, H)
        scores = self.v(energy).squeeze(-1)          # (B, T)
        weights = F.softmax(scores, dim=-1)          # (B, T)

        # Context: weighted sum over all timesteps
        context = torch.bmm(weights.unsqueeze(1), rnn_out)  # (B, 1, H)
        context = context.expand_as(rnn_out)                 # (B, T, H)

        # Concatenate context with original, project back to H
        return rnn_out + context  # residual connection


class CRNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super(CRNN, self).__init__()

        # Stage 1 — CNN feature extractor
        # Input: (B, 1, 32, 128) grayscale image
        # Output: (B, 512, 1, W') where W' depends on pooling strides
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2), nn.ReLU()          # collapse remaining height
        )

        # Stage 2 — Bidirectional LSTM (256 hidden per direction → 512 total)
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True, dropout=dropout, num_layers=2)
        self.rnn_dropout = nn.Dropout(dropout)

        # Stage 3 — Attention over LSTM timesteps
        self.attention = Attention(512)

        # Stage 4 — fully-connected classifier for each timestep
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)                      # (B, 512, 1, W')
        x = x.squeeze(2)                     # (B, 512, W')  — remove height dim
        x = x.permute(0, 2, 1)              # (B, W', 512)  — sequence-first for LSTM
        x, _ = self.rnn(x)                  # (B, W', 512)  — bidirectional output
        x = self.rnn_dropout(x)
        x = self.attention(x)                # (B, W', 512)  — attention-refined
        x = self.fc(x)                       # (B, W', num_classes)
        x = x.permute(1, 0, 2)             # (W', B, num_classes) — CTC expects T-first
        return x