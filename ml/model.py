"""TCN for 3-class action recognition from MediaPipe pose sequences.

Small architecture tuned for ~450 clips of HMDB51 data. Uses:
  - 2 temporal blocks (not 3) to reduce overfitting
  - kernel_size=5 for wider receptive field per layer
  - Global average pooling over time (more stable than last-step)
  - Moderate dropout (0.3)

Input: (batch, seq_len, 297) — position + velocity + acceleration
"""

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.padding = padding

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]
        out = self.relu(self.bn1(out))
        out = self.drop(out)
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.relu(self.bn2(out))
        out = self.drop(out)
        if self.downsample is not None:
            res = self.downsample(res)
        return self.relu(out + res)


class ActionTCN(nn.Module):
    def __init__(self, input_dim=297, num_classes=3, channels=(64, 64),
                 kernel_size=5, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.pool_drop = nn.Dropout(dropout)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x.mean(dim=2)
        x = self.pool_drop(x)
        return self.head(x)


ActionLSTM = ActionTCN
