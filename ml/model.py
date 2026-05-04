"""TCN for action recognition from MediaPipe pose sequences.

Architecture: 2 temporal blocks, 64 channels, kernel_size=5, residual connections,
batch normalization, global average pooling. ~93K params with 9-joint input (81-dim).

Input: (batch, seq_len, feat_dim) where feat_dim = num_joints * 3 * 3
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
