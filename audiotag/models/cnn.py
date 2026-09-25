"""ShortChunkCNN: a compact 7-block CNN for audio tagging (Musicnn-style)."""

import torch
from torch import nn


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )


class ShortChunkCNN(nn.Module):
    """Seven conv blocks (Conv2d + BN + ReLU + MaxPool) with global pooling.

    ``embed`` returns the 1024-dim pre-head vector (global max + avg pooled
    features concatenated); ``forward`` returns tag logits.
    """

    CHANNELS = (64, 128, 128, 256, 256, 512, 512)

    def __init__(self, n_classes: int = 50, dropout: float = 0.5) -> None:
        super().__init__()
        blocks = [_ConvBlock(1, self.CHANNELS[0])]
        for cin, cout in zip(self.CHANNELS[:-1], self.CHANNELS[1:], strict=True):
            blocks.append(_ConvBlock(cin, cout))
        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.CHANNELS[-1] * 2, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x_max = x.amax(dim=(2, 3))
        x_avg = x.mean(dim=(2, 3))
        return torch.cat([x_max, x_avg], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.embed(x)))
