"""AudioTransformerTagger: a patch-based transformer for audio tagging."""

import torch
import torch.nn.functional as F
from torch import nn


class AudioTransformerTagger(nn.Module):
    """Patchify the mel spectrogram, project to tokens, CLS-token pool.

    ``embed`` returns the pooled pre-head CLS vector (d_model dims);
    ``forward`` returns tag logits. Swappable with ``ShortChunkCNN``.
    """

    def __init__(
        self,
        n_classes: int = 50,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        patch_size: int = 16,
        n_mels: int = 128,
        max_frames: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        grid_h = n_mels // patch_size
        grid_w = max_frames // patch_size
        self.max_patches = grid_h * grid_w

        self.proj = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_patches + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W] -> [B, n_patches, d_model]
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)  # [B, d_model, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, d_model]
        if x.shape[1] > self.max_patches:
            raise ValueError(f"input produces {x.shape[1]} patches, max is {self.max_patches}")
        return x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self._patchify(x)
        n = x.shape[1]
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : n + 1]
        x = self.encoder(x)
        return self.norm(x[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))
