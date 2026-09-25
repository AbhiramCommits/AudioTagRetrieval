"""Dataset over precomputed log-mel features and 50-dim tag labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from audiotag.config import Config


class MTATDataset(Dataset):
    """Loads precomputed log-mel spectrograms and top-50 tag vectors.

    During training a random 10-second crop is taken; at evaluation the crop
    is centered. Clips shorter than the crop are zero-padded on the right.
    """

    def __init__(
        self,
        split: str = "train",
        config: Config | None = None,
        training: bool | None = None,
    ) -> None:
        self.cfg = config or Config()
        self.split = split
        self.training = split == "train" if training is None else training
        self.n_frames = self.cfg.n_frames
        self.df = pd.read_parquet(self.cfg.processed_dir / f"{split}.parquet")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        mp3_path = str(row["mp3_path"])
        stem = mp3_path[:-4] if mp3_path.endswith(".mp3") else mp3_path
        mel = np.load(self.cfg.mels_dir / f"{stem}.npy").astype(np.float32)  # [1, 128, T]
        mel = self._crop(mel)
        labels = np.asarray(row["labels"], dtype=np.float32)
        return torch.from_numpy(mel), torch.from_numpy(labels)

    def _crop(self, mel: np.ndarray) -> np.ndarray:
        n = self.n_frames
        length = mel.shape[-1]
        if length >= n:
            if self.training:
                start = np.random.randint(0, length - n + 1)
            else:
                start = (length - n) // 2
            return mel[..., start : start + n]
        pad = np.zeros((*mel.shape[:-1], n - length), dtype=mel.dtype)
        return np.concatenate([mel, pad], axis=-1)

    @staticmethod
    def npy_path_for(mels_dir: Path, mp3_path: str) -> Path:
        stem = mp3_path[:-4] if mp3_path.endswith(".mp3") else mp3_path
        return mels_dir / f"{stem}.npy"
