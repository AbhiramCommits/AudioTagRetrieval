"""Plain helper objects shared by tests (no fixtures here; see conftest.py)."""

import io
import wave

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

TAGS = ["guitar", "classical", "slow", "techno", "piano"]


def make_wav_bytes(
    seconds: float = 0.5, sr: int = 44100, channels: int = 1, freq: float = 440.0
) -> bytes:
    """A tiny 16-bit PCM sine wave as wav bytes (no binary files committed)."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    sig = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels == 2:
        sig = np.stack([sig, 0.7 * sig], axis=1)
    else:
        sig = sig[:, None]
    pcm = (sig * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def write_manifest(path, rows: list[dict]) -> None:
    table = pa.table(
        {
            "clip_id": pa.array([r["clip_id"] for r in rows]),
            "mp3_path": pa.array([r["mp3_path"] for r in rows]),
            "labels": pa.array(
                [np.asarray(r["labels"], dtype=np.uint8).tolist() for r in rows],
                type=pa.list_(pa.uint8()),
            ),
        }
    )
    pq.write_table(table, path)


class TinyEmbedModel(nn.Module):
    """Stub tagger: embed -> [B, dim], head -> [B, n_classes] logits."""

    def __init__(self, n_classes: int = 5, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.head = nn.Linear(dim, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3))  # [B, 1]
        return mean.expand(-1, self.dim).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class FakeRetriever:
    """Stub for API tests: returns fixed neighbors."""

    def __init__(self, ntotal: int = 5, dim: int = 8) -> None:
        self.index = type("Index", (), {"ntotal": ntotal})()
        self.dim = dim

    def search(self, embedding, k: int = 5) -> list[list[dict]]:
        rows = np.asarray(embedding).reshape(-1, self.dim).shape[0]
        return [
            [
                {
                    "clip_id": f"clip{j}",
                    "score": round(1.0 - 0.1 * j, 2),
                    "top_tags": ["guitar", "slow"],
                }
                for j in range(min(k, 5))
            ]
            for _ in range(rows)
        ]
