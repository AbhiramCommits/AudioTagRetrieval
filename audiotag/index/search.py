"""Online similarity search over a persisted FAISS index (query step)."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd


class Retriever:
    """Loads a FAISS index + id_map once and serves cosine-similarity search.

    Embeddings are L2-normalized on both sides, so inner-product scores are
    cosine similarities in [-1, 1]. For IVF* indexes ``nprobe`` is set to 16.
    """

    def __init__(self, index_path: str | Path, id_map_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.id_map_path = Path(id_map_path)
        self.index = faiss.read_index(str(self.index_path))
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = 16
        self.id_map = pd.read_parquet(self.id_map_path)
        self.dim = self.index.d

    def search(self, embedding, k: int = 10) -> list[list[dict]]:
        """Search for the k nearest neighbors of one or more embeddings.

        Args:
            embedding: float array of shape [D] or [B, D].
            k: number of neighbors per query.

        Returns:
            One list of hit dicts per query, sorted by descending cosine
            score: ``{"clip_id", "mp3_path", "score", "top_tags"}``.
        """
        x = np.asarray(embedding, dtype=np.float32).reshape(-1, self.dim)
        faiss.normalize_L2(x)
        scores, indices = self.index.search(x, k)
        results = []
        for row_scores, row_indices in zip(scores, indices, strict=True):
            hits = []
            for score, idx in zip(row_scores, row_indices, strict=True):
                if idx < 0:
                    continue
                row = self.id_map.iloc[int(idx)]
                hits.append(
                    {
                        "clip_id": str(row["clip_id"]),
                        "mp3_path": str(row["mp3_path"]),
                        "score": float(score),
                        "top_tags": list(row["top_tags"]),
                    }
                )
            results.append(hits)
        return results
