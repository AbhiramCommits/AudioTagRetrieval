"""Embedding index build and similarity search.

Note: torch and faiss each bundle their own copy of libomp on macOS; loading
both OpenMP runtimes in one process segfaults unless each is single-threaded,
so we force ``OMP_NUM_THREADS=1`` before either library initializes threads.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from audiotag.index.search import Retriever  # noqa: E402

__all__ = ["Retriever"]
