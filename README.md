# audio-tag-retrieval

Audio tagging and tag-based audio retrieval on MagnaTagATune.

- Features: log-mel spectrograms via torchaudio (see `audiotag/features/melspec.py`)
- Data: MagnaTagATune download + top-50 tag vocabulary (`audiotag/data/`)
- Stack: PyTorch 2.x, torchaudio, FAISS, FastAPI, Python 3.11

```bash
make setup          # create venv + install deps (uv)
make data SUBSET=500  # download (optional subset) + build tags/manifests
make features       # precompute log-mel features
make test           # run tests
```

Model, FAISS index, and serving endpoints are upcoming.
