# audio-tag-retrieval

Audio tagging and tag-based audio retrieval on MagnaTagATune.

- Features: log-mel spectrograms via torchaudio (see `audiotag/features/melspec.py`)
- Data: MagnaTagATune download + top-50 tag vocabulary (`audiotag/data/`)
- Models: `ShortChunkCNN` and `AudioTransformerTagger` (`audiotag/models/`),
  swappable via a shared `embed()` / `forward()` interface
- Training: BCEWithLogitsLoss with per-tag pos_weight, AdamW + cosine schedule
  with warmup, mixed precision (torch.amp), grad clipping, early stopping on
  val mAP, TensorBoard + per-epoch metrics JSONL (`audiotag/train.py`)
- Evaluation: per-tag ROC-AUC / AP, macro AUC, macro/micro mAP, and a
  prior-probability baseline (`audiotag/eval.py`)
- Stack: PyTorch 2.x, torchaudio, FAISS, FastAPI, Python 3.11

```bash
make setup              # create venv + install deps (uv)
make data SUBSET=2000   # download subset + build tags/manifests
make train              # precompute features + train (MODEL=cnn|transformer)
make eval               # evaluate on test split -> artifacts/{model}/metrics.json
make pipeline           # data + features + train + eval
make test               # run tests
```

FAISS index build and serving endpoints are upcoming.
