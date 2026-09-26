# audio-tag-retrieval

Multi-label audio tagging and tag-based audio retrieval on
[MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset/).
The repo trains a CNN or patch-transformer tagger on log-mel spectrograms,
derives 1024-dim clip embeddings from the same model, and serves tag
prediction plus cosine-similarity search over a FAISS index through a
FastAPI service. Everything runs offline-build / online-query: features are
precomputed once, the model is trained once, the index is built once, and
the API just loads checkpoints and indexes at startup.

## Architecture

```mermaid
flowchart LR
    Audio[Audio upload] --> Mel[log-mel 128 x T]
    Mel --> Model[Tagger: CNN or transformer]
    Model --> Logits[Tag logits]
    Model --> Emb[1024-d embedding]
    Logits --> TopK[Top-k tags + sigmoid probs]
    Emb --> FAISS[FAISS flat / IVFPQ]
    FAISS --> Neighbors[Nearest tracks + cosine scores]
    TopK --> API[/analyze response]
    Neighbors --> API
```

## Quickstart

```bash
git clone <repo> && cd audio-tag-retrieval
make setup              # uv venv + deps (Python 3.11)
make data SUBSET=2000   # download MagnaTagATune subset + build tags/manifests
make train              # precompute features + train the CNN (MODEL=transformer for the other one)
make eval               # test-split metrics -> artifacts/cnn/metrics.json
make index              # build flat + IVFPQ FAISS indexes
make serve              # docker compose up the API
make test               # pytest with >=75% coverage
```

`make train`/`make eval`/`make index`/`make serve` default to the CNN model
(`MODEL=cnn`); change via `make train MODEL=transformer`.

## Results

Model results are real measurements from `artifacts/{cnn,transformer}/metrics.json`,
trained on a 2000-clip subset (1446 train / 141 val / 413 test) and evaluated
on 413 held-out test clips:

| model | macro AUC | macro mAP | micro mAP | baseline macro mAP (prior-only) | lift |
|---|---|---|---|---|---|
| CNN (ShortChunkCNN) | 0.848 | 0.268 | 0.222 | 0.052 | +0.217 |
| Transformer (patch-based) | 0.764 | 0.203 | 0.189 | 0.052 | +0.151 |

FAISS index benchmark (`reports/index_benchmark.md`), corpus of 3970 train+val
embeddings, 1000 held-out queries, k=10:

| index | size | mean query latency | recall@10 vs flat |
|---|---|---|---|
| flat (`IndexFlatIP`) | 16.26 MB | 0.01 ms | 1.0000 (reference) |
| ivfpq (`IndexIVFPQ`, nlist=256, m=32, nbits=8, nprobe=16) | 2.26 MB | 0.26 ms | 0.7716 |

Serving benchmark (`reports/serving_benchmark.md`), containerized API on a
shared Docker VM (8 vCPU, 8 GB), `POST /analyze`:

| concurrency | QPS | error rate | p50 | p95 |
|---|---|---|---|---|
| 1 | 5.92 | 0% | 146 ms | 319 ms |
| 8 | 9.48 | 0% | 758 ms | 1547 ms |
| 32 | 4.55 | 0% | 5921 ms | 12531 ms |

## API example

```bash
curl -X POST -F "file=@data/raw/9/the_wretch-ambulatory-02-hope_in_living_stereo-30-59.mp3;type=audio/mpeg" \
     http://localhost:8070/analyze
```

Real response (truncated):

```json
{
  "tags": [
    {"tag": "beat",  "probability": 0.8902},
    {"tag": "drum",  "probability": 0.8804},
    {"tag": "rock",  "probability": 0.8718}
  ],
  "neighbors": [
    {"clip_id": "9/the_wretch-ambulatory-02-hope_in_living_stereo-30-59",
     "score": 1.0, "tags": ["vocal", "male"]},
    {"clip_id": "2/mrdc-timecode-01-lust-30-59",
     "score": 0.975, "tags": []},
    {"clip_id": "6/dj_markitos-slower_emotions138_bpm_remixes-08-love_peace_and_ecstasy_138_bpm_remix-117-146",
     "score": 0.972, "tags": []}
  ]
}
```

Other endpoints: `POST /tag` (tags only), `POST /similar` (neighbors only),
`GET /healthz`, `GET /metrics` (Prometheus). Uploads are capped at 20 MB and
unsupported content types return 415.

## Dataset

MagnaTagATune: 25,863 29-second mp3 clips with 188 binary tags, released by
the City University London Music Informatics Research Group for
**non-commercial research and education**; the Magnatune audio is distributed
under Creative Commons-style licenses (per-track attribution applies). See the
[MagnaTagATune page](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset/)
for the full terms. The downloader uses a HuggingFace mirror
(`confit/magnatagatune`) with the City mirror as fallback; `make data`
fetches only what you ask for (e.g. `SUBSET=2000` keeps a stratified sample
of 2000 clips across the train/val/test directories).

## Limitations

- **Tag imbalance.** The top-50 vocabulary is heavily skewed (e.g. *guitar*
  has thousands of positives, *metal* hundreds). Training uses per-tag
  `pos_weight` and evaluation skips tags with zero test positives, but rare
  tags are still poorly calibrated.
- **10-second clip assumption.** The tagger expects 10 s of audio
  (313 mel frames); the API center-crops longer uploads and zero-pads shorter
  ones, which can hurt very short clips. Corpus embeddings share the same
  crop/pad convention.
- **Training scale.** All committed numbers come from a 2000-clip subset, not
  the full 25k dataset; full-data training should improve mAP substantially.
- **CPU serving.** On CPU the forward pass dominates request latency and high
  concurrency is queue-bound; the API bounds in-flight inference
  (`AUDIOTAG_MAX_CONCURRENT`) rather than thrashing.
- **IVFPQ scores.** With an inner-product coarse quantizer, FAISS returns
  internal distances (ascending) for `IndexIVFPQ`, not cosine similarities;
  the ranking is correct but scores are only comparable for the flat index.

## Development

`make test` runs the full pytest suite (synthetic data only, >=75% coverage)
and CI (`make` + GitHub Actions) runs lint, format, tests, and a docker build
without any dataset present; the few tests that need real data are marked
`requires_data`.
