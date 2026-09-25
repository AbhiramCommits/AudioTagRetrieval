# FAISS index benchmark

Measured with the cnn model (1024-dim L2-normalized embeddings),
corpus of 3970 train+val clips, 1000 held-out test clips
as queries, k=10, faiss-cpu on an Apple Silicon laptop.

- flat: `IndexFlatIP` — exact search (reference)
- ivfpq: `IndexIVFPQ` (nlist=256, m=32, nbits=8), nprobe=16

| index | vectors | index size (MB) | mean query latency (ms) | recall@10 vs flat |
|---|---|---|---|---|
| flat | 3970 | 16.26 | 0.01 | 1.0000 (reference) |
| ivfpq | 3970 | 2.26 | 0.26 | 0.7716 |

Compression ratio: 7.2x smaller on disk
(86% reduction). id_map.parquet is
shared and takes 0.19 MB. Latency is the mean per-query time for
a batched `index.search` over all 1000 queries after warmup.
