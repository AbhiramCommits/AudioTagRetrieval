# Retrieval quality

Mean Jaccard overlap between query ground-truth tags and neighbor
ground-truth tags, over 500 held-out test queries (cnn
model, exact flat index, corpus of 3970 train+val clips).
The random-neighbor baseline uses uniform random neighbors from the same
corpus.

| k | model Jaccard | random-neighbor Jaccard |
|---|---|---|
| 1 | 0.1527 | 0.0365 |
| 5 | 0.1499 | 0.0363 |
| 10 | 0.1492 | 0.0366 |

The model's neighbors share substantially more tags with the query than
random clips do, showing the embedding space is semantically organized
rather than arbitrary.
