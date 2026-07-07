# horosvec

Embedded approximate-nearest-neighbor (ANN) vector index in **pure Go**, backed
by a single SQLite file. One dependency: [modernc.org/sqlite](https://pkg.go.dev/modernc.org/sqlite)
(pure Go SQLite — no CGO, static binaries).

- **Vamana** proximity graph (DiskANN-style greedy search, alpha-RNG robust pruning)
- **RaBitQ** binary quantization for cheap approximate distances during traversal
- **Two-stage search**: RaBitQ beam preselection, then exact L2 rerank on the true vectors
- Transactional inserts (in-memory state applied only after a successful commit)
- `context.Context` throughout — cancellation is an error, never a silent empty result
- Hardened binary import (header validation, bounded allocations on hostile blobs)
- Centroid drift detection triggering async rebuilds

## Install

```sh
go get github.com/hazyhaar/horosvec
```

## Quick start

```go
db, _ := sql.Open("sqlite", "vectors.db")
idx, err := horosvec.New(db, horosvec.DefaultConfig())
if err != nil { /* … */ }

// Build from an iterator of (vector, id) pairs.
if err := idx.Build(ctx, iter); err != nil { /* … */ }

// Incremental inserts are transactional.
if err := idx.Insert(ctx, vecs, ids); err != nil { /* … */ }

// Two-stage ANN search.
results, err := idx.Search(ctx, query, 10)
```

## Measured recall

Recall@10 against exact brute force (deterministic benches in this repository,
N=2000 base vectors, 50 queries, default configuration):

| Dataset | dim | recall@10 mean | worst query |
|---|---|---|---|
| uniform synthetic | 128 | **1.000** | 1.000 |
| gaussian clusters (σ=0.05) | 128 | **0.982** | 0.900 |
| real bge-m3 embeddings (code-session texts) | 1024 | **1.000** | 1.000 |

The real-data bench (`recall_real_test.go`) is opt-in: export
`HOROSVEC_REAL_VECS=/path/to/vectors.json` (a JSON `[][]float64`) and run
`go test -run TestRecallMeasure_RealEmbeddings -v`. The synthetic bench
(`recall_measure_test.go`) always runs.

**Honesty note.** This implementation omits the random rotation of the RaBitQ
paper, so the paper's theoretical bounds do not transfer as-is; the two-stage
design absorbs the estimator noise in the measured regimes. The figures above
are measurements on ~2×10³ vectors, not guarantees at 10⁵+. See `doc.go`
(*Accuracy*, *Known limits*) for the full contract, including the
graceful-degradation behaviour of `SearchWithRerank`.

## Status

v0.1.0 — extracted from the [horos55](https://hazyhaar.fr) ecosystem, where it
serves RAG shard search and code-map embeddings in production. Coverage 85.9 %,
41 tests including commit-failure injection, corrupt-blob hardening, LRU
eviction, drift-triggered rebuilds and cancellation mid-search.

## License

MIT
