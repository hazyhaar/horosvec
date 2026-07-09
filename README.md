# horosvec

Embedded approximate-nearest-neighbor (ANN) vector index in **pure Go**, backed
by a single SQLite file. One dependency: [modernc.org/sqlite](https://pkg.go.dev/modernc.org/sqlite)
(pure Go SQLite — no CGO, static binaries).

- **Vamana** proximity graph (DiskANN-style greedy search, alpha-RNG robust pruning)
- **RaBitQ** binary quantization with **randomized Hadamard rotation** (seed
  persisted with the index) for cheap, unbiased approximate distances during traversal
- **Two-stage search**: RaBitQ beam preselection, then exact L2 rerank on the true vectors
- **fp16 arena** (`HVARENA1` sidecar file, mmap): vector-less SQLite at scale —
  the graph and codes live in the DB, the raw vectors live in a flat fp16 file
  read on demand; tested to **26.7M vectors** (see below)
- **Streaming build** from an arena (memory-bounded: no O(N×dim×4) buffer) and
  **parallel graph construction** (sharded neighborhood locks)
- **External-adjacency import** (`ImportAdjacency`): bring a graph built
  elsewhere (e.g. GPU CAGRA), get a complete horosvec index — encoding,
  medoid, hot plane and persistence reuse the standard path
- Transactional inserts (in-memory state applied only after a successful commit)
- `context.Context` throughout — cancellation is an error, never a silent empty result
- Hardened binary import (header validation, bounded allocations on hostile blobs)
- Centroid drift detection; rebuild refuses arena-backed indexes fail-loud
  (rebuild-in-place is the small-index path only)

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

// Incremental inserts are transactional (DB-blob mode).
if err := idx.Insert(ctx, vecs, ids); err != nil { /* … */ }

// Two-stage ANN search.
results, err := idx.Search(ctx, query, 10)
```

At scale, set `Config.ArenaPath`: `Build` then streams vectors to an fp16 arena
and keeps SQLite vector-less; `Search` reranks from the mmap'd arena. See
`docs/ARCHITECTURE.md` for the full data layout and the import pipeline.

## Measured quality and latency

Recall floors are asserted per distribution in CI (`recall_measure_test.go`,
deterministic seed, `BuildWorkers=1`): **≥ 0.90 uniform**, **≥ 0.60 tight
gaussian clusters** (measured 0.930 / 0.678 — a 1-bit-quantization ceiling on
σ=0.05 clusters, documented rather than hidden).

Production-scale validation (26,691,317 real HackerNews embeddings,
qwen3-embedding-0.6B, 512-dim MRL, fp16 arena; 20 real queries vs exact brute
force over the full arena):

| Metric | Value |
|---|---|
| overlap@10 vs exact | **0.99** (18/20 queries at 1.0) |
| rerank SQL fallbacks | 0 (arena-served) |
| Search p50 / p99 (NVMe, warm cache) | **7.8 ms / 9.2 ms** |
| Search p50 / p99 (NVMe, cold-ish) | 27.6 ms / 28.8 ms |
| Index build (GPU CAGRA adjacency + import) | 17.1 min + 21.7 min |

Latency is medium-dominated: the same index served from a rotational disk
measures p50 ≈ 2.9 s. Keep arenas on SSD/NVMe.

**Why no ann-benchmarks entry.** The reference leaderboard measures a
single-client, in-RAM, mostly low-dimension protocol — orthogonal to what this
engine is built for (concurrent serving, SQLite persistence, off-heap fp16
vectors, high-dim real embeddings). On their axis, hnswlib wins ~×2
single-client and ~×5 at iso-recall on 128-dim data — measured and published
here rather than hidden. On the axes they cannot see, the numbers above apply.
Rationale in full: `docs/ARCHITECTURE.md` §9.

**Conformity note.** The asymmetric RaBitQ estimator was audited line-by-line
against the paper (2026-07-08): the implementation is the canonical unbiased
estimator — the √d and ‖q‖ factors cancel algebraically and the L1 division is
the paper's ⟨ō,o⟩ correction. The symmetric `rabitqDistance` is a benchmark
helper only, not the production estimator. Full derivation in `rabitq.go`.

## Status

Extracted from the [horos55](https://hazyhaar.fr) ecosystem, where it serves
RAG shard search and code-map embeddings in production. Suite includes
commit-failure injection, corrupt-blob hardening, LRU eviction, arena
round-trip parity, import fail-loud (truncated/out-of-range/over-degree) and
cancellation mid-search.

## License

MIT
