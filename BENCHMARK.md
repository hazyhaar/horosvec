# horosvec — Performance Report

**Pure Go ANN vector search engine: Vamana (DiskANN) + RaBitQ 1-bit quantization + SQLite**

`CGO_ENABLED=0` — zero C dependencies, single binary. Only external dependency: `modernc.org/sqlite`.

**Platform**: Intel Xeon Platinum 8581C @ 2.10GHz, 16 cores, Linux amd64, Go 1.24.7

---

## Architecture Overview

```
                        horosvec search pipeline
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   query ──→ dim check                                       │
    │              │                                              │
    │              ├─ count ≤ 50K ──→ bruteForceFlat()            │
    │              │     flat []float32 scan → L2 exact           │
    │              │     100% recall, O(N), 1 alloc               │
    │              │                                              │
    │              └─ count > 50K ──→ vamanaSearch() (2-stage)    │
    │                    │                                        │
    │                    ├─ Stage 1: rabitqGreedySearch()          │
    │                    │    pooled searchState (sync.Pool)       │
    │                    │    bitset visited (not map)             │
    │                    │    typed min-heap (no interface boxing) │
    │                    │    RaBitQ 1-bit distances (~166ns/pair) │
    │                    │    graph traversal via Vamana edges     │
    │                    │                                        │
    │                    └─ Stage 2: L2 rerank                    │
    │                         top-500 candidates × exact L2       │
    │                         ext_id from LRU cache (no SQL)      │
    │                         12 allocs total                     │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

### Key implementation details

**RaBitQ 1-bit quantization** — Each float32 vector is compressed to 1 bit per dimension: `sign(vec[i] - centroid[i])`. A 128-dimensional vector (512 bytes) becomes a 16-byte binary code. Distance estimation uses a corrected dot product: `dist² ≈ ||q'||² + ||o'||² - 2·||o'||²·<sign(o'), q'>/L1(o')`. The L1 norm correction factor compensates for the quantization loss.

**Vamana graph** — A flat navigable graph (no hierarchy like HNSW). Each node has up to R=64 neighbors, selected via robust pruning with the α-RNG rule (α=1.2 promotes long-range edges). Graph is built in 2 passes with Fisher-Yates random initialization.

**Pooled search state** — A `sync.Pool` of `searchState` objects eliminates per-query allocations:
- **Bitset** (`[]uint64`): O(1) visited checks, 1.2KB for 10K nodes (vs. map rehashing)
- **Typed min-heap**: Hand-rolled sift-up/sift-down on `[]searchCandidate`, eliminating `interface{}` boxing from `container/heap` (2 allocs per Push/Pop → 0)
- **Pre-allocated best list**: Binary-search insertion into pre-sized slice

**Flat vector storage** — All vectors stored contiguously in `[]float32` for brute-force. Cache-line friendly sequential scan. Zero deserialization overhead (no SQLite BLOB decode).

**Read-only cache access** — `getReadOnly()` skips LRU promotion during search (read lock only, no write lock contention). Safe because cache is pre-warmed and eviction only happens during writes.

---

## 1. Search Latency — Scaling by Dataset Size

`go test -bench='BenchmarkReport_Search' -benchmem`

```
BenchmarkReport_Search_1K_d128     10000     122 µs/op     352 B/op    1 allocs/op
BenchmarkReport_Search_5K_d128      1668     658 µs/op     352 B/op    1 allocs/op
BenchmarkReport_Search_10K_d128      902    1326 µs/op     352 B/op    1 allocs/op
```

The brute-force path is used below 50K vectors. It scans flat in-memory vectors — no SQLite I/O, no deserialization. Memory is constant at 352 bytes (the `[]Result` slice for top-10 results).

**Scaling**: Linear in N as expected for brute-force. 1K → 10K = 10× data, 10.8× latency.

```
                         Brute-force search latency (dim=128)
     Latency
     1400 µs ┤                                                     ╭
     1200 µs ┤                                              ╭──────╯
     1000 µs ┤                                       ╭──────╯
      800 µs ┤                                ╭──────╯
      600 µs ┤                        ╭───────╯
      400 µs ┤                ╭───────╯
      200 µs ┤        ╭──────╯
      100 µs ┤────────╯
              └───┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬
                  1K    2K    3K    4K    5K    6K    7K    8K   10K
```

## 2. Search Latency — Scaling by Dimension

`go test -bench='BenchmarkReport_Search_5K_d' -benchmem`

```
BenchmarkReport_Search_5K_d64       4003     306 µs/op     352 B/op    1 allocs/op
BenchmarkReport_Search_5K_d128      1668     658 µs/op     352 B/op    1 allocs/op
BenchmarkReport_Search_5K_d256       867    1365 µs/op     352 B/op    1 allocs/op
BenchmarkReport_Search_5K_d512       412    2808 µs/op     352 B/op    1 allocs/op
BenchmarkReport_Search_5K_d1024      100   10757 µs/op     352 B/op    1 allocs/op
```

Linear in dimension for brute-force (L2 distance is O(dim)). Memory stays constant — the flat vector array is pre-allocated at build time.

## 3. Vamana+RaBitQ Path (forced, BruteForceThreshold=0)

`go test -bench='BenchmarkReport_Vamana' -benchmem`

```
BenchmarkReport_Vamana_1K_d128      1003    1173 µs/op    8639 B/op   12 allocs/op
BenchmarkReport_Vamana_5K_d128       298    4071 µs/op    8902 B/op   12 allocs/op
BenchmarkReport_Vamana_10K_d128      181    6452 µs/op    9421 B/op   12 allocs/op
```

The Vamana path uses RaBitQ for graph traversal (Stage 1) then L2 rerank (Stage 2). At 10K, it's slower than brute-force (6.5ms vs 1.3ms) — this is expected and why the dynamic threshold exists. The crossover point is ~50K vectors where brute-force O(N) exceeds Vamana's O(log N) traversal.

12 allocations = 1 result copy + 1 searchState initial resize (subsequent queries = 0 allocs from pool) + 10 ext_id copies for results.

**Memory**: ~9KB regardless of dataset size — the pooled searchState is reused.

## 4. RaBitQ Primitives

`go test -bench='BenchmarkReport_RaBitQ' -benchmem`

### Encoding (float32 vector → 1-bit code)

```
BenchmarkReport_RaBitQ_Encode_d64       9330345     129 ns/op      8 B/op    1 allocs/op
BenchmarkReport_RaBitQ_Encode_d128      4773116     250 ns/op     16 B/op    1 allocs/op
BenchmarkReport_RaBitQ_Encode_d512      1222698     977 ns/op     64 B/op    1 allocs/op
BenchmarkReport_RaBitQ_Encode_d1024      653524    1951 ns/op    128 B/op    1 allocs/op
```

Single allocation = the output code byte slice. ~1.9ns per dimension.

### Asymmetric distance (query → code, zero alloc)

```
BenchmarkReport_RaBitQ_AsymDist_d64    10561677     114 ns/op      0 B/op    0 allocs/op
BenchmarkReport_RaBitQ_AsymDist_d128    5373136     225 ns/op      0 B/op    0 allocs/op
BenchmarkReport_RaBitQ_AsymDist_d512    1360734     876 ns/op      0 B/op    0 allocs/op
BenchmarkReport_RaBitQ_AsymDist_d1024    702417    1743 ns/op      0 B/op    0 allocs/op
```

Zero allocations. ~1.7ns per dimension. This is the hot-path function used during graph traversal.

### Pre-computed asymmetric distance (batch query optimization)

```
BenchmarkReport_RaBitQ_Precomp_d128     7415584     166 ns/op      0 B/op    0 allocs/op
BenchmarkReport_RaBitQ_Precomp_d1024     915882    1336 ns/op      0 B/op    0 allocs/op
```

26% faster than non-precomputed at d128 (166ns vs 225ns) — the query centering is done once and reused for every node in the graph traversal.

## 5. L2 Exact Distance (baseline)

`go test -bench='BenchmarkReport_L2Exact' -benchmem`

```
BenchmarkReport_L2Exact_d64     22588704      54 ns/op    0 B/op    0 allocs/op
BenchmarkReport_L2Exact_d128    11095918     108 ns/op    0 B/op    0 allocs/op
BenchmarkReport_L2Exact_d512     2755099     436 ns/op    0 B/op    0 allocs/op
BenchmarkReport_L2Exact_d1024    1366338     878 ns/op    0 B/op    0 allocs/op
```

8× unrolled loop, ~0.86ns per dimension. Used in Stage 2 (rerank) and brute-force.

### Distance function comparison at d128

```
L2 Exact              108 ns/op    (full precision, used in rerank)
RaBitQ Precomp        166 ns/op    (1-bit, used in graph traversal)
RaBitQ Asym           225 ns/op    (1-bit, cold query)
```

RaBitQ precomputed is only 1.5× slower than exact L2 while using 32× less memory per vector (16 bytes vs 512 bytes for d128). The graph traversal visits ~500-1000 nodes, so RaBitQ saves significant memory bandwidth.

## 6. POPCOUNT — Symmetric RaBitQ Distance

`go test -bench='BenchmarkReport_POPCOUNT' -benchmem`

```
BenchmarkReport_POPCOUNT_8B      202197297     5.9 ns/op    0 B/op    0 allocs/op
BenchmarkReport_POPCOUNT_16B     122373369     9.9 ns/op    0 B/op    0 allocs/op
BenchmarkReport_POPCOUNT_64B      41690125    28.9 ns/op    0 B/op    0 allocs/op
BenchmarkReport_POPCOUNT_128B     21879190    55.9 ns/op    0 B/op    0 allocs/op
```

Uses `math/bits.OnesCount64` (compiled to hardware POPCNT on amd64). 64-byte codes (dim=512) in 29ns = ~0.45ns per byte.

## 7. Concurrent Search Throughput

`go test -bench='BenchmarkReport_ConcurrentSearch' -benchmem`

```
BenchmarkReport_ConcurrentSearch_10K-16    13898    89.8 µs/op    377 B/op    1 allocs/op
```

With 16 goroutines searching simultaneously on a 10K index: **89.8µs per query**, which translates to **~11,100 queries/second per core** or **~178,000 queries/second** aggregate on 16 cores.

This is 14.8× faster than single-threaded (1.3ms → 89.8µs) thanks to:
- `sync.RWMutex` allowing parallel readers
- `getReadOnly()` cache access (no write lock contention)
- `sync.Pool` per-goroutine search state (no sharing, no contention)
- Flat vectors in shared memory (read-only after build)

## 8. Search State Pool Efficiency

```
BenchmarkReport_SearchStatePool-16      56042978    21.5 ns/op    0 B/op    0 allocs/op
BenchmarkReport_SearchStateBitset-16      564516    2177 ns/op    0 B/op    0 allocs/op
```

Pool acquire+release: **21.5ns** (negligible vs search latency). Bitset reset + 1000 visits: **2.2µs**, zero allocations. The bitset replaces `map[int64]bool` which would allocate ~50 times for 1000 entries.

## 9. Recall@10

`go test -run=TestReport_RecallAtScale`

```
┌──────────┬─────┬────────────┬────────────┬─────────────┐
│  Vectors │ Dim │ Recall@10  │  Path      │ Queries     │
├──────────┼─────┼────────────┼────────────┼─────────────┤
│     100  │  64 │   100.0%   │ brute-force │      20     │
│     500  │ 128 │   100.0%   │ brute-force │      50     │
│    1000  │ 128 │   100.0%   │ brute-force │      50     │
│    5000  │ 128 │   100.0%   │ brute-force │      50     │
│   10000  │ 128 │   100.0%   │ brute-force │      50     │
└──────────┴─────┴────────────┴────────────┴─────────────┘
```

100% recall at all tested scales. The brute-force path scans all vectors with exact L2 distance — no approximation. The Vamana+RaBitQ path achieves 98-100% recall on separate audit tests with `BruteForceThreshold=0`.

## 8. RaBitQ Approximation Quality

`go test -run=TestReport_RaBitQCorrelation`

```
┌──────┬──────────────┬──────────────┬─────────────┐
│ Dim  │ Spearman ρ   │ Recall@10/50 │  Code Size  │
├──────┼──────────────┼──────────────┼─────────────┤
│   64 │    0.8461     │     70.0%     │      8 B    │
│  128 │    0.8342     │     70.0%     │     16 B    │
│  256 │    0.8429     │     80.0%     │     32 B    │
│  512 │    0.8357     │     60.0%     │     64 B    │
│ 1024 │    0.8249     │     90.0%     │    128 B    │
└──────┴──────────────┴──────────────┴─────────────┘
```

**Spearman ρ ≈ 0.83** across all dimensions — the rank correlation between RaBitQ approximate distances and exact L2 distances. This measures how well the 1-bit codes preserve distance ordering.

**Recall@10 with 50 candidates** = 60-90% from RaBitQ alone (no graph). The Vamana graph + beam search + L2 rerank bring this to 98-100%.

**Compression ratio**: 32× at d128 (512B float32 → 16B code), 128× at d1024 (4096B → 128B).

## 9. Memory Profile

`go test -run=TestReport_MemoryProfile`

```
┌──────────┬─────┬──────────────┬──────────────┬──────────────┐
│  Vectors │ Dim │   DB Size    │  Flat Vecs   │  Cache Est   │
├──────────┼─────┼──────────────┼──────────────┼──────────────┤
│    1000  │ 128 │     1.3 MB   │     0.5 MB   │     1.1 MB   │
│    5000  │ 128 │     6.6 MB   │     2.6 MB   │     5.7 MB   │
│   10000  │ 128 │    13.4 MB   │     5.1 MB   │    11.4 MB   │
└──────────┴─────┴──────────────┴──────────────┴──────────────┘
```

- **DB Size**: SQLite file on disk (vectors + graph + codes + metadata)
- **Flat Vecs**: Contiguous `[]float32` for brute-force (N × dim × 4 bytes)
- **Cache Est**: LRU cache holding all nodes (vectors + codes + neighbors)

Per-vector memory at d128: ~1.3KB in DB, 0.5KB flat, ~1.1KB in cache.

## 10. End-to-End Timing (10K × 128d)

`go test -run=TestReport_EndToEndTimings`

```
╔══════════════════════════════════════════════════════════════╗
║              END-TO-END TIMING REPORT (10K×128d)            ║
╠══════════════════════════════════════════════════════════════╣
║  New()                                    26ms              ║
║  Build (10K vectors)                 1m12.995s              ║
║  Search brute-force (100 queries)    138.441ms   1.384ms/q   ║
║  Search Vamana+RaBitQ (100 queries)  673.642ms   6.736ms/q   ║
║  Insert (100 vectors, batch)             505ms   5.053ms/vec ║
╚══════════════════════════════════════════════════════════════╝
```

Build time is dominated by the Vamana graph construction (2 passes × 10K greedy searches). Build is currently single-threaded.

## 11. Allocation Analysis

### Before optimization (v1)
```
Search 10K:  18.8ms    16,500,000 B/op    90,088 allocs/op
```

Sources: SQLite row scanning (10K `Scan` calls), `deserializeFloat32s` (10K `[]float32` allocations), `map[int64]bool` rehashing, `container/heap` interface boxing.

### After optimization (v2)
```
Search 10K:   1.3ms          352 B/op         1 allocs/op
```

The single allocation is the `[]Result` output slice. All internal state is pooled and reused.

```
                     Allocation reduction
    ┌─────────────────────────────────────────┐
    │  Before   90,088 allocs   16.5 MB       │
    │  After         1 alloc      352 B       │
    │                                         │
    │  Reduction:  90,000×       47,000×      │
    └─────────────────────────────────────────┘
```

### How each optimization contributes

| Optimization | Allocs eliminated | Bytes eliminated |
|---|---|---|
| Flat vectors (no SQLite scan) | ~30K (Scan + deserialize) | ~15 MB |
| SQLite driver overhead eliminated | ~60K (internal driver allocs) | ~1 MB |
| Pooled searchState (sync.Pool) | ~3 (map + heap + best) | ~40 KB |
| Typed heap (no interface boxing) | ~2K (Push/Pop boxing) | ~32 KB |
| Cached ext_id (no SQL per result) | ~10 (QueryRow per result) | ~2 KB |

## 12. Build and Insert

```
BenchmarkReport_Build_1K_d128      1    2847 ms/op    250 MB/op    971K allocs/op
BenchmarkReport_Insert_Into5K    270    4598 µs/op     19 KB/op     65 allocs/op
```

**Build 1K**: 2.8 seconds. Dominated by 2 passes of greedy search + robust pruning over the full graph. Currently single-threaded.

**Insert**: 4.6ms per vector into a 5K-node index. Performs greedy search to find neighbors, saves to SQLite, updates reverse edges. 65 allocations per insert (mostly SQLite driver internals).

## 13. Design Decisions

### Why brute-force below 50K?

At 10K vectors, brute-force takes 1.3ms with 100% recall. Vamana+RaBitQ takes 6.5ms with ~98% recall. The crossover where Vamana becomes faster is around 50K vectors:

| N | Brute-force | Vamana+RaBitQ |
|---|---|---|
| 1K | 0.12ms | 1.17ms |
| 5K | 0.66ms | 4.07ms |
| 10K | 1.33ms | 6.45ms |
| 50K (est.) | ~6.6ms | ~8ms |
| 100K (est.) | ~13ms | ~9ms |

### Why RaBitQ without rotation?

Standard RaBitQ uses a random rotation before quantization to improve approximation quality. horosvec uses centering + sign only (no rotation). This gives Spearman ρ ≈ 0.83 instead of ρ ≈ 0.90+ with rotation, but:
- No matrix multiplication needed (O(D) vs O(D²))
- Simpler implementation (pure sign of centered components)
- The Vamana graph + L2 rerank compensate for the lower RaBitQ accuracy

### Why sync.Pool for search state?

The search state (bitset + heap + best list) is ~15KB at 10K nodes. Creating and GC-ing this on every query adds latency and GC pressure. `sync.Pool` reuses states across queries — after the first few queries, all subsequent queries make 0 allocations in the search hot path.

### Why a typed heap instead of container/heap?

Go's `container/heap` uses `interface{}` for Push/Pop, causing 2 allocations per operation (the value is boxed into an interface). With ~1000 Push/Pop operations per search, that's ~2000 unnecessary allocations. The hand-rolled typed heap operates directly on `[]searchCandidate` with zero allocations.

---

## Running the benchmarks

```bash
# Full benchmark suite
go test -run='^$' -bench='BenchmarkReport_' -benchmem -timeout 600s ./...

# Recall and quality reports
go test -v -run='TestReport_' -timeout 600s ./...

# Quick search benchmark
go test -run='^$' -bench='BenchmarkReport_Search_10K' -benchmem ./...
```
