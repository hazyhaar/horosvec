# Benchmarking a pure-Go vector index against hnswlib and sqlite-vec — every number I got wrong first

*July 2026. Harness, raw JSONL results and methodology:
[hazyhaar/horosvec-bench](https://github.com/hazyhaar/horosvec-bench). Engine:
[hazyhaar/horosvec](https://github.com/hazyhaar/horosvec) v0.7.0. Machine:
32 cores, 64 GB RAM, NVMe, RTX 5090 (GPU used for corpus embedding and graph
construction only — all search benchmarks are CPU).*

The headline first, with its exact perimeter so nobody has to dig for it: on
1M real 512-dimensional embeddings (Hacker News text, qwen3-embedding-0.6B),
under 32 concurrent clients, this pure-Go index sustains **~×1.9 hnswlib's
throughput at every recall point of the sweep, at equal-or-better recall**.
Single-client, hnswlib wins ~×2. On 128-dim SIFT at iso-recall, hnswlib wins
~×5. All curves — including every one where I lose — are in the public JSONL.

But the final numbers matter less than how many of them were wrong the first
time I measured them. This write-up is mostly about that.

## What I got wrong, in order

### 1. The disk

First end-to-end latency on a 26.7M-vector index (all of Hacker News,
validated at 99% overlap@10 against exact brute force over the full corpus):
p50 **2.9 seconds** per query. A "warm" rerun six hours later: 2.7 s — barely
faster, which should have been the tell: *a warm run that isn't faster than
cold isn't warm*. The working set (27 GB fp16 arena + 19 GB SQLite index) sat
on an 18 TB rotational archive disk; every rerank page fault paid a mechanical
seek, and the page cache had been evicted between runs.

Same index, same 20 queries, same oracles, moved to NVMe: **27.6 ms**
cold-ish, **7.8 ms** warm. A ×100-370 error, signed entirely by the storage
medium. The engine had been innocent all along — and I had almost consigned
"p50 ≈ 3 s at 26.7M" as its measured profile. Lesson: a latency verdict
carries its storage medium as silently as its dataset; name the medium in
every published number.

### 2. The code path

The three-engine benchmark then produced two damning findings *against my own
engine*: a throughput cliff (×5.7 collapse past a beam-width threshold, where
hnswlib lost ×1.8 on the same sweep) and dead concurrency scaling (+19% at
8 clients, regression at 32 — while hnswlib scaled ×6.6 and even sqlite-vec's
exact scan scaled ×9.5).

Instruction: reproduce at 300k instead of 1M (6-minute builds make profiling
cheap) — and the cliff **moved**, from ef 256 to ef 512. A defect that moves
with corpus size is a resource regime, not an algorithmic cost. Then `perf`
on the query window only: **41% garbage collection, 22% sweep, 21.6%
`database/sql.withLock`**. The search path was doing SQL.

One grep later: my harness enabled the production configuration — fp16 mmap
arena rerank — through a silent opt-in environment variable. Unset. Every
horosvec line in the benchmark had measured the fallback path, where each
rerank candidate is a row-by-row SQL blob read: an allocation storm whose GC
cost grows with the live heap (hence the cliff moving with N), behind a
process-wide pool lock (hence dead client scaling).

Rerun with the arena enabled: the cliff flattened into hnswlib-shaped smooth
decay, and 32-client throughput multiplied by **56** (1,581 → 88,218 QPS at
ef 64 on SIFT). Both "findings" dissolved. I kept the DB-blob curves published
anyway — they are true measurements of a real mode, and the mode now gets
batch-rerank work precisely because of them. Lessons: a harness that selects
its configuration through a silent opt-in lies by default (the measured mode
now belongs in the output record); and a finding against a system must name
the code path it measured.

### 3. The forged constants

Two smaller ones, same family. A hard size oracle in my own run plan
(expected byte size of a 26.7M×64 adjacency file) was arithmetically wrong by
+256,000 bytes — computed in my head at planning time; the file was right,
the plan was wrong. And a recall floor of 0.90, measured on uniform synthetic
data, turned out unreachable on tight gaussian clusters (0.678, intrinsic to
1-bit quantization at 128 dim) while trivially true on the distribution it
came from. Thresholds inherit the distribution they were measured on; golden
numbers get computed by scripts, not by heads.

## Where the numbers landed

Common protocol for all engines: queries never in the base, warm-up excluded,
monotonic-clock windows ≥ 3 s, recall measured sequentially (concurrency never
changes result contents — asserted), exact ground truth, one engine running at
a time. Concurrency is closed-loop client goroutines. hnswlib runs through its
cgo binding — a fairness caveat worth stating: part of its concurrency deficit
may be cgo crossing overhead rather than the C++ core.

**Real corpus — 1M × 512-dim (HN text embeddings), 500 held-out queries,
32 clients:**

| ef | horosvec QPS | hnswlib QPS | horosvec recall@10 | hnswlib recall@10 |
|---|---|---|---|---|
| 64 | 39,599 | 20,501 | 0.9468 | 0.9138 |
| 128 | 22,638 | 11,525 | 0.9774 | 0.9612 |
| 256 | 11,975 | 6,356 | 0.9892 | 0.9812 |
| 512 | 6,582 | 3,423 | 0.9938 | 0.9892 |

Single-client on the same corpus: hnswlib 4,128 QPS vs horosvec 2,086 at
ef 64 — the C++ core is ~×2 faster per thread; the pure-Go engine wins by
parallelizing better (RLock over a pointer-free flat "hot plane", per-query
state from a pool, zero allocations in the hot loop).

**SIFT-1M (128-dim), where I lose:** at iso-recall ~0.95, hnswlib delivers
~×5 the throughput (its ef 64 already recalls 0.9635 where horosvec needs
ef 512 for 0.9458). One bit per dimension is information-starved at 128 dims.
Structural, documented, on the roadmap (exact-walk mode for low dimension).

**sqlite-vec** (exact scan): recall ~1.0 always, 2.5 QPS at 1M×512
(395 ms/query), 10 QPS at 1M×128. It is the recall control of the harness and
a fine choice below ~50k vectors; it is not an ANN competitor at scale.

**Why not just submit to ann-benchmarks?** Deliberately declined: its
protocol — single client, index in RAM, mostly low/mid-dimension datasets —
is orthogonal on every axis to what this engine is for (concurrent serving,
SQLite persistence, off-heap fp16 vectors via mmap, high-dim real
embeddings). On their axis, the honest expectation is exactly the SIFT
numbers above, published here rather than hidden. Full rationale in
[ARCHITECTURE.md §9](./ARCHITECTURE.md).

## The 26.7M context

The corpus that motivated all this: all of Hacker News (28.7M items, 26.7M
after eligibility), embedded in 6h40 on one GPU, graph built by cuVS/CAGRA in
17.1 minutes, imported into horosvec (RaBitQ re-encoding, SQLite persistence,
zero vectors on the Go heap) in 21.7 minutes, validated at 99% overlap@10
against exact brute force over the full arena, served at p50 7.8 ms with
~14 GB of heap. The index is a SQLite file plus an fp16 mmap sidecar; it
opens without a rebuild and survives restarts.

## Reproducing

Everything needed is in [horosvec-bench](https://github.com/hazyhaar/horosvec-bench):
the three engine runners behind one protocol, the concurrency-sweep mode, the
exact-ground-truth computation, and `results/2026-07-09/` with every JSONL
this write-up cites — both horosvec modes labeled, defeats included.
