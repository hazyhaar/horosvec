# horosvec — architecture reference

Status: 2026-07-09, post-HNbook (26.7M-vector production validation).
Companion to `doc.go` (API contract) and the campaign retrospective kept in the
horos55 ecosystem (`audits/2026-07-07_horosvec_RETEX_unified_EN.md`).

## 1. The two-stage design

Every search runs the same two stages:

1. **Walk** — greedy beam search over a Vamana proximity graph, scored by a
   cheap RaBitQ estimator (1 bit per rotated dimension). The walk never touches
   raw vectors: it reads binary codes, per-node norms and adjacency, all
   resident in RAM (the *hot plane*).
2. **Rerank** — the beam's candidates are re-scored with exact L2 on the true
   vectors, read from wherever the vectors live (SQLite blobs, fp16 arena via
   mmap, or an in-RAM cache). The returned `Result.Score` is always an exact
   distance; the estimator is an ordering device, never a contract.

Below `Config.BruteForceThreshold` (default 50k) the index skips the graph
entirely and brute-forces exactly — perfect recall on small shards.

## 2. RaBitQ estimator (canonical, audited)

Vectors are centered on the build centroid **in rotated space** and quantized
to their signs. Stored per node: the sign code, ‖o−c‖² and L1(o−c). The
asymmetric estimate implemented is:

    dist² ≈ ‖q−c‖² + ‖o−c‖² − 2·‖o−c‖²·signDot / L1(o−c)

which is exactly the paper's unbiased estimator ⟨ō,q'⟩/⟨ō,o'⟩ after algebraic
cancellation of both √d factors and ‖q−c‖ (signDot carries the query's real
coordinates). The L1 division **is** the per-vector correction ⟨ō,o⟩ — the
thing that distinguishes RaBitQ from naive sign quantization. Audited
line-by-line against the paper on 2026-07-08 (verdict: conformant; full
derivation inlined in `rabitq.go`). A LUT (256 entries per code byte)
reproduces signDot exactly for fast scanning; `rabitqDistance` (symmetric,
popcount cosine) is a benchmark helper only.

## 3. Randomized Hadamard rotation

`rotation.go` applies r rounds of D·H (random ±1 diagonal, normalized fast
Walsh-Hadamard transform, zero-padding to the next power of two). Exact
isometry; the seed and round count are persisted in `vindex_meta` and reloaded,
so codes encoded under one seed can never be estimated under another. The
rotation lifted the SIFT recall ceiling from 0.955 to 0.987.

## 4. Storage topologies — two artifacts, not two modes

Product frontier (decided 2026-07-11, anchored to the 2026-07-10 measurements
archived in `audits/`, commit b1ac836):

- **DB-blob(-flat) is the WORKING artifact** — the incremental, transactional
  mode for live indexes. Its justification is incrementality and simplicity,
  not speed: after the flatVecs fix it runs at throughput PARITY with the
  arena (40,619 vs 39,599 QPS at concurrency 32, a ~2.5% gap within
  measurement noise — do not quote either mode as "the fastest").
  The qualifier "incremental" is VALIDATED (P0 verdict, 2026-07-12, archived
  in `horosvec-bench/audits/2026-07-12_p0_vamana_validation.md`): under
  inserts up to +50% of the initial build, max recall@10 drop vs a full
  rebuild is 0.0050 (alert threshold 0.01) at equivalent search throughput;
  month-shard RAM is bounded (175k build: 1.38 GiB heap; +87.5k inserts:
  4.5 GiB Sys). Known cost: insert throughput (~42 ms/vector) is ~5× the
  build cost per vector — daily deltas are fine (~8 min for 11.6k), monthly
  refits should prefer a rebuild.
- **The arena is the PUBLICATION artifact** — a frozen, per-SHARD serving
  format (analogy: a table vs. a Parquet file). Per shard only: the 26.7M
  monolith build is dead by construction on CPU (~3 days AND >62 GiB RAM,
  measured 2026-07-10); publication targets daily/monthly shards.

Consumers impact note: the known importers — `horos55/internal/silo_retrieval/
retriever.go`, `horos55/internal/codemap/embedder/horosvec_writer.go`,
`horos55/cmd/horos55-codemap/similar_cmd.go` — use the DB-blob working mode
through the public API; this frontier is documentation of intent and changes
no engine code and no API contract.

The periodic arena↔blob bridge is kept as a DOCUMENTED CONDITIONAL FALLBACK
(veille), not an active backlog item: it only becomes relevant if db-blob-flat
fails at a scale a consumer actually needs. The growing segmented arena
design (proven at prototype level: 435M verified reads under append churn,
zero torn reads — see `horosvec-bench/audits/2026-07-10_design_arene_
croissante.md`) is archived as NOT INTEGRATED for lack of a consumer; it is
the documented reopening path if the fallback is ever needed.

| Mode | Vectors live in | Insert | Scale ceiling |
|---|---|---|---|
| DB-blob (default) | `vindex_nodes.vector` (fp32 blobs) | yes, transactional | RAM + blob weight |
| **Arena** (`Config.ArenaPath`) | `HVARENA1` flat fp16 file, mmap | refused (fail-loud) | 26.7M proven; RAM holds hot plane only |
| Standalone import | in-RAM cache (`ImportBinary`) | no | RAM |

The arena file layout: 24-byte header (magic `HVARENA1`, version, dim, count)
then count×dim little-endian fp16; rank in file = node_id. A sibling `.ids`
file maps rank → external id (uint64 LE). The arena is opened read-only
(`PROT_READ`) and is never rewritten; builds and imports are replayable by
full overwrite of the SQLite side.

The fp16 conversion is IEEE-conformant round-to-nearest-even, subnormals
included (audited 2026-07-09 against a flush-to-zero counter-proposal:
rejected, values below 2⁻²⁵ round to zero anyway).

At 26.7M×512 the arena weighs 27.3 GB on disk; the hot plane (adjacency,
codes, norms) costs ~14 GB of anonymous heap; SQLite (graph + codes, no
vectors) 19.3 GB.

## 5. Build paths

- **`Build` (DB-blob)** — in-memory Vamana construction, float64 centroid
  accumulation, parallel graph construction (`BuildWorkers`, sharded
  neighborhood mutexes, ~×9 measured).
- **`Build` with `ArenaPath` (streaming)** — pass 1 streams vectors to the
  arena (checkpointed `ArenaWriter`, resumable); pass 2 (`BuildFromArena`)
  mmaps the arena and builds with a flat build-time adjacency (~854 B/node
  measured), never materializing the fp32 dataset.
- **`ImportAdjacency` (external graph)** — see §6.
- **Rebuild** — `RebuildAsync` is the small-index drift response; it refuses
  arena-backed indexes fail-loud (the legacy full-RAM path would destroy the
  vector-less layout at scale).

## 6. External-adjacency import (the GPU pipeline)

`ImportAdjacency(ctx, …)` turns `(arena fp16, flat u32 adjacency)` into a
complete index: it re-encodes RaBitQ codes and norms from the arena (rotation
included), computes the medoid via the streaming selector, persists the
vector-less graph, and rebuilds the hot plane. Guards, all fail-loud: exact
adjacency file size (N×degree×4), neighbor ids < N, degree ≤ MaxDegree,
`0xFFFFFFFF` padding sentinels (RAFT/cuVS convention) and self-loops filtered
at read, and a **normalization oracle** (≥10k sampled vectors, |‖v‖−1| < 0.01
for ≥99.9%) because an inner-product-built graph is only rank-equivalent to
the L2 walk on normalized vectors — the import refuses rather than assumes.

Production pipeline proven on HackerNews (26,691,317 items):

    parquet ─(DuckDB, eligibility+concat)→ NDJSON
      ─(hnbook-embed → vLLM, fp16, checkpointed)→ arena 27.3 GB   [3.6 h GPU]
      ─(cuVS CAGRA, managed memory, degree 64)→ adjacency 6.8 GB  [17.1 min GPU]
      ─(hnbook-import → ImportAdjacency)→ index 19.3 GB           [21.7 min CPU]
      ─(hnbook-validate: 20 real queries vs exact brute force)→
         overlap@10 = 0.99, rerank SQL = 0, p50 7.8 ms (NVMe warm)

The CPU build path (§5) remains the portable, GPU-free fallback; it is not on
the production critical path.

## 7. Concurrency and durability

One RWMutex over the index: `Search` takes RLock (concurrent searches;
`Rotate` is reentrant via per-call scratch), `Insert`/`Build` take the
exclusive lock. Inserts commit to SQLite first, then extend the hot plane
(`planePatch` overlay); on crash the plane is rebuilt from SQLite — the DB is
the single source of truth. `rebuildInternal` documents its lock-everything
tradeoff explicitly (correctness over latency; shadow-rebuild-and-swap is the
known evolution, see backlog).

## 8. Operational envelope (measured, 2026-07-09)

- Serving medium dominates latency: identical index, identical queries —
  p50 2.9 s from a rotational disk, 27.6 ms NVMe cold-ish, 7.8 ms NVMe warm.
  Rule: arenas and indexes live on SSD/NVMe; pinning the arena in RAM buys
  only the cold-cache gap.
- Recall floors are per-distribution (uniform ≥0.90, tight clusters ≥0.60):
  a single floor measured on one distribution is fiction for the others.
- Never truncate embedding inputs to a model's exact context window
  (vLLM pooling scheduler race at full window; keep ~200 tokens of margin).

## 9. Why horosvec is not on ann-benchmarks (a deliberate non-submission)

ann-benchmarks is the reference leaderboard of the ANN field, and horosvec is
deliberately absent from it — not because the numbers would be embarrassing,
but because **the benchmark and this engine measure orthogonal things**. Being
explicit about that orthogonality matters more than a leaderboard position.

What ann-benchmarks measures: the recall/throughput Pareto front of an
**in-memory index**, driven by a **single sequential client**, on standard
datasets that are mostly **low-to-mid dimensional** (SIFT-128, GloVe-100/200).
That protocol isolates the algorithmic kernel — and on that axis the honest
expectation is known and measured: a C++ graph over resident fp32 vectors
(hnswlib) beats horosvec ~×2 single-client, and ~×5 at iso-recall on 128-dim
data, where 1-bit-per-dimension codes simply carry too few bits.

What horosvec is built for — and what that protocol structurally cannot see:

- **Concurrent serving.** Under 32 closed-loop clients on a real 512-dim
  corpus (1M qwen embeddings), horosvec sustains ×1.9 the throughput of
  hnswlib at every sweep point, at higher recall at every point (measured
  2026-07-09, harness and JSONL in horosvec-bench). ann-benchmarks runs one
  query at a time; this axis does not exist there.
- **Persistence and memory envelope.** A horosvec index is a SQLite file plus
  an fp16 mmap arena: it opens without a rebuild, survives restarts, and
  serves 26.7M vectors with ~14 GB of heap because the vectors live off-heap.
  ann-benchmarks assumes the index fits in RAM as a process-lifetime object;
  durability and heap discipline are out of scope.
- **High-dimensional real embeddings.** The 1-bit quantization pays off
  precisely where modern embedding models live (512-1024+ dims) — the regime
  underrepresented in the standard datasets.
- **Zero native dependencies.** Pure Go, static binaries. The submission
  itself would have required wrapping the engine in a cgo shared library —
  benchmarking a build configuration the project explicitly refuses.

A leaderboard entry that ranked horosvec on the single-client, in-RAM,
low-dimension axis would be a *true measurement of the wrong object* — the
same failure mode this project's own benchmark campaign documented and
corrected twice (a rotational disk measured as engine latency; a non-production
code path measured as the engine). The comparative numbers horosvec publishes
instead (see `README.md` and the bench harness) state their exact perimeter:
corpus, dimension, concurrency, storage medium, code path — including the
configurations where hnswlib wins.

## 10. Known limits / backlog (engraved, not hidden)

- No online writes at scale: arena mode is build-once/serve-many BY DESIGN
  (see §4 — publication artifact). The appendable segmented arena is designed
  and prototype-proven but deliberately not integrated (no consumer); it is
  the reopening path, not a debt.
- The tri-modal storage branching wants a `NodeStore` abstraction before a
  fourth mode is added.
- Filtered search (predicate/allow-list during the walk) is the top functional
  item next.
- L2 kernel is pure Go; multi-accumulator float32 (~×2) before any assembly.
- Single medoid entry point; multiple sampled pivots are on the shortlist.
