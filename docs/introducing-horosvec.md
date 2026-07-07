# horosvec: an embedded ANN vector index in pure Go, on top of a single SQLite file

*July 2026 — also published in French at [hazyhaar.fr](https://hazyhaar.fr/articles/horosvec-index-vectoriel-ann-go-pur).*

Vector similarity search has become the silent building block of every modern
document system: finding, among tens of thousands of passages, the ones that
*talk about the same thing* as a query without sharing a single word with it.
The engines that do this are usually heavy services — dedicated servers, native
dependencies, orchestration. **horosvec** takes the opposite stance: an
approximate-nearest-neighbor index **embedded in your process**, written in
**pure Go**, with all of its state living in **one SQLite file**.

One dependency: [modernc.org/sqlite](https://pkg.go.dev/modernc.org/sqlite),
the pure-Go SQLite port — no CGO, static binaries. MIT licensed.

```sh
go get github.com/hazyhaar/horosvec
```

## Two algorithms working in tandem

horosvec combines two ideas from the recent ANN literature.

**Vamana**, the proximity graph popularized by DiskANN: every vector becomes a
node connected to its relevant neighbors, and search navigates this graph
greedily instead of comparing the query against the whole base. Robust
alpha-RNG pruning keeps long-range shortcuts that avoid local minima.

**RaBitQ**, an extreme binary quantization: every coordinate of a vector is
reduced to its sign — one bit per dimension, plus two norms per vector. The
approximate distances computed on these codes are crude but almost free, and
they are enough to steer the navigation.

The architectural decision that makes the whole thing hold is the **two-stage
search**: graph traversal preselects candidates using RaBitQ distances, then
the final ranking is recomputed with exact L2 distance on the true float32
vectors. The estimator is allowed to be noisy: it only needs to place the true
neighbors inside the beam — the exact rerank does the rest.

## Measured, not promised

The implementation deliberately deviates from the RaBitQ paper on one point:
the random rotation step, on which the paper's theoretical guarantees rest, is
not implemented. Rather than invoking bounds that no longer apply, the
repository ships **deterministic, replayable benches** and publishes their
numbers.

Recall@10 against exact brute-force ground truth, 2,000 base vectors, 50
queries, default configuration:

| Dataset | dim | mean recall@10 | worst query |
|---|---|---|---|
| uniform synthetic | 128 | 1.000 | 1.000 |
| tight gaussian clusters | 128 | 0.982 | 0.900 |
| **real bge-m3 embeddings** (code-session texts) | 1024 | **1.000** | **1.000** |

The third line is the one that matters: on **real data** — two thousand
messages from software-development sessions, embedded by an actual embedding
model in dimension 1024 — the index does not miss a single neighbor. The
theoretical concern about the anisotropy of embedding spaces did not
materialize at this scale. The honest limits (2×10³ vectors, queries drawn
from the same distribution as the base) are documented in the package, and the
real-data bench is replayable by anyone: export `HOROSVEC_REAL_VECS` pointing
at a JSON array of vectors and run
`go test -run TestRecallMeasure_RealEmbeddings -v`.

## Hardened by production, and by adversity

horosvec is not a weekend prototype: it is the extraction of the engine that
serves RAG shard search and code-map embeddings in production inside the
horos55 ecosystem. Its preparation for publication went through a full
adversarial audit, whose findings became tested properties:

- **bounded deserialization**: a corrupt or hostile binary blob fails cleanly —
  no panic, no unbounded allocation;
- **inserts are transactional all the way into memory**: internal state (node
  cache, counters, flat mirror) is applied only after the SQLite commit — a
  failed commit leaves zero phantom neighbors;
- **cancellation is an error, never a silence**: a context cancelled in the
  middle of graph traversal returns an explicit error instead of an empty
  result indistinguishable from "no neighbors";
- 42 tests, 85.9% coverage — including commit-failure injection, blob
  corruption, LRU eviction and drift-triggered rebuilds.

Known limits are stated in the documentation rather than glossed over: no
delete API (full rebuild instead), an unbounded in-memory mirror for the
brute-force path, and a graceful-degradation contract on external reranking
that may become explicit error propagation in a future major version.

## Who is it for?

Any Go program that wants semantic search **without an external service**: a
CLI that indexes your notes, a server searching its own documents, a pipeline
deduplicating by similarity. If you need a distributed vector database, this
is not it; if you need an index that fits in your binary and in one file, this
is exactly it.
