// Package horosvec is an embedded approximate-nearest-neighbor (ANN) vector
// index in pure Go, backed by a single SQLite file.
//
// It combines two building blocks from the ANN literature:
//
//   - a Vamana proximity graph (DiskANN-style greedy search with alpha-RNG
//     robust pruning) for navigation;
//   - RaBitQ binary quantization (one sign bit per dimension plus per-vector
//     norms) for cheap approximate distances during graph traversal.
//
// Search runs in two stages: a beam search over the graph using RaBitQ
// approximate distances preselects candidates, then the final ranking is
// recomputed with exact L2 distance on the full float32 vectors. Small
// indexes (below Config.BruteForceThreshold) skip the graph entirely and are
// scanned exactly.
//
// # Storage and dependencies
//
// All state lives in a SQLite database (modernc.org/sqlite, pure Go — no
// CGO). The index can be rebuilt from the stored vectors, exported to a
// compact binary blob (ExportBinary) and reloaded with ImportBinary.
// ImportBinary validates header fields and bounds every length read from the
// stream before allocating, so a corrupt or hostile blob fails cleanly
// instead of panicking or over-allocating.
//
// # Concurrency and transactional semantics
//
// An Index is safe for concurrent use: Search takes a read lock, Build,
// Insert and rebuilds take the write lock. Insert applies its in-memory
// effects (node cache, flat-vector mirror, id counter, centroid) only after
// the SQLite transaction commits; if the commit fails, the in-memory index
// is untouched and Search never serves ids that were not persisted.
//
// Search, SearchWithRerank, Insert and Build honour context cancellation:
// cancellation during graph traversal returns a wrapped context error, never
// a silent empty result.
//
// # Accuracy: deviation from the RaBitQ paper
//
// This implementation deliberately omits the random rotation step of the
// RaBitQ paper. The paper's theoretical error bounds therefore do not
// transfer as-is: sign-bit quantization without rotation is sensitive to
// strongly axis-aligned (anisotropic) data in principle. In practice the
// two-stage design (wide beam preselection + exact L2 rerank) absorbs the
// estimator noise. Measured recall@10 against exact brute force, N=2000
// base vectors, 50 queries, defaults:
//
//   - uniform synthetic, dim 128:            mean 1.000
//   - gaussian clusters (sigma 0.05), 128:   mean 0.982 (worst query 0.90)
//   - real bge-m3 embeddings, dim 1024:      mean 1.000 (real code-session
//     texts; see recall_real_test.go, opt-in via HOROSVEC_REAL_VECS)
//
// These figures come from the repository's deterministic benches
// (recall_measure_test.go, recall_real_test.go). They are measurements, not
// guarantees; scale beyond ~10^4 vectors is not covered by them.
//
// # Known limits (v0.1)
//
//   - SearchWithRerank degrades gracefully when the reranker callback fails:
//     it returns the approximate candidates truncated to topK with a nil
//     error. Callers that need to distinguish degraded results must wrap the
//     reranker and track the failure themselves. This contract may change to
//     explicit error propagation in a future major version.
//   - There is no delete API: nodes are removed only by a full rebuild
//     (RebuildAsync, typically triggered by centroid drift — NeedsRebuild).
//   - The in-memory flat-vector mirror used by brute-force search grows with
//     the index and is not bounded; very large indexes should rely on the
//     graph path and budget memory accordingly.
//   - PRAGMA tuning applies per pooled connection; see pragma.go if you
//     manage the *sql.DB pool yourself.
package horosvec
