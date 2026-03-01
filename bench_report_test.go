// Comprehensive benchmark suite for horosvec performance report.
// Run with: go test -run='^$' -bench=BenchmarkReport -benchmem -timeout 600s ./...
package horosvec

import (
	"context"
	"database/sql"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"

	_ "modernc.org/sqlite"
)

// ════════════════════════════════════════════════════════════════
// Section 1: SEARCH LATENCY — scaling by dataset size
// ════════════════════════════════════════════════════════════════

func benchSearchAtScale(b *testing.B, n, dim int) {
	b.Helper()
	rng := rand.New(rand.NewPCG(42, 0))
	vecs, ids := generateVecs(rng, n, dim)

	db := newTestDB2(b)
	cfg := DefaultConfig()
	cfg.CacheCapacity = n + 1000
	idx, err := New(db, cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		b.Fatal(err)
	}

	queries := make([][]float32, 100)
	for i := range queries {
		queries[i] = vecs[rng.IntN(n)]
	}

	b.ResetTimer()
	qi := 0
	for b.Loop() {
		_, _ = idx.Search(queries[qi%100], 10)
		qi++
	}
}

func BenchmarkReport_Search_1K_d128(b *testing.B)  { benchSearchAtScale(b, 1000, 128) }
func BenchmarkReport_Search_5K_d128(b *testing.B)  { benchSearchAtScale(b, 5000, 128) }
func BenchmarkReport_Search_10K_d128(b *testing.B) { benchSearchAtScale(b, 10000, 128) }

// ════════════════════════════════════════════════════════════════
// Section 2: SEARCH LATENCY — scaling by dimension
// ════════════════════════════════════════════════════════════════

func BenchmarkReport_Search_5K_d64(b *testing.B)   { benchSearchAtScale(b, 5000, 64) }
func BenchmarkReport_Search_5K_d256(b *testing.B)  { benchSearchAtScale(b, 5000, 256) }
func BenchmarkReport_Search_5K_d512(b *testing.B)  { benchSearchAtScale(b, 5000, 512) }
func BenchmarkReport_Search_5K_d1024(b *testing.B) { benchSearchAtScale(b, 5000, 1024) }

// ════════════════════════════════════════════════════════════════
// Section 3: VAMANA PATH — forced Vamana search (BruteForceThreshold=0)
// ════════════════════════════════════════════════════════════════

func benchVamanaAtScale(b *testing.B, n, dim int) {
	b.Helper()
	rng := rand.New(rand.NewPCG(42, 0))
	vecs, ids := generateVecs(rng, n, dim)

	db := newTestDB2(b)
	cfg := DefaultConfig()
	cfg.CacheCapacity = n + 1000
	cfg.BruteForceThreshold = 0 // force Vamana+RaBitQ path
	idx, err := New(db, cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		b.Fatal(err)
	}

	queries := make([][]float32, 100)
	for i := range queries {
		queries[i] = vecs[rng.IntN(n)]
	}

	b.ResetTimer()
	qi := 0
	for b.Loop() {
		_, _ = idx.Search(queries[qi%100], 10)
		qi++
	}
}

func BenchmarkReport_Vamana_1K_d128(b *testing.B)  { benchVamanaAtScale(b, 1000, 128) }
func BenchmarkReport_Vamana_5K_d128(b *testing.B)  { benchVamanaAtScale(b, 5000, 128) }
func BenchmarkReport_Vamana_10K_d128(b *testing.B) { benchVamanaAtScale(b, 10000, 128) }

// ════════════════════════════════════════════════════════════════
// Section 4: RABITQ PRIMITIVES — encoding + distance computation
// ════════════════════════════════════════════════════════════════

func benchRaBitQEncode(b *testing.B, dim int) {
	rng := rand.New(rand.NewPCG(42, 0))
	centroid := make([]float32, dim)
	for i := range dim {
		centroid[i] = float32(rng.NormFloat64() * 0.01)
	}
	enc := NewEncoder(centroid)
	vec := make([]float32, dim)
	for i := range dim {
		vec[i] = float32(rng.NormFloat64())
	}

	b.ResetTimer()
	for b.Loop() {
		enc.Encode(vec)
	}
}

func BenchmarkReport_RaBitQ_Encode_d64(b *testing.B)   { benchRaBitQEncode(b, 64) }
func BenchmarkReport_RaBitQ_Encode_d128(b *testing.B)  { benchRaBitQEncode(b, 128) }
func BenchmarkReport_RaBitQ_Encode_d512(b *testing.B)  { benchRaBitQEncode(b, 512) }
func BenchmarkReport_RaBitQ_Encode_d1024(b *testing.B) { benchRaBitQEncode(b, 1024) }

func benchRaBitQAsymDist(b *testing.B, dim int) {
	rng := rand.New(rand.NewPCG(42, 0))
	centroid := make([]float32, dim)
	enc := NewEncoder(centroid)
	query := make([]float32, dim)
	vec := make([]float32, dim)
	for i := range dim {
		query[i] = float32(rng.NormFloat64())
		vec[i] = float32(rng.NormFloat64())
	}
	code, sqNorm, l1Norm := enc.Encode(vec)

	b.ResetTimer()
	for b.Loop() {
		rabitqDistanceAsym(query, centroid, code, sqNorm, l1Norm)
	}
}

func BenchmarkReport_RaBitQ_AsymDist_d64(b *testing.B)   { benchRaBitQAsymDist(b, 64) }
func BenchmarkReport_RaBitQ_AsymDist_d128(b *testing.B)  { benchRaBitQAsymDist(b, 128) }
func BenchmarkReport_RaBitQ_AsymDist_d512(b *testing.B)  { benchRaBitQAsymDist(b, 512) }
func BenchmarkReport_RaBitQ_AsymDist_d1024(b *testing.B) { benchRaBitQAsymDist(b, 1024) }

func benchRaBitQPrecomp(b *testing.B, dim int) {
	rng := rand.New(rand.NewPCG(42, 0))
	centroid := make([]float32, dim)
	enc := NewEncoder(centroid)
	query := make([]float32, dim)
	vec := make([]float32, dim)
	for i := range dim {
		query[i] = float32(rng.NormFloat64())
		vec[i] = float32(rng.NormFloat64())
	}
	code, sqNorm, l1Norm := enc.Encode(vec)

	// Pre-compute centered query
	queryCentered := make([]float64, dim)
	var querySqNorm float64
	for i := range dim {
		c := float64(query[i]) - float64(centroid[i])
		queryCentered[i] = c
		querySqNorm += c * c
	}

	b.ResetTimer()
	for b.Loop() {
		rabitqDistanceAsymPrecomp(queryCentered, querySqNorm, code, sqNorm, l1Norm)
	}
}

func BenchmarkReport_RaBitQ_Precomp_d128(b *testing.B)  { benchRaBitQPrecomp(b, 128) }
func BenchmarkReport_RaBitQ_Precomp_d1024(b *testing.B) { benchRaBitQPrecomp(b, 1024) }

// ════════════════════════════════════════════════════════════════
// Section 5: L2 EXACT — baseline distance computation
// ════════════════════════════════════════════════════════════════

func benchL2(b *testing.B, dim int) {
	rng := rand.New(rand.NewPCG(42, 0))
	a := make([]float32, dim)
	bb := make([]float32, dim)
	for i := range dim {
		a[i] = float32(rng.NormFloat64())
		bb[i] = float32(rng.NormFloat64())
	}

	b.ResetTimer()
	for b.Loop() {
		l2DistanceSquared(a, bb)
	}
}

func BenchmarkReport_L2Exact_d64(b *testing.B)   { benchL2(b, 64) }
func BenchmarkReport_L2Exact_d128(b *testing.B)  { benchL2(b, 128) }
func BenchmarkReport_L2Exact_d512(b *testing.B)  { benchL2(b, 512) }
func BenchmarkReport_L2Exact_d1024(b *testing.B) { benchL2(b, 1024) }

// ════════════════════════════════════════════════════════════════
// Section 6: POPCOUNT — symmetric RaBitQ distance (bit-level)
// ════════════════════════════════════════════════════════════════

func benchPOPCOUNT(b *testing.B, nBytes int) {
	rng := rand.New(rand.NewPCG(42, 0))
	a := make([]byte, nBytes)
	bb := make([]byte, nBytes)
	for i := range a {
		a[i] = byte(rng.IntN(256))
		bb[i] = byte(rng.IntN(256))
	}

	b.ResetTimer()
	for b.Loop() {
		rabitqDistance(a, bb, 100.0, 100.0)
	}
}

func BenchmarkReport_POPCOUNT_8B(b *testing.B)   { benchPOPCOUNT(b, 8) }
func BenchmarkReport_POPCOUNT_16B(b *testing.B)  { benchPOPCOUNT(b, 16) }
func BenchmarkReport_POPCOUNT_64B(b *testing.B)  { benchPOPCOUNT(b, 64) }
func BenchmarkReport_POPCOUNT_128B(b *testing.B) { benchPOPCOUNT(b, 128) }

// ════════════════════════════════════════════════════════════════
// Section 7: CONCURRENT SEARCH THROUGHPUT
// ════════════════════════════════════════════════════════════════

func BenchmarkReport_ConcurrentSearch_10K(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 0))
	n := 10000
	dim := 128
	vecs, ids := generateVecs(rng, n, dim)

	db := newTestDB2(b)
	cfg := DefaultConfig()
	cfg.CacheCapacity = n + 1000
	idx, err := New(db, cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		b.Fatal(err)
	}

	queries := make([][]float32, 100)
	for i := range queries {
		queries[i] = vecs[rng.IntN(n)]
	}

	b.SetParallelism(runtime.GOMAXPROCS(0))
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		qi := 0
		for pb.Next() {
			_, _ = idx.Search(queries[qi%100], 10)
			qi++
		}
	})
}

// ════════════════════════════════════════════════════════════════
// Section 8: BUILD — index construction cost
// ════════════════════════════════════════════════════════════════

func benchBuild(b *testing.B, n, dim int) {
	rng := rand.New(rand.NewPCG(42, 0))
	vecs, ids := generateVecs(rng, n, dim)

	for b.Loop() {
		dir, err := os.MkdirTemp("", "horosvec-bench-*")
		if err != nil {
			b.Fatal(err)
		}
		dbPath := filepath.Join(dir, "bench.db")
		db, err := sql.Open("sqlite", dbPath)
		if err != nil {
			b.Fatal(err)
		}
		cfg := DefaultConfig()
		idx, err := New(db, cfg)
		if err != nil {
			b.Fatal(err)
		}
		iter := &sliceIterator{vecs: vecs, ids: ids}
		if err := idx.Build(context.Background(), iter); err != nil {
			b.Fatal(err)
		}
		_ = idx.Close()
		_ = db.Close()
		os.RemoveAll(dir)
	}
}

func BenchmarkReport_Build_1K_d128(b *testing.B) { benchBuild(b, 1000, 128) }
func BenchmarkReport_Build_5K_d128(b *testing.B) { benchBuild(b, 5000, 128) }

// ════════════════════════════════════════════════════════════════
// Section 9: INSERT — incremental insertion cost
// ════════════════════════════════════════════════════════════════

func BenchmarkReport_Insert_Into5K(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 0))
	n := 5000
	dim := 128
	vecs, ids := generateVecs(rng, n, dim)

	db := newTestDB2(b)
	cfg := DefaultConfig()
	cfg.CacheCapacity = n + 1000
	idx, err := New(db, cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		b.Fatal(err)
	}

	// Pre-generate insert vectors
	insertVecs := make([][]float32, 100)
	insertIDs := make([][]byte, 100)
	for i := range insertVecs {
		insertVecs[i] = make([]float32, dim)
		for j := range dim {
			insertVecs[i][j] = float32(rng.NormFloat64())
		}
		v := n + i
		insertIDs[i] = []byte{byte(v >> 24), byte(v >> 16), byte(v >> 8), byte(v)}
	}

	b.ResetTimer()
	qi := 0
	for b.Loop() {
		i := qi % 100
		_ = idx.Insert([][]float32{insertVecs[i]}, [][]byte{insertIDs[i]})
		qi++
	}
}

// ════════════════════════════════════════════════════════════════
// Section 10: RECALL MEASUREMENT (not a benchmark, but reported)
// ════════════════════════════════════════════════════════════════

func TestReport_RecallAtScale(t *testing.T) {
	scales := []struct {
		n, dim int
	}{
		{100, 64},
		{500, 128},
		{1000, 128},
		{5000, 128},
		{10000, 128},
	}

	fmt.Println()
	fmt.Println("┌──────────┬─────┬────────────┬────────────┬─────────────┐")
	fmt.Println("│  Vectors │ Dim │ Recall@10  │  Path      │ Queries     │")
	fmt.Println("├──────────┼─────┼────────────┼────────────┼─────────────┤")

	for _, s := range scales {
		db := newTestDB(t)
		rng := rand.New(rand.NewPCG(42, 0))
		vecs, ids := generateVecs(rng, s.n, s.dim)

		cfg := DefaultConfig()
		cfg.EfSearch = 128
		cfg.CacheCapacity = s.n + 1000

		idx, err := New(db, cfg)
		if err != nil {
			t.Fatal(err)
		}

		iter := &sliceIterator{vecs: vecs, ids: ids}
		if err := idx.Build(context.Background(), iter); err != nil {
			t.Fatal(err)
		}

		k := 10
		numQueries := 50
		if numQueries > s.n/5 {
			numQueries = s.n / 5
		}
		if numQueries < 5 {
			numQueries = 5
		}

		totalRecall := 0.0
		for q := range numQueries {
			qi := q * (s.n / numQueries)
			query := vecs[qi]

			// Exact brute-force
			type idDist struct {
				idx  int
				dist float64
			}
			dists := make([]idDist, s.n)
			for i, v := range vecs {
				dists[i] = idDist{i, l2DistanceSquared(query, v)}
			}
			sort.Slice(dists, func(a, b int) bool {
				return dists[a].dist < dists[b].dist
			})
			trueTopK := make(map[string]bool, k)
			for i := range k {
				trueTopK[string(ids[dists[i].idx])] = true
			}

			results, err := idx.Search(query, k)
			if err != nil {
				t.Fatalf("search failed: %v", err)
			}

			hits := 0
			for _, r := range results {
				if trueTopK[string(r.ID)] {
					hits++
				}
			}
			totalRecall += float64(hits) / float64(k)
		}

		avgRecall := totalRecall / float64(numQueries)
		path := "brute-force"
		if idx.cfg.BruteForceThreshold > 0 && int(idx.nextID) <= idx.cfg.BruteForceThreshold {
			path = "brute-force"
		} else {
			path = "Vamana+RaBitQ"
		}

		fmt.Printf("│  %6d  │ %3d │   %5.1f%%   │ %-10s │     %3d     │\n",
			s.n, s.dim, avgRecall*100, path, numQueries)
		idx.Close()
	}

	fmt.Println("└──────────┴─────┴────────────┴────────────┴─────────────┘")
	fmt.Println()
}

// ════════════════════════════════════════════════════════════════
// Section 11: MEMORY — SQLite database size
// ════════════════════════════════════════════════════════════════

func TestReport_MemoryProfile(t *testing.T) {
	scales := []struct {
		n, dim int
	}{
		{1000, 128},
		{5000, 128},
		{10000, 128},
	}

	fmt.Println()
	fmt.Println("┌──────────┬─────┬──────────────┬──────────────┬──────────────┐")
	fmt.Println("│  Vectors │ Dim │   DB Size    │  Flat Vecs   │  Cache Est   │")
	fmt.Println("├──────────┼─────┼──────────────┼──────────────┼──────────────┤")

	for _, s := range scales {
		dir := t.TempDir()
		dbPath := filepath.Join(dir, "mem.db")
		db, err := sql.Open("sqlite", dbPath)
		if err != nil {
			t.Fatal(err)
		}

		rng := rand.New(rand.NewPCG(42, 0))
		vecs, ids := generateVecs(rng, s.n, s.dim)

		cfg := DefaultConfig()
		cfg.CacheCapacity = s.n + 1000
		idx, err := New(db, cfg)
		if err != nil {
			t.Fatal(err)
		}

		iter := &sliceIterator{vecs: vecs, ids: ids}
		if err := idx.Build(context.Background(), iter); err != nil {
			t.Fatal(err)
		}

		idx.Close()
		db.Close()

		fi, err := os.Stat(dbPath)
		if err != nil {
			t.Fatal(err)
		}
		dbSize := fi.Size()

		// flatVecs memory: n * dim * 4 bytes
		flatMem := int64(s.n) * int64(s.dim) * 4
		// Cache estimate: each node ≈ dim*4 (vec) + dim/8 (code) + 64*8 (neighbors) + 100 (overhead)
		cachePerNode := int64(s.dim)*4 + int64(s.dim)/8 + 64*8 + 100
		cacheMem := int64(s.n) * cachePerNode

		fmt.Printf("│  %6d  │ %3d │  %6.1f MB   │  %6.1f MB   │  %6.1f MB   │\n",
			s.n, s.dim, float64(dbSize)/1e6, float64(flatMem)/1e6, float64(cacheMem)/1e6)
	}

	fmt.Println("└──────────┴─────┴──────────────┴──────────────┴──────────────┘")
	fmt.Println()
}

// ════════════════════════════════════════════════════════════════
// Section 12: SPEARMAN CORRELATION — RaBitQ approximation quality
// ════════════════════════════════════════════════════════════════

func TestReport_RaBitQCorrelation(t *testing.T) {
	dims := []int{64, 128, 256, 512, 1024}

	fmt.Println()
	fmt.Println("┌──────┬──────────────┬──────────────┬─────────────┐")
	fmt.Println("│ Dim  │ Spearman ρ   │ Recall@10/50 │  Code Size  │")
	fmt.Println("├──────┼──────────────┼──────────────┼─────────────┤")

	for _, dim := range dims {
		rng := rand.New(rand.NewPCG(42, 0))
		n := 1000
		vecs := make([][]float32, n)
		for i := range n {
			vecs[i] = make([]float32, dim)
			for j := range dim {
				vecs[i][j] = float32(rng.NormFloat64())
			}
		}

		centroid := computeCentroid(vecs)
		enc := NewEncoder(centroid)

		codes := make([][]byte, n)
		sqNorms := make([]float64, n)
		l1Norms := make([]float64, n)
		for i, v := range vecs {
			codes[i], sqNorms[i], l1Norms[i] = enc.Encode(v)
		}

		query := vecs[0]
		exactDists := make([]float64, n)
		approxDists := make([]float64, n)
		for i, v := range vecs {
			exactDists[i] = l2DistanceSquared(query, v)
			approxDists[i] = rabitqDistanceAsym(query, centroid, codes[i], sqNorms[i], l1Norms[i])
		}

		rho := spearmanRank(exactDists, approxDists)
		recall := recallAtK(exactDists, approxDists, 10, 50)
		codeSize := (dim + 7) / 8

		fmt.Printf("│ %4d │    %.4f     │    %5.1f%%     │   %4d B    │\n",
			dim, rho, recall*100, codeSize)
	}

	fmt.Println("└──────┴──────────────┴──────────────┴─────────────┘")
	fmt.Println()
}

// ════════════════════════════════════════════════════════════════
// Section 13: SEARCH STATE POOL — reuse efficiency
// ════════════════════════════════════════════════════════════════

func BenchmarkReport_SearchStatePool(b *testing.B) {
	maxNodes := int64(10000)
	L := 128
	dim := 128

	b.ResetTimer()
	for b.Loop() {
		s := acquireSearchState(maxNodes, L, dim)
		releaseSearchState(s)
	}
}

func BenchmarkReport_SearchStateBitset(b *testing.B) {
	maxNodes := int64(100000)
	L := 128
	dim := 128
	s := acquireSearchState(maxNodes, L, dim)
	defer releaseSearchState(s)

	b.ResetTimer()
	for b.Loop() {
		s.reset(maxNodes, L, dim)
		for i := int64(0); i < 1000; i++ {
			s.visit(i)
		}
	}
}

// ════════════════════════════════════════════════════════════════
// Section 14: END-TO-END TIMING with detailed breakdown
// ════════════════════════════════════════════════════════════════

func TestReport_EndToEndTimings(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0))
	n := 10000
	dim := 128
	vecs, ids := generateVecs(rng, n, dim)

	// --- Build phase ---
	db := newTestDB(t)
	cfg := DefaultConfig()
	cfg.CacheCapacity = n + 1000

	t0 := time.Now()
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	tNew := time.Since(t0)

	t0 = time.Now()
	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}
	tBuild := time.Since(t0)

	// --- Search phase ---
	numQueries := 100
	queries := make([][]float32, numQueries)
	for i := range queries {
		queries[i] = vecs[rng.IntN(n)]
	}

	// Warm run
	for _, q := range queries[:10] {
		idx.Search(q, 10)
	}

	// Timed brute-force path (default threshold)
	t0 = time.Now()
	for _, q := range queries {
		idx.Search(q, 10)
	}
	tBruteTotal := time.Since(t0)

	// Timed Vamana path
	idx.cfg.BruteForceThreshold = 0
	t0 = time.Now()
	for _, q := range queries {
		idx.Search(q, 10)
	}
	tVamanaTotal := time.Since(t0)
	idx.cfg.BruteForceThreshold = cfg.BruteForceThreshold // restore

	// --- Insert phase ---
	insertVecs := make([][]float32, 100)
	insertIDs := make([][]byte, 100)
	for i := range insertVecs {
		insertVecs[i] = make([]float32, dim)
		for j := range dim {
			insertVecs[i][j] = float32(rng.NormFloat64())
		}
		v := n + i
		insertIDs[i] = []byte{byte(v >> 24), byte(v >> 16), byte(v >> 8), byte(v)}
	}

	t0 = time.Now()
	if err := idx.Insert(insertVecs, insertIDs); err != nil {
		t.Fatal(err)
	}
	tInsert := time.Since(t0)

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║              END-TO-END TIMING REPORT (10K×128d)            ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  New()                              %10s              ║\n", tNew.Round(time.Millisecond))
	fmt.Printf("║  Build (10K vectors)                %10s              ║\n", tBuild.Round(time.Millisecond))
	fmt.Printf("║  Search brute-force (100 queries)   %10s   %6s/q   ║\n",
		tBruteTotal.Round(time.Microsecond), (tBruteTotal / time.Duration(numQueries)).Round(time.Microsecond))
	fmt.Printf("║  Search Vamana+RaBitQ (100 queries) %10s   %6s/q   ║\n",
		tVamanaTotal.Round(time.Microsecond), (tVamanaTotal / time.Duration(numQueries)).Round(time.Microsecond))
	fmt.Printf("║  Insert (100 vectors, batch)        %10s   %6s/vec ║\n",
		tInsert.Round(time.Millisecond), (tInsert / 100).Round(time.Microsecond))
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()
}
