// Audit tests — validate RaBitQ and Vamana implementations independently.
// Each test isolates one algorithm property against known ground truth.
package horosvec

import (
	"bytes"
	"context"
	"database/sql"
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"sort"
	"testing"

	_ "modernc.org/sqlite"
)

// --- RaBitQ audit ---

// TestAudit_RaBitQ_CorrectionFactor checks that the simplified correction
// factor (using L1 norm directly) preserves distance ordering.
// The real RaBitQ paper uses L1/(sqrt(D)*L2) — we test how much the
// simplification costs in ranking quality.
func TestAudit_RaBitQ_CorrectionFactor(t *testing.T) {
	// WHAT: Measure rank correlation between RaBitQ asymmetric distance and exact L2.
	// WHY: The correction factor is simplified — we need to know the cost.
	dims := []int{64, 128, 512, 1024}
	for _, dim := range dims {
		t.Run(fmt.Sprintf("dim%d", dim), func(t *testing.T) {
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

			// Exact L2 distances
			exactDists := make([]float64, n)
			for i, v := range vecs {
				exactDists[i] = l2DistanceSquared(query, v)
			}

			// RaBitQ distances
			approxDists := make([]float64, n)
			for i := range n {
				approxDists[i] = rabitqDistanceAsym(query, centroid, codes[i], sqNorms[i], l1Norms[i])
			}

			// Spearman rank correlation
			rho := spearmanRank(exactDists, approxDists)
			t.Logf("dim=%d: Spearman ρ = %.4f", dim, rho)
			if rho < 0.80 {
				t.Errorf("dim=%d: rank correlation %.4f < 0.80 — RaBitQ correction is too lossy", dim, rho)
			}

			// Recall@10 with 5x oversampling (brute-force on codes, no graph)
			recall := recallAtK(exactDists, approxDists, 10, 50)
			t.Logf("dim=%d: recall@10 (50 candidates) = %.1f%%", dim, recall*100)
		})
	}
}

// TestAudit_RaBitQ_NegativeDistances checks that RaBitQ never produces
// negative distance estimates (which would corrupt search ordering).
func TestAudit_RaBitQ_NegativeDistances(t *testing.T) {
	// WHAT: Scan for negative distance estimates across random pairs.
	// WHY: The simplified formula could go negative if correction is wrong.
	rng := rand.New(rand.NewPCG(99, 0))
	dim := 1024
	n := 5000

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

	negatives := 0
	queries := 200
	for q := range queries {
		query := vecs[q]
		for i := range n {
			d := rabitqDistanceAsym(query, centroid, codes[i], sqNorms[i], l1Norms[i])
			if d < 0 {
				negatives++
			}
		}
	}
	total := queries * n
	t.Logf("Negative distances: %d / %d (%.4f%%)", negatives, total, float64(negatives)/float64(total)*100)
	if negatives > total/100 { // > 1% is a problem
		t.Errorf("too many negative distances: %d / %d", negatives, total)
	}
}

// TestAudit_RaBitQ_PaperFormula compares the simplified correction with
// the full RaBitQ paper formula: correction = L1 / (sqrt(D) * L2).
func TestAudit_RaBitQ_PaperFormula(t *testing.T) {
	// WHAT: Quantify the gap between simplified and paper formula.
	// WHY: If the gap is small, the simplification is justified.
	rng := rand.New(rand.NewPCG(42, 0))
	dim := 1024
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
	sqrtD := math.Sqrt(float64(dim))

	query := vecs[0]

	// Compare ranking with simplified vs paper correction
	type distPair struct {
		simplified float64
		paper      float64
		exact      float64
	}
	pairs := make([]distPair, n)
	for i, v := range vecs {
		code, sqNorm, l1Norm := enc.Encode(v)

		// Simplified (current code): uses l1Norm directly
		simplified := rabitqDistanceAsym(query, centroid, code, sqNorm, l1Norm)

		// Paper formula: correction = L1 / (sqrt(D) * L2)
		l2Norm := math.Sqrt(sqNorm)
		var paperCorrection float64
		if l2Norm > 0 {
			paperCorrection = l1Norm / (sqrtD * l2Norm)
		}

		// Recompute with paper correction
		var signDot float64
		for j := range dim {
			centered := float64(query[j]) - float64(centroid[j])
			if code[j/8]&(1<<uint(j%8)) != 0 {
				signDot += centered
			} else {
				signDot -= centered
			}
		}
		var querySqNorm float64
		for j := range dim {
			c := float64(query[j]) - float64(centroid[j])
			querySqNorm += c * c
		}

		var paperDist float64
		if paperCorrection > 0 {
			paperDist = querySqNorm + sqNorm - 2.0*math.Sqrt(sqNorm*querySqNorm)*signDot*paperCorrection/l1Norm
		} else {
			paperDist = querySqNorm + sqNorm
		}

		pairs[i] = distPair{
			simplified: simplified,
			paper:      paperDist,
			exact:      l2DistanceSquared(query, v),
		}
	}

	// Compare rank correlations
	exact := make([]float64, n)
	simplified := make([]float64, n)
	paper := make([]float64, n)
	for i, p := range pairs {
		exact[i] = p.exact
		simplified[i] = p.simplified
		paper[i] = p.paper
	}

	rhoSimplified := spearmanRank(exact, simplified)
	rhoPaper := spearmanRank(exact, paper)

	t.Logf("Spearman ρ (simplified vs exact): %.4f", rhoSimplified)
	t.Logf("Spearman ρ (paper vs exact):      %.4f", rhoPaper)
	t.Logf("Gap: %.4f", rhoPaper-rhoSimplified)
}

// --- Vamana audit ---

// TestAudit_Vamana_GraphConnectivity checks that the built graph is
// fully connected (every node reachable from medoid via BFS).
func TestAudit_Vamana_GraphConnectivity(t *testing.T) {
	// WHAT: BFS from medoid must reach all nodes.
	// WHY: Disconnected nodes are invisible to search — silent data loss.
	sizes := []int{100, 500, 1000}
	for _, n := range sizes {
		t.Run(fmt.Sprintf("n%d", n), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(42, 0))
			dim := 64
			vecs, _ := generateVecs(rng, n, dim)

			nodes := make([]graphNode, n)
			centroid := computeCentroid(vecs)
			enc := NewEncoder(centroid)
			for i, v := range vecs {
				code, sqNorm, l1Norm := enc.Encode(v)
				nodes[i] = graphNode{id: int64(i), vec: v, code: code, sqNorm: sqNorm, l1Norm: l1Norm}
			}

			medoid := findMedoid(nodes)
			buildGraph(context.Background(), nodes, medoid, 64, 128, 1.2, 2)

			// BFS from medoid
			visited := make(map[int64]bool)
			queue := []int64{medoid}
			visited[medoid] = true
			for len(queue) > 0 {
				cur := queue[0]
				queue = queue[1:]
				for _, nbr := range nodes[cur].neighbors {
					if !visited[nbr] {
						visited[nbr] = true
						queue = append(queue, nbr)
					}
				}
			}

			reachable := len(visited)
			t.Logf("n=%d: reachable from medoid = %d/%d (%.1f%%)", n, reachable, n, float64(reachable)/float64(n)*100)
			if reachable < n {
				t.Errorf("graph not fully connected: %d/%d nodes reachable", reachable, n)
			}
		})
	}
}

// TestAudit_Vamana_DegreeDistribution checks that node degrees stay within bounds.
func TestAudit_Vamana_DegreeDistribution(t *testing.T) {
	// WHAT: Verify degree bounds (0 < degree <= R for all nodes after build).
	// WHY: Over-capacity nodes waste memory; zero-degree nodes are dead.
	rng := rand.New(rand.NewPCG(42, 0))
	n := 2000
	dim := 64
	maxDeg := 64

	vecs, _ := generateVecs(rng, n, dim)
	nodes := make([]graphNode, n)
	centroid := computeCentroid(vecs)
	enc := NewEncoder(centroid)
	for i, v := range vecs {
		code, sqNorm, l1Norm := enc.Encode(v)
		nodes[i] = graphNode{id: int64(i), vec: v, code: code, sqNorm: sqNorm, l1Norm: l1Norm}
	}

	medoid := findMedoid(nodes)
	buildGraph(context.Background(), nodes, medoid, maxDeg, 128, 1.2, 2)

	minNodeDeg, maxNodeDeg := n, 0
	totalDeg := 0
	overCapacity := 0
	zeroDegCount := 0
	for _, node := range nodes {
		deg := len(node.neighbors)
		if deg < minNodeDeg {
			minNodeDeg = deg
		}
		if deg > maxNodeDeg {
			maxNodeDeg = deg
		}
		totalDeg += deg
		if deg > maxDeg {
			overCapacity++
		}
		if deg == 0 {
			zeroDegCount++
		}
	}

	avgDeg := float64(totalDeg) / float64(n)
	t.Logf("Degree: min=%d, max=%d, avg=%.1f, over_R=%d, zero=%d", minNodeDeg, maxNodeDeg, avgDeg, overCapacity, zeroDegCount)
	if overCapacity > 0 {
		t.Errorf("%d nodes exceed max degree R=%d", overCapacity, maxDeg)
	}
	if zeroDegCount > 0 {
		t.Errorf("%d nodes have zero neighbors", zeroDegCount)
	}
}

// TestAudit_Vamana_SearchVsBruteForce compares Vamana search quality
// against brute-force at increasing scales.
func TestAudit_Vamana_SearchVsBruteForce(t *testing.T) {
	// WHAT: Recall@10 must stay >= 90% at all tested scales.
	// WHY: Measures actual approximation quality vs exact NN.
	scales := []struct {
		n   int
		dim int
	}{
		{100, 64},
		{500, 64},
		{1000, 64},
	}

	for _, s := range scales {
		t.Run(fmt.Sprintf("n%d_d%d", s.n, s.dim), func(t *testing.T) {
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
			defer idx.Close()

			iter := &sliceIterator{vecs: vecs, ids: ids}
			if err := idx.Build(context.Background(), iter); err != nil {
				t.Fatal(err)
			}

			k := 10
			numQueries := 50
			if numQueries > s.n/10 {
				numQueries = s.n / 10
			}
			if numQueries < 5 {
				numQueries = 5
			}

			totalRecall := 0.0
			for q := range numQueries {
				qi := q * (s.n / numQueries)
				query := vecs[qi]

				// Brute-force exact
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

				// Vamana search
				results, err := idx.Search(query, k)
				if err != nil {
					t.Fatalf("query %d: %v", q, err)
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
			t.Logf("n=%d dim=%d: recall@%d = %.1f%% (%d queries)", s.n, s.dim, k, avgRecall*100, numQueries)
			if avgRecall < 0.90 {
				t.Errorf("recall@%d = %.1f%% < 90%%", k, avgRecall*100)
			}
		})
	}
}

// TestAudit_Vamana_RaBitQUsed proves that search DOES use RaBitQ codes for graph traversal.
func TestAudit_Vamana_RaBitQUsed(t *testing.T) {
	// WHAT: Corrupt all RaBitQ codes after build, search results must change.
	// WHY: Confirms RaBitQ is active in 2-stage search (RaBitQ beam + L2 rerank).
	rng := rand.New(rand.NewPCG(42, 0))
	n := 1000
	dim := 128
	k := 10

	db := newTestDB(t)
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0 // force Vamana path for this test
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if buildErr := idx.Build(context.Background(), iter); buildErr != nil {
		t.Fatal(buildErr)
	}

	// Search before corruption
	query := vecs[42]
	resultsBefore, err := idx.Search(query, k)
	if err != nil {
		t.Fatal(err)
	}

	// Corrupt ALL RaBitQ codes in the database
	_, err = db.Exec("UPDATE vindex_nodes SET quantized = zeroblob(128), sq_norm = 0, l1_norm = 0")
	if err != nil {
		t.Fatal(err)
	}

	// Clear cache to force reload from corrupted DB
	idx.cache.clear()
	warmCache(db, idx.cache, idx.medoid, 2)

	// Search after corruption
	resultsAfter, err := idx.Search(query, k)
	if err != nil {
		t.Fatal(err)
	}

	// Results must differ (proves RaBitQ IS used for graph traversal)
	changed := 0
	for i := range min(len(resultsBefore), len(resultsAfter)) {
		if !bytes.Equal(resultsBefore[i].ID, resultsAfter[i].ID) {
			changed++
		}
	}
	t.Logf("Results changed: %d/%d positions differ after corrupting RaBitQ codes", changed, k)
	if changed == 0 {
		t.Errorf("RaBitQ codes are NOT used: corrupting them had zero effect on search results")
	}
}

// TestAudit_Vamana_InsertDegradation measures recall degradation after
// many inserts without rebuild.
func TestAudit_Vamana_InsertDegradation(t *testing.T) {
	// WHAT: Recall after inserting 50% more vectors without rebuild.
	// WHY: Real-world usage adds vectors incrementally — must not degrade too much.
	rng := rand.New(rand.NewPCG(42, 0))
	n := 1000
	dim := 64
	k := 10

	db := newTestDB(t)
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.EfSearch = 128
	cfg.CacheCapacity = 3000
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if buildErr := idx.Build(context.Background(), iter); buildErr != nil {
		t.Fatal(buildErr)
	}

	// Recall before inserts
	recallBefore := measureRecall(idx, vecs, ids, k, 30)
	t.Logf("Recall@%d before inserts: %.1f%%", k, recallBefore*100)

	// Insert 50% more
	nInsert := n / 2
	insertVecs, insertIDs := generateVecs(rng, nInsert, dim)
	for i := range insertIDs {
		v := n + i
		insertIDs[i] = []byte{byte(v >> 24), byte(v >> 16), byte(v >> 8), byte(v)}
	}
	if insertErr := idx.Insert(insertVecs, insertIDs); insertErr != nil {
		t.Fatal(insertErr)
	}

	// Recall after inserts (on ALL vectors including inserted)
	allVecs := make([][]float32, 0, len(vecs)+len(insertVecs))
	allVecs = append(allVecs, vecs...)
	allVecs = append(allVecs, insertVecs...)
	allIDs := make([][]byte, 0, len(ids)+len(insertIDs))
	allIDs = append(allIDs, ids...)
	allIDs = append(allIDs, insertIDs...)
	recallAfter := measureRecall(idx, allVecs, allIDs, k, 30)
	t.Logf("Recall@%d after +50%% inserts: %.1f%%", k, recallAfter*100)

	degradation := recallBefore - recallAfter
	t.Logf("Degradation: %.1f%%", degradation*100)
	if recallAfter < 0.75 {
		t.Errorf("recall after inserts = %.1f%% < 75%% — too much degradation", recallAfter*100)
	}
}

// TestAudit_DynamicBruteForceThreshold verifies the brute-force/Vamana switch.
func TestAudit_DynamicBruteForceThreshold(t *testing.T) {
	// WHAT: Below BruteForceThreshold → brute-force (100% recall). Above → Vamana+RaBitQ.
	// WHY: Dynamic threshold is the core optimization for small vs large shards.
	rng := rand.New(rand.NewPCG(42, 0))
	n := 500
	dim := 64
	k := 10
	numQueries := 20

	db := newTestDB(t)
	vecs, ids := generateVecs(rng, n, dim)

	// --- Part 1: brute-force path (threshold > n) ---
	cfg := DefaultConfig()
	cfg.BruteForceThreshold = n + 1 // force brute-force
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if buildErr := idx.Build(context.Background(), iter); buildErr != nil {
		t.Fatal(buildErr)
	}

	bruteRecall := measureRecall(idx, vecs, ids, k, numQueries)
	t.Logf("Brute-force path (threshold=%d): recall@%d = %.1f%%", cfg.BruteForceThreshold, k, bruteRecall*100)
	if bruteRecall != 1.0 {
		t.Errorf("brute-force must give 100%% recall, got %.1f%%", bruteRecall*100)
	}

	// --- Part 2: Vamana path (threshold = 0) ---
	idx.cfg.BruteForceThreshold = 0 // force Vamana
	vamanaRecall := measureRecall(idx, vecs, ids, k, numQueries)
	t.Logf("Vamana path (threshold=0): recall@%d = %.1f%%", k, vamanaRecall*100)
	if vamanaRecall < 0.90 {
		t.Errorf("Vamana recall = %.1f%% < 90%%", vamanaRecall*100)
	}

	// --- Part 3: Corruption proof — brute-force ignores RaBitQ codes ---
	idx.cfg.BruteForceThreshold = n + 1 // back to brute-force
	resultsBefore, err := idx.Search(vecs[42], k)
	if err != nil {
		t.Fatal(err)
	}

	// Corrupt all codes
	_, err = db.Exec("UPDATE vindex_nodes SET quantized = zeroblob(128), sq_norm = 0, l1_norm = 0")
	if err != nil {
		t.Fatal(err)
	}
	idx.cache.clear()

	resultsAfter, err := idx.Search(vecs[42], k)
	if err != nil {
		t.Fatal(err)
	}

	// Brute-force reads raw vectors, not codes — results must be identical
	for i := range min(len(resultsBefore), len(resultsAfter)) {
		if !bytes.Equal(resultsBefore[i].ID, resultsAfter[i].ID) {
			t.Errorf("brute-force result[%d] changed after corrupting codes — should be immune", i)
		}
	}
	t.Log("Brute-force path is immune to RaBitQ code corruption (reads raw vectors)")
}

// --- Benchmarks ---

func BenchmarkRaBitQ_Encode_1024(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 0))
	dim := 1024
	centroid := make([]float32, dim)
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

func BenchmarkRaBitQ_AsymDist_1024(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 0))
	dim := 1024
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

func BenchmarkL2Exact_1024(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 0))
	dim := 1024
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

func BenchmarkVamana_Search_10K(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 0))
	n := 10000
	dim := 128
	vecs, ids := generateVecs(rng, n, dim)

	db := newTestDB2(b)
	cfg := DefaultConfig()
	cfg.CacheCapacity = 20000
	idx, err := New(db, cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		b.Fatal(err)
	}

	query := vecs[0]
	b.ResetTimer()
	for b.Loop() {
		_, _ = idx.Search(query, 10)
	}
}

func BenchmarkBruteForce_10K(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 0))
	n := 10000
	dim := 128
	vecs, _ := generateVecs(rng, n, dim)
	query := vecs[0]

	b.ResetTimer()
	for b.Loop() {
		dists := make([]float64, n)
		for i, v := range vecs {
			dists[i] = l2DistanceSquared(query, v)
		}
		sort.Float64s(dists)
	}
}

// --- helpers ---

func newTestDB2(tb testing.TB) *sql.DB {
	tb.Helper()
	dir := tb.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { db.Close() })
	return db
}

func computeCentroid(vecs [][]float32) []float32 {
	dim := len(vecs[0])
	centroid := make([]float32, dim)
	for _, v := range vecs {
		for j, val := range v {
			centroid[j] += val
		}
	}
	inv := float32(1.0 / float64(len(vecs)))
	for j := range dim {
		centroid[j] *= inv
	}
	return centroid
}

func spearmanRank(a, b []float64) float64 {
	n := len(a)
	rankA := getRanks(a)
	rankB := getRanks(b)

	var sumD2 float64
	for i := range n {
		d := rankA[i] - rankB[i]
		sumD2 += d * d
	}
	return 1 - 6*sumD2/(float64(n)*(float64(n)*float64(n)-1))
}

func getRanks(vals []float64) []float64 {
	type iv struct {
		idx int
		val float64
	}
	s := make([]iv, len(vals))
	for i, v := range vals {
		s[i] = iv{i, v}
	}
	sort.Slice(s, func(i, j int) bool { return s[i].val < s[j].val })
	ranks := make([]float64, len(vals))
	for r, v := range s {
		ranks[v.idx] = float64(r)
	}
	return ranks
}

func recallAtK(exact, approx []float64, k, candidates int) float64 {
	n := len(exact)
	type iv struct {
		idx int
		val float64
	}

	exactSorted := make([]iv, n)
	approxSorted := make([]iv, n)
	for i := range n {
		exactSorted[i] = iv{i, exact[i]}
		approxSorted[i] = iv{i, approx[i]}
	}
	sort.Slice(exactSorted, func(a, b int) bool { return exactSorted[a].val < exactSorted[b].val })
	sort.Slice(approxSorted, func(a, b int) bool { return approxSorted[a].val < approxSorted[b].val })

	trueTopK := make(map[int]bool, k)
	for i := range k {
		trueTopK[exactSorted[i].idx] = true
	}

	if candidates > n {
		candidates = n
	}
	hits := 0
	for i := range candidates {
		if trueTopK[approxSorted[i].idx] {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

func measureRecall(idx *Index, vecs [][]float32, ids [][]byte, k, numQueries int) float64 { //nolint:unparam // k is parameterized for clarity even if callers currently use same value
	n := len(vecs)
	totalRecall := 0.0

	for q := range numQueries {
		qi := q * (n / numQueries)
		query := vecs[qi]

		// Brute-force
		type idDist struct {
			idx  int
			dist float64
		}
		dists := make([]idDist, n)
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
			continue
		}
		hits := 0
		for _, r := range results {
			if trueTopK[string(r.ID)] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(k)
	}

	return totalRecall / float64(numQueries)
}
