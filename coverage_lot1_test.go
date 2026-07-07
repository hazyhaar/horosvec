package horosvec

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

func exactTopK(vecs [][]float32, ids [][]byte, query []float32, topK int) []Result {
	type idDist struct {
		id   []byte
		dist float64
	}
	dists := make([]idDist, len(vecs))
	for i, v := range vecs {
		dists[i] = idDist{id: ids[i], dist: l2DistanceSquared(query, v)}
	}
	sort.Slice(dists, func(a, b int) bool { return dists[a].dist < dists[b].dist })
	if topK > len(dists) {
		topK = len(dists)
	}
	out := make([]Result, topK)
	for i := range topK {
		out[i] = Result{ID: dists[i].id, Score: dists[i].dist}
	}
	return out
}

func TestSearchWithRerank_NominalExactOrder(t *testing.T) {
	const (
		n    = 80
		dim  = 16
		topK = 5
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(201, 0))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
	cfg.EfSearch = 64
	cfg.RerankTopN = 30

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	query := vecs[0]
	want := exactTopK(vecs, ids, query, topK)

	reranker := func(candidateIDs [][]byte) ([][]float32, error) {
		out := make([][]float32, len(candidateIDs))
		for i, id := range candidateIDs {
			var vecBlob []byte
			if err := db.QueryRow("SELECT vector FROM vindex_nodes WHERE ext_id = ?", id).Scan(&vecBlob); err != nil {
				return nil, fmt.Errorf("reload vector: %w", err)
			}
			out[i] = deserializeFloat32s(vecBlob)
		}
		return out, nil
	}

	results, err := idx.SearchWithRerank(context.Background(), query, topK, reranker)
	if err != nil {
		t.Fatalf("SearchWithRerank: %v", err)
	}
	if len(results) != topK {
		t.Fatalf("got %d results, want topK=%d", len(results), topK)
	}

	for i, r := range results {
		if r.Score != want[i].Score {
			t.Fatalf("result %d score=%f want exact %f", i, r.Score, want[i].Score)
		}
		if string(r.ID) != string(want[i].ID) {
			t.Fatalf("result %d id mismatch: got %v want %v", i, r.ID, want[i].ID)
		}
	}
	for i := 1; i < len(results); i++ {
		if results[i].Score < results[i-1].Score {
			t.Fatalf("results not sorted by exact distance at %d", i)
		}
	}
}

func TestSearchWithRerank_RerankerError(t *testing.T) {
	const (
		n    = 50
		dim  = 16
		topK = 3
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(202, 0))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	rerankErr := errors.New("reranker failed")
	reranker := func([][]byte) ([][]float32, error) {
		return nil, rerankErr
	}

	results, err := idx.SearchWithRerank(context.Background(), vecs[0], topK, reranker)
	if err != nil {
		t.Fatalf("expected graceful fallback without error, got %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected fallback candidates from initial search")
	}
	if len(results) > topK {
		t.Fatalf("got %d results, want at most topK=%d", len(results), topK)
	}
}

func TestBruteForceSQLite_NoFlatVecs(t *testing.T) {
	const (
		n    = 40
		dim  = 8
		topK = 5
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(203, 0))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 1000

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	if idx.flatVecs == nil {
		t.Fatal("expected flatVecs populated after Build")
	}

	idx.flatVecs = nil
	idx.flatIDs = nil

	query := vecs[7]
	want := exactTopK(vecs, ids, query, topK)

	results, err := idx.Search(context.Background(), query, topK)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != topK {
		t.Fatalf("got %d results, want %d", len(results), topK)
	}

	for i, r := range results {
		if r.Score != want[i].Score {
			t.Fatalf("result %d score=%f want %f", i, r.Score, want[i].Score)
		}
		if string(r.ID) != string(want[i].ID) {
			t.Fatalf("result %d id mismatch", i)
		}
	}
}

func TestLRUCache_EvictionAndReload(t *testing.T) {
	const (
		n   = 20
		dim = 8
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(204, 0))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.CacheCapacity = 2
	cfg.BruteForceThreshold = 1000

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	if got := len(idx.cache.items); got != cfg.CacheCapacity {
		t.Fatalf("cache size=%d want capacity=%d", got, cfg.CacheCapacity)
	}

	ctx := context.Background()
	evictedID := int64(0)
	node, err := loadNode(ctx, db, idx.cache, evictedID)
	if err != nil {
		t.Fatalf("reload evicted node %d: %v", evictedID, err)
	}
	if string(node.extID) != string(ids[evictedID]) {
		t.Fatalf("reloaded node extID=%v want %v", node.extID, ids[evictedID])
	}
	if got := len(idx.cache.items); got > cfg.CacheCapacity {
		t.Fatalf("cache size=%d exceeds capacity=%d after reload", got, cfg.CacheCapacity)
	}

	for id := int64(1); id < int64(n); id++ {
		if _, err := loadNode(ctx, db, idx.cache, id); err != nil {
			t.Fatalf("load node %d: %v", id, err)
		}
	}
	if got := len(idx.cache.items); got != cfg.CacheCapacity {
		t.Fatalf("cache size=%d want capacity=%d after churn", got, cfg.CacheCapacity)
	}

	results, err := idx.Search(ctx, vecs[3], 3)
	if err != nil {
		t.Fatalf("Search after eviction: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected search results after LRU eviction")
	}
}

func TestCentroidDrift_NeedsRebuildAndReset(t *testing.T) {
	const dim = 4
	threshold := 0.1

	ct := NewCentroidTracker(dim, threshold, 1.0)
	buildCentroid := []float32{1, 0, 0, 0}
	ct.SetBuildCentroid(buildCentroid, 100)
	ct.SetCentroid(buildCentroid, 100)

	if ct.NeedsRebuild() {
		t.Fatal("should not need rebuild before drift")
	}
	if ct.DriftRatio() != 0 {
		t.Fatalf("initial drift=%f want 0", ct.DriftRatio())
	}

	for i := 0; i < 50; i++ {
		ct.Add([]float32{5, 0, 0, 0})
	}

	drift := ct.DriftRatio()
	if drift <= threshold {
		t.Fatalf("drift=%f want > threshold %f", drift, threshold)
	}
	if !ct.NeedsRebuild() {
		t.Fatal("expected NeedsRebuild after centroid drift")
	}

	cur := ct.Current()
	if cur[0] <= buildCentroid[0] {
		t.Fatalf("current centroid[0]=%f should move beyond build centroid %f", cur[0], buildCentroid[0])
	}

	ct.Reset()
	if ct.NeedsRebuild() {
		t.Fatal("Reset should clear rebuild need")
	}
	if ct.DriftRatio() != 0 {
		t.Fatalf("drift after reset=%f want 0", ct.DriftRatio())
	}
	if ct.Current()[0] != 0 {
		t.Fatalf("current after reset=%v want zeros", ct.Current())
	}
}

func TestRabitqDistance_OrderingVsExactL2(t *testing.T) {
	centroid := []float32{0, 0, 0, 0}
	enc := NewEncoder(centroid)

	query := []float32{0, 0, 0, 0}
	type storedVec struct {
		name string
		vec  []float32
	}
	stored := []storedVec{
		{name: "near", vec: []float32{1, 0, 0, 0}},
		{name: "mid", vec: []float32{5, 0, 0, 0}},
		{name: "far", vec: []float32{10, 0, 0, 0}},
	}

	type scored struct {
		name  string
		exact float64
		asym  float64
	}
	scores := make([]scored, len(stored))
	for i, s := range stored {
		code, sqNorm, l1Norm := enc.Encode(s.vec)
		scores[i] = scored{
			name:  s.name,
			exact: l2DistanceSquared(query, s.vec),
			asym:  rabitqDistanceAsym(query, centroid, code, sqNorm, l1Norm),
		}
	}

	if !(scores[0].exact < scores[1].exact && scores[1].exact < scores[2].exact) {
		t.Fatalf("exact ordering not separated: %+v", scores)
	}
	if !(scores[0].asym < scores[1].asym && scores[1].asym < scores[2].asym) {
		t.Fatalf("asymmetric RaBitQ ordering mismatch: %+v", scores)
	}

	codeA, sqA, _ := enc.Encode(stored[0].vec)
	codeB, sqB, _ := enc.Encode(stored[2].vec)
	symAB := rabitqDistance(codeA, codeB, sqA, sqB)
	symBA := rabitqDistance(codeB, codeA, sqB, sqA)
	if math.Abs(symAB-symBA) > 1e-6 {
		t.Fatalf("symmetric distance not consistent: %f vs %f", symAB, symBA)
	}
	if symAB <= 0 {
		t.Fatalf("symmetric distance should be positive, got %f", symAB)
	}
}
