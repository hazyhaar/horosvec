package horosvec

import (
	"context"
	"database/sql"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"

	_ "modernc.org/sqlite"
)

// sliceIterator implements VectorIterator over in-memory slices.
type sliceIterator struct {
	vecs [][]float32
	ids  [][]byte
	pos  int
}

func (s *sliceIterator) Next() ([]byte, []float32, bool) {
	if s.pos >= len(s.vecs) {
		return nil, nil, false
	}
	id := s.ids[s.pos]
	vec := s.vecs[s.pos]
	s.pos++
	return id, vec, true
}

func (s *sliceIterator) Reset() error {
	s.pos = 0
	return nil
}

func newTestDB(t *testing.T) *sql.DB {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.Exec("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000; PRAGMA synchronous=NORMAL")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func generateVecs(rng *rand.Rand, n, dim int) ([][]float32, [][]byte) {
	vecs := make([][]float32, n)
	ids := make([][]byte, n)
	for i := range n {
		vecs[i] = make([]float32, dim)
		for j := range dim {
			vecs[i][j] = float32(rng.NormFloat64())
		}
		ids[i] = []byte{byte(i >> 24), byte(i >> 16), byte(i >> 8), byte(i)}
	}
	return vecs, ids
}

func TestBuildAndSearchRecall(t *testing.T) {
	const (
		n   = 10000
		dim = 128
		k   = 10
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(42, 0))

	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.EfSearch = 128
	cfg.CacheCapacity = 20000

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	if c := idx.Count(); c != n {
		t.Fatalf("count = %d, want %d", c, n)
	}

	// Test recall@10 over multiple queries
	numQueries := 50
	totalRecall := 0.0

	for q := range numQueries {
		query := vecs[q*100] // pick every 100th vector as query

		// Compute exact top-k
		type idDist struct {
			idx  int
			dist float64
		}
		exactDists := make([]idDist, n)
		for i, v := range vecs {
			var d float64
			for j := range dim {
				diff := float64(query[j]) - float64(v[j])
				d += diff * diff
			}
			exactDists[i] = idDist{i, d}
		}
		sort.Slice(exactDists, func(a, b int) bool {
			return exactDists[a].dist < exactDists[b].dist
		})

		trueTopK := make(map[string]bool, k)
		for i := range k {
			id := ids[exactDists[i].idx]
			trueTopK[string(id)] = true
		}

		// Search with the index
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
	t.Logf("Average recall@%d over %d queries: %.2f%%", k, numQueries, avgRecall*100)
	if avgRecall < 0.85 {
		t.Errorf("recall@%d = %.2f%%, want >= 85%%", k, avgRecall*100)
	}
}

func TestInsertAndFind(t *testing.T) {
	const (
		n      = 1000
		dim    = 128
		nInsert = 200
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(42, 0))

	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.EfSearch = 128

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	// Generate and insert new vectors
	insertVecs, insertIDs := generateVecs(rng, nInsert, dim)
	// Use distinct IDs for inserts
	for i := range insertIDs {
		v := n + i
		insertIDs[i] = []byte{byte(v >> 24), byte(v >> 16), byte(v >> 8), byte(v)}
	}

	if err := idx.Insert(insertVecs, insertIDs); err != nil {
		t.Fatal(err)
	}

	if c := idx.Count(); c != n+nInsert {
		t.Fatalf("count after insert = %d, want %d", c, n+nInsert)
	}

	// Search for the inserted vectors — they should be findable
	found := 0
	for i, vec := range insertVecs {
		results, err := idx.Search(vec, 10)
		if err != nil {
			t.Fatalf("search for inserted vec %d: %v", i, err)
		}
		for _, r := range results {
			if string(r.ID) == string(insertIDs[i]) {
				found++
				break
			}
		}
	}

	findRate := float64(found) / float64(nInsert)
	t.Logf("Find rate for inserted vectors: %.2f%% (%d/%d)", findRate*100, found, nInsert)
	if findRate < 0.80 {
		t.Errorf("find rate = %.2f%%, want >= 80%%", findRate*100)
	}
}

func TestPersistenceRoundTrip(t *testing.T) {
	const (
		n   = 500
		dim = 64
	)

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "persist.db")

	rng := rand.New(rand.NewPCG(42, 0))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.EfSearch = 64

	// Build and close
	{
		db, err := sql.Open("sqlite", dbPath)
		if err != nil {
			t.Fatal(err)
		}
		db.Exec("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000")

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
	}

	// Reopen and search
	{
		db, err := sql.Open("sqlite", dbPath)
		if err != nil {
			t.Fatal(err)
		}
		defer db.Close()
		db.Exec("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000")

		idx, err := New(db, cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer idx.Close()

		if c := idx.Count(); c != n {
			t.Fatalf("count after reload = %d, want %d", c, n)
		}

		// Search should still work
		results, err := idx.Search(vecs[0], 5)
		if err != nil {
			t.Fatal(err)
		}

		if len(results) == 0 {
			t.Fatal("no results after persistence round-trip")
		}

		// The query vector itself should be in the top results
		foundSelf := false
		for _, r := range results {
			if string(r.ID) == string(ids[0]) {
				foundSelf = true
				break
			}
		}
		if !foundSelf {
			t.Error("query vector not found in top-5 after persistence round-trip")
		}
	}
}

func TestConcurrentSearch(t *testing.T) {
	const (
		n   = 2000
		dim = 64
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(42, 0))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	// Run concurrent searches
	const numGoroutines = 10
	const queriesPerGoroutine = 20
	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines*queriesPerGoroutine)

	for g := range numGoroutines {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			for q := range queriesPerGoroutine {
				queryIdx := (gid*queriesPerGoroutine + q) % n
				results, err := idx.Search(vecs[queryIdx], 5)
				if err != nil {
					errors <- err
					return
				}
				if len(results) == 0 {
					errors <- fmt.Errorf("goroutine %d query %d: no results", gid, q)
					return
				}
			}
		}(g)
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Error(err)
	}
}

func TestNeedsRebuild(t *testing.T) {
	const (
		n   = 500
		dim = 64
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(42, 0))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.InsertRatioThreshold = 0.30

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	if idx.NeedsRebuild() {
		t.Error("should not need rebuild right after build")
	}

	// Insert 31% more vectors to trigger rebuild
	nInsert := int(float64(n) * 0.31)
	insertVecs, insertIDs := generateVecs(rng, nInsert, dim)
	for i := range insertIDs {
		v := n + i
		insertIDs[i] = []byte{byte(v >> 24), byte(v >> 16), byte(v >> 8), byte(v)}
	}
	if err := idx.Insert(insertVecs, insertIDs); err != nil {
		t.Fatal(err)
	}

	if !idx.NeedsRebuild() {
		t.Error("should need rebuild after inserting 31% more vectors")
	}
}

func TestEmptyIndex(t *testing.T) {
	db := newTestDB(t)
	cfg := DefaultConfig()

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	// Search on empty index
	_, err = idx.Search([]float32{1, 2, 3}, 5)
	if err == nil {
		t.Error("expected error searching empty index")
	}

	// Count on empty index
	if c := idx.Count(); c != 0 {
		t.Errorf("count = %d, want 0", c)
	}
}

func BenchmarkBuild10K(b *testing.B) {
	const (
		n   = 10000
		dim = 128
	)

	rng := rand.New(rand.NewPCG(42, 0))
	vecs, ids := generateVecs(rng, n, dim)

	for b.Loop() {
		dir, _ := os.MkdirTemp("", "horosvec-bench-*")
		dbPath := filepath.Join(dir, "bench.db")
		db, _ := sql.Open("sqlite", dbPath)
		db.Exec("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000; PRAGMA synchronous=NORMAL")

		cfg := DefaultConfig()
		idx, _ := New(db, cfg)

		iter := &sliceIterator{vecs: vecs, ids: ids}
		idx.Build(context.Background(), iter)

		idx.Close()
		db.Close()
		os.RemoveAll(dir)
	}
}

func BenchmarkSearch10K(b *testing.B) {
	const (
		n   = 10000
		dim = 128
	)

	rng := rand.New(rand.NewPCG(42, 0))
	vecs, ids := generateVecs(rng, n, dim)

	dir, _ := os.MkdirTemp("", "horosvec-bench-*")
	defer os.RemoveAll(dir)
	dbPath := filepath.Join(dir, "bench.db")
	db, _ := sql.Open("sqlite", dbPath)
	defer db.Close()
	db.Exec("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000; PRAGMA synchronous=NORMAL")

	cfg := DefaultConfig()
	cfg.CacheCapacity = 20000
	idx, _ := New(db, cfg)
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	idx.Build(context.Background(), iter)

	query := vecs[0]

	b.ResetTimer()
	for b.Loop() {
		idx.Search(query, 10)
	}
}
