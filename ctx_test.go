package horosvec

import (
	"context"
	"errors"
	"math/rand/v2"
	"testing"
)

func TestInsert_CanceledContext(t *testing.T) {
	const (
		n   = 100
		dim = 32
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(99, 0))
	vecs, ids := generateVecs(rng, n, dim)

	idx, err := New(db, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	countBefore := idx.Count()
	nextIDBefore := idx.nextID

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	insertVec := make([]float32, dim)
	for j := range dim {
		insertVec[j] = float32(rng.NormFloat64())
	}
	insertID := []byte("canceled-insert")

	err = idx.Insert(ctx, [][]float32{insertVec}, [][]byte{insertID})
	if err == nil {
		t.Fatal("expected error from canceled Insert")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	if got := idx.Count(); got != countBefore {
		t.Fatalf("node count changed: before=%d after=%d", countBefore, got)
	}
	if idx.nextID != nextIDBefore {
		t.Fatalf("nextID changed: before=%d after=%d", nextIDBefore, idx.nextID)
	}
}

func TestSearch_CanceledContext(t *testing.T) {
	const (
		n   = 100
		dim = 32
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(101, 0))
	vecs, ids := generateVecs(rng, n, dim)

	idx, err := New(db, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	results, err := idx.Search(ctx, vecs[0], 5)
	if err == nil {
		t.Fatal("expected error from canceled Search")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected zero results, got %d", len(results))
	}
}

func buildVamanaTestIndex(t *testing.T, n, dim int) (*Index, [][]float32) {
	t.Helper()

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(103, 0))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0 // force Vamana path (small index would otherwise brute-force)

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}
	return idx, vecs
}

func TestSearch_CanceledDuringSearch(t *testing.T) {
	const (
		n   = 100
		dim = 32
	)

	idx, vecs := buildVamanaTestIndex(t, n, dim)
	defer idx.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	idx.testDuringGreedySearch = func() { cancel() }
	defer func() { idx.testDuringGreedySearch = nil }()

	results, err := idx.Search(ctx, vecs[0], 5)
	if err == nil {
		t.Fatal("expected error from Search canceled during greedy search")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected zero results, got %d", len(results))
	}
}

func TestSearchWithRerank_CanceledDuringSearch(t *testing.T) {
	const (
		n   = 100
		dim = 32
	)

	idx, vecs := buildVamanaTestIndex(t, n, dim)
	defer idx.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	idx.testDuringGreedySearch = func() { cancel() }
	defer func() { idx.testDuringGreedySearch = nil }()

	results, err := idx.SearchWithRerank(ctx, vecs[0], 5, nil)
	if err == nil {
		t.Fatal("expected error from SearchWithRerank canceled during greedy search")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected zero results, got %d", len(results))
	}
}
