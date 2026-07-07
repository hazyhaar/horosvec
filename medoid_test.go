package horosvec

import (
	"bytes"
	"context"
	"log/slog"
	"math/rand/v2"
	"strings"
	"testing"
)

// TestMedoidInvalidationOnLoad: DB has a valid graph but the stored medoid points to
// a non-existent node_id. New() must recompute and persist a valid medoid.
func TestMedoidInvalidationOnLoad(t *testing.T) {
	db := newTestDB(t)

	// Build a small index.
	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0 // force Vamana path
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(1, 0))
	vecs, ids := generateVecs(rng, 200, 16)
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	idx.Close()

	// Corrupt: write an invalid medoid (node_id 999999 does not exist).
	if _, err := db.Exec(`INSERT OR REPLACE INTO vindex_meta (key, value) VALUES ('medoid', ?)`,
		serializeInt64(999999)); err != nil {
		t.Fatal(err)
	}

	// Capture slog output to verify WARN is emitted.
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)

	// Reopen — must recompute medoid.
	idx2, err := New(db, cfg)
	if err != nil {
		t.Fatalf("New with invalid medoid: %v", err)
	}
	defer idx2.Close()

	if !strings.Contains(buf.String(), "medoid invalid") {
		t.Error("expected WARN log 'medoid invalid' during Load, got: " + buf.String())
	}

	// New medoid must exist in vindex_nodes.
	ok, err := validateMedoid(db, idx2.medoid)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatalf("recomputed medoid %d not found in vindex_nodes", idx2.medoid)
	}

	// Search must return results (not empty) after recompute.
	query := vecs[0]
	results, err := idx2.Search(context.Background(), query, 5)
	if err != nil {
		t.Fatalf("search after medoid recompute: %v", err)
	}
	if len(results) == 0 {
		t.Error("expected non-empty results after medoid recompute")
	}
}

// TestMedoidStaleAfterMassInsert: Build 1000 nodes, insert 1500 more (>50% new).
// After insert, medoid must have been recomputed (insertedSinceMedoid resets to 0).
func TestMedoidStaleAfterMassInsert(t *testing.T) {
	db := newTestDB(t)

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
	cfg.MedoidStaleThreshold = 100_000 // disable periodic; rely on 50% guard

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	rng := rand.New(rand.NewPCG(2, 0))
	buildVecs, buildIDs := generateVecs(rng, 1000, 16)
	if err := idx.Build(context.Background(), &sliceIterator{vecs: buildVecs, ids: buildIDs}); err != nil {
		t.Fatal(err)
	}

	// Capture medoid before insert.
	medoidBefore := idx.medoid

	// Insert 1500 new nodes — exceeds 50% of 2500 total.
	insertVecs, insertIDs := generateVecs(rng, 1500, 16)
	if err := idx.Insert(context.Background(), insertVecs, insertIDs); err != nil {
		t.Fatal(err)
	}

	// insertedSinceMedoid must have reset to 0, meaning recompute happened.
	if idx.insertedSinceMedoid != 0 {
		t.Errorf("expected insertedSinceMedoid=0 after recompute, got %d", idx.insertedSinceMedoid)
	}
	if idx.medoid == medoidBefore {
		// Not necessarily different, but recompute should have run; log for info.
		t.Log("medoid unchanged after mass insert (valid: centroid may not have moved)")
	}

	// New medoid must be valid.
	ok, err := validateMedoid(db, idx.medoid)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatalf("medoid %d invalid after mass insert recompute", idx.medoid)
	}

	// Search must return results.
	query := buildVecs[0]
	results, err := idx.Search(context.Background(), query, 5)
	if err != nil {
		t.Fatalf("search after mass insert: %v", err)
	}
	if len(results) == 0 {
		t.Error("expected results after mass insert")
	}
}

// TestNoRecomputeIfValid: Build 1000 nodes, insert 100 (<50%). Medoid must stay untouched.
func TestNoRecomputeIfValid(t *testing.T) {
	db := newTestDB(t)

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
	cfg.MedoidStaleThreshold = 100_000 // disable periodic threshold

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	rng := rand.New(rand.NewPCG(3, 0))
	buildVecs, buildIDs := generateVecs(rng, 1000, 16)
	if err := idx.Build(context.Background(), &sliceIterator{vecs: buildVecs, ids: buildIDs}); err != nil {
		t.Fatal(err)
	}

	medoidBefore := idx.medoid

	insertVecs, insertIDs := generateVecs(rng, 100, 16)
	if err := idx.Insert(context.Background(), insertVecs, insertIDs); err != nil {
		t.Fatal(err)
	}

	// 100 inserts out of 1100 total = 9% < 50%: no recompute expected.
	if idx.insertedSinceMedoid == 0 {
		t.Error("insertedSinceMedoid should not be 0 (no recompute should have occurred)")
	}
	if idx.medoid != medoidBefore {
		t.Logf("medoid changed from %d to %d without trigger (unexpected but not fatal)", medoidBefore, idx.medoid)
	}
}
