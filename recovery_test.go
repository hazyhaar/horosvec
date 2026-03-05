package horosvec

import (
	"context"
	"database/sql"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite"
)

func TestRecovery_SearchAfterCorruptedIndex(t *testing.T) {
	// Build an index, corrupt the vindex_nodes table, Search should error, not panic.
	db := newTestDB(t)
	idx, err := New(db, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	rng := rand.New(rand.NewPCG(42, 0))
	vecs, ids := generateVecs(rng, 100, 32)

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	// Corrupt: delete all nodes
	if _, err := db.Exec(`DELETE FROM vindex_nodes`); err != nil {
		t.Fatal(err)
	}

	query := make([]float32, 32)
	for i := range query {
		query[i] = float32(rng.NormFloat64())
	}

	// Search should not panic — may return empty results or error
	results, err := idx.Search(query, 10)
	_ = err
	_ = results
}

func TestRecovery_NewOnCorruptedFile(t *testing.T) {
	// Write random bytes to a DB file, New() should error cleanly.
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "corrupt.db")

	if err := os.WriteFile(dbPath, []byte("not a sqlite database!!!"), 0o644); err != nil {
		t.Fatal(err)
	}

	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return // can't even open — that's fine
	}
	defer db.Close()

	_, err = New(db, DefaultConfig())
	if err == nil {
		t.Log("Warning: New succeeded on corrupt DB (may overwrite)")
	}
	// Key: no panic
}

func TestRecovery_TruncatedDBFile(t *testing.T) {
	// Build index, close DB, truncate file, reopen → should error cleanly.
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "trunc.db")

	// Build valid index
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatal(err)
	}

	idx, err := New(db, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(42, 0))
	vecs, ids := generateVecs(rng, 50, 16)

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}
	idx.Close()
	db.Close()

	// Truncate the DB file
	if err := os.Truncate(dbPath, 16); err != nil {
		t.Fatal(err)
	}

	// Reopen
	db2, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return // clean error
	}
	defer db2.Close()

	_, err = New(db2, DefaultConfig())
	if err != nil {
		return // expected: schema init fails on truncated file
	}
	// If it somehow succeeds, search should handle it
	t.Log("Warning: New succeeded on truncated DB")
}

func TestRecovery_InsertAfterDeletedMeta(t *testing.T) {
	// Build index, delete vindex_meta rows, Insert should error, not panic.
	db := newTestDB(t)
	idx, err := New(db, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	rng := rand.New(rand.NewPCG(42, 0))
	vecs, ids := generateVecs(rng, 50, 16)

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	// Corrupt: delete meta (dimension, medoid, etc.)
	if _, err := db.Exec(`DELETE FROM vindex_meta`); err != nil {
		t.Fatal(err)
	}

	// Try to insert new vectors — should handle missing meta gracefully
	newVecs, newIDs := generateVecs(rng, 10, 16)
	err = idx.Insert(newVecs, newIDs)
	_ = err // may error, must not panic
}

func TestRecovery_SearchOnEmptyIndex(t *testing.T) {
	// New index with no data, Search should return an error or empty results, not panic.
	db := newTestDB(t)
	idx, err := New(db, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	query := make([]float32, 32)
	results, err := idx.Search(query, 10)
	// "index not built" error is the expected behavior for an empty index.
	if err != nil {
		return
	}
	if len(results) != 0 {
		t.Fatalf("expected 0 results on empty index, got %d", len(results))
	}
}
