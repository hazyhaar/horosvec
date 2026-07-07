package horosvec

import (
	"bytes"
	"context"
	"database/sql"
	"math/rand/v2"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite"
)

// TestInsert_CommitFailureNoGhostNodes verifies that a failed commit leaves no in-memory ghosts.
func TestInsert_CommitFailureNoGhostNodes(t *testing.T) {
	const (
		n   = 200
		dim = 32
	)

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "commit_fail.db")
	dsn := "file:" + dbPath + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)"

	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	rng := rand.New(rand.NewPCG(77, 0))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.EfSearch = 64

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	nextIDBefore := idx.nextID

	insertVec := make([]float32, dim)
	for j := range dim {
		insertVec[j] = float32(rng.NormFloat64())
	}
	insertID := []byte{0xDE, 0xAD, 0xBE, 0xEF}

	idx.testBeforeInsertCommit = func(tx *sql.Tx) {
		if err := tx.Rollback(); err != nil {
			t.Errorf("sabotage rollback: %v", err)
		}
	}
	defer func() { idx.testBeforeInsertCommit = nil }()

	err = idx.Insert(context.Background(), [][]float32{insertVec}, [][]byte{insertID})
	idx.testBeforeInsertCommit = nil
	if err == nil {
		t.Fatal("expected insert to fail on commit")
	}

	if idx.nextID != nextIDBefore {
		t.Fatalf("nextID = %d, want %d after failed commit", idx.nextID, nextIDBefore)
	}

	results, err := idx.Search(context.Background(), insertVec, 10)
	if err != nil {
		t.Fatalf("search after failed insert: %v", err)
	}
	for _, r := range results {
		if bytes.Equal(r.ID, insertID) {
			t.Fatal("ghost node: failed insert id returned by Search")
		}
	}

	if err := idx.Insert(context.Background(), [][]float32{insertVec}, [][]byte{insertID}); err != nil {
		t.Fatalf("insert after failed commit: %v", err)
	}

	results, err = idx.Search(context.Background(), insertVec, 10)
	if err != nil {
		t.Fatalf("search after successful insert: %v", err)
	}
	found := false
	for _, r := range results {
		if bytes.Equal(r.ID, insertID) {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("vector not found after successful insert following commit failure")
	}
}

// TestInsert_IntraBatchVisibility verifies multi-vector inserts see prior batch nodes and link back.
func TestInsert_IntraBatchVisibility(t *testing.T) {
	const (
		n       = 150
		dim     = 32
		nInsert = 5
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(88, 0))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.EfSearch = 64
	cfg.MaxDegree = 16

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	firstBatchID := idx.nextID

	// Tight cluster so intra-batch nodes become mutual nearest neighbors.
	base := make([]float32, dim)
	for j := range dim {
		base[j] = float32(rng.NormFloat64())
	}
	insertVecs := make([][]float32, nInsert)
	insertIDs := make([][]byte, nInsert)
	for i := range nInsert {
		vec := make([]float32, dim)
		copy(vec, base)
		vec[i%dim] += float32(i+1) * 1e-4
		insertVecs[i] = vec
		v := n + i
		insertIDs[i] = []byte{byte(v >> 24), byte(v >> 16), byte(v >> 8), byte(v)}
	}

	if err := idx.Insert(context.Background(), insertVecs, insertIDs); err != nil {
		t.Fatal(err)
	}

	for i, vec := range insertVecs {
		results, err := idx.Search(context.Background(), vec, 10)
		if err != nil {
			t.Fatalf("search insert vec %d: %v", i, err)
		}
		found := false
		for _, r := range results {
			if bytes.Equal(r.ID, insertIDs[i]) {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("inserted vec %d not found in top-10", i)
		}
	}

	// At least one batch node must have a reverse edge to another node from the same batch.
	hasIntraBatchEdge := false
	for nodeID := firstBatchID; nodeID < firstBatchID+int64(nInsert); nodeID++ {
		var neighborsBlob []byte
		err := db.QueryRow("SELECT neighbors FROM vindex_nodes WHERE node_id = ?", nodeID).Scan(&neighborsBlob)
		if err != nil {
			t.Fatalf("load neighbors for node %d: %v", nodeID, err)
		}
		neighbors := deserializeInt64s(neighborsBlob)
		for _, nbr := range neighbors {
			if nbr >= firstBatchID && nbr < firstBatchID+int64(nInsert) && nbr != nodeID {
				hasIntraBatchEdge = true
				break
			}
		}
		if hasIntraBatchEdge {
			break
		}
	}
	if !hasIntraBatchEdge {
		t.Error("expected at least one intra-batch neighbor edge among inserted nodes")
	}
}
