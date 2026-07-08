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

func TestHotPlane_ConsistencyWithSQL(t *testing.T) {
	const (
		n   = 500
		dim = 128
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(901, 0))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
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
	if idx.plane == nil {
		t.Fatal("plane nil after Build")
	}
	if idx.plane.n != n {
		t.Fatalf("plane.n = %d, want %d", idx.plane.n, n)
	}

	picks := make(map[int]bool)
	for len(picks) < 50 {
		picks[rng.IntN(n)] = true
	}

	ctx := context.Background()
	for nodeID := range picks {
		var extID, neighborsBlob, code []byte
		var sqNorm, l1Norm float64
		err := db.QueryRowContext(ctx,
			"SELECT ext_id, neighbors, quantized, sq_norm, l1_norm FROM vindex_nodes WHERE node_id = ?",
			nodeID,
		).Scan(&extID, &neighborsBlob, &code, &sqNorm, &l1Norm)
		if err != nil {
			t.Fatalf("sql node %d: %v", nodeID, err)
		}

		planeCode := idx.plane.codeAt(nodeID)
		if !bytes.Equal(planeCode, code) {
			t.Fatalf("node %d: code mismatch", nodeID)
		}
		if idx.plane.sqNorm[nodeID] != sqNorm {
			t.Fatalf("node %d: sqNorm plane=%v sql=%v", nodeID, idx.plane.sqNorm[nodeID], sqNorm)
		}
		if idx.plane.l1Norm[nodeID] != l1Norm {
			t.Fatalf("node %d: l1Norm plane=%v sql=%v", nodeID, idx.plane.l1Norm[nodeID], l1Norm)
		}
		if !bytes.Equal(idx.plane.extIDAt(nodeID), extID) {
			t.Fatalf("node %d: extID mismatch", nodeID)
		}

		sqlNbrs := deserializeInt64s(neighborsBlob)
		planeNbrs := idx.plane.neighborsAt(nodeID)
		if len(planeNbrs) != len(sqlNbrs) {
			t.Fatalf("node %d: neighbor count plane=%d sql=%d", nodeID, len(planeNbrs), len(sqlNbrs))
		}
		for i := range sqlNbrs {
			if int64(planeNbrs[i]) != sqlNbrs[i] {
				t.Fatalf("node %d neighbor[%d]: plane=%d sql=%d", nodeID, i, planeNbrs[i], sqlNbrs[i])
			}
		}
	}
}

func TestHotPlane_InsertExtendsPlaneAndPatch(t *testing.T) {
	const (
		n       = 500
		dim     = 64
		nInsert = 40
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(902, 0))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
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
	planeNBefore := idx.plane.n

	insertVecs, insertIDs := generateVecs(rng, nInsert, dim)
	for i := range insertIDs {
		v := n + i
		insertIDs[i] = []byte{byte(v >> 24), byte(v >> 16), byte(v >> 8), byte(v)}
	}

	if err := idx.Insert(context.Background(), insertVecs, insertIDs); err != nil {
		t.Fatal(err)
	}

	if idx.plane.n != planeNBefore+nInsert {
		t.Fatalf("plane.n = %d, want %d", idx.plane.n, planeNBefore+nInsert)
	}

	for i := range nInsert {
		nodeID := planeNBefore + i
		if !bytes.Equal(idx.plane.extIDAt(nodeID), insertIDs[i]) {
			t.Fatalf("inserted node %d extID mismatch", nodeID)
		}
	}

	// planePatch holds re-cabled neighbor lists for existing nodes (reverse edges).
	for id, patch := range idx.planePatch {
		var neighborsBlob []byte
		err := db.QueryRow("SELECT neighbors FROM vindex_nodes WHERE node_id = ?", id).Scan(&neighborsBlob)
		if err != nil {
			t.Fatalf("sql neighbors for patched node %d: %v", id, err)
		}
		sqlNbrs := deserializeInt64s(neighborsBlob)
		if len(patch) != len(sqlNbrs) {
			t.Fatalf("patch node %d: len %d != sql %d", id, len(patch), len(sqlNbrs))
		}
		for i := range sqlNbrs {
			if int64(patch[i]) != sqlNbrs[i] {
				t.Fatalf("patch node %d neighbor[%d]: patch=%d sql=%d", id, i, patch[i], sqlNbrs[i])
			}
		}
	}

	found := 0
	for i, vec := range insertVecs {
		results, err := idx.Search(context.Background(), vec, 10)
		if err != nil {
			t.Fatalf("search insert %d: %v", i, err)
		}
		for _, r := range results {
			if bytes.Equal(r.ID, insertIDs[i]) {
				found++
				break
			}
		}
	}
	findRate := float64(found) / float64(nInsert)
	t.Logf("find rate via plane path: %.2f%% (%d/%d)", findRate*100, found, nInsert)
	if findRate < 0.80 {
		t.Errorf("find rate = %.2f%%, want >= 80%%", findRate*100)
	}
}

func TestHotPlane_RecallClustersUnchanged(t *testing.T) {
	rng := rand.New(rand.NewPCG(recallMeasureSeed, 1))
	baseVecs, baseIDs := generateClusterVecs(rng, recallMeasureN, recallMeasureDim, recallMeasureNumClusters, recallMeasureClusterSigma)
	queries := generateClusterQueries(rng, recallMeasureNumQueries, recallMeasureDim, recallMeasureNumClusters, recallMeasureClusterSigma)

	exactTopKs := make([]map[string]bool, len(queries))
	for q, query := range queries {
		exactTopKs[q] = exactTopKSet(baseVecs, baseIDs, query, recallMeasureK)
	}

	cfg := recallMeasureConfig(false)
	stats := measureRecall(t, baseVecs, baseIDs, queries, exactTopKs, cfg)
	if stats.mean < recallMeasureFloorClusters {
		t.Errorf("gaussian_clusters recall@%d mean=%.4f, want >= %.2f (plane must not change results)", recallMeasureK, stats.mean, recallMeasureFloorClusters)
	}
}

func TestHotPlane_CommitFailureNoPlaneExtend(t *testing.T) {
	const (
		n   = 100
		dim = 32
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(903, 0))
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
	planeNBefore := idx.plane.n
	patchBefore := len(idx.planePatch)

	insertVec := make([]float32, dim)
	insertID := []byte{0xCA, 0xFE}

	idx.testBeforeInsertCommit = func(tx *sql.Tx) {
		_ = tx.Rollback()
	}
	defer func() { idx.testBeforeInsertCommit = nil }()

	err = idx.Insert(context.Background(), [][]float32{insertVec}, [][]byte{insertID})
	if err == nil {
		t.Fatal("expected commit failure")
	}

	if idx.plane.n != planeNBefore {
		t.Fatalf("plane.n = %d after failed commit, want %d", idx.plane.n, planeNBefore)
	}
	if len(idx.planePatch) != patchBefore {
		t.Fatalf("planePatch len = %d after failed commit, want %d", len(idx.planePatch), patchBefore)
	}
}

const benchPlaneN = 2000
const benchPlaneDim = 128

func benchmarkSearchIndex(b *testing.B, usePlane bool) {
	dbPath := filepath.Join(b.TempDir(), "bench.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		b.Fatal(err)
	}
	defer db.Close()

	rng := rand.New(rand.NewPCG(904, 0))
	vecs, ids := generateVecs(rng, benchPlaneN, benchPlaneDim)

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
	cfg.EfSearch = 128
	cfg.CacheCapacity = 50000

	idx, err := New(db, cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		b.Fatal(err)
	}
	if !usePlane {
		idx.plane = nil
		idx.planePatch = nil
	}

	query := vecs[0]
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := idx.Search(ctx, query, 10); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSearchPlane(b *testing.B) {
	benchmarkSearchIndex(b, true)
}

func BenchmarkSearchNoPlane(b *testing.B) {
	benchmarkSearchIndex(b, false)
}
