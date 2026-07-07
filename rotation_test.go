package horosvec

import (
	"bytes"
	"context"
	"database/sql"
	"math"
	"math/rand/v2"
	"path/filepath"
	"sort"
	"testing"

	_ "modernc.org/sqlite"
)

func TestRotation_Orthogonality(t *testing.T) {
	seeds := []uint64{1, 42, 0xDEADBEEF}
	dims := []int{64, 100, 128}
	roundsList := []int{1, 2}

	for _, dim := range dims {
		for _, rounds := range roundsList {
			for _, seed := range seeds {
				r := NewRotator(dim, rounds, seed)
				src := make([]float32, dim)
				for i := range dim {
					src[i] = float32(i)*0.01 - 0.5
				}
				dst := make([]float32, r.CodeDim())
				r.Rotate(src, dst)

				srcNorm := l2NormSquared(src, dim)
				dstNorm := l2NormSquared(dst, r.CodeDim())
				if math.Abs(srcNorm-dstNorm) > 1e-6 {
					t.Fatalf("dim=%d rounds=%d seed=%d: ||src||²=%g ||dst||²=%g diff=%g",
						dim, rounds, seed, srcNorm, dstNorm, math.Abs(srcNorm-dstNorm))
				}
			}
		}
	}
}

func TestRotation_Determinism(t *testing.T) {
	const dim = 100
	r1 := NewRotator(dim, 1, 99)
	r2 := NewRotator(dim, 1, 99)
	src := make([]float32, dim)
	for i := range dim {
		src[i] = float32(math.Sin(float64(i)))
	}
	out1 := make([]float32, r1.CodeDim())
	out2 := make([]float32, r2.CodeDim())
	r1.Rotate(src, out1)
	r2.Rotate(src, out2)
	for i := range r1.CodeDim() {
		if out1[i] != out2[i] {
			t.Fatalf("dim %d: out1=%g out2=%g", i, out1[i], out2[i])
		}
	}
}

func TestRotation_IdentityRoundsZero(t *testing.T) {
	const dim = 100
	r := NewRotator(dim, 0, 42)
	if r.CodeDim() != dim {
		t.Fatalf("identity codeDim=%d want %d", r.CodeDim(), dim)
	}
	src := make([]float32, dim)
	for i := range dim {
		src[i] = float32(i)
	}
	dst := make([]float32, dim)
	r.Rotate(src, dst)
	for i := range dim {
		if dst[i] != src[i] {
			t.Fatalf("identity mismatch at %d: got %g want %g", i, dst[i], src[i])
		}
	}
}

func TestRotation_NonPow2DimBuildSearch(t *testing.T) {
	const (
		n           = 500
		dim         = 100
		k           = 10
		numClusters = 10
		sigma       = 0.05
	)
	rng := rand.New(rand.NewPCG(77, 0))
	vecs, ids := generateClusterVecs(rng, n, dim, numClusters, sigma)

	db := newTestDB(t)
	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
	cfg.EfSearch = 128
	cfg.RotationRounds = 1

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	if idx.codeDim != 128 {
		t.Fatalf("codeDim=%d want 128", idx.codeDim)
	}

	numQueries := 20
	totalRecall := 0.0
	for q := range numQueries {
		query := vecs[q*n/numQueries]
		exact := bruteForceTopK(vecs, ids, query, k)
		results, err := idx.Search(context.Background(), query, k)
		if err != nil {
			t.Fatalf("query %d: %v", q, err)
		}
		hits := 0
		for _, r := range results {
			if exact[string(r.ID)] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(k)
	}
	avg := totalRecall / float64(numQueries)
	t.Logf("dim=%d padding recall@%d mean=%.2f%%", dim, k, avg*100)
	if avg < 0.90 {
		t.Errorf("recall@%d = %.2f%%, want >= 90%% on clustered dim=100", k, avg*100)
	}
}

func TestCompat_LegacyDBWithoutRotationMeta(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "legacy.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	rng := rand.New(rand.NewPCG(11, 0))
	vecs, ids := generateVecs(rng, 200, 64)

	cfg := Config{
		MaxDegree:            64,
		SearchListSize:       128,
		BuildPasses:          2,
		EfSearch:             128,
		RerankTopN:           500,
		CacheCapacity:        10000,
		BruteForceThreshold:  0,
		Alpha:                1.2,
		DriftThreshold:       0.05,
		InsertRatioThreshold: 1.0,
		MedoidStaleThreshold: 10_000,
		RotationRounds:       0,
	}
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	// Simulate pre-rotation DB: strip rotation keys written by saveGraph.
	for _, key := range []string{"rotation_seed", "rotation_rounds", "code_dim", "db_format_version"} {
		if _, err := db.Exec("DELETE FROM vindex_meta WHERE key = ?", key); err != nil {
			t.Fatal(err)
		}
	}
	idx.Close()

	reloaded, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer reloaded.Close()
	if reloaded.rotator.Rounds() != 0 {
		t.Fatalf("legacy rounds=%d want 0", reloaded.rotator.Rounds())
	}
	if reloaded.codeDim != 64 {
		t.Fatalf("legacy codeDim=%d want 64", reloaded.codeDim)
	}

	query := vecs[0]
	exact := bruteForceTopK(vecs, ids, query, 5)
	results, err := reloaded.Search(context.Background(), query, 5)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range results {
		if !exact[string(r.ID)] {
			t.Fatalf("legacy search id %v not in exact top-5", r.ID)
		}
	}
}

func TestCompat_LegacyBinaryV1Import(t *testing.T) {
	db := newTestDB(t)
	cfg := DefaultConfig()
	cfg.RotationRounds = 0
	cfg.BruteForceThreshold = 0

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(3, 0))
	vecs, ids := generateVecs(rng, 50, 32)
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if _, err := idx.ExportBinary(&buf); err != nil {
		t.Fatal(err)
	}
	// Rewrite version byte to v1 (identity import path).
	data := buf.Bytes()
	data[4] = binaryVersionV1

	imported, err := ImportBinary(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if imported.rotator.Rounds() != 0 {
		t.Fatalf("v1 import rounds=%d want 0", imported.rotator.Rounds())
	}
	query := vecs[0]
	results, err := imported.Search(context.Background(), query, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results", len(results))
	}
}

func TestDBFormatVersion_UnknownRejects(t *testing.T) {
	db := newTestDB(t)
	cfg := DefaultConfig()
	cfg.RotationRounds = 0
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(5, 0))
	vecs, ids := generateVecs(rng, 10, 16)
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	idx.Close()

	if _, err := db.Exec(
		"INSERT OR REPLACE INTO vindex_meta (key, value) VALUES (?, ?)",
		"db_format_version", serializeInt64(99),
	); err != nil {
		t.Fatal(err)
	}
	_, err = New(db, cfg)
	if err == nil {
		t.Fatal("expected error opening DB with unknown format version")
	}
}

func bruteForceTopK(vecs [][]float32, ids [][]byte, query []float32, k int) map[string]bool {
	type idDist struct {
		id   string
		dist float64
	}
	dists := make([]idDist, len(vecs))
	for i, v := range vecs {
		dists[i] = idDist{string(ids[i]), l2DistanceSquared(query, v)}
	}
	sort.Slice(dists, func(a, b int) bool { return dists[a].dist < dists[b].dist })
	topK := make(map[string]bool, k)
	for i := range k {
		topK[dists[i].id] = true
	}
	return topK
}

func BenchmarkRotate(b *testing.B) {
	const dim = 1024
	r := NewRotator(dim, 1, 42)
	src := make([]float32, dim)
	for i := range dim {
		src[i] = float32(i) * 0.001
	}
	dst := make([]float32, r.CodeDim())
	b.ResetTimer()
	for range b.N {
		r.Rotate(src, dst)
	}
}
