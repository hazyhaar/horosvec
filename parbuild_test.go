package horosvec

import (
	"context"
	"database/sql"
	"errors"
	"math/rand/v2"
	"sort"
	"testing"
	"time"

	_ "modernc.org/sqlite"
)

func cloneGraphNodes(src []graphNode) []graphNode {
	dst := make([]graphNode, len(src))
	for i, n := range src {
		dst[i] = graphNode{
			id:     n.id,
			extID:  append([]byte(nil), n.extID...),
			vec:    append([]float32(nil), n.vec...),
			code:   append([]byte(nil), n.code...),
			sqNorm: n.sqNorm,
			l1Norm: n.l1Norm,
		}
	}
	return dst
}

func makeParBuildNodes(rng *rand.Rand, n, dim int) []graphNode {
	vecs, ids := generateClusterVecs(rng, n, dim, recallMeasureNumClusters, recallMeasureClusterSigma)
	nodes := make([]graphNode, n)
	for i := range n {
		nodes[i] = graphNode{
			id:    int64(i),
			extID: ids[i],
			vec:   vecs[i],
		}
	}
	return nodes
}

func sortedInt64s(s []int64) []int64 {
	out := append([]int64(nil), s...)
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// TestParBuild_QualityEquivalence checks parallel build recall@10 is within 0.02 of sequential.
func TestParBuild_QualityEquivalence(t *testing.T) {
	rng := rand.New(rand.NewPCG(parBuildSeed, 0))
	baseVecs, baseIDs := generateClusterVecs(rng, recallMeasureN, recallMeasureDim, recallMeasureNumClusters, recallMeasureClusterSigma)
	queries := generateClusterQueries(rng, recallMeasureNumQueries, recallMeasureDim, recallMeasureNumClusters, recallMeasureClusterSigma)

	exactTopKs := make([]map[string]bool, len(queries))
	for q, query := range queries {
		exactTopKs[q] = exactTopKSet(baseVecs, baseIDs, query, recallMeasureK)
	}

	cfgSeq := recallMeasureConfig(false)
	cfgSeq.BuildWorkers = 1
	statsSeq := measureRecall(t, baseVecs, baseIDs, queries, exactTopKs, cfgSeq)

	cfgPar := recallMeasureConfig(false)
	cfgPar.BuildWorkers = 8
	statsPar := measureRecall(t, baseVecs, baseIDs, queries, exactTopKs, cfgPar)

	t.Logf("recall@%d sequential=%.4f parallel=%.4f", recallMeasureK, statsSeq.mean, statsPar.mean)
	if statsPar.mean < statsSeq.mean-0.02 {
		t.Errorf("parallel recall %.4f < sequential %.4f - 0.02", statsPar.mean, statsSeq.mean)
	}
}

const parBuildSeed = 77

// TestParBuild_LegacyDeterministic verifies BuildWorkers=1 matches the sequential reference graph.
func TestParBuild_LegacyDeterministic(t *testing.T) {
	const (
		n   = 500
		dim = 64
	)

	rng := rand.New(rand.NewPCG(parBuildSeed, 1))
	nodesRef := makeParBuildNodes(rng, n, dim)
	nodesLegacy := cloneGraphNodes(nodesRef)

	medoid := findMedoid(nodesRef)
	ctx := context.Background()
	cfg := DefaultConfig()

	if err := buildGraph(ctx, nodesRef, medoid, cfg.MaxDegree, cfg.SearchListSize, cfg.Alpha, cfg.BuildPasses, 1); err != nil {
		t.Fatal(err)
	}
	if err := buildGraph(ctx, nodesLegacy, medoid, cfg.MaxDegree, cfg.SearchListSize, cfg.Alpha, cfg.BuildPasses, 1); err != nil {
		t.Fatal(err)
	}

	checkNodes := []int{0, 1, 2, 7, 13, 21, 42, 99, 123, 200, 256, 300, 333, 400, 417, 444, 455, 466, 488, 499}
	for _, id := range checkNodes {
		got := sortedInt64s(nodesLegacy[id].neighbors)
		want := sortedInt64s(nodesRef[id].neighbors)
		if len(got) != len(want) {
			t.Fatalf("node %d: neighbor count %d != %d", id, len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("node %d: neighbors differ at %d: got %v want %v", id, i, got, want)
			}
		}
	}
}

// TestParBuild_RaceBuild exercises parallel graph construction under the race detector.
func TestParBuild_RaceBuild(t *testing.T) {
	const (
		n   = 2000
		dim = 32
	)

	rng := rand.New(rand.NewPCG(parBuildSeed, 2))
	nodes := makeParBuildNodes(rng, n, dim)
	medoid := findMedoid(nodes)
	cfg := DefaultConfig()

	if err := buildGraph(context.Background(), nodes, medoid, cfg.MaxDegree, cfg.SearchListSize, cfg.Alpha, cfg.BuildPasses, 8); err != nil {
		t.Fatal(err)
	}
}

// TestParBuild_CanceledContext verifies canceled ctx returns error without hanging.
func TestParBuild_CanceledContext(t *testing.T) {
	const (
		n   = 2000
		dim = 64
	)

	rng := rand.New(rand.NewPCG(parBuildSeed, 3))
	vecs, ids := generateClusterVecs(rng, n, dim, 10, 0.05)

	db := newTestDB(t)
	cfg := DefaultConfig()
	cfg.BuildWorkers = 8
	cfg.BruteForceThreshold = 0

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- idx.Build(ctx, iter)
	}()

	time.Sleep(5 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected error from canceled Build")
		}
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("Build did not return after context cancel (deadlock?)")
	}
}

const (
	parBenchN   = 5000
	parBenchDim = 128
)

func parBenchDataset() ([][]float32, [][]byte) {
	rng := rand.New(rand.NewPCG(parBuildSeed, 4))
	return generateVecs(rng, parBenchN, parBenchDim)
}

func benchmarkBuild(b *testing.B, workers int) {
	vecs, ids := parBenchDataset()
	cfg := DefaultConfig()
	cfg.BuildWorkers = workers

	b.ResetTimer()
	for range b.N {
		dir := b.TempDir()
		dbPath := dir + "/bench.db"
		db, err := sql.Open("sqlite", dbPath)
		if err != nil {
			b.Fatal(err)
		}
		idx, err := New(db, cfg)
		if err != nil {
			b.Fatal(err)
		}
		iter := &sliceIterator{vecs: vecs, ids: ids}
		if err := idx.Build(context.Background(), iter); err != nil {
			b.Fatal(err)
		}
		idx.Close()
		db.Close()
	}
}

func BenchmarkBuildSequential(b *testing.B) {
	benchmarkBuild(b, 1)
}

func BenchmarkBuildParallel(b *testing.B) {
	benchmarkBuild(b, 8)
}
