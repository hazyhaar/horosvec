package horosvec

import (
	"context"
	"math/rand/v2"
	"testing"
)

const (
	recallMeasureDim          = 128
	recallMeasureN            = 2000
	recallMeasureNumQueries   = 50
	recallMeasureK            = 10
	recallMeasureSeed         = 55
	recallMeasureNumClusters  = 20
	recallMeasureClusterSigma = 0.05

	// Planchers de rappel par distribution, avec build DÉTERMINISTE (BuildWorkers=1) et graine
	// fixée : le rappel est alors une constante reproductible sur tout matériel. Valeurs
	// mesurées à N=2000, dim=128, k=10 : uniform ≈ 0,930, gaussian_clusters ≈ 0,678. Les
	// planchers sont posés sous ces valeurs avec marge, tout en restant très supérieurs à
	// l'ancien 0,50 (qui n'assertait rien). Le rappel bien plus bas des clusters gaussiens
	// serrés (sigma 0,05) est intrinsèque à la quantification RaBitQ 1-bit sur des vecteurs
	// quasi identiques — pas une régression ; un plancher 0,90 y serait inatteignable sans
	// changer le comportement de Search.
	recallMeasureFloorUniform  = 0.90
	recallMeasureFloorClusters = 0.60
)

// TestRecallMeasure_VamanaRabitq benchmarks recall@k on Vamana+RaBitQ before M5/M3 arbitration.
// Vectors are deterministic (fixed PCG seed); ground truth is exact brute-force L2.
func TestRecallMeasure_VamanaRabitq(t *testing.T) {
	t.Run("uniform", func(t *testing.T) {
		rng := rand.New(rand.NewPCG(recallMeasureSeed, 0))
		baseVecs, baseIDs := generateUniformVecs(rng, recallMeasureN, recallMeasureDim)
		queries := generateUniformQueries(rng, recallMeasureNumQueries, recallMeasureDim)
		runRecallMeasure(t, "uniform", baseVecs, baseIDs, queries, recallMeasureFloorUniform)
	})

	t.Run("gaussian_clusters", func(t *testing.T) {
		rng := rand.New(rand.NewPCG(recallMeasureSeed, 1))
		baseVecs, baseIDs := generateClusterVecs(rng, recallMeasureN, recallMeasureDim, recallMeasureNumClusters, recallMeasureClusterSigma)
		queries := generateClusterQueries(rng, recallMeasureNumQueries, recallMeasureDim, recallMeasureNumClusters, recallMeasureClusterSigma)
		runRecallMeasure(t, "gaussian_clusters", baseVecs, baseIDs, queries, recallMeasureFloorClusters)
	})
}

type recallStats struct {
	mean float64
	min  float64
	max  float64
}

func runRecallMeasure(t *testing.T, label string, baseVecs [][]float32, baseIDs [][]byte, queries [][]float32, floor float64) {
	t.Helper()

	exactTopKs := make([]map[string]bool, len(queries))
	for q, query := range queries {
		exactTopKs[q] = exactTopKSet(baseVecs, baseIDs, query, recallMeasureK)
	}

	cfgDefault := recallMeasureConfig(false)
	statsDefault := measureRecall(t, baseVecs, baseIDs, queries, exactTopKs, cfgDefault)
	t.Logf("[%s] config=default SearchListSize=%d EfSearch=%d MaxDegree=%d BuildPasses=%d: recall@%d mean=%.4f min=%.4f max=%.4f (N=%d queries=%d dim=%d)",
		label, cfgDefault.SearchListSize, cfgDefault.EfSearch, cfgDefault.MaxDegree, cfgDefault.BuildPasses,
		recallMeasureK, statsDefault.mean, statsDefault.min, statsDefault.max,
		recallMeasureN, recallMeasureNumQueries, recallMeasureDim)
	if statsDefault.mean < floor {
		t.Errorf("[%s] default config: recall@%d mean=%.4f, want >= %.2f", label, recallMeasureK, statsDefault.mean, floor)
	}

	cfgWide := recallMeasureConfig(true)
	statsWide := measureRecall(t, baseVecs, baseIDs, queries, exactTopKs, cfgWide)
	t.Logf("[%s] config=SearchListSize_x2 SearchListSize=%d EfSearch=%d MaxDegree=%d BuildPasses=%d: recall@%d mean=%.4f min=%.4f max=%.4f (N=%d queries=%d dim=%d)",
		label, cfgWide.SearchListSize, cfgWide.EfSearch, cfgWide.MaxDegree, cfgWide.BuildPasses,
		recallMeasureK, statsWide.mean, statsWide.min, statsWide.max,
		recallMeasureN, recallMeasureNumQueries, recallMeasureDim)
	if statsWide.mean < floor {
		t.Errorf("[%s] SearchListSize_x2: recall@%d mean=%.4f, want >= %.2f", label, recallMeasureK, statsWide.mean, floor)
	}
}

func recallMeasureConfig(doubleSearchList bool) Config {
	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0 // force Vamana+RaBitQ path
	cfg.BuildWorkers = 1        // build séquentiel bit-déterministe (la graine des données ne
	//                             suffit pas : le build parallèle par défaut interleave et
	//                             rend le graphe non reproductible — cf. Config.BuildWorkers).
	if doubleSearchList {
		cfg.SearchListSize *= 2
	}
	return cfg
}

func measureRecall(
	t *testing.T,
	baseVecs [][]float32,
	baseIDs [][]byte,
	queries [][]float32,
	exactTopKs []map[string]bool,
	cfg Config,
) recallStats {
	t.Helper()

	db := newTestDB(t)
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: baseVecs, ids: baseIDs}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	recalls := make([]float64, len(queries))
	for q, query := range queries {
		results, err := idx.Search(context.Background(), query, recallMeasureK)
		if err != nil {
			t.Fatalf("query %d: %v", q, err)
		}
		hits := 0
		for _, r := range results {
			if exactTopKs[q][string(r.ID)] {
				hits++
			}
		}
		recalls[q] = float64(hits) / float64(recallMeasureK)
	}

	stats := recallStats{mean: 0, min: 1, max: 0}
	for _, r := range recalls {
		stats.mean += r
		if r < stats.min {
			stats.min = r
		}
		if r > stats.max {
			stats.max = r
		}
	}
	stats.mean /= float64(len(recalls))
	return stats
}

func exactTopKSet(vecs [][]float32, ids [][]byte, query []float32, k int) map[string]bool {
	results := exactTopK(vecs, ids, query, k)
	topK := make(map[string]bool, len(results))
	for _, r := range results {
		topK[string(r.ID)] = true
	}
	return topK
}

func generateUniformVecs(rng *rand.Rand, n, dim int) ([][]float32, [][]byte) {
	vecs := make([][]float32, n)
	ids := make([][]byte, n)
	for i := range n {
		vecs[i] = make([]float32, dim)
		for j := range dim {
			vecs[i][j] = rng.Float32()
		}
		ids[i] = recallMeasureID(i)
	}
	return vecs, ids
}

func generateUniformQueries(rng *rand.Rand, n, dim int) [][]float32 {
	queries := make([][]float32, n)
	for i := range n {
		queries[i] = make([]float32, dim)
		for j := range dim {
			queries[i][j] = rng.Float32()
		}
	}
	return queries
}

func generateClusterVecs(rng *rand.Rand, n, dim, numClusters int, sigma float64) ([][]float32, [][]byte) {
	centers := make([][]float32, numClusters)
	for c := range numClusters {
		centers[c] = make([]float32, dim)
		for j := range dim {
			centers[c][j] = rng.Float32()
		}
	}

	vecs := make([][]float32, n)
	ids := make([][]byte, n)
	for i := range n {
		center := centers[rng.IntN(numClusters)]
		vecs[i] = gaussianAround(rng, center, dim, sigma)
		ids[i] = recallMeasureID(i)
	}
	return vecs, ids
}

func generateClusterQueries(rng *rand.Rand, n, dim, numClusters int, sigma float64) [][]float32 {
	centers := make([][]float32, numClusters)
	for c := range numClusters {
		centers[c] = make([]float32, dim)
		for j := range dim {
			centers[c][j] = rng.Float32()
		}
	}

	queries := make([][]float32, n)
	for i := range n {
		center := centers[rng.IntN(numClusters)]
		queries[i] = gaussianAround(rng, center, dim, sigma)
	}
	return queries
}

func gaussianAround(rng *rand.Rand, center []float32, dim int, sigma float64) []float32 {
	vec := make([]float32, dim)
	for j := range dim {
		vec[j] = center[j] + float32(rng.NormFloat64()*sigma)
	}
	return vec
}

func recallMeasureID(i int) []byte {
	return []byte{byte(i >> 24), byte(i >> 16), byte(i >> 8), byte(i)}
}
