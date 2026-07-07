package horosvec

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

func TestRabitqLUT_Equivalence(t *testing.T) {
	dims := []int{8, 64, 1024, 60}
	const pairsPerDim = 200
	const seed uint64 = 0xFAB17C0DE

	for _, dim := range dims {
		dim := dim
		t.Run(dimLabel(dim), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(seed, uint64(dim)))
			nBytes := (dim + 7) / 8
			lut := make([]float64, nBytes*256)

			for pair := range pairsPerDim {
				queryCentered := make([]float64, dim)
				var querySqNorm float64
				for i := range dim {
					v := rng.NormFloat64()
					queryCentered[i] = v
					querySqNorm += v * v
				}
				buildRabitqLUT(queryCentered, lut)

				code := make([]byte, nBytes)
				for i := range nBytes {
					code[i] = byte(rng.IntN(256))
				}
				storedSqNorm := rng.Float64()*10 + 0.1
				storedL1Norm := rng.Float64()*10 + 0.1

				got := rabitqDistanceLUT(lut, querySqNorm, code, storedSqNorm, storedL1Norm)
				want := rabitqDistanceAsymPrecomp(queryCentered, querySqNorm, code, storedSqNorm, storedL1Norm)
				if math.Abs(got-want) > 1e-9 {
					t.Fatalf("pair %d dim %d: LUT=%g precomp=%g diff=%g", pair, dim, got, want, got-want)
				}
			}
		})
	}
}

func TestRabitqLUT_ZeroL1Norm(t *testing.T) {
	dims := []int{8, 60, 1024}
	const seed uint64 = 0xBADA5500

	for _, dim := range dims {
		dim := dim
		t.Run(dimLabel(dim), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(seed, uint64(dim)))
			nBytes := (dim + 7) / 8
			lut := make([]float64, nBytes*256)

			queryCentered := make([]float64, dim)
			for i := range dim {
				queryCentered[i] = rng.NormFloat64()
			}
			buildRabitqLUT(queryCentered, lut)

			code := make([]byte, nBytes)
			for i := range nBytes {
				code[i] = byte(rng.IntN(256))
			}
			storedSqNorm := rng.Float64()*5 + 1.0

			gotLUT := rabitqDistanceLUT(lut, 1.0, code, storedSqNorm, 0)
			gotPrecomp := rabitqDistanceAsymPrecomp(queryCentered, 1.0, code, storedSqNorm, 0)
			if gotLUT != storedSqNorm {
				t.Fatalf("LUT with L1=0: got %g want storedSqNorm %g", gotLUT, storedSqNorm)
			}
			if gotPrecomp != storedSqNorm {
				t.Fatalf("precomp with L1=0: got %g want storedSqNorm %g", gotPrecomp, storedSqNorm)
			}
		})
	}
}

func TestRabitqLUT_SearchMatchesBruteForce(t *testing.T) {
	const (
		n                  = 64
		dim                = 32
		k                  = 5
		numClusters        = 8
		sigma              = 0.02
		seed        uint64 = 0xFA5C4A00
	)

	rng := rand.New(rand.NewPCG(seed, 0))
	vecs, ids := generateClusterVecs(rng, n, dim, numClusters, sigma)

	db := newTestDB(t)
	cfg := DefaultConfig()
	cfg.EfSearch = 64
	cfg.SearchListSize = 32
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

	query := vecs[0]
	type idDist struct {
		id   string
		dist float64
	}
	exact := make([]idDist, n)
	for i, v := range vecs {
		var d float64
		for j := range dim {
			diff := float64(query[j]) - float64(v[j])
			d += diff * diff
		}
		exact[i] = idDist{string(ids[i]), d}
	}
	sort.Slice(exact, func(a, b int) bool { return exact[a].dist < exact[b].dist })

	wantTopK := make(map[string]bool, k)
	for i := range k {
		wantTopK[exact[i].id] = true
	}

	results, err := idx.Search(context.Background(), query, k)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != k {
		t.Fatalf("got %d results, want %d", len(results), k)
	}

	hits := 0
	for _, r := range results {
		if wantTopK[string(r.ID)] {
			hits++
		}
	}
	if hits != k {
		t.Fatalf("recall@%d = %d/%d (want perfect recall on separated clusters)", k, hits, k)
	}
}

func BenchmarkRabitqDistanceAsym(b *testing.B) {
	const dim = 1024
	queryCentered := make([]float64, dim)
	var querySqNorm float64
	for i := range dim {
		v := float64(i%17) - 8
		queryCentered[i] = v
		querySqNorm += v * v
	}
	code := make([]byte, dim/8)
	for i := range code {
		code[i] = byte(i*37 + 13)
	}
	storedSqNorm := 42.0
	storedL1Norm := 17.5

	b.ResetTimer()
	for range b.N {
		rabitqDistanceAsymPrecomp(queryCentered, querySqNorm, code, storedSqNorm, storedL1Norm)
	}
}

func BenchmarkRabitqDistanceLUT(b *testing.B) {
	const dim = 1024
	queryCentered := make([]float64, dim)
	var querySqNorm float64
	for i := range dim {
		v := float64(i%17) - 8
		queryCentered[i] = v
		querySqNorm += v * v
	}
	lut := make([]float64, (dim/8)*256)
	buildRabitqLUT(queryCentered, lut)
	code := make([]byte, dim/8)
	for i := range code {
		code[i] = byte(i*37 + 13)
	}
	storedSqNorm := 42.0
	storedL1Norm := 17.5

	b.ResetTimer()
	for range b.N {
		rabitqDistanceLUT(lut, querySqNorm, code, storedSqNorm, storedL1Norm)
	}
}

func dimLabel(dim int) string {
	return fmt.Sprintf("dim%d", dim)
}
