package horosvec

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

func TestRaBitQOrderPreservation(t *testing.T) {
	const (
		n   = 1000
		dim = 128
		k   = 10
	)

	rng := rand.New(rand.NewPCG(42, 0))

	vecs := make([][]float32, n)
	for i := range n {
		vecs[i] = make([]float32, dim)
		for j := range dim {
			vecs[i][j] = float32(rng.NormFloat64())
		}
	}

	centroid := make([]float32, dim)
	for _, v := range vecs {
		for j, val := range v {
			centroid[j] += val
		}
	}
	for j := range dim {
		centroid[j] /= float32(n)
	}

	enc := NewEncoder(centroid)

	codes := make([][]byte, n)
	sqNorms := make([]float64, n)
	l1Norms := make([]float64, n)
	for i, v := range vecs {
		codes[i], sqNorms[i], l1Norms[i] = enc.Encode(v)
	}

	query := vecs[0]

	// Exact L2 distances
	type idDist struct {
		id   int
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

	// Corrected asymmetric RaBitQ distances
	approxDists := make([]idDist, n)
	for i := range n {
		d := rabitqDistanceAsym(query, centroid, codes[i], sqNorms[i], l1Norms[i])
		approxDists[i] = idDist{i, d}
	}
	sort.Slice(approxDists, func(a, b int) bool {
		return approxDists[a].dist < approxDists[b].dist
	})

	trueTopK := make(map[int]bool, k)
	for i := range k {
		trueTopK[exactDists[i].id] = true
	}

	candidateSize := k * 5
	if candidateSize > n {
		candidateSize = n
	}
	hits := 0
	for i := range candidateSize {
		if trueTopK[approxDists[i].id] {
			hits++
		}
	}

	recall := float64(hits) / float64(k)
	t.Logf("RaBitQ corrected asymmetric recall@%d (candidates=%d): %.2f%%", k, candidateSize, recall*100)
	// 1-bit quantization without rotation achieves ~70% recall@10 at 5x oversampling.
	// This is a fundamental limit; higher recall requires the Vamana graph + larger beam.
	if recall < 0.60 {
		t.Errorf("recall@%d = %.2f%%, want >= 60%%", k, recall*100)
	}
}

func TestRaBitQEncodeConsistency(t *testing.T) {
	centroid := []float32{1.0, 2.0, 3.0, 4.0}
	enc := NewEncoder(centroid)

	vec := []float32{1.5, 1.5, 3.5, 3.5}
	code1, norm1, l1a := enc.Encode(vec)
	code2, norm2, l1b := enc.Encode(vec)

	if norm1 != norm2 {
		t.Errorf("sqNorms differ: %f vs %f", norm1, norm2)
	}
	if l1a != l1b {
		t.Errorf("l1Norms differ: %f vs %f", l1a, l1b)
	}
	if len(code1) != len(code2) {
		t.Fatalf("code lengths differ: %d vs %d", len(code1), len(code2))
	}
	for i := range code1 {
		if code1[i] != code2[i] {
			t.Errorf("codes differ at byte %d: %08b vs %08b", i, code1[i], code2[i])
		}
	}
}

func TestRaBitQDistanceSelfIsSmall(t *testing.T) {
	centroid := make([]float32, 64)
	enc := NewEncoder(centroid)

	vec := make([]float32, 64)
	for i := range vec {
		vec[i] = float32(i) * 0.1
	}
	code, sqNorm, l1Norm := enc.Encode(vec)

	// Symmetric: self vs self should be ~0
	d := rabitqDistance(code, code, sqNorm, sqNorm)
	if d > 1e-6 {
		t.Errorf("symmetric self-distance = %f, want ~0", d)
	}

	// Corrected asymmetric: self-distance should be small
	dAsym := rabitqDistanceAsym(vec, centroid, code, sqNorm, l1Norm)
	t.Logf("corrected asymmetric self-distance = %f (sqNorm=%f, ratio=%.2f%%)", dAsym, sqNorm, dAsym/sqNorm*100)
	if dAsym > sqNorm*0.3 {
		t.Errorf("asymmetric self-distance = %f, too large relative to sqNorm=%f", dAsym, sqNorm)
	}
}

func BenchmarkPOPCOUNT192(b *testing.B) {
	code1 := make([]byte, 192)
	code2 := make([]byte, 192)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range code1 {
		code1[i] = byte(rng.IntN(256))
		code2[i] = byte(rng.IntN(256))
	}

	b.ResetTimer()
	for b.Loop() {
		rabitqDistance(code1, code2, 100.0, 100.0)
	}
}

func BenchmarkPOPCOUNT16(b *testing.B) {
	code1 := make([]byte, 16)
	code2 := make([]byte, 16)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range code1 {
		code1[i] = byte(rng.IntN(256))
		code2[i] = byte(rng.IntN(256))
	}

	b.ResetTimer()
	for b.Loop() {
		rabitqDistance(code1, code2, 100.0, 100.0)
	}
}

func BenchmarkAsymDistance128(b *testing.B) {
	dim := 128
	rng := rand.New(rand.NewPCG(42, 0))
	query := make([]float32, dim)
	centroid := make([]float32, dim)
	for i := range dim {
		query[i] = float32(rng.NormFloat64())
	}
	enc := NewEncoder(centroid)
	code, sqNorm, l1Norm := enc.Encode(query)

	b.ResetTimer()
	for b.Loop() {
		rabitqDistanceAsym(query, centroid, code, sqNorm, l1Norm)
	}
}

func TestRaBitQDistanceCorrelation(t *testing.T) {
	const dim = 128
	rng := rand.New(rand.NewPCG(99, 0))

	centroid := make([]float32, dim)
	enc := NewEncoder(centroid)

	query := make([]float32, dim)
	for i := range query {
		query[i] = float32(rng.NormFloat64())
	}

	close := make([]float32, dim)
	far := make([]float32, dim)
	for i := range dim {
		close[i] = query[i] + float32(rng.NormFloat64()*0.1)
		far[i] = float32(rng.NormFloat64() * 10.0)
	}

	closeCode, closeSqNorm, closeL1 := enc.Encode(close)
	farCode, farSqNorm, farL1 := enc.Encode(far)

	dClose := rabitqDistanceAsym(query, centroid, closeCode, closeSqNorm, closeL1)
	dFar := rabitqDistanceAsym(query, centroid, farCode, farSqNorm, farL1)

	var exactClose, exactFar float64
	for i := range dim {
		dc := float64(query[i]) - float64(close[i])
		df := float64(query[i]) - float64(far[i])
		exactClose += dc * dc
		exactFar += df * df
	}

	t.Logf("Close: exact=%.2f, approx=%.2f", math.Sqrt(exactClose), dClose)
	t.Logf("Far:   exact=%.2f, approx=%.2f", math.Sqrt(exactFar), dFar)

	if dClose >= dFar {
		t.Errorf("expected close vector (%f) to have smaller approx distance than far vector (%f)", dClose, dFar)
	}
}
