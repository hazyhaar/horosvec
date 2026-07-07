package horosvec

import (
	"context"
	"encoding/json"
	"os"
	"sort"
	"testing"
)

// TestRecallMeasure_RealEmbeddings measures recall@10 on REAL embeddings
// (bge-m3, dim 1024) supplied via HOROSVEC_REAL_VECS (path to a JSON
// [][]float64). Skipped without the variable: the synthetic bench
// (recall_measure_test.go) is the default measurement; this one exercises the
// anisotropic case of real embedding spaces, the blind spot of synthetic data
// (RaBitQ random rotation absent — see doc.go, Accuracy).
func TestRecallMeasure_RealEmbeddings(t *testing.T) {
	path := os.Getenv("HOROSVEC_REAL_VECS")
	if path == "" {
		t.Skip("HOROSVEC_REAL_VECS not set: real-data bench is opt-in")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var vecs64 [][]float64
	if err := json.Unmarshal(raw, &vecs64); err != nil {
		t.Fatalf("json: %v", err)
	}
	if len(vecs64) < 300 {
		t.Fatalf("too few vectors: %d", len(vecs64))
	}
	dim := len(vecs64[0])
	all := make([][]float32, len(vecs64))
	for i, v := range vecs64 {
		if len(v) != dim {
			t.Fatalf("vec[%d] dim %d != %d", i, len(v), dim)
		}
		f := make([]float32, dim)
		for j, x := range v {
			f[j] = float32(x)
		}
		all[i] = f
	}

	const nQueries, k = 50, 10
	base, queries := all[:len(all)-nQueries], all[len(all)-nQueries:]
	ids := make([][]byte, len(base))
	for i := range base {
		ids[i] = []byte{byte(i >> 16), byte(i >> 8), byte(i)}
	}

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0 // force the Vamana+RaBitQ path
	db := newTestDB(t)
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := idx.Build(context.Background(), &sliceIterator{vecs: base, ids: ids}); err != nil {
		t.Fatalf("Build: %v", err)
	}

	var sum, minR, maxR float64
	minR = 1
	for qi, q := range queries {
		type cand struct {
			i int
			d float64
		}
		exact := make([]cand, len(base))
		for i, b := range base {
			exact[i] = cand{i, l2DistanceSquared(q, b)}
		}
		sort.Slice(exact, func(a, b int) bool { return exact[a].d < exact[b].d })
		truth := make(map[string]bool, k)
		for _, c := range exact[:k] {
			truth[string(ids[c.i])] = true
		}

		got, err := idx.Search(context.Background(), q, k)
		if err != nil {
			t.Fatalf("Search q%d: %v", qi, err)
		}
		hits := 0
		for _, r := range got {
			if truth[string(r.ID)] {
				hits++
			}
		}
		r := float64(hits) / float64(k)
		sum += r
		if r < minR {
			minR = r
		}
		if r > maxR {
			maxR = r
		}
	}
	mean := sum / float64(nQueries)
	t.Logf("[real_embeddings] dim=%d N=%d queries=%d k=%d: recall@10 mean=%.4f min=%.4f max=%.4f",
		dim, len(base), nQueries, k, mean, minR, maxR)
	if mean < 0.50 {
		t.Errorf("mean recall %.4f below sanity floor 0.50", mean)
	}
}
