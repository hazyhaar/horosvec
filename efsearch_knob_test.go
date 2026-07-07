package horosvec

import (
	"context"
	"math/rand/v2"
	"testing"
)

// TestSearch_EfSearchKnobEffective would have caught the decorative-knob defect
// found on the SIFT bench: EfSearch was silently floored to RerankTopN (500),
// so every beam from 64 to 512 ran the exact same traversal. The decidable
// oracle is the heap-pop count (via the existing testDuringGreedySearch hook):
// a wider beam MUST pop strictly more nodes on the same index and query.
func TestSearch_EfSearchKnobEffective(t *testing.T) {
	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(42, 0))
	base, ids := generateClusterVecs(rng, 2000, 64, 20, 0.05)

	buildCfg := DefaultConfig()
	buildCfg.BruteForceThreshold = 0
	idx, err := New(db, buildCfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := idx.Build(context.Background(), &sliceIterator{vecs: base, ids: ids}); err != nil {
		t.Fatalf("Build: %v", err)
	}

	query := base[0]
	popsAt := func(ef int) int {
		cfg := DefaultConfig()
		cfg.BruteForceThreshold = 0
		cfg.EfSearch = ef
		reloaded, err := New(db, cfg)
		if err != nil {
			t.Fatalf("reload ef=%d: %v", ef, err)
		}
		pops := 0
		reloaded.testDuringGreedySearch = func() { pops++ }
		defer func() { reloaded.testDuringGreedySearch = nil }()
		if _, err := reloaded.Search(context.Background(), query, 10); err != nil {
			t.Fatalf("Search ef=%d: %v", ef, err)
		}
		return pops
	}

	narrow := popsAt(32)
	wide := popsAt(512)
	t.Logf("pops: ef=32 -> %d, ef=512 -> %d", narrow, wide)
	if wide <= narrow {
		t.Fatalf("EfSearch décoratif : ef=512 dépile %d nœuds, ef=32 en dépile %d — le bouton doit élargir le parcours", wide, narrow)
	}
	// Le régime étroit ne doit plus être gonflé au plancher RerankTopN=500 :
	// à ef=32 sur 2000 nœuds, un parcours de ~500 dépilements signerait le retour
	// de l'ancien couplage.
	if narrow >= 400 {
		t.Fatalf("ef=32 dépile %d nœuds : le plancher RerankTopN semble à nouveau gonfler le faisceau", narrow)
	}
}
