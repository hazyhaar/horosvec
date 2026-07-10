package horosvec

import (
	"context"
	"math/rand/v2"
	"testing"
)

// TestRerankFlatVecsParity prouve que le re-classement db-blob servi par le miroir plat
// fp32 (flatVecs, lu sans verrou) rend un top-K STRICTEMENT identique — mêmes ext_id dans
// le même ordre, mêmes scores — au re-classement de repli via SQL (loadNodeReadOnly), sur
// le même index. Les deux chemins lisent les mêmes vecteurs fp32 : divergence attendue = 0.
//
// Le test force le chemin Vamana (BruteForceThreshold=0) afin d'exercer réellement la boucle
// de rerank, et non le court-circuit brute-force. Il vérifie en outre que la recherche servie
// par flatVecs n'incrémente PAS le compteur RerankSQLLoads (oracle : le nouveau chemin ne
// passe plus par SQL), puis annule flatVecs pour forcer le repli SQL et comparer.
func TestRerankFlatVecsParity(t *testing.T) {
	const (
		n   = 1200
		dim = 64
		k   = 10
	)

	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(7, 0))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0 // force le chemin Vamana + rerank (pas de court-circuit flat)
	cfg.BuildWorkers = 1        // build séquentiel déterministe

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}
	if idx.flatVecs == nil {
		t.Fatalf("flatVecs devrait être chargé en mode db-blob (ArenaPath vide)")
	}

	queries, _ := generateVecs(rng, 50, dim)

	// Passe 1 : rerank servi par flatVecs (sans verrou, sans SQL).
	before := idx.RerankSQLLoads()
	flatResults := make([][]Result, len(queries))
	for q, query := range queries {
		res, err := idx.Search(context.Background(), query, k)
		if err != nil {
			t.Fatalf("search flatVecs q=%d: %v", q, err)
		}
		flatResults[q] = res
	}
	// C6 : oracle du non-recours au SQL — aucune charge de rerank SQL sur le chemin flatVecs.
	if got := idx.RerankSQLLoads() - before; got != 0 {
		t.Fatalf("chemin flatVecs: RerankSQLLoads a bougé de %d, attendu 0", got)
	}

	// Passe 2 : annuler flatVecs force le repli loadNodeReadOnly (SQL) — chemin de référence.
	idx.flatVecs = nil
	beforeSQL := idx.RerankSQLLoads()
	for q, query := range queries {
		sqlRes, err := idx.Search(context.Background(), query, k)
		if err != nil {
			t.Fatalf("search SQL q=%d: %v", q, err)
		}
		fr := flatResults[q]
		if len(sqlRes) != len(fr) {
			t.Fatalf("q=%d: longueur top-K flat=%d vs sql=%d", q, len(fr), len(sqlRes))
		}
		for i := range fr {
			if string(fr[i].ID) != string(sqlRes[i].ID) {
				t.Fatalf("q=%d rang=%d: ext_id diverge flat=%x sql=%x", q, i, fr[i].ID, sqlRes[i].ID)
			}
			if fr[i].Score != sqlRes[i].Score {
				t.Fatalf("q=%d rang=%d: score diverge flat=%v sql=%v", q, i, fr[i].Score, sqlRes[i].Score)
			}
		}
	}
	// Le repli SQL DOIT avoir chargé des vecteurs (sinon la comparaison ne prouve rien).
	if idx.RerankSQLLoads()-beforeSQL == 0 {
		t.Fatalf("repli SQL: RerankSQLLoads n'a pas bougé, le chemin de référence n'a pas été exercé")
	}
}
