package horosvec

import (
	"context"
	"math/rand/v2"
	"path/filepath"
	"testing"
)

// Le préchargement du lot de re-classement est une INDICATION d'accès donnée au
// noyau : il change la façon dont les pages arrivent en mémoire, jamais ce
// qu'elles contiennent. Les tests ci-dessous gardent cette propriété, ainsi que
// les cas où le lot contient des identifiants que l'arène ne couvre pas.
//
// Le gain de performance, lui, ne se mesure pas ici : il n'apparaît que sur un
// index plus grand que la mémoire disponible, hors de portée d'une suite de
// tests. Les mesures et leur protocole sont consignés dans
// arena_prefetch_unix.go et docs/MESURES-2026-08-prefetch.md.

// indexAvecArene construit un index puis son arène, et rouvre l'ensemble avec la
// configuration demandée.
func indexAvecArene(t *testing.T, n, dim int, cfg Config) *Index {
	t.Helper()
	dir := t.TempDir()
	arenaPath := filepath.Join(dir, "test.arena")

	db := newTestDB(t)
	base, err := New(db, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(12, 34))
	vecs, ids := generateVecs(rng, n, dim)
	if err := base.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	if err := base.ExportArena(arenaPath); err != nil {
		t.Fatal(err)
	}
	base.Close()

	cfg.ArenaPath = arenaPath
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { idx.Close() })
	return idx
}

// TestPrefetchNeChangePasLesResultats est la garde centrale : le conseil d'accès
// ne doit jamais déplacer un résultat. Les deux configurations sont interrogées
// avec les mêmes requêtes et doivent rendre exactement les mêmes voisins, dans
// le même ordre, avec les mêmes distances.
func TestPrefetchNeChangePasLesResultats(t *testing.T) {
	const n, dim, topK = 400, 32, 10

	// Le seuil de recherche exhaustive doit être abaissé : au-dessous, Search
	// balaie tout l'index sans passer par le re-classement, et le chemin de
	// préchargement ne serait jamais emprunté — le test serait creux. Vérifié par
	// mutation : sans cet abaissement, une panique posée dans prefetchRerank ne
	// fait pas échouer ce test.
	avec := DefaultConfig()
	avec.PrefetchRerank = true
	avec.BruteForceThreshold = 10
	sans := DefaultConfig()
	sans.PrefetchRerank = false
	sans.BruteForceThreshold = 10

	idxAvec := indexAvecArene(t, n, dim, avec)
	idxSans := indexAvecArene(t, n, dim, sans)

	rng := rand.New(rand.NewPCG(56, 78))
	for q := 0; q < 50; q++ {
		query := make([]float32, dim)
		for i := range query {
			query[i] = float32(rng.NormFloat64())
		}
		ra, err := idxAvec.Search(context.Background(), query, topK)
		if err != nil {
			t.Fatal(err)
		}
		rs, err := idxSans.Search(context.Background(), query, topK)
		if err != nil {
			t.Fatal(err)
		}
		if len(ra) != len(rs) {
			t.Fatalf("requete %d : %d resultats avec prefetch, %d sans", q, len(ra), len(rs))
		}
		for i := range ra {
			if string(ra[i].ID) != string(rs[i].ID) || ra[i].Score != rs[i].Score {
				t.Fatalf("requete %d rang %d : avec=(%s,%v) sans=(%s,%v)",
					q, i, ra[i].ID, ra[i].Score, rs[i].ID, rs[i].Score)
			}
		}
	}
}

// TestPrefetchIdentifiantsHorsCouverture vérifie que le préchargement ignore
// proprement les candidats absents de l'arène, au lieu de calculer une plage
// hors des bornes de la cartographie. Un identifiant hors couverture est un cas
// nominal : la boucle de re-classement bascule alors sur sa voie de repli.
func TestPrefetchIdentifiantsHorsCouverture(t *testing.T) {
	a, _ := benchArena(64, benchDim)
	cands := []searchCandidate{
		{nodeID: 0},
		{nodeID: -1},
		{nodeID: a.count},
		{nodeID: a.count + 1000},
		{nodeID: 63},
	}
	// Une plage mal calculée ferait paniquer sur le découpage de a.data.
	a.prefetchRerank(cands)

	// Le dernier nœud est le cas limite du calcul de fin de plage : sa page finale
	// dépasse la taille utile de l'arène et doit être bornée.
	a.prefetchRerank([]searchCandidate{{nodeID: a.count - 1}})
}

// TestPrefetchLotVide couvre le cas d'un re-classement sans candidat.
func TestPrefetchLotVide(t *testing.T) {
	a, _ := benchArena(8, benchDim)
	a.prefetchRerank(nil)
	a.prefetchRerank([]searchCandidate{})
}

// TestPrefetchActifParDefaut grave le choix : le réglage vaut mieux activé,
// puisque le seul cas défavorable mesuré coûte 4,4 % quand toute l'arène est
// résidente, contre un facteur 8,5 à 11,8 gagné dès qu'elle ne l'est pas.
func TestPrefetchActifParDefaut(t *testing.T) {
	if !DefaultConfig().PrefetchRerank {
		t.Fatal("le prechargement du lot de reclassement doit etre actif par defaut")
	}
}
