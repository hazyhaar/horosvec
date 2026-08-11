package horosvec

import (
	"context"
	"math/rand/v2"
	"sort"
	"testing"
)

// TestMultiBitBoutEnBout construit le MÊME corpus à plusieurs largeurs de code
// et mesure le rappel réel contre une vérité terrain exacte. C'est le seul
// oracle qui dise si le dispositif tient sa promesse : la qualité de sélection
// mesurée hors moteur ne prouve rien tant qu'elle n'a pas traversé le graphe.
func TestMultiBitBoutEnBout(t *testing.T) {
	const n, dim, topK, nq = 3000, 128, 10, 40

	rng := rand.New(rand.NewPCG(17, 23))
	vecs := make([][]float32, n)
	ids := make([][]byte, n)
	for i := range vecs {
		v := make([]float32, dim)
		// Grappes : corpus plus exigeant qu'un bruit uniforme, où la
		// quantification grossière se voit.
		centre := float64(i % 12)
		for j := range v {
			v[j] = float32(centre + rng.NormFloat64())
		}
		vecs[i] = v
		ids[i] = []byte{byte(i >> 8), byte(i)}
	}
	queries := make([][]float32, nq)
	for i := range queries {
		src := vecs[rng.IntN(n)]
		q := make([]float32, dim)
		for j := range q {
			q[j] = float32(float64(src[j]) + 0.5*rng.NormFloat64())
		}
		queries[i] = q
	}

	// Vérité terrain exacte, indépendante de toute quantification.
	verite := make([]map[string]bool, nq)
	for qi, q := range queries {
		type pd struct {
			id string
			d  float64
		}
		tous := make([]pd, n)
		for i, v := range vecs {
			tous[i] = pd{string(ids[i]), l2DistanceSquared(q, v)}
		}
		sort.Slice(tous, func(a, b int) bool { return tous[a].d < tous[b].d })
		m := map[string]bool{}
		for _, p := range tous[:topK] {
			m[p.id] = true
		}
		verite[qi] = m
	}

	mesure := func(bits int) float64 {
		db := newTestDB(t)
		cfg := DefaultConfig()
		cfg.CodeBits = bits
		cfg.BruteForceThreshold = 10 // forcer le passage par le graphe
		idx, err := New(db, cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer idx.Close()
		if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
			t.Fatal(err)
		}
		if idx.codeBits != bits {
			t.Fatalf("largeur non appliquee : %d au lieu de %d", idx.codeBits, bits)
		}
		var somme float64
		for qi, q := range queries {
			res, err := idx.Search(context.Background(), q, topK)
			if err != nil {
				t.Fatal(err)
			}
			trouve := 0
			for _, r := range res {
				if verite[qi][string(r.ID)] {
					trouve++
				}
			}
			somme += float64(trouve) / float64(topK)
		}
		return somme / float64(nq)
	}

	base := mesure(1)
	t.Logf("bits=1 : rappel@%d = %.4f (reference, schema d'origine)", topK, base)
	for _, b := range []int{2, 3, 4} {
		r := mesure(b)
		t.Logf("bits=%d : rappel@%d = %.4f  (%+.4f)", b, topK, r, r-base)
		if r < base-0.02 {
			t.Fatalf("bits=%d degrade le rappel : %.4f contre %.4f a un bit", b, r, base)
		}
	}
}

// TestMultiBitPersisteLaLargeur vérifie qu'un index rouvert relit la largeur de
// SA construction, et non celle que la configuration réclame — sans quoi les
// codes seraient relus au mauvais format.
func TestMultiBitPersisteLaLargeur(t *testing.T) {
	const n, dim = 200, 64
	rng := rand.New(rand.NewPCG(2, 3))
	vecs, ids := generateVecs(rng, n, dim)

	db := newTestDB(t)
	cfg := DefaultConfig()
	cfg.CodeBits = 3
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	idx.Close()

	// Réouverture en RÉCLAMANT une autre largeur : l'index doit garder la sienne.
	cfg2 := DefaultConfig()
	cfg2.CodeBits = 1
	idx2, err := New(db, cfg2)
	if err != nil {
		t.Fatal(err)
	}
	defer idx2.Close()
	if idx2.codeBits != 3 {
		t.Fatalf("largeur relue %d, attendu 3 — la configuration a pris le pas sur le format persiste", idx2.codeBits)
	}
}
