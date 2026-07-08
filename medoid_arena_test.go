package horosvec

import (
	"context"
	"math"
	"testing"
)

// expectedArenaMedoid calcule le médoïde de référence directement depuis l'arène fp16, via
// l'API publique ArenaReader — oracle INDÉPENDANT de recomputeMedoid : centroïde des vecteurs
// décodés puis argmin de la distance L2 au centroïde. Reproduit l'arithmétique de sélection
// (centroïde en float32, distance carrée) pour écarter tout écart de tie-break, sans réutiliser
// findMedoidSource.
func expectedArenaMedoid(t *testing.T, arenaPath string) int64 {
	t.Helper()
	r, err := OpenArenaReader(arenaPath)
	if err != nil {
		t.Fatal(err)
	}
	n := int(r.Count())
	dim := r.Dim()
	if n == 0 {
		t.Fatal("arène vide")
	}
	vecs := make([][]float32, n)
	centroid := make([]float64, dim)
	buf := make([]float32, dim)
	for i := 0; i < n; i++ {
		if !r.VecInto(int64(i), buf) {
			t.Fatalf("VecInto rang %d échoue", i)
		}
		vecs[i] = append([]float32(nil), buf...)
		for j := 0; j < dim; j++ {
			centroid[j] += float64(buf[j])
		}
	}
	centroidF32 := make([]float32, dim)
	invN := 1.0 / float64(n)
	for j := range centroid {
		centroidF32[j] = float32(centroid[j] * invN)
	}
	var best = math.MaxFloat64
	var bestID int64
	for i, v := range vecs {
		d := l2DistanceSquared(centroidF32, v)
		if d < best {
			best = d
			bestID = int64(i)
		}
	}
	return bestID
}

// TestMedoidRecomputeArenaMode : oracle du correctif C2. En mode arène (SQLite vector-less),
// un médoïde périmé stocké doit être recomputé à partir de l'ARÈNE fp16, jamais des blobs
// vecteur vides du SQLite. Avant le correctif, recomputeMedoid matérialisait
// `SELECT node_id, vector` (blobs vides) et sélectionnait un médoïde faux (premier rang, faute
// de vecteurs) de façon silencieuse. Après correctif, le médoïde recomputé est exactement
// celui que dicte l'arène.
func TestMedoidRecomputeArenaMode(t *testing.T) {
	const n, dim = 1500, 32
	idx, db, arenaPath, _ := buildStreamingTestIndex(t, n, dim)
	idx.Close()

	// Médoïde de référence, calculé indépendamment depuis l'arène.
	want := expectedArenaMedoid(t, arenaPath)

	// Corruption : médoïde stocké pointant vers un node_id inexistant → force la recomputation.
	if _, err := db.Exec(`INSERT OR REPLACE INTO vindex_meta (key, value) VALUES ('medoid', ?)`,
		serializeInt64(999999)); err != nil {
		t.Fatal(err)
	}

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
	cfg.ArenaPath = arenaPath
	idx2, err := New(db, cfg)
	if err != nil {
		t.Fatalf("réouverture avec médoïde périmé : %v", err)
	}
	defer idx2.Close()

	if idx2.medoid != want {
		t.Errorf("médoïde recomputé = %d, attendu (arène) = %d — la recomputation n'a pas lu l'arène", idx2.medoid, want)
	}

	// Le médoïde recomputé doit exister dans vindex_nodes et être persisté en base.
	ok, err := validateMedoid(db, idx2.medoid)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatalf("médoïde recomputé %d absent de vindex_nodes", idx2.medoid)
	}
	var stored []byte
	if err := db.QueryRow(`SELECT value FROM vindex_meta WHERE key='medoid'`).Scan(&stored); err != nil {
		t.Fatal(err)
	}
	if got := deserializeInt64(stored); got != want {
		t.Errorf("médoïde persisté = %d, attendu = %d", got, want)
	}

	// Search doit rester fonctionnelle après recomputation depuis l'arène.
	res, err := idx2.Search(context.Background(), make([]float32, dim), 5)
	if err != nil {
		t.Fatalf("search après recomputation : %v", err)
	}
	_ = res
}
