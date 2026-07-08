package horosvec

import (
	"context"
	"database/sql"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// genUnitVecs produit n vecteurs de dimension dim, chacun normalisé à ‖v‖₂ = 1 (tirage gaussien
// puis division par la norme), et des ext_ids = représentation ASCII décimale du rang (0..n-1).
// Ce choix d'ext_id rend le round-trip d'import exact : readArenaIDs (uint64 LE → ASCII décimal)
// reproduit les mêmes octets d'ext_id que ceux insérés dans l'index d'origine.
func genUnitVecs(rng *rand.Rand, n, dim int) ([][]float32, [][]byte) {
	vecs := make([][]float32, n)
	ids := make([][]byte, n)
	for i := range n {
		v := make([]float32, dim)
		var sum float64
		for j := range dim {
			v[j] = float32(rng.NormFloat64())
			sum += float64(v[j]) * float64(v[j])
		}
		inv := float32(1.0 / math.Sqrt(sum))
		for j := range dim {
			v[j] *= inv
		}
		vecs[i] = v
		ids[i] = []byte(strconv.Itoa(i))
	}
	return vecs, ids
}

// writeUint64IDsFile écrit un fichier d'ids finalisé (uint64 LE, rang = valeur), lu par
// readArenaIDs. Reproduit le format écrit par hnbook-embed.
func writeUint64IDsFile(t *testing.T, path string, n int) {
	t.Helper()
	buf := make([]byte, n*8)
	for i := range n {
		binary.LittleEndian.PutUint64(buf[i*8:], uint64(i))
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}

// buildUnitArenaIndex construit un index Vamana réel sur n vecteurs unitaires via le chemin
// arène streaming (ArenaPath posé) : l'arène fp16 est produite sur disque. Retourne l'index
// ouvert (arène résidente), le chemin d'arène et les vecteurs sources.
func buildUnitArenaIndex(t *testing.T, dir string, n, dim int) (*Index, string, [][]float32) {
	t.Helper()
	dbPath := filepath.Join(dir, "orig.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })

	arenaPath := filepath.Join(dir, "orig.arena")
	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0 // force le chemin Vamana + rerank
	cfg.ArenaPath = arenaPath   // chemin streaming vector-less → arène produite sur disque
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(20260709, 1))
	vecs, ids := genUnitVecs(rng, n, dim)
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	return idx, arenaPath, vecs
}

func overlapAt(a, b []Result) int {
	set := make(map[string]bool, len(a))
	for _, r := range a {
		set[string(r.ID)] = true
	}
	n := 0
	for _, r := range b {
		if set[string(r.ID)] {
			n++
		}
	}
	return n
}

// TestImportAdjacencyRoundtripParity est le prototype de couture E2E SANS GPU (O3) : un index
// Vamana réel est construit, son adjacence exportée au format plat u32, puis RÉIMPORTÉE via
// ImportAdjacency sur la même arène et les mêmes ext_ids. On vérifie l'identité du node_count,
// la validité du médoïde, et la parité de recherche (overlap@10 ≥ 0,95 sur 20 requêtes).
func TestImportAdjacencyRoundtripParity(t *testing.T) {
	const n, dim, k = 20000, 64, 10
	dir := t.TempDir()

	idxOrig, arenaPath, vecs := buildUnitArenaIndex(t, dir, n, dim)
	defer idxOrig.Close()

	degree := idxOrig.cfg.MaxDegree
	adjPath := filepath.Join(dir, "adjacency.u32")
	if err := idxOrig.ExportAdjacency(adjPath, degree); err != nil {
		t.Fatalf("export adjacency: %v", err)
	}
	// Taille exacte attendue de l'adjacence plate (O1).
	st, err := os.Stat(adjPath)
	if err != nil {
		t.Fatal(err)
	}
	if want := int64(n) * int64(degree) * 4; st.Size() != want {
		t.Fatalf("adjacency size = %d, want %d", st.Size(), want)
	}

	idsPath := filepath.Join(dir, "ids.u64")
	writeUint64IDsFile(t, idsPath, n)

	// Réimport sur une base neuve.
	db2Path := filepath.Join(dir, "imported.db")
	db2, err := sql.Open("sqlite", db2Path)
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()
	cfg2 := DefaultConfig()
	cfg2.BruteForceThreshold = 0
	idxImp, err := New(db2, cfg2)
	if err != nil {
		t.Fatal(err)
	}
	defer idxImp.Close()

	if err := idxImp.ImportAdjacency(context.Background(), arenaPath, idsPath, adjPath, degree); err != nil {
		t.Fatalf("import adjacency: %v", err)
	}

	// node_count identique.
	cntBytes, err := loadMeta(db2, "node_count")
	if err != nil {
		t.Fatal(err)
	}
	if got := int(deserializeInt64(cntBytes)); got != n {
		t.Fatalf("imported node_count = %d, want %d", got, n)
	}
	// Médoïde valide (dans les bornes) et cohérent avec l'original (même findMedoidSource).
	if idxImp.medoid < 0 || idxImp.medoid >= int64(n) {
		t.Fatalf("imported medoid %d out of range [0,%d)", idxImp.medoid, n)
	}
	if idxImp.medoid != idxOrig.medoid {
		t.Fatalf("imported medoid %d != original medoid %d", idxImp.medoid, idxOrig.medoid)
	}

	// Parité de recherche : overlap@10 ≥ 0,95 sur 20 requêtes.
	total := 0
	for q := 0; q < 20; q++ {
		query := vecs[q*(n/20)]
		rOrig, err := idxOrig.Search(context.Background(), query, k)
		if err != nil {
			t.Fatal(err)
		}
		rImp, err := idxImp.Search(context.Background(), query, k)
		if err != nil {
			t.Fatal(err)
		}
		total += overlapAt(rOrig, rImp)
	}
	frac := float64(total) / float64(20*k)
	if frac < 0.95 {
		t.Fatalf("overlap@10 parité = %.4f, want ≥ 0.95", frac)
	}
}

// TestImportAdjacencyFailLoud vérifie les gardes fail-loud (O1) : taille tronquée, id voisin
// hors borne (décalage d'un cran = le bug attendu), et degré > MaxDegree.
func TestImportAdjacencyFailLoud(t *testing.T) {
	const n, dim = 2000, 32
	dir := t.TempDir()
	idxOrig, arenaPath, _ := buildUnitArenaIndex(t, dir, n, dim)
	defer idxOrig.Close()
	degree := idxOrig.cfg.MaxDegree

	adjPath := filepath.Join(dir, "adj.u32")
	if err := idxOrig.ExportAdjacency(adjPath, degree); err != nil {
		t.Fatal(err)
	}
	idsPath := filepath.Join(dir, "ids.u64")
	writeUint64IDsFile(t, idsPath, n)

	newImp := func(t *testing.T) *Index {
		db, err := sql.Open("sqlite", filepath.Join(t.TempDir(), "imp.db"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { db.Close() })
		cfg := DefaultConfig()
		cfg.BruteForceThreshold = 0
		idx, err := New(db, cfg)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { idx.Close() })
		return idx
	}

	// (a) Adjacence tronquée : taille != n×degree×4.
	t.Run("truncated", func(t *testing.T) {
		raw, err := os.ReadFile(adjPath)
		if err != nil {
			t.Fatal(err)
		}
		badPath := filepath.Join(t.TempDir(), "trunc.u32")
		if err := os.WriteFile(badPath, raw[:len(raw)-4], 0o644); err != nil {
			t.Fatal(err)
		}
		if err := newImp(t).ImportAdjacency(context.Background(), arenaPath, idsPath, badPath, degree); err == nil {
			t.Fatal("expected fail-loud on truncated adjacency")
		}
	})

	// (b) Id voisin hors borne (== n) : décalage arène↔node_id↔ligne d'adjacence.
	t.Run("out_of_range_neighbor", func(t *testing.T) {
		raw, err := os.ReadFile(adjPath)
		if err != nil {
			t.Fatal(err)
		}
		bad := make([]byte, len(raw))
		copy(bad, raw)
		binary.LittleEndian.PutUint32(bad[0:], uint32(n)) // premier voisin du nœud 0 → n (hors borne)
		badPath := filepath.Join(t.TempDir(), "oor.u32")
		if err := os.WriteFile(badPath, bad, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := newImp(t).ImportAdjacency(context.Background(), arenaPath, idsPath, badPath, degree); err == nil {
			t.Fatal("expected fail-loud on out-of-range neighbor id")
		}
	})

	// (c) Degré déclaré > MaxDegree de la config.
	t.Run("degree_over_maxdegree", func(t *testing.T) {
		if err := newImp(t).ImportAdjacency(context.Background(), arenaPath, idsPath, adjPath, degree+1); err == nil {
			t.Fatal("expected fail-loud on degree > MaxDegree")
		}
	})
}

// TestArenaVerifyNormalized est l'oracle métrique (O2) : ≥ 10 000 vecteurs échantillonnés,
// ≥ 99,9 % unitaires → succès sur une arène normalisée, échec sur une arène non normalisée.
func TestArenaVerifyNormalized(t *testing.T) {
	const dim = 64
	dir := t.TempDir()

	writeArena := func(name string, vecs [][]float32) string {
		p := filepath.Join(dir, name)
		w, err := NewArenaWriter(p, dim)
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range vecs {
			if err := w.WriteVec(v); err != nil {
				t.Fatal(err)
			}
		}
		if err := w.Finalize(); err != nil {
			t.Fatal(err)
		}
		return p
	}

	rng := rand.New(rand.NewPCG(20260709, 2))
	unit, _ := genUnitVecs(rng, 12000, dim)
	unitPath := writeArena("unit.arena", unit)
	r, err := OpenArenaReader(unitPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.VerifyNormalized(defaultNormSampleN, defaultNormSeed); err != nil {
		t.Fatalf("expected normalized arena to pass O2 guard: %v", err)
	}

	// Arène non normalisée (vecteurs gaussiens bruts, ‖v‖ ~ sqrt(dim)) → échec attendu.
	rawVecs := make([][]float32, 12000)
	for i := range rawVecs {
		v := make([]float32, dim)
		for j := range dim {
			v[j] = float32(rng.NormFloat64())
		}
		rawVecs[i] = v
	}
	rawPath := writeArena("raw.arena", rawVecs)
	r2, err := OpenArenaReader(rawPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := r2.VerifyNormalized(defaultNormSampleN, defaultNormSeed); err == nil {
		t.Fatal("expected non-normalized arena to fail O2 guard")
	}
}
