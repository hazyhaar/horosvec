package horosvec

import (
	"context"
	"database/sql"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

// buildStreamingTestIndex construit un index en mode streaming vector-less (ArenaPath posé)
// et retourne l'index ouvert, le chemin d'arène et les vecteurs source.
func buildStreamingTestIndex(t *testing.T, n, dim int) (*Index, *sql.DB, string, [][]float32) {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "stream_idx.db")
	arenaPath := dbPath + ".arena"
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0 // chemin Vamana + rerank
	cfg.ArenaPath = arenaPath   // opt-in streaming vector-less
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(42, 0))
	vecs, ids := generateVecs(rng, n, dim)
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	return idx, db, arenaPath, vecs
}

// TestStreamingBuildVectorLessSQLite : le build streaming n'écrit plus de blob vecteur en
// SQLite (colonne vide), l'arène couvre le rerank (0 accès SQL au rerank), et Search rend un
// rappel correct.
func TestStreamingBuildVectorLessSQLite(t *testing.T) {
	const n, dim, k = 2000, 32, 10
	idx, db, arenaPath, vecs := buildStreamingTestIndex(t, n, dim)
	defer idx.Close()

	// (C5) Tous les nœuds ont un blob vecteur VIDE.
	var maxVecLen, rowCount int
	if err := db.QueryRow("SELECT COUNT(*), COALESCE(MAX(LENGTH(vector)),0) FROM vindex_nodes").Scan(&rowCount, &maxVecLen); err != nil {
		t.Fatal(err)
	}
	if rowCount != n {
		t.Fatalf("node count = %d, want %d", rowCount, n)
	}
	if maxVecLen != 0 {
		t.Fatalf("max vector blob length = %d, want 0 (vector-less SQLite)", maxVecLen)
	}

	// L'arène existe et couvre tous les nœuds.
	ar, err := openArena(arenaPath)
	if err != nil {
		t.Fatal(err)
	}
	if ar.count != int64(n) || ar.dim != dim {
		t.Fatalf("arena (count=%d,dim=%d), want (%d,%d)", ar.count, ar.dim, n, dim)
	}

	// (C5) Search passe par l'arène : RerankSQLLoads reste à 0.
	for q := 0; q < 100; q++ {
		if _, err := idx.Search(context.Background(), vecs[q*(n/100)], k); err != nil {
			t.Fatal(err)
		}
	}
	if got := idx.RerankSQLLoads(); got != 0 {
		t.Fatalf("RerankSQLLoads = %d, want 0 (rerank must read the arena, not SQL)", got)
	}

	// Rappel : le graphe (fp16) doit retrouver au moins le voisin exact (soi-même) sur une
	// large majorité de requêtes.
	selfHits := 0
	for q := 0; q < 200; q++ {
		query := vecs[q]
		res, err := idx.Search(context.Background(), query, k)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range res {
			if string(r.ID) == string([]byte{byte(q >> 24), byte(q >> 16), byte(q >> 8), byte(q)}) {
				selfHits++
				break
			}
		}
	}
	if selfHits < 190 {
		t.Fatalf("self-recall %d/200 too low", selfHits)
	}
}

// TestStreamingBuildReloadRetroCompat : un index vector-less se recharge via New avec le même
// ArenaPath et Search fonctionne (arène rouverte, plan chaud reconstruit sans blob vecteur).
func TestStreamingBuildReloadRetroCompat(t *testing.T) {
	const n, dim, k = 1500, 32, 10
	idx, db, arenaPath, vecs := buildStreamingTestIndex(t, n, dim)
	idx.Close()

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
	cfg.ArenaPath = arenaPath
	reloaded, err := New(db, cfg)
	if err != nil {
		t.Fatalf("reload vector-less index: %v", err)
	}
	defer reloaded.Close()
	if reloaded.Count() != n {
		t.Fatalf("reloaded count = %d, want %d", reloaded.Count(), n)
	}
	res, err := reloaded.Search(context.Background(), vecs[0], k)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		t.Fatal("reloaded search returned no results")
	}
	if got := reloaded.RerankSQLLoads(); got != 0 {
		t.Fatalf("reloaded RerankSQLLoads = %d, want 0", got)
	}
}

// TestStreamingArenaTruncationDetected : une arène tronquée (fichier coupé au milieu) est
// détectée proprement à la relecture (garde de longueur d'openArena), jamais lue corrompue.
func TestStreamingArenaTruncationDetected(t *testing.T) {
	const n, dim = 800, 32
	idx, _, arenaPath, _ := buildStreamingTestIndex(t, n, dim)
	idx.Close()

	info, err := os.Stat(arenaPath)
	if err != nil {
		t.Fatal(err)
	}
	// Couper le fichier au milieu de la charge utile.
	if err := os.Truncate(arenaPath, info.Size()-int64(dim)); err != nil {
		t.Fatal(err)
	}
	if _, err := openArena(arenaPath); err == nil {
		t.Fatal("expected openArena to reject a truncated arena")
	}
}

// TestStreamingBuildNoTmpLeak : aucun fichier temporaire d'arène ne subsiste après un build
// réussi (fsync + rename atomique, pas de fuite de descripteur/fichier tmp).
func TestStreamingBuildNoTmpLeak(t *testing.T) {
	const n, dim = 600, 32
	idx, _, arenaPath, _ := buildStreamingTestIndex(t, n, dim)
	idx.Close()

	if _, err := os.Stat(arenaPath + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("leftover .tmp file after successful build (err=%v)", err)
	}
}

// TestStreamingSmallIndexDefaultThreshold : un index streaming vector-less SOUS le seuil
// force-brute PAR DÉFAUT (50000) ne doit pas paniquer à la Search — le brute-force lit
// l'arène, pas les blobs vides. Régression du finding hard d'audit V2 (les autres tests
// forçaient BruteForceThreshold=0, esquivant ce chemin).
func TestStreamingSmallIndexDefaultThreshold(t *testing.T) {
	const n, dim, k = 1000, 32, 10
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "small_stream.db")
	arenaPath := dbPath + ".arena"
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })

	cfg := DefaultConfig() // BruteForceThreshold = 50000 (défaut) : n=1000 → chemin brute-force
	cfg.ArenaPath = arenaPath
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(7, 0))
	vecs, ids := generateVecs(rng, n, dim)
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}

	// Brute-force via arène : rappel exact (self-hit) sur toutes les requêtes, aucune panique.
	for q := 0; q < n; q++ {
		res, err := idx.Search(context.Background(), vecs[q], k)
		if err != nil {
			t.Fatalf("search q=%d: %v", q, err)
		}
		want := string([]byte{byte(q >> 24), byte(q >> 16), byte(q >> 8), byte(q)})
		if len(res) == 0 || string(res[0].ID) != want {
			t.Fatalf("q=%d: nearest = %q (score %.4f), want self %q", q, res[0].ID, res[0].Score, want)
		}
	}
	idx.Close()

	// Reload sous seuil par défaut : loadFlatVectors NE doit PAS s'exécuter (arène mode) ;
	// la Search rechargée passe par l'arène sans paniquer.
	reloaded, err := New(db, cfg)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	defer reloaded.Close()
	res, err := reloaded.Search(context.Background(), vecs[0], k)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || string(res[0].ID) != string([]byte{0, 0, 0, 0}) {
		t.Fatalf("reloaded nearest = %q, want self", res[0].ID)
	}
}

// TestStreamingSearchTopKZero : Search topK<=0 sur un index arène ne panique pas (garde
// partagée) et rend un résultat vide. Régression finding A d'audit V2.
func TestStreamingSearchTopKZero(t *testing.T) {
	const n, dim = 800, 32
	idx, _, _, vecs := buildStreamingTestIndex(t, n, dim)
	defer idx.Close()
	res, err := idx.Search(context.Background(), vecs[0], 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 0 {
		t.Fatalf("topK=0 returned %d results, want 0", len(res))
	}
	// Idem sur le chemin brute-force arène (petit index, seuil par défaut).
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "tk0.db")
	db, _ := sql.Open("sqlite", dbPath)
	t.Cleanup(func() { db.Close() })
	cfg := DefaultConfig()
	cfg.ArenaPath = dbPath + ".arena"
	sidx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer sidx.Close()
	rng := rand.New(rand.NewPCG(1, 0))
	sv, si := generateVecs(rng, 500, dim)
	if err := sidx.Build(context.Background(), &sliceIterator{vecs: sv, ids: si}); err != nil {
		t.Fatal(err)
	}
	if res, err := sidx.Search(context.Background(), sv[0], 0); err != nil || len(res) != 0 {
		t.Fatalf("brute-force arena topK=0: res=%d err=%v", len(res), err)
	}
}

// TestStreamingInsertRejected : Insert sur un index arène est refusé fail-loud (pas de perte
// de rappel silencieuse). Régression finding B d'audit V2.
func TestStreamingInsertRejected(t *testing.T) {
	const n, dim = 600, 32
	idx, _, _, vecs := buildStreamingTestIndex(t, n, dim)
	defer idx.Close()
	err := idx.Insert(context.Background(), [][]float32{vecs[0]}, [][]byte{[]byte("x")})
	if err == nil {
		t.Fatal("expected Insert on arena-backed index to be rejected")
	}
}

// TestStreamingSearchWithRerankTopKGuard : SearchWithRerank avec topK<=0 ne panique pas
// (garde propre, la troncature candidates[:topK] ne voit jamais un topK négatif).
// Régression finding C d'audit V2.
func TestStreamingSearchWithRerankTopKGuard(t *testing.T) {
	const n, dim = 800, 32
	idx, _, _, vecs := buildStreamingTestIndex(t, n, dim)
	defer idx.Close()
	for _, tk := range []int{0, -1, -5} {
		res, err := idx.SearchWithRerank(context.Background(), vecs[0], tk, nil)
		if err != nil {
			t.Fatalf("topK=%d: %v", tk, err)
		}
		if len(res) != 0 {
			t.Fatalf("topK=%d returned %d results, want 0", tk, len(res))
		}
	}
}

// TestStreamingBuildEmptyIterator : un itérateur vide échoue proprement sans laisser d'arène.
func TestStreamingBuildEmptyIterator(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "empty.db")
	arenaPath := dbPath + ".arena"
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
	cfg.ArenaPath = arenaPath
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	if err := idx.Build(context.Background(), &sliceIterator{}); err == nil {
		t.Fatal("expected error on empty iterator")
	}
	if _, err := os.Stat(arenaPath); !os.IsNotExist(err) {
		t.Fatalf("empty build left an arena file (err=%v)", err)
	}
}
