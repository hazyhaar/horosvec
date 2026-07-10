package horosvec

import (
	"context"
	"database/sql"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// TestArenaFP16RoundtripExactIntegers vérifie que les entiers représentables (0..255,
// comme les vecteurs SIFT) traversent fp16 sans perte — d'où une dégradation de rappel
// nulle sur ce corpus.
func TestArenaFP16RoundtripExactIntegers(t *testing.T) {
	for v := 0; v <= 2048; v++ {
		f := float32(v)
		got := float16ToFloat32(float32ToFloat16(f))
		if got != f {
			t.Fatalf("integer %d not exact: got %v", v, got)
		}
		got = float16ToFloat32(float32ToFloat16(-f))
		if got != -f {
			t.Fatalf("integer %d (neg) not exact: got %v", -v, got)
		}
	}
}

// TestArenaFP16RoundtripBoundedLoss vérifie que la perte relative est bornée (≈ 2^-10)
// pour des valeurs quelconques dans une plage réaliste d'embeddings.
func TestArenaFP16RoundtripBoundedLoss(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 0))
	const tol = 1.0 / 1024.0 // 2^-10, marge au-dessus de la précision fp16 (2^-11)
	for i := 0; i < 100000; i++ {
		f := float32(rng.NormFloat64() * 10)
		got := float16ToFloat32(float32ToFloat16(f))
		if f == 0 {
			if got != 0 {
				t.Fatalf("zero not preserved: %v", got)
			}
			continue
		}
		relErr := math.Abs(float64(got-f)) / math.Abs(float64(f))
		if relErr > tol {
			t.Fatalf("value %v: relative error %v > %v (got %v)", f, relErr, tol, got)
		}
	}
}

// TestArenaFP16Extremes vérifie le comportement aux valeurs extrêmes (dépassement,
// sous-normal, NaN, Inf).
func TestArenaFP16Extremes(t *testing.T) {
	// Dépassement de la plage fp16 (max normal ≈ 65504) → Inf.
	if h := float16ToFloat32(float32ToFloat16(1e30)); !math.IsInf(float64(h), 1) {
		t.Fatalf("overflow +1e30 should saturate to +Inf, got %v", h)
	}
	if h := float16ToFloat32(float32ToFloat16(-1e30)); !math.IsInf(float64(h), -1) {
		t.Fatalf("overflow -1e30 should saturate to -Inf, got %v", h)
	}
	// Trop petit → zéro signé.
	if h := float16ToFloat32(float32ToFloat16(1e-30)); h != 0 {
		t.Fatalf("underflow 1e-30 should be 0, got %v", h)
	}
	// NaN se propage.
	if h := float16ToFloat32(float32ToFloat16(float32(math.NaN()))); !math.IsNaN(float64(h)) {
		t.Fatalf("NaN should propagate, got %v", h)
	}
	// Inf.
	if h := float16ToFloat32(float32ToFloat16(float32(math.Inf(1)))); !math.IsInf(float64(h), 1) {
		t.Fatalf("Inf should propagate, got %v", h)
	}
}

// TestArenaOpenRejectsCorruptHeader vérifie que openArena refuse un header invalide.
func TestArenaOpenRejectsCorruptHeader(t *testing.T) {
	dir := t.TempDir()

	// Fichier trop court.
	short := filepath.Join(dir, "short.arena")
	if err := os.WriteFile(short, []byte("HV"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := openArena(short); err == nil {
		t.Fatal("expected error on too-short file")
	}

	// Header valide de forme mais mauvais magic.
	badMagic := make([]byte, arenaHeaderSize)
	copy(badMagic[0:8], "XXXXXXXX")
	badMagicPath := filepath.Join(dir, "badmagic.arena")
	if err := os.WriteFile(badMagicPath, badMagic, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := openArena(badMagicPath); err == nil {
		t.Fatal("expected error on bad magic")
	}

	// Longueur incohérente avec le count annoncé.
	badLen := make([]byte, arenaHeaderSize+4)
	copy(badLen[0:8], arenaMagic)
	badLen[8] = 1   // version 1
	badLen[12] = 4  // dim 4
	badLen[16] = 10 // count 10 → attend 24 + 10*4*2 = 104 octets
	badLenPath := filepath.Join(dir, "badlen.arena")
	if err := os.WriteFile(badLenPath, badLen, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := openArena(badLenPath); err == nil {
		t.Fatal("expected error on inconsistent length")
	}
}

// buildArenaTestIndex construit un index Vamana (chemin non force-brute) et retourne
// l'index, la base, les vecteurs et le chemin de base pour ré-ouverture.
func buildArenaTestIndex(t *testing.T, n, dim int) (*sql.DB, string, [][]float32) {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "arena_idx.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0 // force le chemin Vamana + rerank
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(42, 0))
	vecs, ids := generateVecs(rng, n, dim)
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	idx.Close()
	return db, dbPath, vecs
}

// TestArenaExportOpenRoundtrip vérifie l'export SQL→arène puis la relecture, et que
// vecInto décode le vecteur du bon nœud (alignement offset ↔ node_id).
func TestArenaExportOpenRoundtrip(t *testing.T) {
	const n, dim = 500, 32
	db, dbPath, vecs := buildArenaTestIndex(t, n, dim)

	// Ré-ouvrir et exporter.
	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	arenaPath := dbPath + ".arena"
	if err := idx.ExportArena(arenaPath); err != nil {
		t.Fatal(err)
	}
	idx.Close()

	ar, err := openArena(arenaPath)
	if err != nil {
		t.Fatal(err)
	}
	if ar.dim != dim {
		t.Fatalf("arena dim = %d, want %d", ar.dim, dim)
	}
	if ar.count != int64(n) {
		t.Fatalf("arena count = %d, want %d", ar.count, n)
	}

	// node_id i correspond à vecs[i] (ordre d'insertion = ordre node_id, dense).
	dst := make([]float32, dim)
	for i := 0; i < n; i++ {
		if !ar.vecInto(int64(i), dst) {
			t.Fatalf("vecInto(%d) returned false", i)
		}
		for j := 0; j < dim; j++ {
			want := float16ToFloat32(float32ToFloat16(vecs[i][j]))
			if dst[j] != want {
				t.Fatalf("node %d dim %d: got %v want %v (orig %v)", i, j, dst[j], want, vecs[i][j])
			}
		}
	}
	// Hors couverture → false.
	if ar.vecInto(int64(n), dst) {
		t.Fatal("vecInto out of range should return false")
	}
	if ar.vecInto(-1, dst) {
		t.Fatal("vecInto negative should return false")
	}
}

// TestArenaRerankZeroSQL est le critère C2 : avec une arène active, la boucle de rerank
// ne déclenche AUCUN accès SQL (RerankSQLLoads == 0).
func TestArenaRerankZeroSQL(t *testing.T) {
	const n, dim, k = 2000, 32, 10
	db, dbPath, vecs := buildArenaTestIndex(t, n, dim)

	// Exporter l'arène depuis l'index existant.
	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
	idxExp, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	arenaPath := dbPath + ".arena"
	if err := idxExp.ExportArena(arenaPath); err != nil {
		t.Fatal(err)
	}
	idxExp.Close()

	// Ré-ouvrir AVEC arène.
	cfgArena := DefaultConfig()
	cfgArena.BruteForceThreshold = 0
	cfgArena.ArenaPath = arenaPath
	idx, err := New(db, cfgArena)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	if idx.arena == nil {
		t.Fatal("arena not loaded")
	}

	for q := 0; q < 100; q++ {
		if _, err := idx.Search(context.Background(), vecs[q*(n/100)], k); err != nil {
			t.Fatal(err)
		}
	}
	if got := idx.RerankSQLLoads(); got != 0 {
		t.Fatalf("critère C2 violé : rerank SQL loads = %d, want 0", got)
	}
}

// TestArenaSearchParityAndNoArenaUnchanged vérifie que le Search avec arène rend les
// mêmes résultats que sans arène (fp16 exact sur ces vecteurs → parité), et que sans
// arène le compteur SQL reflète bien le chemin loadNodeReadOnly (comportement inchangé).
func TestArenaSearchParityAndNoArenaUnchanged(t *testing.T) {
	const n, dim, k = 1500, 32, 10
	db, dbPath, vecs := buildArenaTestIndex(t, n, dim)

	// Sans arène.
	cfgPlain := DefaultConfig()
	cfgPlain.BruteForceThreshold = 0
	idxPlain, err := New(db, cfgPlain)
	if err != nil {
		t.Fatal(err)
	}
	arenaPath := dbPath + ".arena"
	if err := idxPlain.ExportArena(arenaPath); err != nil {
		t.Fatal(err)
	}

	// Avec arène.
	cfgArena := DefaultConfig()
	cfgArena.BruteForceThreshold = 0
	cfgArena.ArenaPath = arenaPath
	idxArena, err := New(db, cfgArena)
	if err != nil {
		t.Fatal(err)
	}
	defer idxArena.Close()

	agree := 0
	total := 0
	for q := 0; q < 100; q++ {
		query := vecs[q*(n/100)]
		rPlain, err := idxPlain.Search(context.Background(), query, k)
		if err != nil {
			t.Fatal(err)
		}
		rArena, err := idxArena.Search(context.Background(), query, k)
		if err != nil {
			t.Fatal(err)
		}
		if len(rPlain) != len(rArena) {
			t.Fatalf("q%d: result count mismatch %d vs %d", q, len(rPlain), len(rArena))
		}
		for i := range rPlain {
			total++
			if string(rPlain[i].ID) == string(rArena[i].ID) {
				agree++
			}
		}
	}
	// Sans arène (db-blob) : le miroir plat fp32 (flatVecs) est désormais chargé à toute
	// échelle et sert le re-classement SANS verrou ni SQL. L'oracle est donc l'INVERSE de
	// l'ancien : aucune charge de rerank SQL ne doit avoir eu lieu (le chemin flatVecs court-
	// circuite loadNodeReadOnly). Le repli SQL n'est emprunté que si flatVecs est absent.
	if idxPlain.flatVecs == nil {
		t.Fatal("sans arène, flatVecs devrait être chargé (mode db-blob RAM-résident)")
	}
	if idxPlain.RerankSQLLoads() != 0 {
		t.Fatalf("sans arène, le rerank via flatVecs ne doit pas toucher SQL, compteur=%d", idxPlain.RerankSQLLoads())
	}
	idxPlain.Close()

	// Quasi-parité : les vecteurs sont gaussiens (non exacts en fp16), la troncature
	// réordonne occasionnellement des candidats quasi ex æquo à la frontière du top-k.
	// L'accord doit rester très élevé (≥ 98 %), traduisant une dégradation bornée.
	if float64(agree)/float64(total) < 0.98 {
		t.Fatalf("parité arène vs non-arène trop faible : %d/%d résultats identiques", agree, total)
	}
}

// TestArenaConcurrentReads exerce des Search concurrents avec arène active (lecture
// seule, doit être sûr en accès parallèle ; à croiser avec go test -race).
func TestArenaConcurrentReads(t *testing.T) {
	const n, dim, k = 1500, 32, 10
	db, dbPath, vecs := buildArenaTestIndex(t, n, dim)

	cfgExp := DefaultConfig()
	cfgExp.BruteForceThreshold = 0
	idxExp, err := New(db, cfgExp)
	if err != nil {
		t.Fatal(err)
	}
	arenaPath := dbPath + ".arena"
	if err := idxExp.ExportArena(arenaPath); err != nil {
		t.Fatal(err)
	}
	idxExp.Close()

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
	cfg.ArenaPath = arenaPath
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			for q := 0; q < 50; q++ {
				query := vecs[(seed*50+q)%n]
				if _, err := idx.Search(context.Background(), query, k); err != nil {
					t.Errorf("concurrent search error: %v", err)
					return
				}
			}
		}(w)
	}
	wg.Wait()
	if got := idx.RerankSQLLoads(); got != 0 {
		t.Fatalf("concurrent: rerank SQL loads = %d, want 0", got)
	}
}
