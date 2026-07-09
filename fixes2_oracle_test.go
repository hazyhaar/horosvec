package horosvec

import (
	"context"
	"database/sql"
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite"
)

// fixes2 oracle tests — vague de correctifs cumulés (plan 2026-07-10).
// Chaque test exerce le MÉCANISME exact d'un correctif et était rouge sur HEAD (8a330d1).

// buildFileIndex builds an index backed by a file DB and returns the path + a live index.
func buildFileIndex(t *testing.T, n, dim int, seed uint64) (string, *sql.DB, *Index) {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "fixes2.db")
	dsn := "file:" + dbPath + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(seed, 0))
	vecs, ids := generateVecs(rng, n, dim)
	cfg := DefaultConfig()
	cfg.EfSearch = 64
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	return dbPath, db, idx
}

// --- A2 : nextID / getMaxNodeID error propagated ---

// TestA2_NextIDNeverZeroOnReload : contrat POSITIF que garde A2. Sur un index non vide
// rechargé, nextID vaut max(node_id)+1, jamais 0 — un nextID=0 réécrirait les nœuds existants
// au prochain Insert. La branche d'erreur d'A2 (getMaxNodeID en échec → New rend une erreur)
// n'est pas isolément atteignable par une fixture de DB corrompue : toute corruption assez
// grave pour faire échouer MAX(node_id) fait d'abord échouer initSchema (index sur node_id),
// et une table vide rend (-1, nil) et non une erreur. A2 reste une propagation défensive
// correcte ; ce test verrouille le contrat observable (aucun avalement silencieux → nextID=0).
func TestA2_NextIDNeverZeroOnReload(t *testing.T) {
	dbPath, db, idx := buildFileIndex(t, 100, 16, 1)
	wantNext := idx.nextID
	if wantNext == 0 {
		t.Fatal("precondition: freshly built index has nextID 0")
	}
	idx.Close()
	db.Close()

	dsn := "file:" + dbPath + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)"
	db2, err := sql.Open("sqlite", dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()
	idx2, err := New(db2, DefaultConfig())
	if err != nil {
		t.Fatalf("A2: reload of a valid index must succeed: %v", err)
	}
	defer idx2.Close()
	if idx2.nextID != wantNext {
		t.Fatalf("A2: reloaded nextID = %d, want %d (never 0/zombie)", idx2.nextID, wantNext)
	}
}

// --- A3 : medoid unreadable → Search errors ; neighbor unreadable → tolerated + counter ---

// TestA3_UnreadableMedoidPropagates exerce EXACTEMENT le chemin médoïde de
// rabitqGreedySearchInternal (horosvec.go:849-853) via un loadFn qui échoue sur le seul
// médoïde. Sur HEAD, ce chemin renvoyait `return nil, nil` (résultat vide indistinct d'un
// corpus vide) ; après A3, l'erreur est propagée.
func TestA3_UnreadableMedoidPropagates(t *testing.T) {
	dim := 16
	_, _, idx := buildFileIndex(t, 300, dim, 2)
	defer idx.Close()

	queryRotated := make([]float32, idx.codeDim)
	q := make([]float32, dim)
	idx.rotator.Rotate(q, queryRotated)

	badMedoid := func(id int64) (*cachedNode, error) {
		if id == idx.medoid {
			return nil, sql.ErrConnDone // médoïde illisible
		}
		return loadNode(context.Background(), idx.db, idx.cache, id)
	}
	_, err := idx.rabitqGreedySearchInternal(context.Background(), queryRotated, 32, idx.nextID, badMedoid)
	if err == nil {
		t.Fatal("A3: unreadable medoid must propagate an error, got nil (empty result masks a broken index)")
	}
}

// TestA3_UnreadableNeighborTolerated : un nœud VOISIN illisible est une dégradation tolérée —
// la marche continue, le résultat est rendu, et l'incident est compté (jamais avalé).
func TestA3_UnreadableNeighborTolerated(t *testing.T) {
	dim := 16
	_, _, idx := buildFileIndex(t, 300, dim, 12)
	defer idx.Close()

	queryRotated := make([]float32, idx.codeDim)
	q := make([]float32, dim)
	idx.rotator.Rotate(q, queryRotated)

	// Échoue sur un nœud voisin arbitraire (jamais le médoïde) pour forcer la dégradation.
	victim := idx.medoid + 1
	if victim >= idx.nextID {
		victim = 0
	}
	before := idx.DegradedNeighborLoads()
	badNeighbor := func(id int64) (*cachedNode, error) {
		if id == victim {
			return nil, sql.ErrConnDone
		}
		return loadNode(context.Background(), idx.db, idx.cache, id)
	}
	res, err := idx.rabitqGreedySearchInternal(context.Background(), queryRotated, 32, idx.nextID, badNeighbor)
	if err != nil {
		t.Fatalf("A3: a single unreadable neighbor must NOT fail the search: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("A3: search should still return candidates despite one bad neighbor")
	}
	if idx.DegradedNeighborLoads() <= before {
		t.Fatal("A3: DegradedNeighborLoads counter should have incremented")
	}
}

// --- A4 : malformed vector blob in bruteForceSQLite → skipped, no panic, counter ---

func TestA4_BruteForceSQLiteSkipsMalformedBlob(t *testing.T) {
	dim := 16
	dbPath, db, idx := buildFileIndex(t, 40, dim, 3) // < BruteForceThreshold → brute-force path
	idx.Close()
	db.Close()

	dsn := "file:" + dbPath + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)"
	db2, err := sql.Open("sqlite", dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()
	// Blob vecteur tronqué sur un nœud : sur HEAD, deserializeFloat32s produit un vecteur court
	// puis l2DistanceSquared lit hors borne (panique). Après A4, la ligne est sautée.
	if _, err := db2.Exec("UPDATE vindex_nodes SET vector = X'0102' WHERE node_id = 5"); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig() // ArenaPath vide, flat désactivé sur blob court (cf. A4 flat), reste SQLite
	idx2, err := New(db2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx2.Close()
	// A4 (chargement flat) : un blob court au chargement DÉSACTIVE le flat (fail-loud), jamais
	// un flatVecs désaligné silencieux — la recherche retombe alors sur le scan SQLite.
	if idx2.flatVecs != nil {
		t.Fatal("A4: flat vectors must be disabled when a malformed blob is seen at load")
	}
	q := make([]float32, dim)
	res, sErr := idx2.Search(context.Background(), q, 10) // ne doit pas paniquer
	if sErr != nil {
		t.Fatalf("A4: brute force search errored: %v", sErr)
	}
	if len(res) == 0 {
		t.Fatal("A4: expected results from the 39 valid nodes")
	}
	if idx2.MalformedVectorSkips() == 0 {
		t.Fatal("A4: malformed vector skip counter should be > 0")
	}
}

// --- O3 : node_count meta reconciled with COUNT(*) at load ---

// TestO3_NodeCountMetaReconciledGovernsLoadDecision : une méta node_count MENSONGÈRE (gonflée
// au-dessus du seuil brute-force) ne doit plus gouverner les décisions de chargement. Après O3,
// nodeCount est réconcilié au COUNT(*) réel (120), qui repasse sous le seuil (500) → les
// vecteurs plats sont chargés. Sur HEAD, la méta (9999) était crue → nodeCount > seuil → flat
// NON chargé. L'observable est donc idx.flatVecs != nil.
func TestO3_NodeCountMetaReconciledGovernsLoadDecision(t *testing.T) {
	dim := 16
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "o3.db")
	dsn := "file:" + dbPath + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(4, 0))
	vecs, ids := generateVecs(rng, 120, dim)
	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 500 // 120 < 500 < 9999
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	idx.Close()
	db.Close()

	db2, err := sql.Open("sqlite", dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()
	// Désynchronise délibérément la méta : node_count ment (9999, au-dessus du seuil).
	if _, err := db2.Exec("INSERT OR REPLACE INTO vindex_meta (key, value) VALUES ('node_count', ?)", serializeInt64(9999)); err != nil {
		t.Fatal(err)
	}
	idx2, err := New(db2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx2.Close()
	if idx2.flatVecs == nil {
		t.Fatal("O3: flat vectors should be loaded (nodeCount reconciled to 120 <= threshold 500); meta 9999 was trusted")
	}
}

// --- A1 : node_count meta stays == COUNT(*) after a successful insert (written in tx) ---

// TestA1_NodeCountMetaWrittenInsideTx exerce le MÉCANISME différenciant d'A1 : l'atomicité.
// Via le hook testBeforeInsertCommit, on lit node_count DANS la transaction, avant Commit. Après
// A1, la méta y vaut déjà le nouveau compte (60+10=70). Sur HEAD, la méta n'était écrite qu'en
// post-commit (best-effort, hors tx) → dans la tx elle valait encore l'ancien compte (60) :
// rouge-avant. Un test de simple consistance d'état final serait satisfait par les deux versions
// (l'ancien post-commit écrivait aussi méta=COUNT(*)) et ne prouverait pas l'atomicité.
func TestA1_NodeCountMetaWrittenInsideTx(t *testing.T) {
	dim := 16
	_, _, idx := buildFileIndex(t, 60, dim, 5)
	defer idx.Close()

	const wantInTx = int64(70) // 60 + 10
	var seenInTx int64 = -1
	idx.testBeforeInsertCommit = func(tx *sql.Tx) {
		var raw []byte
		if err := tx.QueryRow("SELECT value FROM vindex_meta WHERE key='node_count'").Scan(&raw); err != nil {
			t.Errorf("A1: read node_count in tx: %v", err)
			return
		}
		seenInTx = deserializeInt64(raw)
	}
	defer func() { idx.testBeforeInsertCommit = nil }()

	rng := rand.New(rand.NewPCG(99, 0))
	vecs, ids := generateVecs(rng, 10, dim)
	// ext_ids distincts de ceux du build (0..59) : ext_id est UNIQUE, un doublon
	// déclencherait un REPLACE et le compte n'augmenterait pas (artefact de test).
	for i := range ids {
		ids[i] = []byte{0xFF, byte(i)}
	}
	if err := idx.Insert(context.Background(), vecs, ids); err != nil {
		t.Fatal(err)
	}
	if seenInTx != wantInTx {
		t.Fatalf("A1: node_count inside tx before commit = %d, want %d (meta must be written in the tx, atomically with the nodes)", seenInTx, wantInTx)
	}
}

// --- B2 : cancelled context aborts brute-force scan ---

func TestB2_BruteForceFlatHonorsCancelledContext(t *testing.T) {
	dim := 16
	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(6, 0))
	vecs, ids := generateVecs(rng, 40, dim) // < BruteForceThreshold → flat path
	cfg := DefaultConfig()
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	if idx.flatVecs == nil {
		t.Skip("flat vectors not loaded, path not exercised")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	q := make([]float32, dim)
	_, err = idx.bruteForceFlat(ctx, q, 10)
	if err == nil {
		t.Fatal("B2: bruteForceFlat should error on a cancelled context")
	}
}

// --- B3 : int32 offset guard rejects cumulative overflow (pure function, no 33M build) ---

func TestB3_Int32OffsetGuard(t *testing.T) {
	if err := checkInt32Offset(0, "neighbors"); err != nil {
		t.Fatalf("B3: zero offset must pass: %v", err)
	}
	if err := checkInt32Offset(maxInt32Offset, "neighbors"); err != nil {
		t.Fatalf("B3: exactly 2^31-1 must pass: %v", err)
	}
	if err := checkInt32Offset(maxInt32Offset+1, "neighbors"); err == nil {
		t.Fatal("B3: 2^31 cumulative offset must fail-loud, got nil")
	}
	if err := checkInt32Offset(int64(1)<<40, "ext_ids"); err == nil {
		t.Fatal("B3: huge ext_ids cumulative offset must fail-loud, got nil")
	}
}

// --- A5 : hot plane extension failure counted + plane invalidated, plane repli remains correct ---

func TestA5_PlaneDegradationCountedOnAppendFailure(t *testing.T) {
	dim := 16
	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(7, 0))
	vecs, ids := generateVecs(rng, 200, dim)
	cfg := DefaultConfig()
	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	if idx.plane == nil {
		t.Skip("hot plane not built")
	}
	// Sabote le plan : force un codeBytes incompatible pour que appendNode échoue à la
	// prochaine extension (A5).
	idx.plane.codeBytes++
	before := idx.PlaneDegraded()
	iv, iid := generateVecs(rand.New(rand.NewPCG(8, 0)), 1, dim)
	if err := idx.Insert(context.Background(), iv, iid); err != nil {
		t.Fatalf("A5: insert itself must still succeed (data committed): %v", err)
	}
	if idx.PlaneDegraded() != before+1 {
		t.Fatalf("A5: expected PlaneDegraded to increment, before=%d after=%d", before, idx.PlaneDegraded())
	}
	if idx.plane != nil {
		t.Fatal("A5: plane should be invalidated (nil) after append failure")
	}
	// Le repli reste correct : la recherche fonctionne encore.
	q := make([]float32, dim)
	if _, err := idx.Search(context.Background(), q, 5); err != nil {
		t.Fatalf("A5: search after plane degradation must still work: %v", err)
	}
}

// --- B1 : float64 centroid accumulation stays close to reference over many adds ---

func TestB1_CentroidFloat64AccumulationNoDrift(t *testing.T) {
	dim := 8
	const n = 200000
	ct := NewCentroidTracker(dim, 0.05, 1.0)
	rng := rand.New(rand.NewPCG(11, 0))
	// Référence float64 exacte : somme puis division.
	sum := make([]float64, dim)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for j := 0; j < dim; j++ {
			// Valeurs autour de 1000 : amplifie l'erreur relative float32 de la moyenne mobile.
			x := 1000.0 + rng.NormFloat64()
			v[j] = float32(x)
			sum[j] += float64(v[j])
		}
		ct.Add(v)
	}
	// Lecture de l'accumulateur INTERNE (ct.current) et non de Current(), qui recaste en
	// float32 et masquerait la précision d'accumulation. Après B1 (accumulation float64),
	// l'écart à la référence float64 exacte est ~1e-9 ; sur HEAD (accumulation float32), il
	// dépasse 1e-6 (au moins un ulp float32 à magnitude ~1000, soit ~6e-5).
	ct.mu.Lock()
	internal := make([]float64, dim)
	for j := 0; j < dim; j++ {
		internal[j] = float64(ct.current[j])
	}
	ct.mu.Unlock()
	for j := 0; j < dim; j++ {
		ref := sum[j] / float64(n)
		diff := math.Abs(internal[j] - ref)
		if diff > 1e-6 {
			t.Fatalf("B1: centroid[%d] accumulation drift %.6g exceeds 1e-6 (got %.9f, ref %.9f)", j, diff, internal[j], ref)
		}
	}
}
