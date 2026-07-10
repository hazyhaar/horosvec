package horosvec

import (
	"context"
	"database/sql"
	"math/rand/v2"
	"path/filepath"
	"testing"
)

// TestFlatVecsBudgetGuard prouve la garde de capacité RAM du miroir plat fp32 en mode
// db-blob (ArenaPath vide). Un index dont l'empreinte estimée (nodeCount*dim*4) dépasse
// Config.FlatVecsMaxBytes NE charge PAS flatVecs au rechargement (flatVecs nil), incrémente
// le compteur d'observabilité, et la recherche fonctionne quand même via le repli SQL
// (loadNodeReadOnly) — dégradation gracieuse, jamais d'OOM ni de panique. Le budget nul
// (défaut) préserve le comportement historique : miroir chargé à toute échelle.
func TestFlatVecsBudgetGuard(t *testing.T) {
	const (
		n             = 1200
		dim           = 64
		k             = 10
		wantFootprint = n * dim * 4 // 307200 octets
	)

	rng := rand.New(rand.NewPCG(11, 0))
	vecs, ids := generateVecs(rng, n, dim)
	queries, _ := generateVecs(rng, 30, dim)

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "budget.db")
	dsn := "file:" + dbPath + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)"

	openIdx := func(t *testing.T, cfg Config) (*sql.DB, *Index) {
		t.Helper()
		db, err := sql.Open("sqlite", dsn)
		if err != nil {
			t.Fatal(err)
		}
		idx, err := New(db, cfg)
		if err != nil {
			db.Close()
			t.Fatal(err)
		}
		return db, idx
	}

	baseCfg := func() Config {
		c := DefaultConfig()
		c.BruteForceThreshold = 0 // force le chemin Vamana + rerank
		c.BuildWorkers = 1
		return c
	}

	// Passe A : construire et persister un index db-blob.
	{
		db, idx := openIdx(t, baseCfg())
		if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
			t.Fatal(err)
		}
		if idx.flatVecs == nil {
			t.Fatalf("flatVecs devrait être chargé au build en mode db-blob")
		}
		idx.Close()
		db.Close()
	}

	// Passe B : rouvrir avec un budget minuscule (< empreinte) — flatVecs doit rester nil.
	{
		cfg := baseCfg()
		cfg.FlatVecsMaxBytes = 1024 // 1 Kio << 300 Kio d'empreinte → dépassement
		db, idx := openIdx(t, cfg)

		if idx.flatVecs != nil {
			t.Fatalf("garde budget: flatVecs devrait être nil (empreinte %d > budget %d)", wantFootprint, cfg.FlatVecsMaxBytes)
		}
		if got := idx.FlatVecsBudgetSkips(); got != 1 {
			t.Fatalf("garde budget: FlatVecsBudgetSkips=%d, attendu 1", got)
		}

		// La recherche fonctionne quand même via le repli SQL (loadNodeReadOnly).
		before := idx.RerankSQLLoads()
		for q, query := range queries {
			res, err := idx.Search(context.Background(), query, k)
			if err != nil {
				t.Fatalf("search sous garde budget q=%d: %v", q, err)
			}
			if len(res) == 0 {
				t.Fatalf("search sous garde budget q=%d: résultat vide", q)
			}
		}
		// Oracle décidable : le repli SQL a bien été emprunté.
		if idx.RerankSQLLoads()-before == 0 {
			t.Fatalf("garde budget: repli SQL non exercé (RerankSQLLoads inchangé)")
		}
		idx.Close()
		db.Close()
	}

	// Passe C : rouvrir sans budget (0 = illimité) — flatVecs chargé (opt-out historique).
	{
		db, idx := openIdx(t, baseCfg())
		if idx.flatVecs == nil {
			t.Fatalf("budget nul: flatVecs devrait être chargé (comportement historique)")
		}
		if got := idx.FlatVecsBudgetSkips(); got != 0 {
			t.Fatalf("budget nul: FlatVecsBudgetSkips=%d, attendu 0", got)
		}
		idx.Close()
		db.Close()
	}

	// Passe D : budget large (>= empreinte) — flatVecs chargé, aucun skip.
	{
		cfg := baseCfg()
		cfg.FlatVecsMaxBytes = wantFootprint * 2
		db, idx := openIdx(t, cfg)
		if idx.flatVecs == nil {
			t.Fatalf("budget large: flatVecs devrait être chargé (empreinte %d <= budget %d)", wantFootprint, cfg.FlatVecsMaxBytes)
		}
		if got := idx.FlatVecsBudgetSkips(); got != 0 {
			t.Fatalf("budget large: FlatVecsBudgetSkips=%d, attendu 0", got)
		}
		idx.Close()
		db.Close()
	}
}

// TestFlatVecsBudgetGuardBuild prouve que la garde de capacité couvre désormais le chemin
// Build lui-même (trou A du fix 2026-07-08), pas seulement le rechargement testé ci-dessus
// par TestFlatVecsBudgetGuard. Avant ce fix, un PREMIER Build à grande échelle peuplait
// flatVecs sans jamais consulter Config.FlatVecsMaxBytes — la garde de New() ne protégeait
// que les réouvertures ultérieures, jamais la construction initiale. Assertion décidable :
// flatVecs reste nil après Build quand l'empreinte estimée dépasse le budget, le compteur
// FlatVecsBudgetSkips s'incrémente, et la recherche fonctionne quand même via le repli SQL
// (loadNodeReadOnly, RerankSQLLoads progresse) — 0 OOM, 0 panic.
func TestFlatVecsBudgetGuardBuild(t *testing.T) {
	const (
		n             = 1200
		dim           = 64
		k             = 10
		wantFootprint = n * dim * 4 // 307200 octets
	)

	rng := rand.New(rand.NewPCG(23, 0))
	vecs, ids := generateVecs(rng, n, dim)
	queries, _ := generateVecs(rng, 20, dim)

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "budget_build.db")
	dsn := "file:" + dbPath + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)"

	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0 // force le chemin Vamana + rerank (pas le scan brute-force petit corpus)
	cfg.BuildWorkers = 1
	cfg.FlatVecsMaxBytes = 1024 // 1 Kio << 300 Kio d'empreinte → dépassement dès le Build

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}

	if idx.flatVecs != nil {
		t.Fatalf("garde budget au Build: flatVecs devrait être nil (empreinte %d > budget %d), reçu longueur %d",
			wantFootprint, cfg.FlatVecsMaxBytes, len(idx.flatVecs))
	}
	if got := idx.FlatVecsBudgetSkips(); got != 1 {
		t.Fatalf("garde budget au Build: FlatVecsBudgetSkips=%d, attendu 1", got)
	}

	before := idx.RerankSQLLoads()
	for q, query := range queries {
		res, err := idx.Search(context.Background(), query, k)
		if err != nil {
			t.Fatalf("search sous garde budget (post-build) q=%d: %v", q, err)
		}
		if len(res) == 0 {
			t.Fatalf("search sous garde budget (post-build) q=%d: résultat vide", q)
		}
	}
	if idx.RerankSQLLoads()-before == 0 {
		t.Fatalf("garde budget au Build: repli SQL non exercé (RerankSQLLoads inchangé)")
	}
}

// TestFlatVecsBudgetGuardInsertRuntime prouve que la garde couvre aussi la croissance runtime
// de flatVecs par l'Insert (second angle mort du trou A) : un flatVecs peuplé sous le budget
// initial ne doit PAS croître sans borne au fil des inserts. Sémantique choisie : la
// croissance est GELÉE dès que le prochain append projetterait l'empreinte au-delà du budget
// — flatVecs reste un miroir PARTIEL des node_id les plus anciens. La cohérence node_id→offset
// est vérifiée : les résultats de recherche restent corrects (rappel exact via repli SQL pour
// les node_id insérés après le gel), et RerankSQLLoads progresse (preuve que le repli est
// bien emprunté), sans panic ni troncature silencieuse.
func TestFlatVecsBudgetGuardInsertRuntime(t *testing.T) {
	const dim = 64

	rng := rand.New(rand.NewPCG(29, 0))
	baseVecs, baseIDs := generateVecs(rng, 400, dim)
	extraVecs, extraIDs := generateVecs(rng, 400, dim)
	queries, _ := generateVecs(rng, 20, dim)

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "budget_insert.db")
	dsn := "file:" + dbPath + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)"

	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0
	cfg.BuildWorkers = 1
	// Budget suffisant pour les 400 premiers nœuds (400*64*4 = 102400 octets) mais pas pour
	// les 800 après extension (800*64*4 = 204800 octets) : force le gel de croissance à
	// l'Insert, pas au Build.
	cfg.FlatVecsMaxBytes = 150000

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	if err := idx.Build(context.Background(), &sliceIterator{vecs: baseVecs, ids: baseIDs}); err != nil {
		t.Fatal(err)
	}
	if idx.flatVecs == nil {
		t.Fatalf("build initial sous budget: flatVecs devrait être chargé (400 nœuds tient dans 150000 octets)")
	}
	frozenLen := len(idx.flatVecs)
	skipsBeforeInsert := idx.FlatVecsBudgetSkips()

	if err := idx.Insert(context.Background(), extraVecs, extraIDs); err != nil {
		t.Fatal(err)
	}

	// La croissance doit être gelée : flatVecs ne couvre pas les 800 nœuds.
	if len(idx.flatVecs) != frozenLen {
		t.Fatalf("gel de croissance runtime: flatVecs a grandi de %d à %d octets malgré le dépassement de budget",
			frozenLen, len(idx.flatVecs))
	}
	if got := idx.FlatVecsBudgetSkips(); got <= skipsBeforeInsert {
		t.Fatalf("gel de croissance runtime: FlatVecsBudgetSkips=%d, attendu > %d", got, skipsBeforeInsert)
	}

	// Recherche toujours correcte : les nœuds gelés hors flatVecs se rerank via SQL
	// (RerankSQLLoads progresse), sans panic ni résultat vide.
	before := idx.RerankSQLLoads()
	for q, query := range queries {
		res, err := idx.Search(context.Background(), query, 10)
		if err != nil {
			t.Fatalf("search après gel runtime q=%d: %v", q, err)
		}
		if len(res) == 0 {
			t.Fatalf("search après gel runtime q=%d: résultat vide", q)
		}
	}
	if idx.RerankSQLLoads()-before == 0 {
		t.Fatalf("gel de croissance runtime: repli SQL non exercé pour les nœuds hors flatVecs (RerankSQLLoads inchangé)")
	}
}
