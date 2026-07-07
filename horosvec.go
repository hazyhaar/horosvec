package horosvec

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"math"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"

	_ "modernc.org/sqlite"
)

// streamingBuildGCPercent bride le GOGC pendant le build streaming (restauré ensuite).
// Une valeur basse plafonne le pic RSS au prix d'un GC plus fréquent — le build est un
// moment froid, la latence n'y importe pas, contrairement au pic mémoire.
const streamingBuildGCPercent = 25

// Config controls the behavior of the Vamana index.
type Config struct {
	MaxDegree      int // R: max neighbors per node (default 64)
	SearchListSize int // L: beam width during build (default 128)
	BuildPasses    int // number of graph construction passes (default 2)
	EfSearch       int // beam width during search (default 128)
	RerankTopN     int // top-N candidates to rerank with exact vectors (default 500)
	CacheCapacity  int // LRU cache capacity in nodes (default 100000)

	BruteForceThreshold int // below this count, skip Vamana and scan all vectors (default 50000)

	Alpha          float64 // pruning parameter, >1 for longer edges (default 1.2)
	DriftThreshold float64 // centroid drift ratio to trigger rebuild (default 0.05)
	// InsertRatioThreshold: inserts/buildCount ratio to trigger rebuild (default 1.0).
	// 1.0 means rebuild only after insert volume matches the built corpus; 0.30 fired too
	// often on incremental backfill and thrashed async rebuilds without recall gain.
	InsertRatioThreshold float64

	// MedoidStaleThreshold: recompute medoid after this many inserts since last build/recompute.
	// 0 disables the periodic recompute (stale-ratio guard still applies).
	MedoidStaleThreshold int // default 10000

	// RotationRounds: randomized Hadamard rounds for RaBitQ (default 1).
	// 0 disables rotation (identity); legacy indexes without rotation meta use 0.
	RotationRounds int
	// RotationSeed: PCG seed for diagonal signs. 0 means defaultRotationSeed (42).
	RotationSeed uint64

	// ArenaPath, when non-empty, points to a flat fp16 arena file (produced by
	// ExportArena). When present and valid at New(), the exact-rerank stage of
	// Search reads raw vectors from the arena (fp16→fp32 on the fly) instead of
	// loadNodeReadOnly — no SQL access, no LRU cache pollution on the hot path.
	// Empty (default): behaviour strictly unchanged.
	ArenaPath string

	// BuildWorkers controls parallel Vamana graph construction.
	// 0 (default) uses ~40% of the CPU capacity available to the process
	// (runtime.GOMAXPROCS(0), which is cgroup-aware since Go 1.25) — horosvec is
	// an embedded library and must not starve its host application during a
	// build; claim more explicitly (e.g. runtime.GOMAXPROCS(0)) for dedicated
	// offline builds. 1 forces the legacy sequential path (bit-identical PCG
	// draws and graph). Parallel builds (BuildWorkers != 1) are not bit-for-bit
	// deterministic due to interleaving order; graph quality is equivalent
	// (standard DiskANN). Use BuildWorkers=1 when exact reproducibility is required.
	BuildWorkers int
}

// effectiveBuildWorkers resolves Config.BuildWorkers: 1 = sequential legacy,
// 0 = polite default (~40% of available CPU capacity, min 1).
func effectiveBuildWorkers(cfg Config) int {
	if cfg.BuildWorkers == 1 {
		return 1
	}
	if cfg.BuildWorkers > 1 {
		return cfg.BuildWorkers
	}
	w := (runtime.GOMAXPROCS(0)*4 + 9) / 10 // ceil(40%)
	if w < 1 {
		w = 1
	}
	return w
}

// encodeGraphNodes rotates and RaBitQ-encodes vectors into graph nodes.
// workers<=1 uses the sequential path; workers>1 parallelizes by contiguous batches.
func encodeGraphNodes(
	ctx context.Context,
	rotator *Rotator,
	encoder *Encoder,
	allVecs [][]float32,
	allIDs [][]byte,
	workers int,
) ([]graphNode, error) {
	n := len(allVecs)
	nodes := make([]graphNode, n)
	if workers <= 1 {
		rotated := make([]float32, rotator.CodeDim())
		for i, v := range allVecs {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			rotator.Rotate(v, rotated)
			code, sqNorm, l1Norm := encoder.Encode(rotated)
			nodes[i] = graphNode{
				id:     int64(i),
				extID:  allIDs[i],
				vec:    v,
				code:   code,
				sqNorm: sqNorm,
				l1Norm: l1Norm,
			}
		}
		return nodes, nil
	}

	// Rotator reuses fhtScratch; clone one rotator per worker for thread safety.
	workerRotators := make([]*Rotator, workers)
	for i := range workers {
		workerRotators[i] = NewRotatorWithCodeDim(
			rotator.Dim(), rotator.CodeDim(), rotator.Rounds(), rotator.Seed(),
		)
	}

	var wg sync.WaitGroup
	var firstErr atomic.Pointer[error]
	batchSize := (n + workers - 1) / workers
	workerIdx := 0
	for batchStart := 0; batchStart < n; batchStart += batchSize {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		batchEnd := batchStart + batchSize
		if batchEnd > n {
			batchEnd = n
		}
		wRot := workerRotators[workerIdx]
		workerIdx++
		wg.Add(1)
		go func(start, end int, wr *Rotator) {
			defer wg.Done()
			rotated := make([]float32, wr.CodeDim())
			for i := start; i < end; i++ {
				if firstErr.Load() != nil {
					return
				}
				if err := ctx.Err(); err != nil {
					firstErr.CompareAndSwap(nil, &err)
					return
				}
				wr.Rotate(allVecs[i], rotated)
				code, sqNorm, l1Norm := encoder.Encode(rotated)
				nodes[i] = graphNode{
					id:     int64(i),
					extID:  allIDs[i],
					vec:    allVecs[i],
					code:   code,
					sqNorm: sqNorm,
					l1Norm: l1Norm,
				}
			}
		}(batchStart, batchEnd, wRot)
	}
	wg.Wait()
	if err := firstErr.Load(); err != nil {
		return nil, *err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return nodes, nil
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		MaxDegree:            64,
		SearchListSize:       128,
		BuildPasses:          2,
		EfSearch:             128,
		RerankTopN:           500,
		CacheCapacity:        100_000,
		BruteForceThreshold:  50_000,
		Alpha:                1.2,
		DriftThreshold:       0.05,
		InsertRatioThreshold: 1.0,
		MedoidStaleThreshold: 10_000,
		RotationRounds:       1,
	}
}

// VectorIterator provides vectors for batch building.
type VectorIterator interface {
	Next() (id []byte, vec []float32, ok bool)
	Reset() error
}

// Result represents a search result.
type Result struct {
	ID    []byte
	Score float64 // lower = closer (L2 distance squared)
}

// Index is a Vamana ANN index backed by SQLite.
type Index struct {
	db                  *sql.DB
	cfg                 Config
	cache               *nodeCache
	encoder             *Encoder
	centroid            *CentroidTracker
	medoid              int64
	dim                 int
	codeDim             int
	rotator             *Rotator
	nextID              int64
	built               bool
	insertedSinceMedoid int64 // resets on Build or medoid recompute

	// Flat vector storage for zero-alloc brute-force search.
	// flatVecs is contiguous: [node0_dim0..node0_dimD, node1_dim0..node1_dimD, ...].
	flatVecs []float32
	flatIDs  [][]byte

	mu        sync.RWMutex // protects searches vs. structural changes
	rebuildMu sync.Mutex   // serializes rebuilds

	// testDuringRebuildPersist is set by tests to run code while idx.mu is held during rebuild persist.
	testDuringRebuildPersist func()

	// testBeforeInsertCommit is set by tests to sabotage the insert transaction before commit.
	testBeforeInsertCommit func(tx *sql.Tx)

	// testDuringGreedySearch is set by tests to run code once per heap iteration in rabitqGreedySearchInternal.
	testDuringGreedySearch func()

	// standalone mode (db == nil): meta key/value pairs loaded from binary export,
	// preserved for round-trip ExportBinary. Insertion order kept via standaloneMetaKeys.
	standaloneMeta     map[string][]byte
	standaloneMetaKeys []string

	// plan chaud : arènes plates pointer-free pour la boucle greedy (fiche 1 hnswlib).
	plane      *hotPlane
	planePatch map[int32][]int32 // overlay voisinages re-câblés depuis dernier buildHotPlane

	// arena : vecteurs bruts fp16 plats en lecture seule pour l'étape de rerank
	// (opt-in via Config.ArenaPath). nil = chemin loadNodeReadOnly (SQL) inchangé.
	arena *arena
	// rerankSQLLoads compte les chargements SQL (loadNodeReadOnly) déclenchés par
	// la boucle de rerank. Avec une arène active couvrant tous les nœuds, ce
	// compteur reste à 0 (critère C2). Instrumentation d'observabilité.
	rerankSQLLoads atomic.Int64
}

// RerankSQLLoads retourne le nombre cumulé de chargements SQL déclenchés par l'étape
// de rerank exact du Search. Avec une arène active couvrant tous les nœuds, ce compteur
// reste à 0 (aucun accès SQL, aucune pollution du cache LRU sur le chemin chaud).
func (idx *Index) RerankSQLLoads() int64 {
	return idx.rerankSQLLoads.Load()
}

// New creates or loads an Index from the given database.
func New(db *sql.DB, cfg Config) (*Index, error) {
	if err := configureSQLite(db); err != nil {
		return nil, fmt.Errorf("horosvec: configure sqlite: %w", err)
	}

	if err := initSchema(db); err != nil {
		return nil, fmt.Errorf("horosvec: init schema: %w", err)
	}

	idx := &Index{
		db:    db,
		cfg:   cfg,
		cache: newNodeCache(cfg.CacheCapacity),
	}

	// Try to load existing index
	medoid, dim, nodeCount, centroid, vectorsAtBuild, err := loadIndex(db)
	if err == nil && nodeCount > 0 {
		rotMeta, rotErr := loadRotationMeta(db, dim)
		if rotErr != nil {
			return nil, rotErr
		}
		idx.medoid = medoid
		idx.dim = dim
		idx.codeDim = rotMeta.codeDim
		idx.rotator = NewRotatorWithCodeDim(dim, rotMeta.codeDim, rotMeta.rounds, rotMeta.seed)
		idx.encoder = NewEncoder(centroid)
		idx.centroid = NewCentroidTracker(dim, cfg.DriftThreshold, cfg.InsertRatioThreshold)
		idx.centroid.SetCentroid(centroid, int64(nodeCount))
		idx.centroid.SetBuildCentroid(centroid, vectorsAtBuild)
		idx.built = true

		maxID, err := getMaxNodeID(db)
		if err == nil {
			idx.nextID = maxID + 1
		}

		// Guard against stale medoid (e.g. node deleted or incremental inserts
		// added new nodes without recomputing). Recompute and persist if needed.
		refreshed, err := checkAndRefreshMedoid(db, medoid)
		if err != nil {
			return nil, fmt.Errorf("horosvec: validate medoid: %w", err)
		}
		idx.medoid = refreshed

		warmCache(context.Background(), db, idx.cache, idx.medoid, 2)

		// Load flat vectors for brute-force search. En mode arène (ArenaPath posé), le
		// SQLite est vector-less (blobs vides) : loadFlatVectors produirait un flatVecs vide
		// et ferait paniquer bruteForceFlat. La source des vecteurs est alors l'arène, lue
		// par bruteForceArena — on NE charge donc PAS le flat.
		if nodeCount <= cfg.BruteForceThreshold && cfg.ArenaPath == "" {
			idx.loadFlatVectors()
		}

		if err := idx.rebuildPlaneLocked(); err != nil {
			return nil, fmt.Errorf("horosvec: build hot plane on load: %w", err)
		}

		// Arène fp16 opt-in : ouverte et validée contre (dim, count) de l'index.
		if cfg.ArenaPath != "" {
			ar, arErr := openArena(cfg.ArenaPath)
			if arErr != nil {
				return nil, arErr
			}
			if ar.dim != idx.dim {
				return nil, fmt.Errorf("horosvec: arena dim %d != index dim %d", ar.dim, idx.dim)
			}
			if ar.count != int64(nodeCount) {
				return nil, fmt.Errorf("horosvec: arena count %d != index node_count %d", ar.count, nodeCount)
			}
			idx.arena = ar
		}
	}

	return idx, nil
}

// loadFlatVectors loads all vectors into contiguous memory for brute-force search.
func (idx *Index) loadFlatVectors() {
	rows, err := idx.db.Query("SELECT ext_id, vector FROM vindex_nodes ORDER BY node_id")
	if err != nil {
		return
	}
	defer rows.Close()

	flatVecs := make([]float32, 0, int(idx.nextID)*idx.dim)
	flatIDs := make([][]byte, 0, idx.nextID)
	for rows.Next() {
		var extID, vecBlob []byte
		if err := rows.Scan(&extID, &vecBlob); err != nil {
			return
		}
		vec := deserializeFloat32s(vecBlob)
		flatVecs = append(flatVecs, vec...)
		flatIDs = append(flatIDs, extID)
	}
	if rows.Err() != nil {
		return
	}
	idx.flatVecs = flatVecs
	idx.flatIDs = flatIDs
}

// Build constructs the full index from the given iterator.
// Takes exclusive mu.Lock — blocks all Search/Insert. Mutates: dim, encoder, centroid, medoid, built, flatVecs, flatIDs, nextID. Must be called before Insert/Search.
func (idx *Index) Build(ctx context.Context, iter VectorIterator) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Chemin streaming vector-less (opt-in via Config.ArenaPath) : consomme l'itérateur
	// en flux vers une arène fp16 écrite au fil de l'eau, construit le graphe depuis
	// l'arène et persiste un SQLite allégé (sans blob vecteur). Le chemin par défaut
	// (ArenaPath vide) reste strictement inchangé — petits index brute-force compris.
	if idx.cfg.ArenaPath != "" {
		return idx.buildStreamingArena(ctx, iter)
	}

	var allVecs [][]float32
	var allIDs [][]byte

	for {
		id, vec, ok := iter.Next()
		if !ok {
			break
		}
		idCopy := make([]byte, len(id))
		copy(idCopy, id)
		vecCopy := make([]float32, len(vec))
		copy(vecCopy, vec)
		allVecs = append(allVecs, vecCopy)
		allIDs = append(allIDs, idCopy)
	}

	if len(allVecs) == 0 {
		return fmt.Errorf("horosvec: no vectors provided")
	}

	dim := len(allVecs[0])
	if dim == 0 {
		return fmt.Errorf("horosvec: first vector is empty")
	}
	for i, v := range allVecs {
		if len(v) != dim {
			return fmt.Errorf("horosvec: heterogeneous vector dim: vec[%d] has %d, want %d", i, len(v), dim)
		}
	}
	idx.dim = dim

	rounds := idx.cfg.RotationRounds
	seed := effectiveRotationSeed(idx.cfg)
	idx.rotator = NewRotator(dim, rounds, seed)
	idx.codeDim = idx.rotator.CodeDim()

	rotated := make([]float32, idx.codeDim)
	centroid := make([]float32, idx.codeDim)
	for _, v := range allVecs {
		idx.rotator.Rotate(v, rotated)
		for j, val := range rotated {
			centroid[j] += val
		}
	}
	invN := float32(1.0 / float64(len(allVecs)))
	for j := range idx.codeDim {
		centroid[j] *= invN
	}

	idx.encoder = NewEncoder(centroid)
	idx.centroid = NewCentroidTracker(dim, idx.cfg.DriftThreshold, idx.cfg.InsertRatioThreshold)
	idx.centroid.AddBatch(allVecs)
	idx.centroid.SnapshotBuild()

	buildWorkers := effectiveBuildWorkers(idx.cfg)
	nodes, err := encodeGraphNodes(ctx, idx.rotator, idx.encoder, allVecs, allIDs, buildWorkers)
	if err != nil {
		return err
	}

	idx.medoid = findMedoid(nodes)
	if err := buildGraph(ctx, nodes, idx.medoid, idx.cfg.MaxDegree, idx.cfg.SearchListSize, idx.cfg.Alpha, idx.cfg.BuildPasses, buildWorkers); err != nil {
		return err
	}

	// Persist
	tx, err := idx.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("horosvec: begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if _, err := tx.Exec("DELETE FROM vindex_nodes"); err != nil {
		return fmt.Errorf("horosvec: clear nodes: %w", err)
	}
	if _, err := tx.Exec("DELETE FROM vindex_meta"); err != nil {
		return fmt.Errorf("horosvec: clear meta: %w", err)
	}

	rotMeta := rotationMeta{
		seed:     seed,
		rounds:   rounds,
		codeDim:  idx.codeDim,
		dbFormat: currentDBFormatVersion,
	}
	if err := saveGraph(tx, nodes, idx.medoid, dim, idx.cfg.MaxDegree, centroid, rotMeta); err != nil {
		return fmt.Errorf("horosvec: save graph: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("horosvec: commit: %w", err)
	}

	idx.nextID = int64(len(nodes))
	idx.built = true
	idx.insertedSinceMedoid = 0

	// Populate cache
	idx.cache.clear()
	for i := range nodes {
		idx.cache.put(&cachedNode{
			nodeID:    nodes[i].id,
			extID:     nodes[i].extID,
			neighbors: nodes[i].neighbors,
			vec:       nodes[i].vec,
			code:      nodes[i].code,
			sqNorm:    nodes[i].sqNorm,
			l1Norm:    nodes[i].l1Norm,
		})
	}

	// Build flat vector array for brute-force search
	idx.flatVecs = make([]float32, len(allVecs)*dim)
	idx.flatIDs = make([][]byte, len(allVecs))
	for i, v := range allVecs {
		copy(idx.flatVecs[i*dim:], v)
		idx.flatIDs[i] = allIDs[i]
	}

	if err := idx.rebuildPlaneLocked(); err != nil {
		return fmt.Errorf("horosvec: build hot plane: %w", err)
	}
	// Payer la collecte des déchets de construction au moment froid et déterministe,
	// jamais pendant les requêtes.
	runtime.GC()

	return nil
}

// buildStreamingArena is the memory-lean build path (opt-in via Config.ArenaPath).
//
// Pass 1 drains the iterator as a stream, writing each vector fp16 into a flat arena on
// disk (fil de l'eau) — never materializing allVecs [][]float32 in fp32. Only the ext_ids
// are kept in RAM. Pass 2 opens the arena, decodes it into ONE contiguous fp32 build buffer
// (freed before returning), builds the Vamana graph on those fp16-rounded vectors, and
// persists a vector-less SQLite (graph + codes + norms + meta, empty vector blob). The arena
// then serves both the graph build and the Search rerank — graph and rerank see the SAME
// fp16 vectors (consistency). Caller must hold idx.mu.Lock (Build does).
func (idx *Index) buildStreamingArena(ctx context.Context, iter VectorIterator) error {
	// --- Pass 1: stream iterator → fp16 arena on disk, keep only ext_ids in RAM. ---
	var (
		dim    int
		writer *arenaWriter
		allIDs [][]byte
	)
	for {
		id, vec, ok := iter.Next()
		if !ok {
			break
		}
		if err := ctx.Err(); err != nil {
			if writer != nil {
				writer.abort()
			}
			return fmt.Errorf("horosvec: %w", err)
		}
		if writer == nil {
			dim = len(vec)
			if dim == 0 {
				return fmt.Errorf("horosvec: first vector is empty")
			}
			var err error
			writer, err = newArenaWriter(idx.cfg.ArenaPath, dim)
			if err != nil {
				return err
			}
		} else if len(vec) != dim {
			writer.abort()
			return fmt.Errorf("horosvec: heterogeneous vector dim: got %d, want %d", len(vec), dim)
		}
		if err := writer.writeVec(vec); err != nil {
			writer.abort()
			return err
		}
		idCopy := make([]byte, len(id))
		copy(idCopy, id)
		allIDs = append(allIDs, idCopy)
	}
	if writer == nil {
		return fmt.Errorf("horosvec: no vectors provided")
	}
	if err := writer.finalize(); err != nil {
		return err
	}

	// --- Pass 2: decode arena into a transient contiguous fp32 build buffer. ---
	// Cap le GOGC pendant le build : le graphe alloue beaucoup de scratch éphémère
	// (greedySearch/robustPrune) ; sans bride, l'amplification GC (GOGC=100 double le tas
	// vivant) fait exploser le pic RSS. Restauré en fin de build (defer).
	prevGC := debug.SetGCPercent(streamingBuildGCPercent)
	defer debug.SetGCPercent(prevGC)

	// Décoder l'arène fp16 en un tampon fp32 contigu, puis LIBÉRER immédiatement les octets
	// fp16 (ar.data) AVANT buildGraph : ne pas maintenir simultanément fp16 (256 Mo/1M) et
	// fp32 (512 Mo/1M). L'arène de rerank est rouverte après la persistance.
	ar, err := openArena(idx.cfg.ArenaPath)
	if err != nil {
		return err
	}
	n := int(ar.count)
	if n != len(allIDs) {
		return fmt.Errorf("horosvec: arena count %d != ext_id count %d", n, len(allIDs))
	}
	buildFlat := make([]float32, n*dim)
	rowsView := make([][]float32, n)
	for i := 0; i < n; i++ {
		row := buildFlat[i*dim : (i+1)*dim]
		if !ar.vecInto(int64(i), row) {
			return fmt.Errorf("horosvec: arena decode node %d out of range", i)
		}
		rowsView[i] = row
	}
	ar = nil // libérer les octets fp16 : buildGraph ne travaille que sur buildFlat (fp32)
	runtime.GC()

	idx.dim = dim
	rounds := idx.cfg.RotationRounds
	seed := effectiveRotationSeed(idx.cfg)
	idx.rotator = NewRotator(dim, rounds, seed)
	idx.codeDim = idx.rotator.CodeDim()

	rotated := make([]float32, idx.codeDim)
	centroid := make([]float32, idx.codeDim)
	for _, v := range rowsView {
		idx.rotator.Rotate(v, rotated)
		for j, val := range rotated {
			centroid[j] += val
		}
	}
	invN := float32(1.0 / float64(n))
	for j := range idx.codeDim {
		centroid[j] *= invN
	}

	idx.encoder = NewEncoder(centroid)
	idx.centroid = NewCentroidTracker(dim, idx.cfg.DriftThreshold, idx.cfg.InsertRatioThreshold)
	idx.centroid.AddBatch(rowsView)
	idx.centroid.SnapshotBuild()

	buildWorkers := effectiveBuildWorkers(idx.cfg)
	nodes, err := encodeGraphNodes(ctx, idx.rotator, idx.encoder, rowsView, allIDs, buildWorkers)
	if err != nil {
		return err
	}

	idx.medoid = findMedoid(nodes)
	if err := buildGraph(ctx, nodes, idx.medoid, idx.cfg.MaxDegree, idx.cfg.SearchListSize, idx.cfg.Alpha, idx.cfg.BuildPasses, buildWorkers); err != nil {
		return err
	}

	// Persist: vector-less SQLite (arena is the source of raw vectors).
	tx, err := idx.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("horosvec: begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if _, err := tx.Exec("DELETE FROM vindex_nodes"); err != nil {
		return fmt.Errorf("horosvec: clear nodes: %w", err)
	}
	if _, err := tx.Exec("DELETE FROM vindex_meta"); err != nil {
		return fmt.Errorf("horosvec: clear meta: %w", err)
	}
	rotMeta := rotationMeta{
		seed:     seed,
		rounds:   rounds,
		codeDim:  idx.codeDim,
		dbFormat: currentDBFormatVersion,
	}
	if err := saveGraphNoVec(tx, nodes, idx.medoid, dim, idx.cfg.MaxDegree, centroid, rotMeta); err != nil {
		return fmt.Errorf("horosvec: save graph: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("horosvec: commit: %w", err)
	}

	idx.nextID = int64(n)
	idx.built = true
	idx.insertedSinceMedoid = 0

	// Gros index vector-less : pas de flatVecs brute-force (le chemin brute-force n'est
	// atteint que sous BruteForceThreshold, jamais en mode arène).
	idx.flatVecs = nil
	idx.flatIDs = nil
	idx.cache.clear()

	// Libérer explicitement le tampon fp32 transitoire AVANT de rouvrir l'arène de rerank :
	// on ne maintient jamais simultanément le fp32 de build (512 Mo/1M) et le fp16 de rerank
	// (256 Mo/1M).
	nodes = nil
	buildFlat = nil
	rowsView = nil
	runtime.GC()

	// Rouvrir l'arène fp16 en lecture seule : seule source résidente des vecteurs bruts au
	// rerank. (dim, count) sont garantis cohérents par construction.
	rerankArena, err := openArena(idx.cfg.ArenaPath)
	if err != nil {
		return err
	}

	if err := idx.rebuildPlaneLocked(); err != nil {
		return fmt.Errorf("horosvec: build hot plane: %w", err)
	}
	idx.arena = rerankArena

	// Payer la collecte des déchets de construction au moment froid et déterministe.
	runtime.GC()
	return nil
}

// Search finds the topK nearest neighbors.
// For small indices (<= BruteForceThreshold): exact brute-force scan, 100% recall.
// For large indices: 2-stage RaBitQ beam search + L2 rerank.
// RLock allows concurrent Search but not Insert. vamanaSearch uses idx.nextID as bitset capacity — if Insert grows nextID concurrently, bitset undersized.
func (idx *Index) Search(ctx context.Context, query []float32, topK int) ([]Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("horosvec: %w", err)
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if !idx.built {
		return nil, fmt.Errorf("horosvec: index not built")
	}
	if len(query) != idx.dim {
		return nil, fmt.Errorf("horosvec: query dim %d != index dim %d", len(query), idx.dim)
	}
	// Garde topK<=0 : les variantes brute-force (flat/arène/SQL) lisent best[len(best)-1]
	// après un append-puis-tronque à best[:0], ce qui paniquerait sur best[-1] pour topK=0.
	// Aucun voisin demandé → résultat vide.
	if topK <= 0 {
		return nil, nil
	}

	// Dynamic threshold: brute-force for small shards, Vamana+RaBitQ for large
	if idx.cfg.BruteForceThreshold > 0 && int(idx.nextID) <= idx.cfg.BruteForceThreshold {
		return idx.bruteForceSearch(ctx, query, topK)
	}

	return idx.vamanaSearch(ctx, query, topK)
}

// bruteForceSearch scans all vectors with exact L2. 100% recall, O(N).
// Uses flat in-memory vectors when available (zero-alloc hot path),
// falls back to SQL scan otherwise.
func (idx *Index) bruteForceSearch(ctx context.Context, query []float32, topK int) ([]Result, error) {
	if idx.flatVecs != nil {
		return idx.bruteForceFlat(query, topK), nil
	}
	// Mode arène (SQLite vector-less) : les blobs vecteur sont vides ; la source des
	// vecteurs est l'arène fp16. Sans ce routage, bruteForceSQLite lirait des blobs vides
	// et l2DistanceSquared paniquerait.
	if idx.arena != nil {
		return idx.bruteForceArena(query, topK), nil
	}
	return idx.bruteForceSQLite(ctx, query, topK)
}

// bruteForceArena scanne exactement tous les vecteurs depuis l'arène fp16 (100 % rappel).
// Utilisé quand l'index est en mode arène (SQLite sans blob vecteur) et sous le seuil
// force-brute. Les ext_ids sont résolus depuis le plan chaud (couverture dense garantie).
func (idx *Index) bruteForceArena(query []float32, topK int) []Result {
	n := int(idx.arena.count)
	buf := make([]float32, idx.dim)
	best := make([]Result, 0, topK+1)
	worstDist := math.MaxFloat64

	for i := 0; i < n; i++ {
		if !idx.arena.vecInto(int64(i), buf) {
			continue
		}
		d := l2DistanceSquared(query, buf)
		if len(best) < topK || d < worstDist {
			extID := idx.planeExtID(int64(i))
			if extID == nil {
				continue
			}
			idCopy := make([]byte, len(extID))
			copy(idCopy, extID)
			best = append(best, Result{ID: idCopy, Score: d})
			sortResults(best)
			if len(best) > topK {
				best = best[:topK]
			}
			worstDist = best[len(best)-1].Score
		}
	}
	return best
}

// bruteForceFlat scans contiguous in-memory vectors. Zero allocs on hot path.
func (idx *Index) bruteForceFlat(query []float32, topK int) []Result {
	n := len(idx.flatIDs)
	dim := idx.dim
	best := make([]Result, 0, topK+1)
	worstDist := math.MaxFloat64

	for i := range n {
		vec := idx.flatVecs[i*dim : (i+1)*dim]
		d := l2DistanceSquared(query, vec)

		if len(best) < topK || d < worstDist {
			best = append(best, Result{ID: idx.flatIDs[i], Score: d})
			sortResults(best)
			if len(best) > topK {
				best = best[:topK]
			}
			worstDist = best[len(best)-1].Score
		}
	}
	return best
}

// bruteForceSQLite scans all vectors from the database. Used when flat vectors aren't available.
func (idx *Index) bruteForceSQLite(ctx context.Context, query []float32, topK int) ([]Result, error) {
	rows, err := idx.db.QueryContext(ctx, "SELECT ext_id, vector FROM vindex_nodes")
	if err != nil {
		return nil, fmt.Errorf("horosvec: brute force scan: %w", err)
	}
	defer rows.Close()

	best := make([]Result, 0, topK+1)
	worstDist := math.MaxFloat64

	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("horosvec: %w", err)
		}
		var extID, vecBlob []byte
		if err := rows.Scan(&extID, &vecBlob); err != nil {
			continue
		}
		vec := deserializeFloat32s(vecBlob)
		d := l2DistanceSquared(query, vec)

		if len(best) < topK || d < worstDist {
			idCopy := make([]byte, len(extID))
			copy(idCopy, extID)
			best = append(best, Result{ID: idCopy, Score: d})
			sortResults(best)
			if len(best) > topK {
				best = best[:topK]
			}
			worstDist = best[len(best)-1].Score
		}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("horosvec: brute force scan: %w", err)
	}

	return best, nil
}

// vamanaSearch does 2-stage RaBitQ beam search + L2 rerank.
// Called under RLock. loadNodeReadOnly fallback to loadNode may promote cache entry (LRU side effect under read path).
func (idx *Index) vamanaSearch(ctx context.Context, query []float32, topK int) ([]Result, error) {
	// Beam width is the user's speed/recall knob; it must at least feed the
	// rerank floor (3*topK) but is NEVER inflated to RerankTopN — the old
	// coupling (efSearch = max(efSearch, rerankN=500)) silently floored every
	// beam below 500, making EfSearch decorative across its useful range
	// (found on the SIFT bench: identical recall and QPS from ef=64 to 512).
	// Rerank adapts to the beam, not the other way around.
	efSearch := idx.cfg.EfSearch
	if efSearch < topK*3 {
		efSearch = topK * 3
	}
	rerankN := idx.cfg.RerankTopN
	if rerankN < topK*3 {
		rerankN = topK * 3
	}
	if rerankN > efSearch {
		rerankN = efSearch
	}

	// Stage 1: RaBitQ beam search on rotated/padded query (separate from raw query)
	queryRotated := make([]float32, idx.codeDim)
	idx.rotator.Rotate(query, queryRotated)
	candidates, err := idx.rabitqGreedySearch(ctx, queryRotated, efSearch)
	if err != nil {
		return nil, err
	}

	// Stage 2: Re-rank top candidates with exact L2
	if len(candidates) > rerankN {
		candidates = candidates[:rerankN]
	}
	// Buffer réutilisé pour décoder un vecteur d'arène (fp16→fp32) sans allocation
	// par candidat. Alloué une fois par Search.
	var arenaVec []float32
	if idx.arena != nil {
		arenaVec = make([]float32, idx.dim)
	}
	for i := range candidates {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("horosvec: %w", err)
		}
		// Chemin recâblé : lire le vecteur brut depuis l'arène fp16 (aucun SQL,
		// aucune pollution du cache LRU). Sans arène, chemin loadNodeReadOnly inchangé.
		if idx.arena != nil && idx.arena.vecInto(candidates[i].nodeID, arenaVec) {
			candidates[i].dist = l2DistanceSquared(query, arenaVec)
			continue
		}
		node, err := loadNodeReadOnly(ctx, idx.db, idx.cache, candidates[i].nodeID)
		idx.rerankSQLLoads.Add(1)
		if err != nil {
			candidates[i].dist = math.MaxFloat64
			continue
		}
		candidates[i].dist = l2DistanceSquared(query, node.vec)
	}
	sortCandidates(candidates)

	if len(candidates) > topK {
		candidates = candidates[:topK]
	}

	// Resolve ext_ids from hot plane when covered, else cache/SQL.
	results := make([]Result, 0, len(candidates))
	for _, c := range candidates {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("horosvec: %w", err)
		}
		var extID []byte
		if extID = idx.planeExtID(c.nodeID); extID == nil {
			node, err := loadNodeReadOnly(ctx, idx.db, idx.cache, c.nodeID)
			if err != nil || node.extID == nil {
				continue
			}
			extID = node.extID
		}
		idCopy := make([]byte, len(extID))
		copy(idCopy, extID)
		results = append(results, Result{ID: idCopy, Score: c.dist})
	}

	return results, nil
}

// sortResults sorts results by score (ascending).
func sortResults(results []Result) {
	for i := 1; i < len(results); i++ {
		key := results[i]
		j := i - 1
		for j >= 0 && results[j].Score > key.Score {
			results[j+1] = results[j]
			j--
		}
		results[j+1] = key
	}
}

// rabitqGreedySearch performs best-first beam search on the Vamana graph.
func (idx *Index) rabitqGreedySearch(ctx context.Context, query []float32, beamWidth int) ([]searchCandidate, error) {
	return idx.rabitqGreedySearchInternal(ctx, query, beamWidth, idx.nextID, nil)
}

// rabitqGreedySearchInternal performs beam search with an optional node loader and capacity.
// When loadFn is nil, committed nodes are loaded via loadNodeReadOnly.
// Acquires searchState from sync.Pool. If panic before defer release, state leaked. Pool may return dirty state from previous query.
func (idx *Index) rabitqGreedySearchInternal(ctx context.Context, query []float32, beamWidth int, maxNodes int64, loadFn func(int64) (*cachedNode, error)) ([]searchCandidate, error) {
	state := acquireSearchState(maxNodes, beamWidth, idx.codeDim)
	defer releaseSearchState(state)

	// Pre-compute query centering and squared norm (into pooled buffer).
	// query is already rotated/padded; encoder centroid lives in the same space.
	var querySqNorm float64
	for i := range idx.codeDim {
		c := float64(query[i]) - float64(idx.encoder.centroid[i])
		state.queryCentered[i] = c
		querySqNorm += c * c
	}
	buildRabitqLUT(state.queryCentered, state.lut)

	medoidCode, medoidSq, medoidL1, err := idx.greedyNodeCodeNorms(ctx, idx.medoid, loadFn)
	if err != nil {
		return nil, nil
	}
	state.visit(idx.medoid)

	startDist := rabitqDistanceLUT(state.lut, querySqNorm, medoidCode, medoidSq, medoidL1)

	state.pushHeap(searchCandidate{nodeID: idx.medoid, dist: startDist})
	state.insertBest(searchCandidate{nodeID: idx.medoid, dist: startDist}, beamWidth)
	worstBest := startDist

	for state.heapLen > 0 {
		if idx.testDuringGreedySearch != nil {
			idx.testDuringGreedySearch()
		}
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("horosvec: greedy search: %w", err)
		}

		cur := state.popHeap()

		if len(state.best) >= beamWidth && cur.dist > worstBest {
			break
		}

		if err := idx.greedyEachNeighbor(ctx, cur.nodeID, loadFn, func(nbr int64) bool {
			if !state.visit(nbr) {
				return true
			}

			nbrCode, nbrSq, nbrL1, err := idx.greedyNodeCodeNorms(ctx, nbr, loadFn)
			if err != nil {
				return true
			}

			d := rabitqDistanceLUT(state.lut, querySqNorm, nbrCode, nbrSq, nbrL1)

			if len(state.best) < beamWidth || d < worstBest {
				state.pushHeap(searchCandidate{nodeID: nbr, dist: d})
				state.insertBest(searchCandidate{nodeID: nbr, dist: d}, beamWidth)
				worstBest = state.worstBestDist()
			}
			return true
		}); err != nil {
			continue
		}
	}

	// Copy results out (state is pooled and will be reused)
	result := make([]searchCandidate, len(state.best))
	copy(result, state.best)
	return result, nil
}

// SearchWithRerank searches and then reranks top candidates using exact vectors.
func (idx *Index) SearchWithRerank(ctx context.Context, query []float32, topK int, reranker func([][]byte) ([][]float32, error)) ([]Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("horosvec: %w", err)
	}
	// Garde topK<=0 : SearchWithRerank tronque les candidats avec le topK de l'appelant
	// (candidates[:topK]) indépendamment de la garde de Search ; un topK négatif y ferait
	// paniquer candidates[:-1]. Aucun voisin demandé → résultat vide.
	if topK <= 0 {
		return nil, nil
	}

	rerankN := idx.cfg.RerankTopN
	if rerankN < topK*3 {
		rerankN = topK * 3
	}

	candidates, err := idx.Search(ctx, query, rerankN)
	if err != nil {
		return nil, err
	}

	if reranker == nil || len(candidates) == 0 {
		if len(candidates) > topK {
			candidates = candidates[:topK]
		}
		return candidates, nil
	}

	ids := make([][]byte, len(candidates))
	for i, c := range candidates {
		ids[i] = c.ID
	}

	exactVecs, err := reranker(ids)
	if err != nil {
		if len(candidates) > topK {
			candidates = candidates[:topK]
		}
		return candidates, nil
	}

	type scored struct {
		id   []byte
		dist float64
	}
	reranked := make([]scored, 0, len(exactVecs))
	for i, vec := range exactVecs {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("horosvec: %w", err)
		}
		if vec == nil {
			continue
		}
		d := l2DistanceSquared(query, vec)
		reranked = append(reranked, scored{id: ids[i], dist: d})
	}

	for i := 1; i < len(reranked); i++ {
		key := reranked[i]
		j := i - 1
		for j >= 0 && reranked[j].dist > key.dist {
			reranked[j+1] = reranked[j]
			j--
		}
		reranked[j+1] = key
	}

	if len(reranked) > topK {
		reranked = reranked[:topK]
	}

	results := make([]Result, len(reranked))
	for i, r := range reranked {
		results[i] = Result{ID: r.id, Score: r.dist}
	}
	return results, nil
}

// Insert adds new vectors to the index incrementally.
// Takes mu.Lock. In-memory effects are deferred until after a successful tx.Commit via a local overlay.
func (idx *Index) Insert(ctx context.Context, vecs [][]float32, ids [][]byte) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("horosvec: %w", err)
	}

	if len(vecs) != len(ids) {
		return fmt.Errorf("horosvec: vecs/ids length mismatch: %d vs %d", len(vecs), len(ids))
	}
	if len(vecs) == 0 {
		return nil
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	if !idx.built {
		return fmt.Errorf("horosvec: index not built, call Build first")
	}
	// Mode arène (SQLite vector-less) : l'arène fp16 est figée après le build et couvre les
	// seuls nœuds 0..count-1. Un Insert écrirait un blob vecteur réel mais resterait
	// invisible au brute-force arène et briserait le contrôle count arène==node_count au
	// reload. Refus fail-loud plutôt que perte de rappel silencieuse : un index arène se
	// reconstruit (Build), il ne s'insère pas incrémentalement en V2.
	if idx.arena != nil {
		return fmt.Errorf("horosvec: incremental Insert is not supported on an arena-backed index; rebuild instead")
	}
	for i, v := range vecs {
		if len(v) != idx.dim {
			return fmt.Errorf("horosvec: vec[%d] dim %d != index dim %d", i, len(v), idx.dim)
		}
	}

	tx, err := idx.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("horosvec: begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	next := idx.nextID
	insertedOrder := make([]int64, 0, len(vecs))
	pendingNodes := make(map[int64]*cachedNode, len(vecs))
	pendingNeighbors := make(map[int64][]int64)
	var pendingFlatVecs []float32
	var pendingFlatIDs [][]byte
	pendingCentroidVecs := make([][]float32, 0, len(vecs))

	loadNodeOverlay := func(nodeID int64) (*cachedNode, error) {
		if pn, ok := pendingNodes[nodeID]; ok {
			return pn, nil
		}
		n, err := loadNode(ctx, idx.db, idx.cache, nodeID)
		if err != nil {
			return nil, err
		}
		if nb, ok := pendingNeighbors[nodeID]; ok {
			overlay := *n
			overlay.neighbors = nb
			return &overlay, nil
		}
		return n, nil
	}

	for i, vec := range vecs {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("horosvec: %w", err)
		}
		nodeID := next
		next++
		insertedOrder = append(insertedOrder, nodeID)

		queryRotated := make([]float32, idx.codeDim)
		idx.rotator.Rotate(vec, queryRotated)
		code, sqNorm, l1Norm := idx.encoder.Encode(queryRotated)

		if err := saveNode(tx, nodeID, ids[i], nil, vec, code, sqNorm, l1Norm); err != nil {
			return fmt.Errorf("horosvec: save new node: %w", err)
		}

		getNeighbors := func(id int64) []int64 {
			if pn, ok := pendingNodes[id]; ok {
				return pn.neighbors
			}
			if nb, ok := pendingNeighbors[id]; ok {
				return nb
			}
			n, err := loadNode(ctx, idx.db, idx.cache, id)
			if err != nil {
				return nil
			}
			return n.neighbors
		}

		setNeighbors := func(id int64, neighbors []int64) error {
			if err := updateNeighbors(tx, id, neighbors); err != nil {
				return err
			}
			if pn, ok := pendingNodes[id]; ok {
				pn.neighbors = neighbors
				return nil
			}
			pendingNeighbors[id] = neighbors
			return nil
		}

		// Find nearest neighbors via rabitq search on rotated query (overlay supplies intra-batch visibility).
		candidates, err := idx.rabitqGreedySearchInternal(ctx, queryRotated, idx.cfg.SearchListSize, next, loadNodeOverlay)
		if err != nil {
			return err
		}

		neighbors := make([]int64, 0, idx.cfg.MaxDegree)
		for _, c := range candidates {
			if len(neighbors) >= idx.cfg.MaxDegree {
				break
			}
			neighbors = append(neighbors, c.nodeID)
		}
		if err := setNeighbors(nodeID, neighbors); err != nil {
			return fmt.Errorf("horosvec: set neighbors for new node: %w", err)
		}

		getVec := func(id int64) []float32 {
			if pn, ok := pendingNodes[id]; ok {
				return pn.vec
			}
			n, err := loadNode(ctx, idx.db, idx.cache, id)
			if err != nil {
				return nil
			}
			return n.vec
		}

		// Add reverse edges (best-effort: don't fail the whole insert)
		for _, nbr := range neighbors {
			nbrNeighbors := getNeighbors(nbr)
			if nbrNeighbors == nil {
				continue
			}
			found := false
			for _, nn := range nbrNeighbors {
				if nn == nodeID {
					found = true
					break
				}
			}
			if found {
				continue
			}
			if len(nbrNeighbors) < idx.cfg.MaxDegree {
				_ = setNeighbors(nbr, append(nbrNeighbors, nodeID))
				continue
			}
			nbrNode, err := loadNodeOverlay(nbr)
			if err != nil {
				continue
			}
			cands := make([]searchCandidate, 0, len(nbrNeighbors)+1)
			for _, nn := range nbrNeighbors {
				nnVec := getVec(nn)
				if nnVec == nil {
					continue
				}
				cands = append(cands, searchCandidate{
					nodeID: nn,
					dist:   l2DistanceSquared(nbrNode.vec, nnVec),
				})
			}
			cands = append(cands, searchCandidate{
				nodeID: nodeID,
				dist:   l2DistanceSquared(nbrNode.vec, vec),
			})
			pruned := robustPrune(nbr, cands, idx.cfg.Alpha, idx.cfg.MaxDegree, getVec)
			_ = setNeighbors(nbr, pruned)
		}

		pendingNodes[nodeID] = &cachedNode{
			nodeID:    nodeID,
			extID:     ids[i],
			neighbors: neighbors,
			vec:       vec,
			code:      code,
			sqNorm:    sqNorm,
			l1Norm:    l1Norm,
		}

		if idx.flatVecs != nil {
			pendingFlatVecs = append(pendingFlatVecs, vec...)
			pendingFlatIDs = append(pendingFlatIDs, ids[i])
		}

		pendingCentroidVecs = append(pendingCentroidVecs, vec)
	}

	if idx.testBeforeInsertCommit != nil {
		idx.testBeforeInsertCommit(tx)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("horosvec: commit: %w", err)
	}

	for _, node := range pendingNodes {
		idx.cache.put(node)
	}
	for id, neighbors := range pendingNeighbors {
		if cached := idx.cache.get(id); cached != nil {
			cached.neighbors = neighbors
		}
	}
	if idx.flatVecs != nil {
		idx.flatVecs = append(idx.flatVecs, pendingFlatVecs...)
		idx.flatIDs = append(idx.flatIDs, pendingFlatIDs...)
	}
	idx.nextID = next
	for _, vec := range pendingCentroidVecs {
		idx.centroid.Add(vec)
	}
	idx.extendPlaneAfterInsert(insertedOrder, pendingNodes, pendingNeighbors)

	idx.insertedSinceMedoid += int64(len(vecs))

	count, _ := getNodeCount(idx.db)
	_ = updateNodeCountInt64(idx.db, int64(count))

	// Recompute medoid when too many nodes have been added since last build/recompute.
	// Two triggers: periodic (MedoidStaleThreshold) or >50% new nodes relative to total.
	if idx.shouldRefreshMedoid(int64(count)) {
		newMedoid, err := recomputeMedoid(idx.db)
		if err != nil {
			slog.Warn("horosvec: medoid recompute after insert failed", "err", err)
		} else {
			idx.medoid = newMedoid
			idx.insertedSinceMedoid = 0
		}
	}

	return nil
}

// shouldRefreshMedoid returns true when the medoid is likely stale due to mass inserts.
// Called under mu.Lock.
func (idx *Index) shouldRefreshMedoid(totalNodes int64) bool {
	if totalNodes == 0 {
		return false
	}
	// Periodic threshold.
	if idx.cfg.MedoidStaleThreshold > 0 && idx.insertedSinceMedoid >= int64(idx.cfg.MedoidStaleThreshold) {
		return true
	}
	// If more than 50% of nodes were inserted after the last build/recompute.
	if idx.insertedSinceMedoid*2 > totalNodes {
		return true
	}
	return false
}

// NeedsRebuild returns true if the index should be rebuilt.
// Insert updates the centroid via ct.Add() under idx.mu.Lock; CentroidTracker has its own mutex.
func (idx *Index) NeedsRebuild() bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if idx.centroid == nil {
		return false
	}
	return idx.centroid.NeedsRebuild()
}

// RebuildAsync starts an asynchronous rebuild.
// The rebuild runs in a background goroutine protected by rebuildMu.
// Panics are recovered to prevent deadlock on rebuildMu.
func (idx *Index) RebuildAsync(ctx context.Context, iter VectorIterator) {
	idx.rebuildMu.Lock()
	go func() {
		defer idx.rebuildMu.Unlock()
		defer func() {
			if r := recover(); r != nil {
				slog.Error("horosvec: RebuildAsync panic recovered", "panic", r)
			}
		}()
		idx.rebuildInternal(ctx, iter)
	}()
}

// rebuildInternal holds idx.mu.Lock across DB commit and in-memory state update so concurrent
// Insert cannot observe a committed rebuild graph with stale nextID.
func (idx *Index) rebuildInternal(ctx context.Context, iter VectorIterator) {
	// Le verrou couvre TOUT le rebuild : drain de l'itérateur, construction du graphe
	// et persistance. Sans cela, un Insert concurrent validé pendant le drain ou le
	// buildGraph (hors verrou) est effacé par le DELETE puis nextID régressé — perte de
	// données. Compromis assumé : les recherches attendent la fin d'un rebuild ; la
	// correction prime sur la latence (rebuild reste asynchrone côté appelant).
	idx.mu.Lock()
	defer idx.mu.Unlock()
	var allVecs [][]float32
	var allIDs [][]byte
	for {
		id, vec, ok := iter.Next()
		if !ok {
			break
		}
		idCopy := make([]byte, len(id))
		copy(idCopy, id)
		vecCopy := make([]float32, len(vec))
		copy(vecCopy, vec)
		allVecs = append(allVecs, vecCopy)
		allIDs = append(allIDs, idCopy)
	}

	if len(allVecs) == 0 {
		return
	}

	dim := len(allVecs[0])
	if dim == 0 {
		slog.Error("horosvec: rebuild aborted: first vector empty")
		return
	}
	for i, v := range allVecs {
		if len(v) != dim {
			slog.Error("horosvec: rebuild aborted: heterogeneous vector dim", "index", i, "got", len(v), "want", dim)
			return
		}
	}

	if idx.rotator == nil {
		rounds := idx.cfg.RotationRounds
		seed := effectiveRotationSeed(idx.cfg)
		idx.rotator = NewRotator(dim, rounds, seed)
		idx.codeDim = idx.rotator.CodeDim()
	}

	rotated := make([]float32, idx.codeDim)
	centroid := make([]float32, idx.codeDim)
	for _, v := range allVecs {
		idx.rotator.Rotate(v, rotated)
		for j, val := range rotated {
			centroid[j] += val
		}
	}
	invN := float32(1.0 / float64(len(allVecs)))
	for j := range idx.codeDim {
		centroid[j] *= invN
	}

	enc := NewEncoder(centroid)

	buildWorkers := effectiveBuildWorkers(idx.cfg)
	nodes, err := encodeGraphNodes(ctx, idx.rotator, enc, allVecs, allIDs, buildWorkers)
	if err != nil {
		return
	}

	medoid := findMedoid(nodes)
	if err := buildGraph(ctx, nodes, medoid, idx.cfg.MaxDegree, idx.cfg.SearchListSize, idx.cfg.Alpha, idx.cfg.BuildPasses, buildWorkers); err != nil {
		return
	}

	if idx.testDuringRebuildPersist != nil {
		idx.testDuringRebuildPersist()
	}

	tx, err := idx.db.BeginTx(ctx, nil)
	if err != nil {
		slog.Error("horosvec: rebuild begin tx failed", "err", err)
		return
	}
	defer func() { _ = tx.Rollback() }()

	// Clear and rebuild in-place
	if _, err := tx.Exec("DELETE FROM vindex_nodes"); err != nil {
		slog.Error("horosvec: rebuild clear nodes failed", "err", err)
		return
	}
	if _, err := tx.Exec("DELETE FROM vindex_meta"); err != nil {
		slog.Error("horosvec: rebuild clear meta failed", "err", err)
		return
	}

	rotMeta := rotationMeta{
		seed:     idx.rotator.Seed(),
		rounds:   idx.rotator.Rounds(),
		codeDim:  idx.codeDim,
		dbFormat: currentDBFormatVersion,
	}
	if err := saveGraph(tx, nodes, medoid, dim, idx.cfg.MaxDegree, centroid, rotMeta); err != nil {
		slog.Error("horosvec: rebuild save graph failed", "err", err)
		return
	}

	if err := tx.Commit(); err != nil {
		slog.Error("horosvec: rebuild commit failed", "err", err)
		return
	}

	idx.medoid = medoid
	idx.dim = dim
	idx.encoder = enc
	idx.codeDim = idx.rotator.CodeDim()
	idx.nextID = int64(len(nodes))
	idx.centroid = NewCentroidTracker(dim, idx.cfg.DriftThreshold, idx.cfg.InsertRatioThreshold)
	idx.centroid.AddBatch(allVecs)
	idx.centroid.SnapshotBuild()

	// Build flat vector array for brute-force search
	idx.flatVecs = make([]float32, len(allVecs)*dim)
	idx.flatIDs = make([][]byte, len(allVecs))
	for i, v := range allVecs {
		copy(idx.flatVecs[i*dim:], v)
		idx.flatIDs[i] = allIDs[i]
	}

	idx.cache.clear()
	warmCache(ctx, idx.db, idx.cache, medoid, 2)

	// Le rebuild a réécrit un graphe à blobs vecteur PLEINS et un flatVecs frais : l'arène
	// fp16 éventuelle est désormais PÉRIMÉE (anciens vecteurs, count potentiellement faux).
	// La lâcher évite un rerank sur vecteurs obsolètes ; le rerank repasse par les blobs SQL
	// (présents) et l'index redevient graphe-backed (Insert de nouveau autorisé).
	idx.arena = nil

	if err := idx.rebuildPlaneLocked(); err != nil {
		slog.Error("horosvec: rebuild hot plane failed", "err", err)
		return
	}
	// Payer la collecte des déchets de construction au moment froid et déterministe,
	// jamais pendant les requêtes.
	runtime.GC()
}

// Count returns the number of vectors in the index.
// RLock protects read, but getNodeCount() SQL query may return stale count if Insert commits between RUnlock and query.
func (idx *Index) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if idx.db == nil {
		// Standalone mode: nextID equals the loaded node count.
		return int(idx.nextID)
	}
	count, err := getNodeCount(idx.db)
	if err != nil {
		return 0
	}
	return count
}

// Close releases resources. Waits for any in-flight RebuildAsync before returning.
func (idx *Index) Close() error {
	idx.rebuildMu.Lock()
	defer idx.rebuildMu.Unlock()
	idx.cache.clear()
	return nil
}
