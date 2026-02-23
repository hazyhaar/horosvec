package horosvec

import (
	"container/heap"
	"context"
	"database/sql"
	"fmt"
	"sync"

	_ "modernc.org/sqlite"
)

// Config controls the behavior of the Vamana index.
type Config struct {
	MaxDegree      int // R: max neighbors per node (default 64)
	SearchListSize int // L: beam width during build (default 128)
	BuildPasses    int // number of graph construction passes (default 2)
	EfSearch       int // beam width during search (default 64)
	RerankTopN     int // top-N candidates to rerank with exact vectors (default 50)
	CacheCapacity  int // LRU cache capacity in nodes (default 100000)

	Alpha                float64 // pruning parameter, >1 for longer edges (default 1.2)
	DriftThreshold       float64 // centroid drift ratio to trigger rebuild (default 0.05)
	InsertRatioThreshold float64 // inserts/buildCount ratio to trigger rebuild (default 0.30)
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		MaxDegree:            64,
		SearchListSize:       128,
		BuildPasses:          2,
		EfSearch:             64,
		RerankTopN:           50,
		CacheCapacity:        100_000,
		Alpha:                1.2,
		DriftThreshold:       0.05,
		InsertRatioThreshold: 0.30,
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
	db       *sql.DB
	cfg      Config
	cache    *nodeCache
	encoder  *Encoder
	centroid *CentroidTracker
	medoid   int32
	dim      int
	nextID   int32
	built    bool

	mu        sync.RWMutex // protects searches vs. structural changes
	rebuildMu sync.Mutex   // serializes rebuilds
}

// New creates or loads an Index from the given database.
func New(db *sql.DB, cfg Config) (*Index, error) {
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
		idx.medoid = medoid
		idx.dim = dim
		idx.encoder = NewEncoder(centroid)
		idx.centroid = NewCentroidTracker(dim, cfg.DriftThreshold, cfg.InsertRatioThreshold)
		idx.centroid.SetCentroid(centroid, int64(nodeCount))
		idx.centroid.SetBuildCentroid(centroid, vectorsAtBuild)
		idx.built = true

		maxID, err := getMaxNodeID(db)
		if err == nil {
			idx.nextID = maxID + 1
		}

		warmCache(db, idx.cache, medoid, 2)
	}

	return idx, nil
}

// Build constructs the full index from the given iterator.
func (idx *Index) Build(ctx context.Context, iter VectorIterator) error {
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
		return fmt.Errorf("horosvec: no vectors provided")
	}

	dim := len(allVecs[0])
	idx.dim = dim

	// Compute centroid
	centroid := make([]float32, dim)
	for _, v := range allVecs {
		for j, val := range v {
			centroid[j] += val
		}
	}
	invN := float32(1.0 / float64(len(allVecs)))
	for j := range dim {
		centroid[j] *= invN
	}

	idx.encoder = NewEncoder(centroid)
	idx.centroid = NewCentroidTracker(dim, idx.cfg.DriftThreshold, idx.cfg.InsertRatioThreshold)
	idx.centroid.AddBatch(allVecs)
	idx.centroid.SnapshotBuild()

	// Create graph nodes with RaBitQ codes
	nodes := make([]graphNode, len(allVecs))
	for i, v := range allVecs {
		code, sqNorm, l1Norm := idx.encoder.Encode(v)
		nodes[i] = graphNode{
			id:     int32(i),
			extID:  allIDs[i],
			vec:    v,
			code:   code,
			sqNorm: sqNorm,
			l1Norm: l1Norm,
		}
	}

	idx.medoid = findMedoid(nodes)
	buildGraph(ctx, nodes, idx.medoid, idx.cfg.MaxDegree, idx.cfg.SearchListSize, idx.cfg.Alpha, idx.cfg.BuildPasses)

	if ctx.Err() != nil {
		return ctx.Err()
	}

	// Persist
	tx, err := idx.db.Begin()
	if err != nil {
		return fmt.Errorf("horosvec: begin tx: %w", err)
	}
	defer tx.Rollback()

	if _, err := tx.Exec("DELETE FROM vec_nodes"); err != nil {
		return fmt.Errorf("horosvec: clear nodes: %w", err)
	}
	if _, err := tx.Exec("DELETE FROM vec_meta"); err != nil {
		return fmt.Errorf("horosvec: clear meta: %w", err)
	}

	if err := saveGraph(tx, "vec_nodes", nodes, idx.medoid, dim, idx.cfg.MaxDegree, centroid); err != nil {
		return fmt.Errorf("horosvec: save graph: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("horosvec: commit: %w", err)
	}

	idx.nextID = int32(len(nodes))
	idx.built = true

	// Populate cache
	idx.cache.clear()
	for i := range nodes {
		idx.cache.put(&cachedNode{
			nodeID:    nodes[i].id,
			neighbors: nodes[i].neighbors,
			vec:       nodes[i].vec,
			code:      nodes[i].code,
			sqNorm:    nodes[i].sqNorm,
			l1Norm:    nodes[i].l1Norm,
		})
	}

	return nil
}

// Search finds the topK nearest neighbors using RaBitQ approximate distances.
func (idx *Index) Search(query []float32, topK int) ([]Result, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if !idx.built {
		return nil, fmt.Errorf("horosvec: index not built")
	}
	if len(query) != idx.dim {
		return nil, fmt.Errorf("horosvec: query dim %d != index dim %d", len(query), idx.dim)
	}

	efSearch := idx.cfg.EfSearch
	if efSearch < topK {
		efSearch = topK
	}

	candidates := idx.rabitqGreedySearch(query, efSearch)

	if len(candidates) > topK {
		candidates = candidates[:topK]
	}

	results := make([]Result, len(candidates))
	for i, c := range candidates {
		var extID []byte
		err := idx.db.QueryRow("SELECT ext_id FROM vec_nodes WHERE node_id = ?", c.nodeID).Scan(&extID)
		if err != nil {
			continue
		}
		results[i] = Result{ID: extID, Score: c.dist}
	}

	return results, nil
}

// rabitqGreedySearch performs best-first beam search on the Vamana graph.
// Uses exact L2 distances from cached vectors for accurate graph navigation.
func (idx *Index) rabitqGreedySearch(query []float32, L int) []searchCandidate {
	visited := make(map[int32]bool)

	medoidNode, err := loadNode(idx.db, idx.cache, idx.medoid)
	if err != nil {
		return nil
	}
	visited[idx.medoid] = true

	startDist := l2DistanceSquared(query, medoidNode.vec)

	h := &candidateHeap{{nodeID: idx.medoid, dist: startDist}}
	heap.Init(h)

	best := []searchCandidate{{nodeID: idx.medoid, dist: startDist}}
	worstBest := startDist

	for h.Len() > 0 {
		cur := heap.Pop(h).(searchCandidate)

		if len(best) >= L && cur.dist > worstBest {
			break
		}

		curNode, err := loadNode(idx.db, idx.cache, cur.nodeID)
		if err != nil {
			continue
		}

		for _, nbr := range curNode.neighbors {
			if visited[nbr] {
				continue
			}
			visited[nbr] = true

			nbrNode, err := loadNode(idx.db, idx.cache, nbr)
			if err != nil {
				continue
			}

			d := l2DistanceSquared(query, nbrNode.vec)

			if len(best) < L || d < worstBest {
				heap.Push(h, searchCandidate{nodeID: nbr, dist: d})
				best = insertSorted(best, searchCandidate{nodeID: nbr, dist: d})
				if len(best) > L {
					best = best[:L]
				}
				worstBest = best[len(best)-1].dist
			}
		}
	}

	return best
}

// SearchWithRerank searches and then reranks top candidates using exact vectors.
func (idx *Index) SearchWithRerank(query []float32, topK int, reranker func([][]byte) ([][]float32, error)) ([]Result, error) {
	rerankN := idx.cfg.RerankTopN
	if rerankN < topK*3 {
		rerankN = topK * 3
	}

	candidates, err := idx.Search(query, rerankN)
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
func (idx *Index) Insert(vecs [][]float32, ids [][]byte) error {
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
	if len(vecs[0]) != idx.dim {
		return fmt.Errorf("horosvec: vec dim %d != index dim %d", len(vecs[0]), idx.dim)
	}

	tx, err := idx.db.Begin()
	if err != nil {
		return fmt.Errorf("horosvec: begin tx: %w", err)
	}
	defer tx.Rollback()

	for i, vec := range vecs {
		nodeID := idx.nextID
		idx.nextID++

		code, sqNorm, l1Norm := idx.encoder.Encode(vec)

		if err := saveNode(tx, "vec_nodes", nodeID, ids[i], nil, vec, code, sqNorm, l1Norm); err != nil {
			return fmt.Errorf("horosvec: save new node: %w", err)
		}

		getNeighbors := func(id int32) []int32 {
			if id == nodeID {
				return nil
			}
			n, err := loadNode(idx.db, idx.cache, id)
			if err != nil {
				return nil
			}
			return n.neighbors
		}

		setNeighbors := func(id int32, neighbors []int32) {
			if err := updateNeighbors(tx, "vec_nodes", id, neighbors); err != nil {
				return
			}
			if cached := idx.cache.get(id); cached != nil {
				cached.neighbors = neighbors
			}
		}

		// Find nearest neighbors via rabitq search
		candidates := idx.rabitqGreedySearch(vec, idx.cfg.SearchListSize)

		neighbors := make([]int32, 0, idx.cfg.MaxDegree)
		for _, c := range candidates {
			if len(neighbors) >= idx.cfg.MaxDegree {
				break
			}
			neighbors = append(neighbors, c.nodeID)
		}
		setNeighbors(nodeID, neighbors)

		// Add reverse edges
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
			if !found && len(nbrNeighbors) < idx.cfg.MaxDegree {
				setNeighbors(nbr, append(nbrNeighbors, nodeID))
			}
		}

		idx.cache.put(&cachedNode{
			nodeID:    nodeID,
			neighbors: neighbors,
			vec:       vec,
			code:      code,
			sqNorm:    sqNorm,
			l1Norm:    l1Norm,
		})

		idx.centroid.Add(vec)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("horosvec: commit: %w", err)
	}

	count, _ := getNodeCount(idx.db)
	_ = updateNodeCountInt64(idx.db, int64(count))

	return nil
}

// NeedsRebuild returns true if the index should be rebuilt.
func (idx *Index) NeedsRebuild() bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if idx.centroid == nil {
		return false
	}
	return idx.centroid.NeedsRebuild()
}

// RebuildAsync starts an asynchronous rebuild.
func (idx *Index) RebuildAsync(ctx context.Context, iter VectorIterator) {
	idx.rebuildMu.Lock()
	go func() {
		defer idx.rebuildMu.Unlock()
		idx.rebuildInternal(ctx, iter)
	}()
}

func (idx *Index) rebuildInternal(ctx context.Context, iter VectorIterator) {
	if err := initSchemaNew(idx.db); err != nil {
		return
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
		return
	}

	dim := len(allVecs[0])

	centroid := make([]float32, dim)
	for _, v := range allVecs {
		for j, val := range v {
			centroid[j] += val
		}
	}
	invN := float32(1.0 / float64(len(allVecs)))
	for j := range dim {
		centroid[j] *= invN
	}

	enc := NewEncoder(centroid)

	nodes := make([]graphNode, len(allVecs))
	for i, v := range allVecs {
		code, sqNorm, l1Norm := enc.Encode(v)
		nodes[i] = graphNode{
			id:     int32(i),
			extID:  allIDs[i],
			vec:    v,
			code:   code,
			sqNorm: sqNorm,
			l1Norm: l1Norm,
		}
	}

	medoid := findMedoid(nodes)
	buildGraph(ctx, nodes, medoid, idx.cfg.MaxDegree, idx.cfg.SearchListSize, idx.cfg.Alpha, idx.cfg.BuildPasses)

	if ctx.Err() != nil {
		return
	}

	tx, err := idx.db.Begin()
	if err != nil {
		return
	}

	if err := saveGraph(tx, "vec_nodes_new", nodes, medoid, dim, idx.cfg.MaxDegree, centroid); err != nil {
		tx.Rollback()
		return
	}

	if err := tx.Commit(); err != nil {
		return
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	if err := swapIndex(idx.db); err != nil {
		return
	}

	idx.medoid = medoid
	idx.dim = dim
	idx.encoder = enc
	idx.nextID = int32(len(nodes))
	idx.centroid = NewCentroidTracker(dim, idx.cfg.DriftThreshold, idx.cfg.InsertRatioThreshold)
	idx.centroid.AddBatch(allVecs)
	idx.centroid.SnapshotBuild()

	idx.cache.clear()
	warmCache(idx.db, idx.cache, medoid, 2)
}

// Count returns the number of vectors in the index.
func (idx *Index) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	count, err := getNodeCount(idx.db)
	if err != nil {
		return 0
	}
	return count
}

// Close releases resources.
func (idx *Index) Close() error {
	idx.cache.clear()
	return nil
}
