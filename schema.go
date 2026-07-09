package horosvec

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"time"
)

const schemaSQL = `
CREATE TABLE IF NOT EXISTS vindex_nodes (
    node_id   INTEGER PRIMARY KEY,
    ext_id    BLOB NOT NULL UNIQUE,
    neighbors BLOB NOT NULL,
    quantized BLOB NOT NULL,
    vector    BLOB NOT NULL,
    sq_norm   REAL NOT NULL,
    l1_norm   REAL NOT NULL DEFAULT 0
) STRICT;
CREATE INDEX IF NOT EXISTS idx_vindex_ext ON vindex_nodes(ext_id);

CREATE TABLE IF NOT EXISTS vindex_meta (
    key   TEXT PRIMARY KEY,
    value BLOB NOT NULL
) STRICT;
`

// initSchema creates the vindex_nodes and vindex_meta tables if they don't exist.
func initSchema(db *sql.DB) error {
	_, err := db.Exec(schemaSQL)
	return err
}

// saveNode persists a single node to the database.
func saveNode(tx *sql.Tx, nodeID int64, extID []byte, neighbors []int64, vec []float32, code []byte, sqNorm float64, l1Norm float64) error {
	neighborsBlob := serializeInt64s(neighbors)
	vectorBlob := serializeFloat32s(vec)
	_, err := tx.Exec(
		"INSERT OR REPLACE INTO vindex_nodes (node_id, ext_id, neighbors, vector, quantized, sq_norm, l1_norm) VALUES (?, ?, ?, ?, ?, ?, ?)",
		nodeID, extID, neighborsBlob, vectorBlob, code, sqNorm, l1Norm,
	)
	return err
}

// updateNeighbors updates just the neighbors of an existing node.
func updateNeighbors(tx *sql.Tx, nodeID int64, neighbors []int64) error {
	neighborsBlob := serializeInt64s(neighbors)
	_, err := tx.Exec(
		"UPDATE vindex_nodes SET neighbors = ? WHERE node_id = ?",
		neighborsBlob, nodeID,
	)
	return err
}

// loadNode loads a single node from the database, checking the cache first.
// cache.get then SQL load (no lock) then cache.put. Between get and put, concurrent Insert may add same nodeID — cache gets stale copy.
func loadNode(ctx context.Context, db *sql.DB, cache *nodeCache, nodeID int64) (*cachedNode, error) {
	if cached := cache.get(nodeID); cached != nil {
		return cached, nil
	}

	// Standalone mode (imported from binary): no DB to fall back to.
	if db == nil {
		return nil, fmt.Errorf("horosvec: node %d not found in standalone cache", nodeID)
	}

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("horosvec: %w", err)
	}

	var extID, neighbors, vectorBlob, code []byte
	var sqNorm, l1Norm float64
	err := db.QueryRowContext(ctx,
		"SELECT ext_id, neighbors, vector, quantized, sq_norm, l1_norm FROM vindex_nodes WHERE node_id = ?",
		nodeID,
	).Scan(&extID, &neighbors, &vectorBlob, &code, &sqNorm, &l1Norm)
	if err != nil {
		return nil, err
	}

	node := &cachedNode{
		nodeID:    nodeID,
		extID:     extID,
		neighbors: deserializeInt64s(neighbors),
		vec:       deserializeFloat32s(vectorBlob),
		code:      code,
		sqNorm:    sqNorm,
		l1Norm:    l1Norm,
	}
	cache.put(node)
	return node, nil
}

// loadNodeReadOnly loads a node using read-only cache access (no LRU promotion).
// Falls back to full loadNode on cache miss.
// getReadOnly (no LRU promote) falls back to loadNode which DOES promote. Same call may or may not update LRU depending on cache hit.
func loadNodeReadOnly(ctx context.Context, db *sql.DB, cache *nodeCache, nodeID int64) (*cachedNode, error) {
	if cached := cache.getReadOnly(nodeID); cached != nil {
		return cached, nil
	}
	return loadNode(ctx, db, cache, nodeID)
}

// rotationMeta holds persisted rotation parameters for vindex_meta.
type rotationMeta struct {
	seed     uint64
	rounds   int
	codeDim  int
	dbFormat int
}

// saveGraph persists all nodes (avec leur blob vecteur) et les métadonnées d'un build en
// mémoire (chemin legacy). Le chemin arène vector-less persiste en streaming via
// saveGraphStreamArena (blob vecteur vide, encodage à la volée).
func saveGraph(tx *sql.Tx, nodes []graphNode, medoid int64, dim int, maxDegree int, centroid []float32, rot rotationMeta) error {
	for _, n := range nodes {
		if err := saveNode(tx, n.id, n.extID, n.neighbors, n.vec, n.code, n.sqNorm, n.l1Norm); err != nil {
			return fmt.Errorf("save node %d: %w", n.id, err)
		}
	}
	return saveGraphMeta(tx, medoid, dim, maxDegree, len(nodes), centroid, rot)
}

// saveGraphMeta persiste les métadonnées d'index dans vindex_meta. Partagé par le save
// legacy (saveGraphImpl) et le save streaming arène (saveGraphStreamArena).
func saveGraphMeta(tx *sql.Tx, medoid int64, dim, maxDegree, nodeCount int, centroid []float32, rot rotationMeta) error {
	metas := map[string][]byte{
		"medoid":            serializeInt64(medoid),
		"dimension":         serializeInt64(int64(dim)),
		"max_degree":        serializeInt64(int64(maxDegree)),
		"node_count":        serializeInt64(int64(nodeCount)),
		"centroid":          serializeFloat32s(centroid),
		"built_at":          []byte(time.Now().UTC().Format(time.RFC3339)),
		"vectors_at_build":  serializeInt64(int64(nodeCount)),
		"db_format_version": serializeInt64(int64(rot.dbFormat)),
		"rotation_seed":     serializeInt64(int64(rot.seed)),
		"rotation_rounds":   serializeInt64(int64(rot.rounds)),
		"code_dim":          serializeInt64(int64(rot.codeDim)),
	}
	for k, v := range metas {
		_, err := tx.Exec(
			"INSERT OR REPLACE INTO vindex_meta (key, value) VALUES (?, ?)",
			k, v,
		)
		if err != nil {
			return fmt.Errorf("save meta %s: %w", k, err)
		}
	}
	return nil
}

// loadMeta reads a metadata value by key.
func loadMeta(db *sql.DB, key string) ([]byte, error) {
	var val []byte
	err := db.QueryRow("SELECT value FROM vindex_meta WHERE key = ?", key).Scan(&val)
	if err != nil {
		return nil, err
	}
	return val, nil
}

// loadIndex loads an existing index from the database.
func loadIndex(db *sql.DB) (medoid int64, dim int, nodeCount int, centroid []float32, vectorsAtBuild int64, err error) {
	var name string
	err = db.QueryRow("SELECT name FROM sqlite_master WHERE type='table' AND name='vindex_meta'").Scan(&name)
	if err != nil {
		return 0, 0, 0, nil, 0, fmt.Errorf("no index found: %w", err)
	}

	medoidBytes, err := loadMeta(db, "medoid")
	if err != nil {
		return 0, 0, 0, nil, 0, fmt.Errorf("load medoid: %w", err)
	}
	medoid = deserializeInt64(medoidBytes)

	dimBytes, err := loadMeta(db, "dimension")
	if err != nil {
		return 0, 0, 0, nil, 0, fmt.Errorf("load dimension: %w", err)
	}
	dim = int(deserializeInt64(dimBytes))

	countBytes, err := loadMeta(db, "node_count")
	if err != nil {
		return 0, 0, 0, nil, 0, fmt.Errorf("load node_count: %w", err)
	}
	nodeCount = int(deserializeInt64(countBytes))

	// O3 : la méta node_count n'est plus crue sur parole. On la réconcilie avec le COUNT(*)
	// réel de vindex_nodes — source de vérité. Un écart (méta obsolète après un crash ou un
	// import) est journalisé et c'est le COUNT(*) qui prime pour toutes les décisions de
	// chargement (seuil brute-force, flat, arène). La réconciliation est best-effort : si la
	// requête échoue (table illisible), on garde la méta et la garde A2 (getMaxNodeID) fera
	// échouer New fail-loud juste après.
	if actual, cErr := getNodeCount(db); cErr == nil && actual != nodeCount {
		slog.Warn("horosvec: node_count meta reconciled with COUNT(*)",
			"meta", nodeCount, "actual", actual)
		nodeCount = actual
	}

	centroidBytes, err := loadMeta(db, "centroid")
	if err != nil {
		return 0, 0, 0, nil, 0, fmt.Errorf("load centroid: %w", err)
	}
	centroid = deserializeFloat32s(centroidBytes)

	vectorsAtBuildBytes, err := loadMeta(db, "vectors_at_build")
	if err != nil {
		vectorsAtBuild = int64(nodeCount)
	} else {
		vectorsAtBuild = deserializeInt64(vectorsAtBuildBytes)
	}

	return medoid, dim, nodeCount, centroid, vectorsAtBuild, nil
}

// loadRotationMeta reads rotation parameters from vindex_meta.
// Absent rotation keys imply identity rotation (rounds=0, codeDim=dim).
func loadRotationMeta(db *sql.DB, dim int) (rotationMeta, error) {
	meta := rotationMeta{
		codeDim:  dim,
		dbFormat: 1,
	}
	if b, err := loadMeta(db, "db_format_version"); err == nil {
		meta.dbFormat = int(deserializeInt64(b))
	}
	if meta.dbFormat > currentDBFormatVersion {
		return meta, fmt.Errorf("horosvec: unsupported db format version %d (max %d)", meta.dbFormat, currentDBFormatVersion)
	}
	roundsBytes, roundsErr := loadMeta(db, "rotation_rounds")
	if roundsErr != nil {
		return meta, nil
	}
	meta.rounds = int(deserializeInt64(roundsBytes))
	if b, err := loadMeta(db, "rotation_seed"); err == nil {
		meta.seed = uint64(deserializeInt64(b))
	}
	if b, err := loadMeta(db, "code_dim"); err == nil {
		meta.codeDim = int(deserializeInt64(b))
	}
	if err := validateRotationMeta(dim, meta.codeDim, meta.rounds); err != nil {
		return meta, err
	}
	return meta, nil
}

// getMaxNodeID returns the current maximum node_id in the table.
func getMaxNodeID(db *sql.DB) (int64, error) {
	var maxID sql.NullInt64
	err := db.QueryRow("SELECT MAX(node_id) FROM vindex_nodes").Scan(&maxID)
	if err != nil {
		return 0, err
	}
	if !maxID.Valid {
		return -1, nil
	}
	return maxID.Int64, nil
}

// getNodeCount returns the number of nodes in vindex_nodes.
func getNodeCount(db *sql.DB) (int, error) {
	var count int
	err := db.QueryRow("SELECT COUNT(*) FROM vindex_nodes").Scan(&count)
	return count, err
}

// warmCache pre-loads the medoid and its neighbors up to depth hops.
// BFS loads nodes one-by-one with separate lock per node. Between loads, eviction may remove already-cached depth-1 nodes. Safe at init (single-threaded).
func warmCache(ctx context.Context, db *sql.DB, cache *nodeCache, medoid int64, depth int) {
	queue := []int64{medoid}
	seen := map[int64]bool{medoid: true}

	for d := 0; d <= depth && len(queue) > 0; d++ {
		var next []int64
		for _, nodeID := range queue {
			if err := ctx.Err(); err != nil {
				return
			}
			node, err := loadNode(ctx, db, cache, nodeID)
			if err != nil {
				continue
			}
			for _, nbr := range node.neighbors {
				if !seen[nbr] {
					seen[nbr] = true
					next = append(next, nbr)
				}
			}
		}
		queue = next
	}
}
