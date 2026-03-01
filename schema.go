// CLAUDE:SUMMARY SQLite schema DDL, node persistence, metadata storage, and cache warming.
package horosvec

import (
	"database/sql"
	"fmt"
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
);
CREATE INDEX IF NOT EXISTS idx_vindex_ext ON vindex_nodes(ext_id);

CREATE TABLE IF NOT EXISTS vindex_meta (
    key   TEXT PRIMARY KEY,
    value BLOB NOT NULL
);
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
func loadNode(db *sql.DB, cache *nodeCache, nodeID int64) (*cachedNode, error) {
	if cached := cache.get(nodeID); cached != nil {
		return cached, nil
	}

	var neighbors, vectorBlob, code []byte
	var sqNorm, l1Norm float64
	err := db.QueryRow(
		"SELECT neighbors, vector, quantized, sq_norm, l1_norm FROM vindex_nodes WHERE node_id = ?",
		nodeID,
	).Scan(&neighbors, &vectorBlob, &code, &sqNorm, &l1Norm)
	if err != nil {
		return nil, err
	}

	node := &cachedNode{
		nodeID:    nodeID,
		neighbors: deserializeInt64s(neighbors),
		vec:       deserializeFloat32s(vectorBlob),
		code:      code,
		sqNorm:    sqNorm,
		l1Norm:    l1Norm,
	}
	cache.put(node)
	return node, nil
}

// saveGraph persists all nodes and metadata from a build.
func saveGraph(tx *sql.Tx, nodes []graphNode, medoid int64, dim int, maxDegree int, centroid []float32) error {
	for _, n := range nodes {
		if err := saveNode(tx, n.id, n.extID, n.neighbors, n.vec, n.code, n.sqNorm, n.l1Norm); err != nil {
			return fmt.Errorf("save node %d: %w", n.id, err)
		}
	}

	metas := map[string][]byte{
		"medoid":           serializeInt64(medoid),
		"dimension":        serializeInt64(int64(dim)),
		"max_degree":       serializeInt64(int64(maxDegree)),
		"node_count":       serializeInt64(int64(len(nodes))),
		"centroid":         serializeFloat32s(centroid),
		"built_at":         []byte(time.Now().UTC().Format(time.RFC3339)),
		"vectors_at_build": serializeInt64(int64(len(nodes))),
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

// updateNodeCountInt64 updates the node_count in vindex_meta using int64 serialization.
func updateNodeCountInt64(db *sql.DB, count int64) error {
	_, err := db.Exec(
		"INSERT OR REPLACE INTO vindex_meta (key, value) VALUES (?, ?)",
		"node_count", serializeInt64(count),
	)
	return err
}

// warmCache pre-loads the medoid and its neighbors up to depth hops.
func warmCache(db *sql.DB, cache *nodeCache, medoid int64, depth int) {
	queue := []int64{medoid}
	seen := map[int64]bool{medoid: true}

	for d := 0; d <= depth && len(queue) > 0; d++ {
		var next []int64
		for _, nodeID := range queue {
			node, err := loadNode(db, cache, nodeID)
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
