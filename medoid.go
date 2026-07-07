package horosvec

import (
	"database/sql"
	"fmt"
	"log/slog"
)

// validateMedoid checks whether node_id=medoid exists in vindex_nodes.
// Returns (false, nil) when the medoid is absent (not a DB error).
func validateMedoid(db *sql.DB, medoid int64) (bool, error) {
	var n int
	err := db.QueryRow(`SELECT 1 FROM vindex_nodes WHERE node_id=? LIMIT 1`, medoid).Scan(&n)
	if err == sql.ErrNoRows {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// recomputeMedoid loads all nodes, recomputes the medoid, persists it, and returns the new value.
// Caller must hold idx.mu.Lock or guarantee single-writer access.
func recomputeMedoid(db *sql.DB) (int64, error) {
	rows, err := db.Query(`SELECT node_id, vector FROM vindex_nodes`)
	if err != nil {
		return 0, fmt.Errorf("horosvec: recompute medoid: query nodes: %w", err)
	}
	defer rows.Close()

	var nodes []graphNode
	for rows.Next() {
		var nodeID int64
		var vecBlob []byte
		if err := rows.Scan(&nodeID, &vecBlob); err != nil {
			return 0, fmt.Errorf("horosvec: recompute medoid: scan: %w", err)
		}
		nodes = append(nodes, graphNode{
			id:  nodeID,
			vec: deserializeFloat32s(vecBlob),
		})
	}
	if err := rows.Err(); err != nil {
		return 0, fmt.Errorf("horosvec: recompute medoid: rows: %w", err)
	}
	if len(nodes) == 0 {
		return 0, fmt.Errorf("horosvec: recompute medoid: no nodes")
	}

	newMedoid := findMedoid(nodes)

	_, err = db.Exec(
		`INSERT OR REPLACE INTO vindex_meta (key, value) VALUES ('medoid', ?)`,
		serializeInt64(newMedoid),
	)
	if err != nil {
		return 0, fmt.Errorf("horosvec: recompute medoid: persist: %w", err)
	}

	slog.Info("horosvec: medoid recomputed", "new_medoid", newMedoid, "node_count", len(nodes))
	return newMedoid, nil
}

// checkAndRefreshMedoid validates the current medoid and recomputes if invalid.
// Returns the (possibly updated) medoid. Logs a WARN on recompute.
func checkAndRefreshMedoid(db *sql.DB, medoid int64) (int64, error) {
	ok, err := validateMedoid(db, medoid)
	if err != nil {
		return medoid, err
	}
	if ok {
		return medoid, nil
	}
	slog.Warn("horosvec: medoid invalid, recomputing", "stale_medoid", medoid)
	return recomputeMedoid(db)
}
