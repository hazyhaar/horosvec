package horosvec

import (
	"context"
	"database/sql"
	"fmt"
)

// hotPlane : vue contiguë et pointer-free des nœuds persistés, pour la boucle chaude.
// Indexée par nodeID (dense 0..n-1). Mutée sous idx.mu (Lock), lue sous RLock.
// À 10M nœuds dim128 : codes ~160 Mo, norms ~160 Mo, voisins ~2,6 Go (int32, degré 64).
type hotPlane struct {
	n         int
	codeBytes int
	codes     []byte
	sqNorm    []float64
	l1Norm    []float64
	nbrOff    []int32 // n+1 (offsets dans nbrs)
	nbrs      []int32
	extOff    []int32 // n+1 (offsets dans extIDs)
	extIDs    []byte
}

func codeBytesForDim(codeDim int) int {
	return (codeDim + 7) / 8
}

// buildHotPlane charge tous les nœuds persistés en un seul SELECT ordonné par node_id.
func buildHotPlane(db *sql.DB, codeDim int) (*hotPlane, error) {
	codeBytes := codeBytesForDim(codeDim)
	rows, err := db.Query(
		"SELECT node_id, ext_id, neighbors, quantized, sq_norm, l1_norm FROM vindex_nodes ORDER BY node_id",
	)
	if err != nil {
		return nil, fmt.Errorf("horosvec: build hot plane: %w", err)
	}
	defer rows.Close()

	p := &hotPlane{codeBytes: codeBytes}
	for rows.Next() {
		var nodeID int64
		var extID, neighborsBlob, code []byte
		var sqNorm, l1Norm float64
		if err := rows.Scan(&nodeID, &extID, &neighborsBlob, &code, &sqNorm, &l1Norm); err != nil {
			return nil, fmt.Errorf("horosvec: scan hot plane row: %w", err)
		}
		if int(nodeID) != p.n {
			return nil, fmt.Errorf("horosvec: non-sequential node_id %d at position %d", nodeID, p.n)
		}
		if len(code) != codeBytes {
			return nil, fmt.Errorf("horosvec: node %d code len %d != %d", nodeID, len(code), codeBytes)
		}
		if err := p.appendRow(code, sqNorm, l1Norm, extID, deserializeInt64s(neighborsBlob)); err != nil {
			return nil, err
		}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("horosvec: build hot plane rows: %w", err)
	}
	return p, nil
}

// buildHotPlaneFromCache construit le plan depuis le cache LRU (mode standalone ImportBinary).
func buildHotPlaneFromCache(cache *nodeCache, codeDim int, n int) (*hotPlane, error) {
	codeBytes := codeBytesForDim(codeDim)
	p := &hotPlane{codeBytes: codeBytes}
	for nodeID := 0; nodeID < n; nodeID++ {
		node := cache.getReadOnly(int64(nodeID))
		if node == nil {
			return nil, fmt.Errorf("horosvec: cache miss building hot plane node %d", nodeID)
		}
		if len(node.code) != codeBytes {
			return nil, fmt.Errorf("horosvec: node %d code len %d != %d", nodeID, len(node.code), codeBytes)
		}
		if err := p.appendRow(node.code, node.sqNorm, node.l1Norm, node.extID, node.neighbors); err != nil {
			return nil, err
		}
	}
	return p, nil
}

func (p *hotPlane) appendRow(code []byte, sqNorm, l1Norm float64, extID []byte, neighbors []int64) error {
	p.codes = append(p.codes, code...)
	p.sqNorm = append(p.sqNorm, sqNorm)
	p.l1Norm = append(p.l1Norm, l1Norm)

	if len(p.nbrOff) == 0 {
		p.nbrOff = []int32{0}
		p.extOff = []int32{0}
	}
	for _, nbr := range neighbors {
		if nbr < 0 || nbr > int64(^uint32(0)>>1) {
			return fmt.Errorf("horosvec: neighbor %d out of int32 range", nbr)
		}
		p.nbrs = append(p.nbrs, int32(nbr))
	}
	p.nbrOff = append(p.nbrOff, int32(len(p.nbrs)))

	p.extOff[len(p.extOff)-1] = int32(len(p.extIDs))
	p.extIDs = append(p.extIDs, extID...)
	p.extOff = append(p.extOff, int32(len(p.extIDs)))

	p.n++
	return nil
}

// appendNode étend le plan avec un nœud inséré (appelé sous idx.mu Lock après commit).
func (p *hotPlane) appendNode(code []byte, sqNorm, l1Norm float64, extID []byte, neighbors []int64) error {
	if len(code) != p.codeBytes {
		return fmt.Errorf("horosvec: append node code len %d != %d", len(code), p.codeBytes)
	}
	return p.appendRow(code, sqNorm, l1Norm, extID, neighbors)
}

func (p *hotPlane) codeAt(nodeID int) []byte {
	off := nodeID * p.codeBytes
	return p.codes[off : off+p.codeBytes]
}

func (p *hotPlane) extIDAt(nodeID int) []byte {
	return p.extIDs[p.extOff[nodeID]:p.extOff[nodeID+1]]
}

func (p *hotPlane) neighborsAt(nodeID int) []int32 {
	return p.nbrs[p.nbrOff[nodeID]:p.nbrOff[nodeID+1]]
}

// rebuildPlaneLocked reconstruit le plan depuis SQLite ou le cache standalone.
// Vide planePatch. Appelé sous idx.mu Lock.
func (idx *Index) rebuildPlaneLocked() error {
	var (
		plane *hotPlane
		err   error
	)
	if idx.db != nil {
		plane, err = buildHotPlane(idx.db, idx.codeDim)
	} else {
		plane, err = buildHotPlaneFromCache(idx.cache, idx.codeDim, int(idx.nextID))
	}
	if err != nil {
		return err
	}
	idx.plane = plane
	idx.planePatch = nil
	return nil
}

// planeNeighborSlice retourne les voisins depuis le plan ou l'overlay planePatch.
func (idx *Index) planeNeighborSlice(nodeID int64) []int32 {
	if patch, ok := idx.planePatch[int32(nodeID)]; ok {
		return patch
	}
	if idx.plane != nil && nodeID < int64(idx.plane.n) {
		return idx.plane.neighborsAt(int(nodeID))
	}
	return nil
}

// planeExtID retourne l'ext_id depuis le plan, ou nil si hors couverture.
func (idx *Index) planeExtID(nodeID int64) []byte {
	if idx.plane == nil || nodeID < 0 || nodeID >= int64(idx.plane.n) {
		return nil
	}
	return idx.plane.extIDAt(int(nodeID))
}

// greedyNodeCodeNorms lit code/sqNorm/l1Norm depuis le plan (chemin chaud) ou loadFn/cache.
func (idx *Index) greedyNodeCodeNorms(ctx context.Context, nodeID int64, loadFn func(int64) (*cachedNode, error)) ([]byte, float64, float64, error) {
	if loadFn != nil {
		n, err := loadFn(nodeID)
		if err != nil {
			return nil, 0, 0, err
		}
		return n.code, n.sqNorm, n.l1Norm, nil
	}
	if idx.plane != nil && nodeID >= 0 && nodeID < int64(idx.plane.n) {
		id := int(nodeID)
		return idx.plane.codeAt(id), idx.plane.sqNorm[id], idx.plane.l1Norm[id], nil
	}
	n, err := loadNodeReadOnly(ctx, idx.db, idx.cache, nodeID)
	if err != nil {
		return nil, 0, 0, err
	}
	return n.code, n.sqNorm, n.l1Norm, nil
}

// greedyEachNeighbor parcourt les voisins via overlay Insert, plan chaud, ou loadFn.
func (idx *Index) greedyEachNeighbor(ctx context.Context, nodeID int64, loadFn func(int64) (*cachedNode, error), fn func(nbr int64) bool) error {
	if loadFn != nil {
		n, err := loadFn(nodeID)
		if err != nil {
			return err
		}
		for _, nbr := range n.neighbors {
			if !fn(nbr) {
				return nil
			}
		}
		return nil
	}
	if neighbors := idx.planeNeighborSlice(nodeID); neighbors != nil {
		for _, nbr := range neighbors {
			if !fn(int64(nbr)) {
				return nil
			}
		}
		return nil
	}
	n, err := loadNodeReadOnly(ctx, idx.db, idx.cache, nodeID)
	if err != nil {
		return err
	}
	for _, nbr := range n.neighbors {
		if !fn(nbr) {
			return nil
		}
	}
	return nil
}

// extendPlaneAfterInsert étend le plan et l'overlay après un commit Insert réussi.
func (idx *Index) extendPlaneAfterInsert(insertedOrder []int64, pendingNodes map[int64]*cachedNode, pendingNeighbors map[int64][]int64) {
	if idx.plane == nil {
		return
	}
	for _, nodeID := range insertedOrder {
		node := pendingNodes[nodeID]
		if err := idx.plane.appendNode(node.code, node.sqNorm, node.l1Norm, node.extID, node.neighbors); err != nil {
			// Incohérence structurelle : rebuilder depuis SQLite au prochain build.
			idx.plane = nil
			idx.planePatch = nil
			return
		}
	}
	for id, neighbors := range pendingNeighbors {
		// Nouveaux nœuds : voisinages déjà dans le plan via appendNode ; pendingNeighbors
		// peut contenir un snapshot obsolète d'avant la création de pendingNodes.
		if _, isNew := pendingNodes[id]; isNew {
			continue
		}
		patch := make([]int32, len(neighbors))
		for i, nbr := range neighbors {
			patch[i] = int32(nbr)
		}
		if idx.planePatch == nil {
			idx.planePatch = make(map[int32][]int32)
		}
		idx.planePatch[int32(id)] = patch
	}
}
