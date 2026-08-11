package horosvec

import (
	"context"
	"database/sql"
	"encoding/binary"
	"fmt"
	"log/slog"
	"math"
)

// hotPlane : vue contiguë et pointer-free des nœuds persistés, pour la boucle chaude.
// Indexée par nodeID (dense 0..n-1). Mutée sous idx.mu (Lock), lue sous RLock.
// À 10M nœuds dim128 : codes ~160 Mo, norms ~160 Mo, voisins ~2,6 Go (int32, degré 64).
type hotPlane struct {
	n         int
	codeBytes int

	// rows entrelace, pour chaque nœud et dans CET ordre, son code, sa norme
	// carrée et sa norme L1, à pas fixe (rowStride).
	//
	// L'entrelacement lui-même ne gagne RIEN, et c'est mesuré. Un micro-banc
	// comparant quatre agencements sur 26,7 M nœuds — trois tranches séparées,
	// fusionné à pas 80, fusionné à pas 128 aligné sur les lignes de cache, code
	// seul plus normes en tranche dense — les donne tous équivalents, entre 62 et
	// 68 ns par accès, les écarts changeant de signe d'un tour à l'autre. La
	// version alignée sur 128, qui devrait gagner si les lignes de cache étaient
	// en cause, ne gagne pas et coûte 60 % de mémoire en plus.
	//
	// La raison : à cette échelle, chaque nœud visité coûte un accès aléatoire en
	// mémoire principale, soit une soixantaine de nanosecondes sur cette machine,
	// que les données soient dans une, deux ou trois régions. Regrouper trois
	// accès en un n'aide pas quand aucun des trois n'était prévisible.
	//
	// Ce qui est conservé de cette forme : elle porte l'alignement ci-dessous,
	// qui lui rend vraiment, et elle remplace trois tranches par une.
	//
	// La taille du code est arrondie au multiple de 8 supérieur (codeStride), ce
	// qui aligne le début de chaque code sur huit octets : la boucle de comptage
	// de bits lit alors des mots de 64 bits sans recomposition. CE point est
	// mesuré gagnant : la recomposition passe de 12,4 % à 2,3 % du temps
	// processeur.
	rows       []byte
	codeStride int // taille du code arrondie au multiple de 8 supérieur
	rowStride  int // codeStride + 16 (norme carrée + norme L1, en float64)

	nbrOff []int32 // n+1 (offsets dans nbrs)
	nbrs   []int32
	extOff []int32 // n+1 (offsets dans extIDs)
	extIDs []byte
}

// initStrides fixe les pas d'entrelacement une fois codeBytes connu.
func (p *hotPlane) initStrides() {
	if p.rowStride != 0 {
		return
	}
	p.codeStride = (p.codeBytes + 7) &^ 7
	p.rowStride = p.codeStride + 16
}

func codeBytesForDim(codeDim int) int {
	return (codeDim + 7) / 8
}

// codeBytesTotal rend la taille COMPLÈTE d'un code : un plan de bits par
// niveau de quantification. À un bit, elle se confond avec codeBytesForDim.
func codeBytesTotal(codeDim, codeBits int) int {
	return codeBytesForDim(codeDim) * normaliseCodeBits(codeBits)
}

// maxInt32Offset est la plus grande valeur représentable dans un offset int32 du plan chaud.
const maxInt32Offset = int64(^uint32(0) >> 1) // 2^31 - 1

// checkInt32Offset refuse fail-loud un cumul (nombre total de voisins N×degré, ou octets
// cumulés d'ext_ids) qui dépasserait la capacité d'un offset int32 du plan chaud (B3). Au-delà,
// la troncature int32(len(...)) produirait un offset négatif/faux et un slicing corrompu.
// Limite pratique : ~33M nœuds à degré 64 (2^31 / 64), documentée dans doc.go et ARCHITECTURE §9.
func checkInt32Offset(cumulative int64, what string) error {
	if cumulative > maxInt32Offset {
		return fmt.Errorf("horosvec: hot plane %s cumulative offset %d exceeds int32 capacity %d "+
			"(~33M nodes at degree 64) — rebuild with a sharded index", what, cumulative, maxInt32Offset)
	}
	return nil
}

// buildHotPlane charge tous les nœuds persistés en un seul SELECT ordonné par node_id.
func buildHotPlane(db *sql.DB, codeDim, codeBits int) (*hotPlane, error) {
	codeBytes := codeBytesTotal(codeDim, codeBits)
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
func buildHotPlaneFromCache(cache *nodeCache, codeDim, codeBits int, n int) (*hotPlane, error) {
	codeBytes := codeBytesTotal(codeDim, codeBits)
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
	p.initStrides()
	base := len(p.rows)
	p.rows = append(p.rows, make([]byte, p.rowStride)...)
	copy(p.rows[base:base+p.codeBytes], code)
	binary.LittleEndian.PutUint64(p.rows[base+p.codeStride:], math.Float64bits(sqNorm))
	binary.LittleEndian.PutUint64(p.rows[base+p.codeStride+8:], math.Float64bits(l1Norm))

	if len(p.nbrOff) == 0 {
		p.nbrOff = []int32{0}
		p.extOff = []int32{0}
	}
	for _, nbr := range neighbors {
		if nbr < 0 || nbr > maxInt32Offset {
			return fmt.Errorf("horosvec: neighbor %d out of int32 range", nbr)
		}
		p.nbrs = append(p.nbrs, int32(nbr))
	}
	// B3 : garde le CUMUL N×degré (offset dans nbrs), pas seulement chaque id de voisin.
	if err := checkInt32Offset(int64(len(p.nbrs)), "neighbors"); err != nil {
		return err
	}
	p.nbrOff = append(p.nbrOff, int32(len(p.nbrs)))

	p.extOff[len(p.extOff)-1] = int32(len(p.extIDs))
	p.extIDs = append(p.extIDs, extID...)
	// B3 : garde le CUMUL des octets d'ext_ids (offset final dans extIDs) avant la troncature.
	if err := checkInt32Offset(int64(len(p.extIDs)), "ext_ids"); err != nil {
		return err
	}
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
	off := nodeID * p.rowStride
	return p.rows[off : off+p.codeBytes]
}

// rowAt rend en une seule lecture le code et les deux normes du nœud. C'est la
// forme utilisée sur le chemin chaud. Elle ne réduit pas le coût de l'accès —
// voir le commentaire du champ rows — mais elle évite d'indexer trois tranches.
func (p *hotPlane) rowAt(nodeID int) ([]byte, float64, float64) {
	off := nodeID * p.rowStride
	row := p.rows[off : off+p.rowStride]
	sq := math.Float64frombits(binary.LittleEndian.Uint64(row[p.codeStride:]))
	l1 := math.Float64frombits(binary.LittleEndian.Uint64(row[p.codeStride+8:]))
	return row[:p.codeBytes], sq, l1
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
		plane, err = buildHotPlane(idx.db, idx.codeDim, idx.codeBits)
	} else {
		plane, err = buildHotPlaneFromCache(idx.cache, idx.codeDim, idx.codeBits, int(idx.nextID))
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
		code, sq, l1 := idx.plane.rowAt(int(nodeID))
		return code, sq, l1, nil
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
			// A5 : incohérence structurelle du plan chaud. Le plan est invalidé (reconstruit
			// depuis SQLite au prochain build) et le repli SQL/cache reste correct : dégradation
			// d'observabilité, jamais avalée. Compteur exposé (PlaneDegraded) + journal d'erreur.
			idx.planeDegraded.Add(1)
			slog.Error("horosvec: hot plane extension failed after insert, plane invalidated",
				"node", nodeID, "err", err)
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
