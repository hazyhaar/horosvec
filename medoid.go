package horosvec

import (
	"database/sql"
	"fmt"
	"log/slog"
)

// medoidBlockSize borne le nombre de vecteurs simultanément décodés en mémoire lors d'une
// recomputation en mode DB-blob : le parcours se fait par fenêtres de cette taille, jamais
// en matérialisant l'intégralité de la table (empreinte ≈ medoidBlockSize × dim × 4 octets).
const medoidBlockSize = 10000

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

// persistMedoid writes the recomputed medoid into vindex_meta.
func persistMedoid(db *sql.DB, medoid int64) error {
	_, err := db.Exec(
		`INSERT OR REPLACE INTO vindex_meta (key, value) VALUES ('medoid', ?)`,
		serializeInt64(medoid),
	)
	if err != nil {
		return fmt.Errorf("horosvec: recompute medoid: persist: %w", err)
	}
	return nil
}

// recomputeMedoid recomputes the medoid without ever materialising the whole vector set in
// memory, then persists it and returns the new value. Two modes, selected by the vector
// storage of the index:
//
//   - Mode arène (vecteurs dans l'arène fp16, SQLite vector-less) : la sélection réutilise
//     findMedoidSource sur un arenaVecSource, qui décode fp16→fp32 à la demande. L'ancienne
//     matérialisation `SELECT node_id, vector` produisait alors des blobs vides et un médoïde
//     faux, silencieusement — c'est corrigé ici. Si aucune arène n'est ouverte mais qu'un
//     ArenaPath est configuré, l'arène est ouverte pour la durée de la recomputation.
//   - Mode DB-blob legacy (vecteurs stockés dans vindex_nodes.vector) : la sélection réutilise
//     findMedoidSource sur un dbBlockVecSource qui parcourt la table par fenêtres bornées
//     (medoidBlockSize), sans jamais détenir tous les vecteurs à la fois.
//
// Un SQLite vector-less SANS arène disponible échoue fail-loud plutôt que de retourner un
// médoïde faux.
//
// Caller must hold idx.mu.Lock or guarantee single-writer access.
func (idx *Index) recomputeMedoid() (int64, error) {
	dim := idx.dim

	// Mode arène : source des vecteurs = arène fp16, jamais les blobs SQLite.
	ar := idx.arena
	if ar == nil && idx.cfg.ArenaPath != "" {
		opened, err := openArena(idx.cfg.ArenaPath)
		if err != nil {
			return 0, fmt.Errorf("horosvec: recompute medoid: open arena %q: %w", idx.cfg.ArenaPath, err)
		}
		defer func() { _ = opened.close() }()
		ar = opened
	}
	if ar != nil {
		n := int(ar.count)
		if n == 0 {
			return 0, fmt.Errorf("horosvec: recompute medoid: empty arena")
		}
		newMedoid := findMedoidSource(n, dim, newArenaVecSource(ar))
		if err := persistMedoid(idx.db, newMedoid); err != nil {
			return 0, err
		}
		slog.Info("horosvec: medoid recomputed (arena)", "new_medoid", newMedoid, "node_count", n)
		return newMedoid, nil
	}

	// Mode DB-blob legacy : parcours par blocs, empreinte mémoire bornée.
	src, err := newDBBlockVecSource(idx.db, dim, medoidBlockSize)
	if err != nil {
		return 0, err
	}
	if len(src.ids) == 0 {
		return 0, fmt.Errorf("horosvec: recompute medoid: no nodes")
	}
	rank := findMedoidSource(len(src.ids), dim, src)
	newMedoid := src.ids[rank] // rang → node_id réel (les node_id ne sont pas forcément 0..n-1)
	if err := persistMedoid(idx.db, newMedoid); err != nil {
		return 0, err
	}
	slog.Info("horosvec: medoid recomputed (db-blob stream)", "new_medoid", newMedoid, "node_count", len(src.ids))
	return newMedoid, nil
}

// checkAndRefreshMedoid validates the current medoid and recomputes if invalid.
// Returns the (possibly updated) medoid. Logs a WARN on recompute.
func (idx *Index) checkAndRefreshMedoid(medoid int64) (int64, error) {
	ok, err := validateMedoid(idx.db, medoid)
	if err != nil {
		return medoid, err
	}
	if ok {
		return medoid, nil
	}
	slog.Warn("horosvec: medoid invalid, recomputing", "stale_medoid", medoid)
	return idx.recomputeMedoid()
}

// dbBlockVecSource est un vecSource adossé à la table vindex_nodes, qui décode les vecteurs
// par fenêtres bornées (blockSize rangs) plutôt que de tout charger. Les node_id NE sont PAS
// supposés contigus 0..n-1 (un rebuild ou un backfill peut laisser un premier node_id non nul,
// cf. incident recall gaussian_clusters) : la source détient la liste TRIÉE des node_id (ids),
// et l'index i manipulé par findMedoidSource est un RANG dans ids, pas un node_id. L'appelant
// traduit le rang retourné en node_id réel via src.ids[rang]. L'empreinte mémoire est bornée à
// ids (O(n) int64, très inférieur à O(n·dim) vecteurs) plus une fenêtre de blockSize vecteurs.
// L'accès de findMedoidSource est strictement séquentiel (0..n-1, deux passes) : une fenêtre
// vivante suffit, chaque bloc est rechargé au plus une fois par passe.
type dbBlockVecSource struct {
	db        *sql.DB
	dim       int
	blockSize int
	ids       []int64 // node_id triés ; ids[rang] = node_id du rang

	start int // rang du premier vecteur de la fenêtre courante (-1 = aucune)
	count int // nombre de vecteurs valides dans buf
	buf   []float32
}

// newDBBlockVecSource prépare la source : charge la liste triée des node_id et vérifie
// fail-loud que la table porte réellement des vecteurs (un SQLite vector-less sans arène doit
// échouer, jamais produire un médoïde faux).
func newDBBlockVecSource(db *sql.DB, dim, blockSize int) (*dbBlockVecSource, error) {
	rows, err := db.Query(`SELECT node_id FROM vindex_nodes ORDER BY node_id`)
	if err != nil {
		return nil, fmt.Errorf("horosvec: recompute medoid: list node_id: %w", err)
	}
	defer rows.Close()
	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("horosvec: recompute medoid: scan node_id: %w", err)
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("horosvec: recompute medoid: node_id rows: %w", err)
	}
	if len(ids) == 0 {
		return &dbBlockVecSource{db: db, dim: dim, blockSize: blockSize, start: -1}, nil
	}
	// Pré-vol fail-loud : un blob vecteur vide signale un SQLite vector-less sans arène.
	var probe []byte
	if err := db.QueryRow(`SELECT vector FROM vindex_nodes ORDER BY node_id LIMIT 1`).Scan(&probe); err != nil {
		return nil, fmt.Errorf("horosvec: recompute medoid: probe vector: %w", err)
	}
	if len(probe)/4 != dim {
		return nil, fmt.Errorf("horosvec: recompute medoid: vector-less SQLite without arena "+
			"(blob de %d octets, dim attendue %d) — aucune source de vecteurs disponible", len(probe), dim)
	}
	return &dbBlockVecSource{db: db, dim: dim, blockSize: blockSize, ids: ids, start: -1}, nil
}

// loadBlock charge la fenêtre de rangs [blockStart, blockStart+blockSize) contenant le rang.
// Les node_id de cette fenêtre étant ids[blockStart..blockEnd-1] (triés et exhaustifs), une
// requête de plage node_id renvoie exactement ces nœuds, dans l'ordre des rangs.
func (s *dbBlockVecSource) loadBlock(rank int) error {
	blockStart := (rank / s.blockSize) * s.blockSize
	blockEnd := blockStart + s.blockSize
	if blockEnd > len(s.ids) {
		blockEnd = len(s.ids)
	}
	lo, hi := s.ids[blockStart], s.ids[blockEnd-1]
	rows, err := s.db.Query(
		`SELECT vector FROM vindex_nodes WHERE node_id >= ? AND node_id <= ? ORDER BY node_id`,
		lo, hi,
	)
	if err != nil {
		return fmt.Errorf("horosvec: recompute medoid: block query: %w", err)
	}
	defer rows.Close()

	if cap(s.buf) < s.blockSize*s.dim {
		s.buf = make([]float32, s.blockSize*s.dim)
	}
	s.buf = s.buf[:s.blockSize*s.dim]
	k := 0
	for rows.Next() {
		var blob []byte
		if err := rows.Scan(&blob); err != nil {
			return fmt.Errorf("horosvec: recompute medoid: block scan: %w", err)
		}
		vec := deserializeFloat32s(blob)
		if len(vec) != s.dim {
			return fmt.Errorf("horosvec: recompute medoid: node has %d comps, want %d", len(vec), s.dim)
		}
		if k*s.dim+s.dim > len(s.buf) {
			return fmt.Errorf("horosvec: recompute medoid: block overflow at rank %d", blockStart+k)
		}
		copy(s.buf[k*s.dim:k*s.dim+s.dim], vec)
		k++
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("horosvec: recompute medoid: block rows: %w", err)
	}
	if k != blockEnd-blockStart {
		return fmt.Errorf("horosvec: recompute medoid: block loaded %d nodes, want %d", k, blockEnd-blockStart)
	}
	s.start = blockStart
	s.count = k
	return nil
}

// vec renvoie le vecteur du RANG rank (0..len(ids)-1), pas d'un node_id.
func (s *dbBlockVecSource) vec(rank int64, _ int) []float32 {
	if rank < 0 || int(rank) >= len(s.ids) {
		return nil
	}
	r := int(rank)
	if s.start < 0 || r < s.start || r >= s.start+s.count {
		if err := s.loadBlock(r); err != nil {
			slog.Error("horosvec: recompute medoid: load block failed", "rank", r, "err", err)
			return nil
		}
	}
	off := (r - s.start) * s.dim
	if off < 0 || off+s.dim > len(s.buf) {
		return nil
	}
	return s.buf[off : off+s.dim]
}

// clone renvoie une source indépendante (buffers propres, ids partagés en lecture seule).
// findMedoidSource est séquentiel et n'appelle pas clone, mais l'interface l'exige.
func (s *dbBlockVecSource) clone() vecSource {
	return &dbBlockVecSource{db: s.db, dim: s.dim, blockSize: s.blockSize, ids: s.ids, start: -1}
}
