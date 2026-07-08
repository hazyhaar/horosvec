package horosvec

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"syscall"
)

// Voie d'IMPORT d'adjacence externe (build GPU CAGRA) vers un index horosvec complet.
//
// Structurellement, l'import est le chemin buildFromOpenArena MOINS la construction du
// graphe Vamana : la rotation, le centroïde, l'encodage RaBitQ, les normes, le médoïde, la
// persistance SQLite vector-less et le plan chaud sont IDENTIQUES au build ; seule la source
// d'adjacence change — au lieu de bâtir le graphe (buildGraphStore), on LIT une adjacence
// plate produite en amont. L'adjacence est un fichier plat d'entiers u32 little-endian, N
// lignes de `degree` voisins, ligne i = node_id i = rang i dans l'arène fp16. À l'échelle
// HNbook (26,7 M × 64 × 4 = 6,8 Go), le fichier n'est JAMAIS chargé sur le tas Go : il est
// cartographié en mémoire (mmap MAP_SHARED, page cache noyau) exactement comme l'arène, et
// les voisins d'un nœud sont décodés à la demande au moment de la persistance streaming.

const (
	// adjPadSentinel marque une entrée de remplissage (padding) dans une ligne d'adjacence :
	// un rang non peuplé. Convention alignée sur RAFT/cuVS (CAGRA) qui remplit avec l'indice
	// invalide maximal. Import : les sentinelles sont ignorées (jamais une erreur de borne).
	adjPadSentinel uint32 = 0xFFFFFFFF

	// Oracle de normalisation (O2) : l'adjacence CAGRA est bâtie en produit scalaire, la marche
	// horosvec est L2 — l'équivalence de classement n'est acquise que sur des vecteurs unitaires.
	defaultNormSampleN = 10000
	defaultNormSeed    = 20260709
	normTolerance      = 0.01  // |‖v‖₂ − 1| toléré par vecteur
	normMinFraction    = 0.999 // fraction minimale de vecteurs unitaires exigée
)

// adjReader est une vue en lecture seule d'une adjacence plate u32 cartographiée en mémoire
// (mmap MAP_SHARED, PROT_READ). Le fichier n'a pas d'en-tête propre : sa forme (count, degree)
// est fournie par l'appelant (arène + méta) et vérifiée contre la taille exacte du fichier.
// Immuable après ouverture, sûr en accès concurrent.
type adjReader struct {
	data   []byte
	degree int
	count  int64
}

// openAdjacency cartographie une adjacence plate u32 et vérifie que la taille du fichier vaut
// EXACTEMENT count × degree × 4 (fail-loud sur mismatch : troncature, mauvaise arène, mauvais
// degré). Ne valide pas encore le CONTENU (bornes des ids) — c'est le rôle de validate.
func openAdjacency(path string, count int64, degree int) (*adjReader, error) {
	if degree <= 0 {
		return nil, fmt.Errorf("horosvec: open adjacency: invalid degree %d", degree)
	}
	if count < 0 {
		return nil, fmt.Errorf("horosvec: open adjacency: invalid count %d", count)
	}
	// Garde anti-overflow avant la multiplication (degree déjà > 0).
	if count > (int64(1)<<62)/(int64(degree)*4) {
		return nil, fmt.Errorf("horosvec: open adjacency: count %d too large for degree %d", count, degree)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("horosvec: open adjacency: %w", err)
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("horosvec: open adjacency: stat: %w", err)
	}
	want := count * int64(degree) * 4
	if st.Size() != want {
		return nil, fmt.Errorf("horosvec: open adjacency: file size %d != expected %d (count=%d degree=%d, u32)", st.Size(), want, count, degree)
	}
	if want == 0 {
		return nil, fmt.Errorf("horosvec: open adjacency: empty adjacency (count=%d degree=%d)", count, degree)
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(want), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("horosvec: open adjacency: mmap: %w", err)
	}
	return &adjReader{data: data, degree: degree, count: count}, nil
}

// close libère la cartographie mémoire (munmap). Idempotent.
func (r *adjReader) close() error {
	if r == nil || r.data == nil {
		return nil
	}
	data := r.data
	r.data = nil
	if err := syscall.Munmap(data); err != nil {
		return fmt.Errorf("horosvec: munmap adjacency: %w", err)
	}
	return nil
}

// rawAt renvoie la valeur u32 brute au rang k (0..degree-1) de la ligne du nœud id.
func (r *adjReader) rawAt(id int64, k int) uint32 {
	off := (id*int64(r.degree) + int64(k)) * 4
	return binary.LittleEndian.Uint32(r.data[off:])
}

// validate parcourt l'intégralité de l'adjacence (lecture séquentielle mmap, aucune rétention
// sur le tas) et impose les invariants O1 : tout id voisin réel est < count (fail-loud sinon,
// c'est LE bug de décalage attendu) ; toute entrée hors borne qui n'est pas la sentinelle de
// padding est refusée ; le degré effectif d'un nœud (voisins réels, hors sentinelles et hors
// auto-boucle) ne dépasse jamais maxDegree de la config.
func (r *adjReader) validate(ctx context.Context, count int64, maxDegree int) error {
	for i := int64(0); i < count; i++ {
		if i%(1<<20) == 0 {
			if err := ctx.Err(); err != nil {
				return fmt.Errorf("horosvec: validate adjacency: %w", err)
			}
		}
		real := 0
		for k := 0; k < r.degree; k++ {
			v := r.rawAt(i, k)
			if v == adjPadSentinel {
				continue
			}
			if int64(v) >= count {
				return fmt.Errorf("horosvec: validate adjacency: node %d slot %d neighbor id %d out of range (count=%d) — arena/node_id/adjacency-row offset mismatch", i, k, v, count)
			}
			if int64(v) == i {
				continue // auto-boucle : ignorée, jamais persistée
			}
			real++
		}
		if real > maxDegree {
			return fmt.Errorf("horosvec: validate adjacency: node %d has %d real neighbors > MaxDegree %d", i, real, maxDegree)
		}
	}
	return nil
}

// loadInto ajoute les voisins réels du nœud id à dst (sentinelles de padding et auto-boucles
// filtrées) et renvoie la slice remplie. Les bornes ont été vérifiées par validate ; loadInto
// est donc sans erreur (contrat neighborLoader). L'appelant détient dst (typiquement dst[:0]).
func (r *adjReader) loadInto(id int64, dst []int64) []int64 {
	for k := 0; k < r.degree; k++ {
		v := r.rawAt(id, k)
		if v == adjPadSentinel {
			continue
		}
		nb := int64(v)
		if nb == id {
			continue
		}
		dst = append(dst, nb)
	}
	return dst
}

// neighborStride borne la capacité du tampon de lecture (au plus degree voisins par nœud).
func (r *adjReader) neighborStride() int { return r.degree }

// verifyArenaNormalized échantillonne sampleN vecteurs de l'arène (tirage PCG déterministe de
// graine seed, avec remise) et vérifie que la fraction de vecteurs unitaires (|‖v‖₂−1| <
// normTolerance) atteint normMinFraction. Fail-loud sinon (jamais un warning) : un graphe bâti
// en produit scalaire sur des vecteurs non normalisés n'est pas classé de façon équivalente par
// la marche L2 d'horosvec — l'import ne doit pas produire un index silencieusement faux.
func verifyArenaNormalized(ar *arena, sampleN int, seed uint64) error {
	n := ar.count
	if n == 0 {
		return fmt.Errorf("horosvec: normalization guard: empty arena")
	}
	if int64(sampleN) > n {
		sampleN = int(n)
	}
	if sampleN <= 0 {
		return fmt.Errorf("horosvec: normalization guard: invalid sample size")
	}
	rng := rand.New(rand.NewPCG(seed, 0x9E3779B97F4A7C15))
	buf := make([]float32, ar.dim)
	unit := 0
	for s := 0; s < sampleN; s++ {
		i := rng.Int64N(n)
		if !ar.vecInto(i, buf) {
			return fmt.Errorf("horosvec: normalization guard: decode node %d out of range", i)
		}
		var sum float64
		for _, v := range buf {
			sum += float64(v) * float64(v)
		}
		if math.Abs(math.Sqrt(sum)-1.0) < normTolerance {
			unit++
		}
	}
	frac := float64(unit) / float64(sampleN)
	if frac < normMinFraction {
		return fmt.Errorf("horosvec: normalization guard: only %.4f of %d sampled vectors are unit-norm (|‖v‖−1|<%.3f), need ≥%.4f — CAGRA inner-product adjacency requires normalized vectors for L2 rank equivalence", frac, sampleN, normTolerance, normMinFraction)
	}
	return nil
}

// ImportAdjacency construit un index horosvec COMPLET à partir d'une arène fp16 déjà finalisée,
// de son fichier d'ext_ids et d'une adjacence plate u32 produite en amont (build GPU CAGRA).
// C'est le chemin buildFromOpenArena avec la construction du graphe Vamana remplacée par la
// lecture de l'adjacence : rotation, centroïde, encodage RaBitQ, normes, médoïde, persistance
// SQLite vector-less et plan chaud sont identiques.
//
// degree est le nombre de voisins par ligne d'adjacence (p.ex. 64 pour CAGRA), lu de la méta
// par l'appelant ; il doit valoir ≤ Config.MaxDegree. L'arène est ouverte en LECTURE SEULE et
// jamais modifiée. L'import est rejouable : il écrase entièrement les tables de l'index.
//
// Deux gardes fail-loud précèdent toute construction : la cohérence arène↔ids (compte exact),
// puis l'oracle métrique (O2) qui exige des vecteurs unitaires — l'adjacence CAGRA étant bâtie
// en produit scalaire, la marche L2 d'horosvec n'est équivalente que sur des vecteurs normés.
func (idx *Index) ImportAdjacency(ctx context.Context, arenaPath, idsPath, adjPath string, degree int) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if degree <= 0 {
		return fmt.Errorf("horosvec: import adjacency: invalid degree %d", degree)
	}
	if degree > idx.cfg.MaxDegree {
		return fmt.Errorf("horosvec: import adjacency: adjacency degree %d > Config.MaxDegree %d", degree, idx.cfg.MaxDegree)
	}

	ar, err := openArena(arenaPath)
	if err != nil {
		return err
	}
	extIDs, err := readArenaIDs(idsPath)
	if err != nil {
		_ = ar.close()
		return err
	}
	if int64(len(extIDs)) != ar.count {
		_ = ar.close()
		return fmt.Errorf("horosvec: import adjacency: ids count %d != arena count %d", len(extIDs), ar.count)
	}

	// Oracle métrique (O2), garde dure : re-prouve au sol la normalisation avant de bâtir.
	if err := verifyArenaNormalized(ar, defaultNormSampleN, defaultNormSeed); err != nil {
		_ = ar.close()
		return fmt.Errorf("horosvec: import adjacency: %w", err)
	}

	idx.cfg.ArenaPath = arenaPath
	return idx.buildFromOpenArena(ctx, ar, extIDs, idx.adjacencyGraphBuilder(adjPath, degree))
}

// adjacencyGraphBuilder renvoie le graphBuilder d'import : il cartographie l'adjacence externe,
// en valide le contenu (bornes des ids, degré), puis la sert en lecture à la persistance
// streaming. La libération munmap la cartographie une fois l'index persisté. Le médoïde reste
// calculé et persisté par buildFromOpenArena (findMedoidSource), indépendamment de l'adjacence.
func (idx *Index) adjacencyGraphBuilder(adjPath string, degree int) graphBuilder {
	return func(ctx context.Context, n int, _ int64) (neighborLoader, func(), error) {
		r, err := openAdjacency(adjPath, int64(n), degree)
		if err != nil {
			return nil, nil, err
		}
		if err := r.validate(ctx, int64(n), idx.cfg.MaxDegree); err != nil {
			_ = r.close()
			return nil, nil, err
		}
		return r, func() { _ = r.close() }, nil
	}
}

// ExportAdjacency écrit l'adjacence du graphe de l'index au format plat u32 little-endian
// (N lignes × degree voisins, ligne i = node_id i), les rangs non peuplés étant remplis par la
// sentinelle de padding. C'est l'inverse exact de ImportAdjacency : il permet d'éprouver la
// voie d'import SANS GPU (round-trip build Vamana → export → réimport → parité de recherche).
// Écriture atomique via fichier temporaire + rename. degree doit être ≥ au degré réel de tout
// nœud (typiquement Config.MaxDegree).
func (idx *Index) ExportAdjacency(path string, degree int) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if !idx.built {
		return fmt.Errorf("horosvec: export adjacency: index not built")
	}
	if idx.db == nil {
		return fmt.Errorf("horosvec: export adjacency: standalone index has no persisted graph")
	}
	if degree <= 0 {
		return fmt.Errorf("horosvec: export adjacency: invalid degree %d", degree)
	}

	rows, err := idx.db.Query("SELECT node_id, neighbors FROM vindex_nodes ORDER BY node_id")
	if err != nil {
		return fmt.Errorf("horosvec: export adjacency scan: %w", err)
	}
	defer rows.Close()

	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("horosvec: export adjacency create: %w", err)
	}
	w := bufio.NewWriterSize(f, 1<<20)
	rowBuf := make([]byte, degree*4)
	var pos int64
	abort := func() {
		f.Close()
		os.Remove(tmp)
	}
	for rows.Next() {
		var nodeID int64
		var blob []byte
		if err := rows.Scan(&nodeID, &blob); err != nil {
			abort()
			return fmt.Errorf("horosvec: export adjacency row: %w", err)
		}
		if nodeID != pos {
			abort()
			return fmt.Errorf("horosvec: export adjacency non-sequential node_id %d at position %d", nodeID, pos)
		}
		nbrs := deserializeInt64s(blob)
		if len(nbrs) > degree {
			abort()
			return fmt.Errorf("horosvec: export adjacency node %d degree %d > export degree %d", nodeID, len(nbrs), degree)
		}
		for k := 0; k < degree; k++ {
			if k < len(nbrs) {
				binary.LittleEndian.PutUint32(rowBuf[k*4:], uint32(nbrs[k]))
			} else {
				binary.LittleEndian.PutUint32(rowBuf[k*4:], adjPadSentinel)
			}
		}
		if _, err := w.Write(rowBuf); err != nil {
			abort()
			return fmt.Errorf("horosvec: export adjacency write: %w", err)
		}
		pos++
	}
	if err := rows.Err(); err != nil {
		abort()
		return fmt.Errorf("horosvec: export adjacency rows: %w", err)
	}
	if err := w.Flush(); err != nil {
		abort()
		return fmt.Errorf("horosvec: export adjacency flush: %w", err)
	}
	if err := f.Sync(); err != nil {
		abort()
		return fmt.Errorf("horosvec: export adjacency sync: %w", err)
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("horosvec: export adjacency close: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("horosvec: export adjacency rename: %w", err)
	}
	return nil
}
