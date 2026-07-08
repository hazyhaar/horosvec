package horosvec

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"sync"
	"sync/atomic"
)

// arenaVecSource décode les vecteurs fp16→fp32 à la demande depuis l'arène mmap, dans
// trois tampons réutilisés (un par slot). Le chemin de build gros-index ne matérialise
// ainsi jamais de tampon fp32 plein O(N×dim×4) : la source des vecteurs est le page cache
// noyau (arène mmap), hors du tas Go. Une instance n'est PAS sûre en accès concurrent (ses
// tampons sont mutés) ; chaque worker de construction reçoit sa propre instance via clone.
type arenaVecSource struct {
	ar   *arena
	bufs [3][]float32
}

func newArenaVecSource(ar *arena) *arenaVecSource {
	return &arenaVecSource{ar: ar}
}

func (a *arenaVecSource) vec(id int64, slot int) []float32 {
	if id < 0 || id >= a.ar.count {
		return nil
	}
	dst := a.bufs[slot]
	if dst == nil {
		dst = make([]float32, a.ar.dim)
		a.bufs[slot] = dst
	}
	if !a.ar.vecInto(id, dst) {
		return nil
	}
	return dst
}

func (a *arenaVecSource) clone() vecSource {
	return &arenaVecSource{ar: a.ar}
}

// readArenaIDs lit un fichier d'ids finalisé (uint64 little-endian, stride 8, rang =
// node_id — format écrit par hnbook-embed) et renvoie un ext_id par rang, encodé en ASCII
// décimal. Ce choix d'encodage est rétro-compatible : le chemin streaming existant grave
// déjà l'ext_id en ASCII décimal (strconv.Itoa du rang), et Search restitue ces mêmes
// octets bruts — un consommateur (hnbook, banc) reconvertit l'ASCII en uint64.
func readArenaIDs(path string) ([][]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("horosvec: read arena ids: %w", err)
	}
	if len(data)%8 != 0 {
		return nil, fmt.Errorf("horosvec: arena ids file %s size %d not a multiple of 8", path, len(data))
	}
	n := len(data) / 8
	out := make([][]byte, n)
	for i := 0; i < n; i++ {
		id := binary.LittleEndian.Uint64(data[i*8:])
		out[i] = []byte(strconv.FormatUint(id, 10))
	}
	return out, nil
}

// BuildFromArena construit l'index Vamana à partir d'une arène fp16 DÉJÀ complète et de son
// fichier d'ids, sans jamais re-streamer ni matérialiser les vecteurs en fp32 : le graphe
// est bâti en lisant les vecteurs à la demande depuis l'arène mmap (page cache, hors tas).
// C'est le point d'entrée du build gros-index (V2 HNbook), où l'entrée est une arène produite
// au préalable (manifest done) et non un itérateur source.
//
// La cohérence header d'arène ↔ compte d'ids est validée avant toute construction (fail-loud
// sur mismatch). Après succès, l'arène reste cartographiée et sert directement l'étape de
// rerank (une seule cartographie couvre build et rerank). Config.ArenaPath est renseigné pour
// qu'une réouverture ultérieure (New/SetParam) retrouve l'arène de rerank.
func (idx *Index) BuildFromArena(ctx context.Context, arenaPath, idsPath string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

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
		return fmt.Errorf("horosvec: BuildFromArena: ids count %d != arena count %d", len(extIDs), ar.count)
	}
	idx.cfg.ArenaPath = arenaPath
	return idx.buildFromOpenArena(ctx, ar, extIDs)
}

// buildFromOpenArena est le cœur mémoire-borné partagé par BuildFromArena et par le chemin
// streaming (buildStreamingArena) : à partir d'une arène mmap ouverte et des ext_ids, il
// calcule la rotation/centroïde en flux, encode les nœuds (codes RaBitQ), construit le graphe
// Vamana en lisant les vecteurs à la demande, persiste un SQLite allégé (sans blob vecteur)
// et conserve l'arène pour le rerank. Aucun tampon fp32 O(N×dim×4) n'est alloué.
//
// L'appelant DOIT détenir idx.mu.Lock. buildFromOpenArena prend possession de ar : il le ferme
// en cas d'erreur, ou le conserve comme arène de rerank en cas de succès.
func (idx *Index) buildFromOpenArena(ctx context.Context, ar *arena, extIDs [][]byte) error {
	n := int(ar.count)
	if n != len(extIDs) {
		_ = ar.close()
		return fmt.Errorf("horosvec: arena count %d != ext_id count %d", n, len(extIDs))
	}
	if n == 0 {
		_ = ar.close()
		return fmt.Errorf("horosvec: no vectors provided")
	}
	dim := ar.dim

	// Bride le GOGC pendant le build : le graphe alloue beaucoup de scratch éphémère
	// (greedySearch/robustPrune) ; sans bride, l'amplification GC ferait gonfler le pic RSS.
	// Restauré en fin de build (defer).
	prevGC := debug.SetGCPercent(streamingBuildGCPercent)
	defer debug.SetGCPercent(prevGC)

	// Libère toute arène de rerank antérieure avant d'en adopter une neuve (build rejouable).
	if idx.arena != nil {
		_ = idx.arena.close()
		idx.arena = nil
	}

	idx.dim = dim
	rounds := idx.cfg.RotationRounds
	seed := effectiveRotationSeed(idx.cfg)
	idx.rotator = NewRotator(dim, rounds, seed)
	idx.codeDim = idx.rotator.CodeDim()

	// Centroïde de rotation en flux : décodage à la demande, un vecteur à la fois (slotProbe),
	// aucun tampon O(N×dim). Sémantique identique au chemin fp32 (mêmes valeurs fp16-arrondies).
	centroidSrc := newArenaVecSource(ar)
	rotated := make([]float32, idx.codeDim)
	centroid := make([]float32, idx.codeDim)
	for i := 0; i < n; i++ {
		if err := ctx.Err(); err != nil {
			_ = ar.close()
			return fmt.Errorf("horosvec: %w", err)
		}
		v := centroidSrc.vec(int64(i), slotProbe)
		if v == nil {
			_ = ar.close()
			return fmt.Errorf("horosvec: arena decode node %d out of range", i)
		}
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
	// Alimente le tracker de dérive en flux (un vecteur à la fois), même séquence 0..n-1 que
	// l'ancien AddBatch → moyenne mobile bit-identique, sans matérialiser [][]float32.
	ctrBuf := make([]float32, dim)
	for i := 0; i < n; i++ {
		if !ar.vecInto(int64(i), ctrBuf) {
			_ = ar.close()
			return fmt.Errorf("horosvec: arena decode node %d out of range", i)
		}
		idx.centroid.Add(ctrBuf)
	}
	idx.centroid.SnapshotBuild()

	buildWorkers := effectiveBuildWorkers(idx.cfg)
	nodes, err := encodeGraphNodesArena(ctx, idx.rotator, idx.encoder, ar, extIDs, buildWorkers)
	if err != nil {
		_ = ar.close()
		return err
	}

	idx.medoid = findMedoidSource(n, dim, newArenaVecSource(ar))
	if err := buildGraphSource(ctx, nodes, newArenaVecSource(ar), idx.medoid,
		idx.cfg.MaxDegree, idx.cfg.SearchListSize, idx.cfg.Alpha, idx.cfg.BuildPasses, buildWorkers); err != nil {
		_ = ar.close()
		return err
	}

	// Persist : SQLite allégé (vector-less) — l'arène est la source des vecteurs bruts.
	tx, err := idx.db.BeginTx(ctx, nil)
	if err != nil {
		_ = ar.close()
		return fmt.Errorf("horosvec: begin tx: %w", err)
	}
	committed := false
	defer func() {
		if !committed {
			_ = tx.Rollback()
		}
	}()

	if _, err := tx.Exec("DELETE FROM vindex_nodes"); err != nil {
		_ = ar.close()
		return fmt.Errorf("horosvec: clear nodes: %w", err)
	}
	if _, err := tx.Exec("DELETE FROM vindex_meta"); err != nil {
		_ = ar.close()
		return fmt.Errorf("horosvec: clear meta: %w", err)
	}
	rotMeta := rotationMeta{
		seed:     seed,
		rounds:   rounds,
		codeDim:  idx.codeDim,
		dbFormat: currentDBFormatVersion,
	}
	if err := saveGraphNoVec(tx, nodes, idx.medoid, dim, idx.cfg.MaxDegree, centroid, rotMeta); err != nil {
		_ = ar.close()
		return fmt.Errorf("horosvec: save graph: %w", err)
	}
	if err := tx.Commit(); err != nil {
		_ = ar.close()
		return fmt.Errorf("horosvec: commit: %w", err)
	}
	committed = true

	idx.nextID = int64(n)
	idx.built = true
	idx.insertedSinceMedoid = 0

	// Gros index vector-less : pas de flatVecs brute-force (jamais atteint en mode arène).
	idx.flatVecs = nil
	idx.flatIDs = nil
	idx.cache.clear()

	// Libère le graphe transitoire avant de bâtir le plan chaud, puis paie la collecte des
	// déchets au moment froid et déterministe.
	nodes = nil
	runtime.GC()

	if err := idx.rebuildPlaneLocked(); err != nil {
		_ = ar.close()
		idx.arena = nil
		return fmt.Errorf("horosvec: build hot plane: %w", err)
	}
	// L'arène mmap reste résidente : seule source des vecteurs bruts au rerank (une seule
	// cartographie sert le build ET le rerank).
	idx.arena = ar
	runtime.GC()
	return nil
}

// encodeGraphNodesArena encode les nœuds du graphe (codes RaBitQ + normes) en lisant les
// vecteurs à la demande depuis l'arène (fp16→fp32), sans matérialiser de tampon fp32 plein.
// node.vec reste nil : la source des vecteurs pour la construction et le rerank est l'arène.
func encodeGraphNodesArena(
	ctx context.Context,
	rotator *Rotator,
	encoder *Encoder,
	ar *arena,
	extIDs [][]byte,
	workers int,
) ([]graphNode, error) {
	n := int(ar.count)
	dim := ar.dim
	nodes := make([]graphNode, n)

	if workers <= 1 {
		rotated := make([]float32, rotator.CodeDim())
		buf := make([]float32, dim)
		for i := 0; i < n; i++ {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			if !ar.vecInto(int64(i), buf) {
				return nil, fmt.Errorf("horosvec: arena decode node %d out of range", i)
			}
			rotator.Rotate(buf, rotated)
			code, sqNorm, l1Norm := encoder.Encode(rotated)
			nodes[i] = graphNode{
				id:     int64(i),
				extID:  extIDs[i],
				vec:    nil,
				code:   code,
				sqNorm: sqNorm,
				l1Norm: l1Norm,
			}
		}
		return nodes, nil
	}

	// Rotator réutilise fhtScratch ; cloner un rotator par worker pour la sûreté concurrente.
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
			buf := make([]float32, dim)
			for i := start; i < end; i++ {
				if firstErr.Load() != nil {
					return
				}
				if err := ctx.Err(); err != nil {
					firstErr.CompareAndSwap(nil, &err)
					return
				}
				if !ar.vecInto(int64(i), buf) {
					err := fmt.Errorf("horosvec: arena decode node %d out of range", i)
					firstErr.CompareAndSwap(nil, &err)
					return
				}
				wr.Rotate(buf, rotated)
				code, sqNorm, l1Norm := encoder.Encode(rotated)
				nodes[i] = graphNode{
					id:     int64(i),
					extID:  extIDs[i],
					vec:    nil,
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
