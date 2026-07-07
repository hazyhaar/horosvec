package horosvec

import (
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// Arène plate fp16 : stockage contigu, pointer-free, en lecture seule, des vecteurs
// bruts pour la boucle de rerank. Format NEUF, distinct de HVEC v2 (binary_format.go) :
// stride fixe (offset = node_id × dim × 2), little-endian, header versionné.
//
// Objectif : recâbler l'étape de rerank exact du Search pour lire les vecteurs depuis
// cette arène (conversion fp16→fp32 à la volée) au lieu de loadNodeReadOnly, ce qui
// évite tout accès SQL et toute pollution du cache LRU sur le chemin chaud. Sans arène
// (Config.ArenaPath vide), le comportement du moteur reste strictement inchangé.
//
// fp16 : les vecteurs SIFT (entiers 0..255) sont représentés exactement (mantisse 11 bits
// → exact jusqu'à 2048), d'où une dégradation de rappel nulle sur ce corpus ; pour des
// vecteurs quelconques la perte relative est bornée (≈ 2^-11).

const (
	// arenaMagic identifie le format d'arène plate fp16.
	arenaMagic = "HVARENA1"
	// arenaVersion est la version courante du format d'arène.
	arenaVersion uint32 = 1
	// arenaHeaderSize = magic(8) + version(4) + dim(4) + count(8).
	arenaHeaderSize = 24
)

// arena est une vue en lecture seule d'un fichier d'arène fp16 chargé en mémoire.
// Sûr en accès concurrent (immutable après ouverture).
type arena struct {
	dim   int
	count int64
	// data contient l'intégralité du fichier (header + payload) ; la charge
	// utile fp16 commence à arenaHeaderSize.
	data []byte
}

// float32ToFloat16 convertit un float32 en fp16 (IEEE 754 half), arrondi au plus proche
// pair (round-half-to-even). Stdlib pure (math.Float32bits). Sature en +/-Inf sur
// dépassement, propage NaN.
func float32ToFloat16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int32((b>>23)&0xff) - 127 + 15
	mant := b & 0x7fffff

	if (b>>23)&0xff == 0xff {
		// Inf ou NaN.
		if mant != 0 {
			return sign | 0x7e00 // NaN
		}
		return sign | 0x7c00 // Inf
	}
	if exp >= 0x1f {
		// Dépassement de l'exposant fp16 → Inf.
		return sign | 0x7c00
	}
	if exp <= 0 {
		// Sous-normal ou zéro.
		if exp < -10 {
			return sign // trop petit → zéro signé
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		half := mant >> shift
		rem := mant & ((1 << shift) - 1)
		halfway := uint32(1) << (shift - 1)
		if rem > halfway || (rem == halfway && (half&1) == 1) {
			half++
		}
		return sign | uint16(half)
	}
	// Cas normal.
	half := sign | uint16(exp<<10) | uint16(mant>>13)
	rem := mant & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && (half&1) == 1) {
		half++ // un report éventuel se propage dans l'exposant, ce qui est correct
	}
	return half
}

// float16ToFloat32 convertit un fp16 (IEEE 754 half) en float32. Stdlib pure.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x03ff)

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Sous-normal : normaliser vers un float32 normal.
		exp32 := uint32(127 - 15 + 1)
		for mant&0x0400 == 0 {
			mant <<= 1
			exp32--
		}
		mant &= 0x03ff
		return math.Float32frombits(sign | exp32<<23 | mant<<13)
	}
	if exp == 0x1f {
		if mant == 0 {
			return math.Float32frombits(sign | 0x7f800000) // Inf
		}
		return math.Float32frombits(sign | 0x7f800000 | mant<<13) // NaN
	}
	exp32 := exp - 15 + 127
	return math.Float32frombits(sign | exp32<<23 | mant<<13)
}

// ExportArena écrit une arène plate fp16 du jeu de vecteurs de l'index vers path.
// Les vecteurs sont lus depuis SQLite dans l'ordre node_id (dense 0..count-1, garanti
// par la construction Vamana et vérifié par le hot plane). Écriture atomique : le
// contenu est produit dans un fichier temporaire puis renommé.
//
// Requiert un index construit et adossé à une base (mode standalone non supporté :
// les vecteurs bruts n'y sont pas conservés hors flatVecs des petits index).
func (idx *Index) ExportArena(path string) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if !idx.built {
		return fmt.Errorf("horosvec: export arena: index not built")
	}
	if idx.db == nil {
		return fmt.Errorf("horosvec: export arena: standalone index has no raw vectors")
	}
	if idx.dim <= 0 {
		return fmt.Errorf("horosvec: export arena: invalid dim %d", idx.dim)
	}
	return exportArenaFromDB(idx.db, idx.dim, path)
}

// arenaWriter écrit une arène plate fp16 au fil de l'eau : header placeholder d'abord,
// puis une ligne fp16 par vecteur (writeVec), puis finalize() réécrit le header avec le
// count exact, fsync le tout et renomme atomiquement le fichier temporaire vers path.
// Crash-consistance : le fichier final n'apparaît (rename atomique) qu'après un fsync
// complet ; un crash avant finalize laisse seulement le .tmp (jamais renommé), et une
// troncature externe est détectée à la relecture par la garde de longueur d'openArena
// (header + count×dim×2). Un seul écrivain à la fois (arène append-only).
type arenaWriter struct {
	f      *os.File
	tmp    string
	path   string
	dim    int
	count  int64
	rowBuf []byte
}

// newArenaWriter crée le fichier temporaire et réserve l'espace du header (réécrit à la
// fin avec le count exact).
func newArenaWriter(path string, dim int) (*arenaWriter, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("horosvec: arena writer: invalid dim %d", dim)
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return nil, fmt.Errorf("horosvec: arena writer create: %w", err)
	}
	if _, err := f.Write(make([]byte, arenaHeaderSize)); err != nil {
		f.Close()
		os.Remove(tmp)
		return nil, fmt.Errorf("horosvec: arena writer header: %w", err)
	}
	return &arenaWriter{f: f, tmp: tmp, path: path, dim: dim, rowBuf: make([]byte, dim*2)}, nil
}

// writeVec encode et append un vecteur fp32→fp16 dans l'ordre node_id (dense 0..count-1).
func (w *arenaWriter) writeVec(vec []float32) error {
	if len(vec) != w.dim {
		return fmt.Errorf("horosvec: arena writer: vec dim %d != %d", len(vec), w.dim)
	}
	for i, v := range vec {
		binary.LittleEndian.PutUint16(w.rowBuf[i*2:], float32ToFloat16(v))
	}
	if _, err := w.f.Write(w.rowBuf); err != nil {
		return fmt.Errorf("horosvec: arena writer write: %w", err)
	}
	w.count++
	return nil
}

// finalize réécrit le header avec le count exact, fsync et renomme atomiquement.
func (w *arenaWriter) finalize() error {
	if err := w.f.Sync(); err != nil {
		w.abort()
		return fmt.Errorf("horosvec: arena writer sync payload: %w", err)
	}
	if _, err := w.f.Seek(0, 0); err != nil {
		w.abort()
		return fmt.Errorf("horosvec: arena writer seek: %w", err)
	}
	header := make([]byte, arenaHeaderSize)
	copy(header[0:8], arenaMagic)
	binary.LittleEndian.PutUint32(header[8:], arenaVersion)
	binary.LittleEndian.PutUint32(header[12:], uint32(w.dim))
	binary.LittleEndian.PutUint64(header[16:], uint64(w.count))
	if _, err := w.f.Write(header); err != nil {
		w.abort()
		return fmt.Errorf("horosvec: arena writer rewrite header: %w", err)
	}
	if err := w.f.Sync(); err != nil {
		w.abort()
		return fmt.Errorf("horosvec: arena writer sync header: %w", err)
	}
	if err := w.f.Close(); err != nil {
		os.Remove(w.tmp)
		return fmt.Errorf("horosvec: arena writer close: %w", err)
	}
	if err := os.Rename(w.tmp, w.path); err != nil {
		os.Remove(w.tmp)
		return fmt.Errorf("horosvec: arena writer rename: %w", err)
	}
	return nil
}

// abort ferme et supprime le fichier temporaire (aucun fichier final produit).
func (w *arenaWriter) abort() {
	w.f.Close()
	os.Remove(w.tmp)
}

// sync force l'écriture sur disque du payload courant (fsync) sans réécrire le header ni
// renommer. Sert de point de durabilité pour un checkpoint : après un sync réussi, tous les
// vecteurs déjà passés à writeVec sont durables dans le .tmp. Le header reste placeholder
// jusqu'à finalize ; la reprise (resumeArenaWriter) tronque au count durci par le manifest.
func (w *arenaWriter) sync() error {
	if err := w.f.Sync(); err != nil {
		return fmt.Errorf("horosvec: arena writer sync: %w", err)
	}
	return nil
}

// resumeArenaWriter rouvre le fichier temporaire d'une arène en cours (path+".tmp") pour
// reprendre l'écriture après une interruption. Tout octet au-delà de count×dim×2 (écriture
// partielle post-checkpoint jamais confirmée au manifest) est tronqué, et le curseur est
// positionné en append au rang count. Le count fourni provient du checkpoint durable du
// manifest ; la garantie de non-doublon/non-trou repose sur cet appariement.
func resumeArenaWriter(path string, dim int, count int64) (*arenaWriter, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("horosvec: arena writer resume: invalid dim %d", dim)
	}
	if count < 0 {
		return nil, fmt.Errorf("horosvec: arena writer resume: invalid count %d", count)
	}
	if count > (int64(1)<<62)/(int64(dim)*2) {
		return nil, fmt.Errorf("horosvec: arena writer resume: count %d too large for dim %d", count, dim)
	}
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_RDWR, 0o644)
	if err != nil {
		return nil, fmt.Errorf("horosvec: arena writer resume open: %w", err)
	}
	want := int64(arenaHeaderSize) + count*int64(dim)*2
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("horosvec: arena writer resume stat: %w", err)
	}
	if st.Size() < want {
		f.Close()
		return nil, fmt.Errorf("horosvec: arena writer resume: file size %d shorter than checkpoint %d", st.Size(), want)
	}
	if err := f.Truncate(want); err != nil {
		f.Close()
		return nil, fmt.Errorf("horosvec: arena writer resume truncate: %w", err)
	}
	if _, err := f.Seek(want, 0); err != nil {
		f.Close()
		return nil, fmt.Errorf("horosvec: arena writer resume seek: %w", err)
	}
	return &arenaWriter{f: f, tmp: tmp, path: path, dim: dim, count: count, rowBuf: make([]byte, dim*2)}, nil
}

// exportArenaFromDB matérialise l'arène depuis vindex_nodes (chemin de rétro-compat : index
// portant encore les blobs vecteur). Un index construit en streaming vector-less n'a plus de
// blob et écrit son arène directement au build ; ExportArena n'y est pas requis.
func exportArenaFromDB(db *sql.DB, dim int, path string) error {
	rows, err := db.Query("SELECT node_id, vector FROM vindex_nodes ORDER BY node_id")
	if err != nil {
		return fmt.Errorf("horosvec: export arena scan: %w", err)
	}
	defer rows.Close()

	w, err := newArenaWriter(path, dim)
	if err != nil {
		return err
	}
	var pos int64
	for rows.Next() {
		var nodeID int64
		var vecBlob []byte
		if err := rows.Scan(&nodeID, &vecBlob); err != nil {
			w.abort()
			return fmt.Errorf("horosvec: export arena row: %w", err)
		}
		if nodeID != pos {
			w.abort()
			return fmt.Errorf("horosvec: export arena non-sequential node_id %d at position %d", nodeID, pos)
		}
		vec := deserializeFloat32s(vecBlob)
		if len(vec) != dim {
			w.abort()
			return fmt.Errorf("horosvec: export arena node %d dim %d != %d", nodeID, len(vec), dim)
		}
		if err := w.writeVec(vec); err != nil {
			w.abort()
			return err
		}
		pos++
	}
	if err := rows.Err(); err != nil {
		w.abort()
		return fmt.Errorf("horosvec: export arena rows: %w", err)
	}
	return w.finalize()
}

// openArena lit et valide un fichier d'arène fp16. Le header (magic, version, dim,
// count) est vérifié et la longueur du fichier doit correspondre exactement à
// header + count × dim × 2.
func openArena(path string) (*arena, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("horosvec: open arena: %w", err)
	}
	if len(data) < arenaHeaderSize {
		return nil, fmt.Errorf("horosvec: open arena: file too short (%d < %d)", len(data), arenaHeaderSize)
	}
	if string(data[0:8]) != arenaMagic {
		return nil, fmt.Errorf("horosvec: open arena: bad magic")
	}
	version := binary.LittleEndian.Uint32(data[8:])
	if version != arenaVersion {
		return nil, fmt.Errorf("horosvec: open arena: unsupported version %d (want %d)", version, arenaVersion)
	}
	dim := int(binary.LittleEndian.Uint32(data[12:]))
	// Borne haute défensive : un dim aberrant (fichier forgé) ferait déborder le calcul
	// d'offset (nodeID*dim*2) et de longueur. 1<<20 dépasse tout modèle d'embedding réel.
	if dim <= 0 || dim > 1<<20 {
		return nil, fmt.Errorf("horosvec: open arena: invalid dim %d", dim)
	}
	count := int64(binary.LittleEndian.Uint64(data[16:]))
	if count < 0 {
		return nil, fmt.Errorf("horosvec: open arena: invalid count %d", count)
	}
	// Garde anti-overflow sur count avant la multiplication (dim déjà borné ci-dessus).
	if count > (int64(1)<<62)/(int64(dim)*2) {
		return nil, fmt.Errorf("horosvec: open arena: count %d too large for dim %d", count, dim)
	}
	want := int64(arenaHeaderSize) + count*int64(dim)*2
	if int64(len(data)) != want {
		return nil, fmt.Errorf("horosvec: open arena: length %d != expected %d (dim=%d count=%d)", len(data), want, dim, count)
	}
	return &arena{dim: dim, count: count, data: data}, nil
}

// vecInto décode le vecteur fp16 du nœud nodeID vers dst (len == dim), en fp32.
// Retourne false si nodeID est hors de la couverture de l'arène (le chemin appelant
// bascule alors sur la voie SQL). Sûr en accès concurrent (lecture seule).
func (a *arena) vecInto(nodeID int64, dst []float32) bool {
	if nodeID < 0 || nodeID >= a.count {
		return false
	}
	off := arenaHeaderSize + int(nodeID)*a.dim*2
	for i := 0; i < a.dim; i++ {
		h := binary.LittleEndian.Uint16(a.data[off+i*2:])
		dst[i] = float16ToFloat32(h)
	}
	return true
}
