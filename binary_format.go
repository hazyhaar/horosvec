package horosvec

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
)

// Binary format (big-endian). Header 32 bytes:
//
//   offset  size  field
//   0       4     magic "HVEC"
//   4       1     version (0x01 legacy, 0x02 with rotation meta)
//   5       1     flags (bit0 vectors present, bit1 quantized present)
//   6       2     dim (uint16)
//   8       2     max_degree (uint16)
//   10      2     medoid_flag (bit0 = medoid present)
//   12      8     medoid (int64)
//   20      8     num_nodes (int64)
//   28      4     meta_count (uint32)
//
// Per node (in node_id ascending order):
//   8     node_id (int64)
//   2     ext_id_len (uint16)
//   N     ext_id
//   2     neighbors_count (uint16)
//   8*N   neighbors (int64 each)
//   4     quantized_len (uint32)
//   N     quantized (raw blob, present if flag bit1)
//   4     vector_len (uint32 — number of float32, 0 if absent)
//   4*N   vector (float32 each, present if flag bit0 and len>0)
//   4     sq_norm (float32)
//   4     l1_norm (float32)
//
// Per meta entry:
//   2     key_len (uint16)
//   N     key (UTF-8)
//   4     value_len (uint32)
//   N     value

var (
	binaryMagic       = [4]byte{'H', 'V', 'E', 'C'}
	binaryVersion     = uint8(0x02)
	binaryVersionV1   = uint8(0x01)
	supportedVersions = map[uint8]struct{}{
		binaryVersionV1: {},
		binaryVersion:   {},
	}
)

const (
	flagVectorsPresent   uint8  = 1 << 0
	flagQuantizedPresent uint8  = 1 << 1
	flagMedoidPresent    uint16 = 1 << 0
)

// maxPreallocBlob bounds a single length-prefixed blob (quantized code, vector,
// meta value) read from an untrusted binary stream; larger claimed lengths are
// rejected before allocation. 64 MiB is far above any legitimate per-node blob.
const maxPreallocBlob = 1 << 26

// maxPreallocNodes bounds eager slice/map capacity hints derived from an
// untrusted node/meta count. The real count still drives the read loop.
const maxPreallocNodes = 1 << 20

// ErrBadMagic indicates the stream does not start with the HVEC magic header.
var ErrBadMagic = errors.New("horosvec: bad binary magic")

// ErrBadVersion indicates an unsupported binary format version.
var ErrBadVersion = errors.New("horosvec: unsupported binary version")

// nodeBinary is an internal carrier between DB rows / RAM cache and the binary writer.
type nodeBinary struct {
	nodeID    int64
	extID     []byte
	neighbors []int64
	quantized []byte
	vector    []float32
	sqNorm    float64
	l1Norm    float64
}

// ExportBinary serializes the index state to w in the HVEC binary format.
// Works for both DB-backed indices and standalone (db == nil) indices.
// Returns the number of bytes written.
func (idx *Index) ExportBinary(w io.Writer) (int64, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	nodes, meta, err := idx.collectForExport()
	if err != nil {
		return 0, err
	}

	// Sort nodes by node_id for deterministic output.
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].nodeID < nodes[j].nodeID })

	// Buffer everything first then flush once.
	buf := bytes.NewBuffer(make([]byte, 0, 64*1024))
	bw := bufio.NewWriter(buf)

	var flags uint8
	flags |= flagVectorsPresent
	flags |= flagQuantizedPresent

	dim := idx.dim
	if dim < 0 || dim > 0xFFFF {
		return 0, fmt.Errorf("horosvec: dim %d out of uint16 range", dim)
	}
	maxDeg := idx.cfg.MaxDegree
	if maxDeg < 0 || maxDeg > 0xFFFF {
		return 0, fmt.Errorf("horosvec: max_degree %d out of uint16 range", maxDeg)
	}

	// Header
	if _, err := bw.Write(binaryMagic[:]); err != nil {
		return 0, err
	}
	if err := bw.WriteByte(binaryVersion); err != nil {
		return 0, err
	}
	if err := bw.WriteByte(flags); err != nil {
		return 0, err
	}
	if err := binWriteU16(bw, uint16(dim)); err != nil {
		return 0, err
	}
	if err := binWriteU16(bw, uint16(maxDeg)); err != nil {
		return 0, err
	}
	medoidFlag := flagMedoidPresent
	if err := binWriteU16(bw, medoidFlag); err != nil {
		return 0, err
	}
	if err := binWriteI64(bw, idx.medoid); err != nil {
		return 0, err
	}
	if err := binWriteI64(bw, int64(len(nodes))); err != nil {
		return 0, err
	}
	if err := binWriteU32(bw, uint32(len(meta))); err != nil {
		return 0, err
	}

	// Nodes
	for i := range nodes {
		n := &nodes[i]
		if err := binWriteI64(bw, n.nodeID); err != nil {
			return 0, err
		}
		if len(n.extID) > 0xFFFF {
			return 0, fmt.Errorf("horosvec: ext_id len %d exceeds uint16", len(n.extID))
		}
		if err := binWriteU16(bw, uint16(len(n.extID))); err != nil {
			return 0, err
		}
		if _, err := bw.Write(n.extID); err != nil {
			return 0, err
		}
		if len(n.neighbors) > 0xFFFF {
			return 0, fmt.Errorf("horosvec: neighbors count %d exceeds uint16", len(n.neighbors))
		}
		if err := binWriteU16(bw, uint16(len(n.neighbors))); err != nil {
			return 0, err
		}
		for _, nb := range n.neighbors {
			if err := binWriteI64(bw, nb); err != nil {
				return 0, err
			}
		}
		if err := binWriteU32(bw, uint32(len(n.quantized))); err != nil {
			return 0, err
		}
		if len(n.quantized) > 0 {
			if _, err := bw.Write(n.quantized); err != nil {
				return 0, err
			}
		}
		if err := binWriteU32(bw, uint32(len(n.vector))); err != nil {
			return 0, err
		}
		for _, f := range n.vector {
			if err := binWriteF32(bw, f); err != nil {
				return 0, err
			}
		}
		if err := binWriteF32(bw, float32(n.sqNorm)); err != nil {
			return 0, err
		}
		if err := binWriteF32(bw, float32(n.l1Norm)); err != nil {
			return 0, err
		}
	}

	// Meta entries
	for _, m := range meta {
		if len(m.key) > 0xFFFF {
			return 0, fmt.Errorf("horosvec: meta key len %d exceeds uint16", len(m.key))
		}
		if err := binWriteU16(bw, uint16(len(m.key))); err != nil {
			return 0, err
		}
		if _, err := bw.WriteString(m.key); err != nil {
			return 0, err
		}
		if err := binWriteU32(bw, uint32(len(m.value))); err != nil {
			return 0, err
		}
		if len(m.value) > 0 {
			if _, err := bw.Write(m.value); err != nil {
				return 0, err
			}
		}
	}

	if err := bw.Flush(); err != nil {
		return 0, err
	}
	n, err := w.Write(buf.Bytes())
	return int64(n), err
}

type metaEntry struct {
	key   string
	value []byte
}

// collectForExport gathers node + meta data from either the DB or the in-RAM cache.
// Caller must hold idx.mu (R or W).
func (idx *Index) collectForExport() ([]nodeBinary, []metaEntry, error) {
	if idx.db != nil {
		return idx.collectFromDB()
	}
	return idx.collectFromCache()
}

func (idx *Index) collectFromDB() ([]nodeBinary, []metaEntry, error) {
	rows, err := idx.db.Query("SELECT node_id, ext_id, neighbors, quantized, vector, sq_norm, l1_norm FROM vindex_nodes ORDER BY node_id")
	if err != nil {
		return nil, nil, fmt.Errorf("horosvec: export query nodes: %w", err)
	}
	defer rows.Close()

	var nodes []nodeBinary
	for rows.Next() {
		var nb nodeBinary
		var extID, neighborsBlob, quantized, vectorBlob []byte
		if err := rows.Scan(&nb.nodeID, &extID, &neighborsBlob, &quantized, &vectorBlob, &nb.sqNorm, &nb.l1Norm); err != nil {
			return nil, nil, fmt.Errorf("horosvec: export scan node: %w", err)
		}
		nb.extID = append([]byte(nil), extID...)
		nb.neighbors = deserializeInt64s(neighborsBlob)
		nb.quantized = append([]byte(nil), quantized...)
		nb.vector = deserializeFloat32s(vectorBlob)
		nodes = append(nodes, nb)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("horosvec: export iter nodes: %w", err)
	}

	metaRows, err := idx.db.Query("SELECT key, value FROM vindex_meta")
	if err != nil {
		return nil, nil, fmt.Errorf("horosvec: export query meta: %w", err)
	}
	defer metaRows.Close()

	var meta []metaEntry
	for metaRows.Next() {
		var m metaEntry
		var val []byte
		if err := metaRows.Scan(&m.key, &val); err != nil {
			return nil, nil, fmt.Errorf("horosvec: export scan meta: %w", err)
		}
		m.value = append([]byte(nil), val...)
		meta = append(meta, m)
	}
	if err := metaRows.Err(); err != nil {
		return nil, nil, fmt.Errorf("horosvec: export iter meta: %w", err)
	}
	return nodes, meta, nil
}

func (idx *Index) collectFromCache() ([]nodeBinary, []metaEntry, error) {
	idx.cache.mu.RLock()
	defer idx.cache.mu.RUnlock()
	nodes := make([]nodeBinary, 0, len(idx.cache.items))
	for id, cn := range idx.cache.items {
		nodes = append(nodes, nodeBinary{
			nodeID:    id,
			extID:     append([]byte(nil), cn.extID...),
			neighbors: append([]int64(nil), cn.neighbors...),
			quantized: append([]byte(nil), cn.code...),
			vector:    append([]float32(nil), cn.vec...),
			sqNorm:    cn.sqNorm,
			l1Norm:    cn.l1Norm,
		})
	}
	meta := make([]metaEntry, 0, len(idx.standaloneMeta))
	for _, k := range idx.standaloneMetaKeys {
		meta = append(meta, metaEntry{key: k, value: append([]byte(nil), idx.standaloneMeta[k]...)})
	}
	return nodes, meta, nil
}

// ImportBinary deserializes an HVEC binary stream into a standalone Index.
// The returned Index has db == nil and is ready for Search calls. It must not
// be used with Build/Insert (which require a DB).
func ImportBinary(r io.Reader) (*Index, error) {
	br := bufio.NewReader(r)

	// Header
	var magic [4]byte
	if _, err := io.ReadFull(br, magic[:]); err != nil {
		return nil, fmt.Errorf("horosvec: read magic: %w", err)
	}
	if magic != binaryMagic {
		return nil, ErrBadMagic
	}
	version, err := br.ReadByte()
	if err != nil {
		return nil, fmt.Errorf("horosvec: read version: %w", err)
	}
	if _, ok := supportedVersions[version]; !ok {
		return nil, fmt.Errorf("%w: got 0x%02x", ErrBadVersion, version)
	}
	flags, err := br.ReadByte()
	if err != nil {
		return nil, fmt.Errorf("horosvec: read flags: %w", err)
	}
	dim, err := binReadU16(br)
	if err != nil {
		return nil, fmt.Errorf("horosvec: read dim: %w", err)
	}
	maxDeg, err := binReadU16(br)
	if err != nil {
		return nil, fmt.Errorf("horosvec: read max_degree: %w", err)
	}
	medoidFlag, err := binReadU16(br)
	if err != nil {
		return nil, fmt.Errorf("horosvec: read medoid_flag: %w", err)
	}
	medoid, err := binReadI64(br)
	if err != nil {
		return nil, fmt.Errorf("horosvec: read medoid: %w", err)
	}
	numNodes, err := binReadI64(br)
	if err != nil {
		return nil, fmt.Errorf("horosvec: read num_nodes: %w", err)
	}
	metaCount, err := binReadU32(br)
	if err != nil {
		return nil, fmt.Errorf("horosvec: read meta_count: %w", err)
	}

	hasVectors := flags&flagVectorsPresent != 0
	hasQuantized := flags&flagQuantizedPresent != 0
	hasMedoid := medoidFlag&flagMedoidPresent != 0

	// Validate header fields before allocating: the stream may be corrupt or
	// hostile. Negative or absurd counts must not reach make() (panic/OOM).
	if numNodes < 0 {
		return nil, fmt.Errorf("horosvec: import: negative num_nodes %d", numNodes)
	}
	if dim == 0 && numNodes > 0 {
		return nil, fmt.Errorf("horosvec: import: %d nodes with zero dim", numNodes)
	}
	// Cap eager pre-allocation hints so a tiny corrupt blob claiming a huge
	// node count cannot OOM before io.ReadFull fails on the truncated body.
	// The read loop still honours the real numNodes; only the initial capacity
	// hint is bounded, growth happens as bytes are actually read.
	capHint := int(numNodes)
	if capHint > maxPreallocNodes {
		capHint = maxPreallocNodes
	}

	cfg := DefaultConfig()
	cfg.MaxDegree = int(maxDeg)
	if numNodes > int64(cfg.CacheCapacity) {
		cfg.CacheCapacity = int(numNodes)
	}

	metaHint := int(metaCount)
	if metaHint > maxPreallocNodes {
		metaHint = maxPreallocNodes
	}

	idx := &Index{
		db:                 nil,
		cfg:                cfg,
		cache:              newNodeCache(capHint + 1),
		dim:                int(dim),
		nextID:             numNodes,
		standaloneMeta:     make(map[string][]byte, metaHint),
		standaloneMetaKeys: make([]string, 0, metaHint),
	}
	// Standalone cache must hold every node (no eviction); set logical capacity
	// to the true count even though the map hint was bounded above.
	idx.cache.capacity = int(numNodes) + 1
	if hasMedoid {
		idx.medoid = medoid
	}

	flatVecs := make([]float32, 0, capHint*int(dim))
	flatIDs := make([][]byte, 0, capHint)

	for i := int64(0); i < numNodes; i++ {
		nodeID, err := binReadI64(br)
		if err != nil {
			return nil, fmt.Errorf("horosvec: read node_id [%d]: %w", i, err)
		}
		extIDLen, err := binReadU16(br)
		if err != nil {
			return nil, fmt.Errorf("horosvec: read ext_id_len [%d]: %w", i, err)
		}
		extID := make([]byte, extIDLen)
		if _, err := io.ReadFull(br, extID); err != nil {
			return nil, fmt.Errorf("horosvec: read ext_id [%d]: %w", i, err)
		}
		nbCount, err := binReadU16(br)
		if err != nil {
			return nil, fmt.Errorf("horosvec: read neighbors_count [%d]: %w", i, err)
		}
		neighbors := make([]int64, nbCount)
		for j := range neighbors {
			v, err := binReadI64(br)
			if err != nil {
				return nil, fmt.Errorf("horosvec: read neighbor [%d,%d]: %w", i, j, err)
			}
			neighbors[j] = v
		}
		quantLen, err := binReadU32(br)
		if err != nil {
			return nil, fmt.Errorf("horosvec: read quantized_len [%d]: %w", i, err)
		}
		if quantLen > maxPreallocBlob {
			return nil, fmt.Errorf("horosvec: quantized_len %d exceeds cap at node %d", quantLen, i)
		}
		var quantized []byte
		if quantLen > 0 {
			quantized = make([]byte, quantLen)
			if _, err := io.ReadFull(br, quantized); err != nil {
				return nil, fmt.Errorf("horosvec: read quantized [%d]: %w", i, err)
			}
		}
		_ = hasQuantized
		vecLen, err := binReadU32(br)
		if err != nil {
			return nil, fmt.Errorf("horosvec: read vector_len [%d]: %w", i, err)
		}
		if vecLen > maxPreallocBlob {
			return nil, fmt.Errorf("horosvec: vector_len %d exceeds cap at node %d", vecLen, i)
		}
		var vec []float32
		if vecLen > 0 {
			vec = make([]float32, vecLen)
			for j := range vec {
				f, err := binReadF32(br)
				if err != nil {
					return nil, fmt.Errorf("horosvec: read vec elem [%d,%d]: %w", i, j, err)
				}
				vec[j] = f
			}
		}
		_ = hasVectors
		sqNorm, err := binReadF32(br)
		if err != nil {
			return nil, fmt.Errorf("horosvec: read sq_norm [%d]: %w", i, err)
		}
		l1Norm, err := binReadF32(br)
		if err != nil {
			return nil, fmt.Errorf("horosvec: read l1_norm [%d]: %w", i, err)
		}

		idx.cache.put(&cachedNode{
			nodeID:    nodeID,
			extID:     extID,
			neighbors: neighbors,
			vec:       vec,
			code:      quantized,
			sqNorm:    float64(sqNorm),
			l1Norm:    float64(l1Norm),
		})
		if len(vec) == int(dim) {
			flatVecs = append(flatVecs, vec...)
			flatIDs = append(flatIDs, extID)
		}
	}

	for i := uint32(0); i < metaCount; i++ {
		keyLen, err := binReadU16(br)
		if err != nil {
			return nil, fmt.Errorf("horosvec: read meta key_len [%d]: %w", i, err)
		}
		keyBuf := make([]byte, keyLen)
		if _, err := io.ReadFull(br, keyBuf); err != nil {
			return nil, fmt.Errorf("horosvec: read meta key [%d]: %w", i, err)
		}
		valLen, err := binReadU32(br)
		if err != nil {
			return nil, fmt.Errorf("horosvec: read meta value_len [%d]: %w", i, err)
		}
		if valLen > maxPreallocBlob {
			return nil, fmt.Errorf("horosvec: meta value_len %d exceeds cap at entry %d", valLen, i)
		}
		val := make([]byte, valLen)
		if valLen > 0 {
			if _, err := io.ReadFull(br, val); err != nil {
				return nil, fmt.Errorf("horosvec: read meta value [%d]: %w", i, err)
			}
		}
		key := string(keyBuf)
		idx.standaloneMeta[key] = val
		idx.standaloneMetaKeys = append(idx.standaloneMetaKeys, key)
	}

	// Rotation meta: v1 exports omit keys → identity rotation, codeDim=dim.
	rotRounds := 0
	rotSeed := uint64(0)
	codeDim := int(dim)
	if version >= binaryVersion {
		if b, ok := idx.standaloneMeta["rotation_rounds"]; ok {
			rotRounds = int(deserializeInt64(b))
		}
		if b, ok := idx.standaloneMeta["rotation_seed"]; ok {
			rotSeed = uint64(deserializeInt64(b))
		}
		if b, ok := idx.standaloneMeta["code_dim"]; ok {
			codeDim = int(deserializeInt64(b))
		}
	}
	if err := validateRotationMeta(int(dim), codeDim, rotRounds); err != nil {
		return nil, err
	}
	idx.codeDim = codeDim
	idx.rotator = NewRotatorWithCodeDim(int(dim), codeDim, rotRounds, rotSeed)

	// Centroid lives in rotated/padded space; length must match codeDim.
	centroid := make([]float32, codeDim)
	if cb, ok := idx.standaloneMeta["centroid"]; ok {
		if len(cb) != codeDim*4 {
			return nil, fmt.Errorf("horosvec: centroid length %d != code_dim*4 (%d)", len(cb), codeDim*4)
		}
		centroid = deserializeFloat32s(cb)
	}
	idx.encoder = NewEncoder(centroid)
	idx.centroid = NewCentroidTracker(int(dim), cfg.DriftThreshold, cfg.InsertRatioThreshold)
	idx.centroid.SetCentroid(centroid, numNodes)
	idx.centroid.SetBuildCentroid(centroid, numNodes)

	if numNodes > 0 {
		idx.built = true
	}

	// Populate flat vectors for brute-force path. Always set in standalone mode
	// so bruteForceSearch doesn't fall into the SQLite branch.
	if len(flatVecs) == int(numNodes)*int(dim) {
		idx.flatVecs = flatVecs
		idx.flatIDs = flatIDs
	}

	return idx, nil
}

// --- big-endian helpers ---

func binWriteU16(w io.ByteWriter, v uint16) error {
	if err := w.WriteByte(byte(v >> 8)); err != nil {
		return err
	}
	return w.WriteByte(byte(v))
}

func binWriteU32(w io.Writer, v uint32) error {
	var b [4]byte
	binary.BigEndian.PutUint32(b[:], v)
	_, err := w.Write(b[:])
	return err
}

func binWriteI64(w io.Writer, v int64) error {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], uint64(v))
	_, err := w.Write(b[:])
	return err
}

func binWriteF32(w io.Writer, v float32) error {
	var b [4]byte
	binary.BigEndian.PutUint32(b[:], math.Float32bits(v))
	_, err := w.Write(b[:])
	return err
}

func binReadU16(r io.Reader) (uint16, error) {
	var b [2]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint16(b[:]), nil
}

func binReadU32(r io.Reader) (uint32, error) {
	var b [4]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint32(b[:]), nil
}

func binReadI64(r io.Reader) (int64, error) {
	var b [8]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return int64(binary.BigEndian.Uint64(b[:])), nil
}

func binReadF32(r io.Reader) (float32, error) {
	var b [4]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return math.Float32frombits(binary.BigEndian.Uint32(b[:])), nil
}
