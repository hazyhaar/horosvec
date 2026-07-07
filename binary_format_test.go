package horosvec

import (
	"bytes"
	"context"
	"errors"
	"math/rand/v2"
	"sort"
	"testing"
)

func TestRoundtrip_EmptyIndex(t *testing.T) {
	db := newTestDB(t)
	idx, err := New(db, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	var buf bytes.Buffer
	n, err := idx.ExportBinary(&buf)
	if err != nil {
		t.Fatalf("ExportBinary: %v", err)
	}
	if n <= 0 {
		t.Fatalf("ExportBinary wrote %d bytes", n)
	}
	if int64(buf.Len()) != n {
		t.Fatalf("returned size %d != buffer len %d", n, buf.Len())
	}

	imported, err := ImportBinary(&buf)
	if err != nil {
		t.Fatalf("ImportBinary: %v", err)
	}
	if imported.db != nil {
		t.Fatal("imported index should have db == nil")
	}
	if c := imported.Count(); c != 0 {
		t.Fatalf("Count = %d, want 0", c)
	}
	if len(imported.standaloneMeta) != 0 {
		t.Fatalf("meta count = %d, want 0", len(imported.standaloneMeta))
	}
}

// buildSmallIndex builds a Vamana index over a deterministic set of vectors and
// returns the index plus the source vectors/ids for verification.
func buildSmallIndex(t *testing.T, n, dim int) (*Index, [][]float32, [][]byte) {
	t.Helper()
	db := newTestDB(t)
	rng := rand.New(rand.NewPCG(7, 11))
	vecs, ids := generateVecs(rng, n, dim)

	cfg := DefaultConfig()
	cfg.BruteForceThreshold = 0 // force Vamana path so the binary roundtrip exercises cache/medoid
	cfg.EfSearch = 64

	idx, err := New(db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = idx.Close() })

	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}
	return idx, vecs, ids
}

func topIDSet(results []Result) map[string]struct{} {
	m := make(map[string]struct{}, len(results))
	for _, r := range results {
		m[string(r.ID)] = struct{}{}
	}
	return m
}

func TestRoundtrip_SmallIndex(t *testing.T) {
	const (
		n   = 200
		dim = 32
		k   = 5
	)
	idx, vecs, _ := buildSmallIndex(t, n, dim)

	var buf bytes.Buffer
	if _, err := idx.ExportBinary(&buf); err != nil {
		t.Fatalf("ExportBinary: %v", err)
	}

	imported, err := ImportBinary(&buf)
	if err != nil {
		t.Fatalf("ImportBinary: %v", err)
	}
	if imported.db != nil {
		t.Fatal("standalone Index must have nil db")
	}
	if got := imported.Count(); got != n {
		t.Fatalf("Count = %d, want %d", got, n)
	}
	if imported.dim != dim {
		t.Fatalf("dim = %d, want %d", imported.dim, dim)
	}
	if imported.medoid != idx.medoid {
		t.Fatalf("medoid = %d, want %d", imported.medoid, idx.medoid)
	}

	// For a sample of queries, the top-K sets from source and imported should overlap heavily.
	const queries = 10
	totalOverlap := 0
	for q := 0; q < queries; q++ {
		query := vecs[q*(n/queries)]
		srcRes, err := idx.Search(context.Background(), query, k)
		if err != nil {
			t.Fatalf("source search: %v", err)
		}
		impRes, err := imported.Search(context.Background(), query, k)
		if err != nil {
			t.Fatalf("imported search: %v", err)
		}
		srcSet := topIDSet(srcRes)
		for _, r := range impRes {
			if _, ok := srcSet[string(r.ID)]; ok {
				totalOverlap++
			}
		}
	}
	// Same graph + same vectors → expect near-perfect overlap.
	minExpected := queries * k * 8 / 10 // 80%
	if totalOverlap < minExpected {
		t.Errorf("top-K overlap %d/%d below %d", totalOverlap, queries*k, minExpected)
	}
}

func TestRoundtrip_WithMeta(t *testing.T) {
	const (
		n   = 50
		dim = 16
	)
	idx, _, _ := buildSmallIndex(t, n, dim)

	// Inject custom meta directly via the DB to verify they survive the roundtrip.
	custom := map[string][]byte{
		"shard_label": []byte("alpha"),
		"corpus_sha":  []byte("0123456789abcdef"),
		"empty_val":   {},
	}
	for k, v := range custom {
		if _, err := idx.db.Exec("INSERT OR REPLACE INTO vindex_meta (key, value) VALUES (?, ?)", k, v); err != nil {
			t.Fatalf("inject meta %s: %v", k, err)
		}
	}

	var buf bytes.Buffer
	if _, err := idx.ExportBinary(&buf); err != nil {
		t.Fatalf("ExportBinary: %v", err)
	}
	imported, err := ImportBinary(&buf)
	if err != nil {
		t.Fatalf("ImportBinary: %v", err)
	}
	for k, want := range custom {
		got, ok := imported.standaloneMeta[k]
		if !ok {
			t.Errorf("meta %q missing after import", k)
			continue
		}
		if !bytes.Equal(got, want) {
			t.Errorf("meta %q = %q, want %q", k, got, want)
		}
	}
	// Sanity: known meta from Build (medoid, centroid, dimension) should also be present.
	for _, k := range []string{"medoid", "centroid", "dimension", "node_count"} {
		if _, ok := imported.standaloneMeta[k]; !ok {
			t.Errorf("expected built-in meta key %q in imported meta", k)
		}
	}
	// Stable key ordering between exports.
	keys := append([]string(nil), imported.standaloneMetaKeys...)
	sort.Strings(keys)
	dedup := make(map[string]struct{}, len(keys))
	for _, k := range keys {
		if _, dup := dedup[k]; dup {
			t.Errorf("duplicate meta key after import: %q", k)
		}
		dedup[k] = struct{}{}
	}
}

func TestImportBinary_BadMagic(t *testing.T) {
	buf := bytes.NewBuffer([]byte("XXXX\x01\x00\x00\x10\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"))
	_, err := ImportBinary(buf)
	if !errors.Is(err, ErrBadMagic) {
		t.Fatalf("got %v, want ErrBadMagic", err)
	}
}

func TestImportBinary_BadVersion(t *testing.T) {
	// valid magic, wrong version byte
	hdr := []byte{'H', 'V', 'E', 'C', 0xFF, 0x00}
	hdr = append(hdr, make([]byte, 32-len(hdr))...)
	_, err := ImportBinary(bytes.NewReader(hdr))
	if !errors.Is(err, ErrBadVersion) {
		t.Fatalf("got %v, want ErrBadVersion", err)
	}
}

func TestImportBinary_Truncated(t *testing.T) {
	idx, _, _ := buildSmallIndex(t, 20, 8)
	var buf bytes.Buffer
	if _, err := idx.ExportBinary(&buf); err != nil {
		t.Fatalf("ExportBinary: %v", err)
	}
	full := buf.Bytes()
	if len(full) < 64 {
		t.Fatalf("export too small to truncate: %d", len(full))
	}
	// Cut off mid-stream after the header so node parsing fails.
	truncated := full[:len(full)/2]
	_, err := ImportBinary(bytes.NewReader(truncated))
	if err == nil {
		t.Fatal("expected error on truncated stream")
	}
}
