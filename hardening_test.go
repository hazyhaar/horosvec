package horosvec

import (
	"bytes"
	"context"
	"encoding/binary"
	"math/rand/v2"
	"strings"
	"testing"
)

// TestImportBinary_NegativeNumNodes ensures a corrupt header with a negative
// node count is rejected cleanly instead of panicking on make() (finding B1).
func TestImportBinary_NegativeNumNodes(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(binaryMagic[:])
	buf.WriteByte(binaryVersion)
	buf.WriteByte(flagVectorsPresent | flagQuantizedPresent)
	_ = binWriteU16(&buf, 8)                 // dim
	_ = binWriteU16(&buf, 4)                 // max_degree
	_ = binWriteU16(&buf, flagMedoidPresent) // medoid_flag
	_ = binWriteI64(&buf, 0)                 // medoid
	_ = binWriteI64(&buf, -1)                // num_nodes (corrupt)
	_ = binWriteU32(&buf, 0)                 // meta_count

	_, err := ImportBinary(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error on negative num_nodes, got nil")
	}
	if !strings.Contains(err.Error(), "num_nodes") {
		t.Fatalf("expected num_nodes error, got %v", err)
	}
}

// TestImportBinary_HugeNodeCountNoOOM ensures a tiny blob claiming an enormous
// node count fails on the truncated body rather than pre-allocating gigabytes
// (finding B1). If pre-allocation were unbounded this would OOM the test binary.
func TestImportBinary_HugeNodeCountNoOOM(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(binaryMagic[:])
	buf.WriteByte(binaryVersion)
	buf.WriteByte(flagVectorsPresent | flagQuantizedPresent)
	_ = binWriteU16(&buf, 8)
	_ = binWriteU16(&buf, 4)
	_ = binWriteU16(&buf, flagMedoidPresent)
	_ = binWriteI64(&buf, 0)
	_ = binWriteI64(&buf, 1<<40) // claim ~1e12 nodes
	_ = binWriteU32(&buf, 0)
	// no node bodies follow -> read must fail on the first node

	_, err := ImportBinary(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error on truncated body, got nil")
	}
}

// TestImportBinary_OversizedBlobLen ensures an absurd length prefix on a node
// blob is rejected before allocation (finding B1).
func TestImportBinary_OversizedBlobLen(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(binaryMagic[:])
	buf.WriteByte(binaryVersion)
	buf.WriteByte(flagVectorsPresent | flagQuantizedPresent)
	_ = binWriteU16(&buf, 8)
	_ = binWriteU16(&buf, 4)
	_ = binWriteU16(&buf, flagMedoidPresent)
	_ = binWriteI64(&buf, 0)
	_ = binWriteI64(&buf, 1) // one node
	_ = binWriteU32(&buf, 0) // no meta
	// node 0
	_ = binWriteI64(&buf, 0) // node_id
	_ = binWriteU16(&buf, 0) // ext_id_len
	_ = binWriteU16(&buf, 0) // neighbors_count
	var b [4]byte
	binary.BigEndian.PutUint32(b[:], 0xFFFFFFFF) // quantized_len absurd
	buf.Write(b[:])

	_, err := ImportBinary(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error on oversized quantized_len, got nil")
	}
	if !strings.Contains(err.Error(), "exceeds cap") {
		t.Fatalf("expected cap error, got %v", err)
	}
}

// TestBuild_HeterogeneousDims ensures Build rejects a batch with mismatched
// vector dimensions instead of panicking during encode (finding B2).
func TestBuild_HeterogeneousDims(t *testing.T) {
	db := newTestDB(t)
	idx, err := New(db, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	vecs := [][]float32{
		{1, 2, 3, 4},
		{1, 2, 3}, // shorter -> would panic without guard
	}
	ids := [][]byte{[]byte("a"), []byte("b")}
	iter := &sliceIterator{vecs: vecs, ids: ids}

	if err := idx.Build(context.Background(), iter); err == nil {
		t.Fatal("expected error on heterogeneous dims, got nil")
	}
}

// TestInsert_HeterogeneousDims ensures Insert validates every vector, not just
// the first (finding B2).
func TestInsert_HeterogeneousDims(t *testing.T) {
	db := newTestDB(t)
	idx, err := New(db, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	rng := rand.New(rand.NewPCG(1, 2))
	vecs, ids := generateVecs(rng, 30, 8)
	iter := &sliceIterator{vecs: vecs, ids: ids}
	if err := idx.Build(context.Background(), iter); err != nil {
		t.Fatal(err)
	}

	bad := [][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{1, 2, 3}, // wrong dim
	}
	badIDs := [][]byte{[]byte("x"), []byte("y")}
	if err := idx.Insert(context.Background(), bad, badIDs); err == nil {
		t.Fatal("expected error on heterogeneous insert dims, got nil")
	}
}
