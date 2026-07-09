package horosvec

import (
	"encoding/binary"
	"math"
)

// serializeFloat32s encodes a slice of float32 values into a byte slice (little-endian).
func serializeFloat32s(vals []float32) []byte {
	buf := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// deserializeFloat32s decodes a byte slice into a slice of float32 values (little-endian).
func deserializeFloat32s(buf []byte) []float32 {
	n := len(buf) / 4
	vals := make([]float32, n)
	for i := range n {
		vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return vals
}

// vecFromBlobChecked désérialise un blob vecteur et vérifie qu'il porte EXACTEMENT dim
// valeurs (len == dim*4). Un blob court, vide ou désaligné (corruption, SQLite vector-less)
// renvoie ok=false : l'appelant SAUTE la ligne au lieu de faire paniquer l2DistanceSquared
// sur un accès hors borne (A4). Le chemin chaud reste sans assertion — la garde est ici.
func vecFromBlobChecked(buf []byte, dim int) ([]float32, bool) {
	if dim <= 0 || len(buf) != dim*4 {
		return nil, false
	}
	return deserializeFloat32s(buf), true
}

// serializeInt64s encodes a slice of int64 values into a byte slice (little-endian).
func serializeInt64s(vals []int64) []byte {
	buf := make([]byte, len(vals)*8)
	for i, v := range vals {
		binary.LittleEndian.PutUint64(buf[i*8:], uint64(v))
	}
	return buf
}

// deserializeInt64s decodes a byte slice into a slice of int64 values (little-endian).
func deserializeInt64s(buf []byte) []int64 {
	n := len(buf) / 8
	vals := make([]int64, n)
	for i := range n {
		vals[i] = int64(binary.LittleEndian.Uint64(buf[i*8:]))
	}
	return vals
}

// serializeInt64 encodes an int64 into 8 bytes (little-endian).
func serializeInt64(v int64) []byte {
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, uint64(v))
	return buf
}

// deserializeInt64 decodes 8 bytes into an int64 (little-endian).
func deserializeInt64(buf []byte) int64 {
	return int64(binary.LittleEndian.Uint64(buf))
}
