// CLAUDE:SUMMARY Binary serialization helpers for float32 and int64 slices (little-endian).
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
