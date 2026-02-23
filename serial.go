package horosvec

import (
	"encoding/binary"
	"math"
)

// serializeInt32s encodes a slice of int32 values into a byte slice (little-endian).
func serializeInt32s(vals []int32) []byte {
	buf := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(v))
	}
	return buf
}

// deserializeInt32s decodes a byte slice into a slice of int32 values (little-endian).
func deserializeInt32s(buf []byte) []int32 {
	n := len(buf) / 4
	vals := make([]int32, n)
	for i := range n {
		vals[i] = int32(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return vals
}

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

// serializeFloat64(v) encodes a float64 into 8 bytes (little-endian).
func serializeFloat64(v float64) []byte {
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, math.Float64bits(v))
	return buf
}

// deserializeFloat64 decodes 8 bytes into a float64 (little-endian).
func deserializeFloat64(buf []byte) float64 {
	return math.Float64frombits(binary.LittleEndian.Uint64(buf))
}

// serializeInt64(v) encodes an int64 into 8 bytes (little-endian).
func serializeInt64(v int64) []byte {
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, uint64(v))
	return buf
}

// deserializeInt64 decodes 8 bytes into an int64 (little-endian).
func deserializeInt64(buf []byte) int64 {
	return int64(binary.LittleEndian.Uint64(buf))
}
