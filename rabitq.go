package horosvec

import (
	"math"
	"math/bits"
)

// Encoder handles RaBitQ 1-bit quantization of vectors.
// It stores the centroid used for centering and the dimension of vectors.
type Encoder struct {
	dim      int
	centroid []float32
}

// NewEncoder creates an Encoder with the given centroid.
func NewEncoder(centroid []float32) *Encoder {
	return &Encoder{
		dim:      len(centroid),
		centroid: centroid,
	}
}

// Encode quantizes a vector into a 1-bit code (sign of centered components)
// and returns the code, squared L2 norm, and L1 norm of the centered vector.
// The L1 norm is needed as the RaBitQ correction factor <o_bar, o>.
func (e *Encoder) Encode(vec []float32) (code []byte, sqNorm float64, l1Norm float64) {
	d := e.dim
	nBytes := (d + 7) / 8
	code = make([]byte, nBytes)

	for i := range d {
		centered := float64(vec[i]) - float64(e.centroid[i])
		sqNorm += centered * centered
		if centered >= 0 {
			l1Norm += centered
			code[i/8] |= 1 << uint(i%8)
		} else {
			l1Norm -= centered // |centered| for negative values
		}
	}
	return code, sqNorm, l1Norm
}

// rabitqDistanceAsym computes the RaBitQ asymmetric distance estimate between
// a raw query vector and a stored 1-bit code.
//
// Based on the RaBitQ paper formula:
//
//	dist² ≈ sqNorm_stored + sqNorm_query - 2 * sqNorm_stored * signDot / L1_stored
//
// Where signDot = Σ sign(stored_i) * (query_i - centroid_i) and L1_stored is
// the L1 norm of the centered stored vector (the correction factor <ô_bar, ô>).
func rabitqDistanceAsym(query []float32, centroid []float32, storedCode []byte, storedSqNorm float64, storedL1Norm float64) float64 {
	dim := len(query)
	if dim == 0 || storedL1Norm == 0 {
		return storedSqNorm
	}

	// Compute signDot = <sign(stored-c), query-c> and ||query-c||²
	var signDot float64
	var querySqNorm float64
	for i := range dim {
		centered := float64(query[i]) - float64(centroid[i])
		querySqNorm += centered * centered
		if storedCode[i/8]&(1<<uint(i%8)) != 0 {
			signDot += centered
		} else {
			signDot -= centered
		}
	}

	// RaBitQ corrected distance estimate:
	// <ô, q̂> ≈ signDot * ||o'|| / (||q'|| * L1_o)
	// dist² = ||o'||² + ||q'||² - 2 * ||o'||² * signDot / L1_o
	dist := querySqNorm + storedSqNorm - 2.0*storedSqNorm*signDot/storedL1Norm
	return dist
}

// rabitqDistanceAsymPrecomp is a faster variant for batch queries where
// the query's centered components are pre-computed.
func rabitqDistanceAsymPrecomp(queryCentered []float64, querySqNorm float64, storedCode []byte, storedSqNorm float64, storedL1Norm float64) float64 {
	if storedL1Norm == 0 {
		return storedSqNorm
	}

	var signDot float64
	dim := len(queryCentered)
	for i := range dim {
		if storedCode[i/8]&(1<<uint(i%8)) != 0 {
			signDot += queryCentered[i]
		} else {
			signDot -= queryCentered[i]
		}
	}

	return querySqNorm + storedSqNorm - 2.0*storedSqNorm*signDot/storedL1Norm
}

// rabitqDistance computes symmetric distance between two RaBitQ codes.
// Uses POPCOUNT on uint64 blocks for speed. Useful for benchmarking.
func rabitqDistance(queryCode []byte, storedCode []byte, querySqNorm float64, storedSqNorm float64) float64 {
	totalBits := len(queryCode) * 8

	xorCount := 0
	i := 0
	n := len(queryCode)

	for ; i+8 <= n; i += 8 {
		qWord := uint64(queryCode[i]) |
			uint64(queryCode[i+1])<<8 |
			uint64(queryCode[i+2])<<16 |
			uint64(queryCode[i+3])<<24 |
			uint64(queryCode[i+4])<<32 |
			uint64(queryCode[i+5])<<40 |
			uint64(queryCode[i+6])<<48 |
			uint64(queryCode[i+7])<<56

		sWord := uint64(storedCode[i]) |
			uint64(storedCode[i+1])<<8 |
			uint64(storedCode[i+2])<<16 |
			uint64(storedCode[i+3])<<24 |
			uint64(storedCode[i+4])<<32 |
			uint64(storedCode[i+5])<<40 |
			uint64(storedCode[i+6])<<48 |
			uint64(storedCode[i+7])<<56

		xorCount += bits.OnesCount64(qWord ^ sWord)
	}

	for ; i < n; i++ {
		xorCount += bits.OnesCount8(queryCode[i] ^ storedCode[i])
	}

	agreement := totalBits - xorCount

	if totalBits == 0 {
		return querySqNorm + storedSqNorm
	}

	cosEst := 2.0*float64(agreement)/float64(totalBits) - 1.0
	dist := querySqNorm + storedSqNorm - 2.0*math.Sqrt(querySqNorm*storedSqNorm)*cosEst
	return dist
}
