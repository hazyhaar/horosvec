package horosvec

import (
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	defaultRotationSeed    uint64 = 42
	currentDBFormatVersion        = 2
)

// nextPow2 returns the smallest power of 2 >= n, for n > 0.
func nextPow2(n int) int {
	if n <= 0 {
		return 1
	}
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// codeDimFor returns the RaBitQ code dimension: padded to a power of 2 when
// rotation is active, otherwise the raw dimension (identity / legacy indexes).
func codeDimFor(dim, rounds int) int {
	if rounds == 0 {
		return dim
	}
	return nextPow2(dim)
}

// Rotator applies randomized Hadamard rotation H·D per round.
// rounds=0 is explicit identity: copy with optional zero-padding to codeDim.
type Rotator struct {
	dim        int
	codeDim    int
	rounds     int
	seed       uint64
	diagonals  [][]float32 // per-round ±1 diagonal, length codeDim
	fhtScratch []float64   // reusable float64 buffer for numerically stable FHT
}

// NewRotator creates a rotator. codeDim is derived from dim and rounds unless
// an explicit codeDim is required for loading a persisted index (use
// NewRotatorWithCodeDim).
func NewRotator(dim, rounds int, seed uint64) *Rotator {
	return NewRotatorWithCodeDim(dim, codeDimFor(dim, rounds), rounds, seed)
}

// NewRotatorWithCodeDim restores a rotator from persisted metadata.
func NewRotatorWithCodeDim(dim, codeDim, rounds int, seed uint64) *Rotator {
	r := &Rotator{
		dim:     dim,
		codeDim: codeDim,
		rounds:  rounds,
		seed:    seed,
	}
	if rounds > 0 {
		r.fhtScratch = make([]float64, codeDim)
		r.diagonals = make([][]float32, rounds)
		for round := range rounds {
			rng := rand.New(rand.NewPCG(seed, uint64(round)))
			d := make([]float32, r.codeDim)
			for i := range d {
				if rng.Uint64()&1 == 0 {
					d[i] = -1
				} else {
					d[i] = 1
				}
			}
			r.diagonals[round] = d
		}
	}
	return r
}

// Dim returns the raw vector dimension.
func (r *Rotator) Dim() int { return r.dim }

// CodeDim returns the padded code dimension.
func (r *Rotator) CodeDim() int { return r.codeDim }

// Rounds returns the number of H·D rounds (0 = identity).
func (r *Rotator) Rounds() int { return r.rounds }

// Seed returns the persisted PCG seed.
func (r *Rotator) Seed() uint64 { return r.seed }

// Rotate copies src (length dim) into dst (length codeDim), zero-padding beyond
// dim, then applies D·H per round. Identity (rounds=0) is copy + padding only.
func (r *Rotator) Rotate(src []float32, dst []float32) {
	if len(dst) < r.codeDim {
		panic("horosvec: rotate dst too small")
	}
	clear(dst[:r.codeDim])
	n := r.dim
	if len(src) < n {
		n = len(src)
	}
	copy(dst, src[:n])

	if r.rounds == 0 {
		return
	}

	for round := range r.rounds {
		d := r.diagonals[round]
		for i := range r.codeDim {
			dst[i] *= d[i]
		}
		fhtInPlace(dst[:r.codeDim], r.fhtScratch)
	}
}

// fhtInPlace applies a normalized fast Walsh-Hadamard transform in place.
// len(v) must be a power of 2. scratch must have len >= len(v).
func fhtInPlace(v []float32, scratch []float64) {
	n := len(v)
	if n == 0 || n&(n-1) != 0 {
		return
	}
	for i := range n {
		scratch[i] = float64(v[i])
	}
	h := 1
	for h < n {
		for i := 0; i < n; i += h * 2 {
			for j := i; j < i+h; j++ {
				x := scratch[j]
				y := scratch[j+h]
				scratch[j] = x + y
				scratch[j+h] = x - y
			}
		}
		h <<= 1
	}
	invSqrt := 1.0 / math.Sqrt(float64(n))
	for i := range n {
		v[i] = float32(scratch[i] * invSqrt)
	}
}

// l2NormSquared computes ||v||₂² over the first n elements.
func l2NormSquared(v []float32, n int) float64 {
	var s float64
	for i := range n {
		x := float64(v[i])
		s += x * x
	}
	return s
}

func effectiveRotationSeed(cfg Config) uint64 {
	if cfg.RotationSeed != 0 {
		return cfg.RotationSeed
	}
	return defaultRotationSeed
}

// validateRotationMeta checks persisted rotation metadata for consistency.
func validateRotationMeta(dim, codeDim, rounds int) error {
	if rounds == 0 {
		if codeDim != dim {
			return fmt.Errorf("horosvec: identity rotation (rounds=0) requires code_dim=%d == dim=%d", codeDim, dim)
		}
		return nil
	}
	want := nextPow2(dim)
	if codeDim != want {
		return fmt.Errorf("horosvec: rotation active requires code_dim=%d, got %d", want, codeDim)
	}
	return nil
}
