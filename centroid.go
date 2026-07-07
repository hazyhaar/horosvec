package horosvec

import (
	"math"
	"sync"
)

// CentroidTracker maintains a running average centroid and detects drift
// from the centroid that was used at build time.
// All methods are safe for concurrent use.
type CentroidTracker struct {
	mu             sync.Mutex
	dim            int
	current        []float32 // running average
	buildCentroid  []float32 // centroid at last build
	count          int64     // total vectors seen
	buildCount     int64     // vectors at last build
	insertsSince   int64     // inserts since last build
	driftThreshold float64
	insertRatio    float64 // rebuild if inserts/buildCount > this
}

// NewCentroidTracker creates a tracker for vectors of the given dimension.
func NewCentroidTracker(dim int, driftThreshold, insertRatio float64) *CentroidTracker {
	return &CentroidTracker{
		dim:            dim,
		current:        make([]float32, dim),
		buildCentroid:  make([]float32, dim),
		driftThreshold: driftThreshold,
		insertRatio:    insertRatio,
	}
}

// Add updates the running average with a new vector.
func (ct *CentroidTracker) Add(vec []float32) {
	ct.mu.Lock()
	ct.count++
	ct.insertsSince++
	inv := float32(1.0 / float64(ct.count))
	for i := 0; i < ct.dim; i++ {
		ct.current[i] += (vec[i] - ct.current[i]) * inv
	}
	ct.mu.Unlock()
}

// AddBatch updates the running average with multiple vectors.
func (ct *CentroidTracker) AddBatch(vecs [][]float32) {
	ct.mu.Lock()
	for _, v := range vecs {
		ct.count++
		inv := float32(1.0 / float64(ct.count))
		for i := 0; i < ct.dim; i++ {
			ct.current[i] += (v[i] - ct.current[i]) * inv
		}
	}
	ct.mu.Unlock()
}

// SnapshotBuild saves the current centroid as the build centroid and resets insert counter.
func (ct *CentroidTracker) SnapshotBuild() {
	ct.mu.Lock()
	ct.buildCentroid = make([]float32, ct.dim)
	copy(ct.buildCentroid, ct.current)
	ct.buildCount = ct.count
	ct.insertsSince = 0
	ct.mu.Unlock()
}

// DriftRatio returns the L2 distance between current and build centroids,
// normalized by the L2 norm of the build centroid. Returns 0 if build centroid is zero.
func (ct *CentroidTracker) DriftRatio() float64 {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	var driftSq, normSq float64
	for i := 0; i < ct.dim; i++ {
		d := float64(ct.current[i]) - float64(ct.buildCentroid[i])
		driftSq += d * d
		normSq += float64(ct.buildCentroid[i]) * float64(ct.buildCentroid[i])
	}
	if normSq == 0 {
		return 0
	}
	return math.Sqrt(driftSq) / math.Sqrt(normSq)
}

// NeedsRebuild returns true if centroid drift exceeds threshold
// or if inserts since build exceed the insert ratio threshold.
func (ct *CentroidTracker) NeedsRebuild() bool {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	if ct.buildCount == 0 {
		return false
	}
	// Inline drift ratio to avoid double-locking.
	var driftSq, normSq float64
	for i := 0; i < ct.dim; i++ {
		d := float64(ct.current[i]) - float64(ct.buildCentroid[i])
		driftSq += d * d
		normSq += float64(ct.buildCentroid[i]) * float64(ct.buildCentroid[i])
	}
	if normSq > 0 {
		drift := math.Sqrt(driftSq) / math.Sqrt(normSq)
		if drift > ct.driftThreshold {
			return true
		}
	}
	return float64(ct.insertsSince)/float64(ct.buildCount) > ct.insertRatio
}

// Current returns the current running centroid.
func (ct *CentroidTracker) Current() []float32 {
	ct.mu.Lock()
	out := make([]float32, ct.dim)
	copy(out, ct.current)
	ct.mu.Unlock()
	return out
}

// Reset clears the tracker state.
func (ct *CentroidTracker) Reset() {
	ct.mu.Lock()
	ct.current = make([]float32, ct.dim)
	ct.buildCentroid = make([]float32, ct.dim)
	ct.count = 0
	ct.buildCount = 0
	ct.insertsSince = 0
	ct.mu.Unlock()
}

// SetCentroid sets the current centroid directly (used when loading from DB).
func (ct *CentroidTracker) SetCentroid(centroid []float32, count int64) {
	ct.mu.Lock()
	ct.current = make([]float32, ct.dim)
	copy(ct.current, centroid)
	ct.count = count
	ct.mu.Unlock()
}

// SetBuildCentroid sets the build centroid directly (used when loading from DB).
func (ct *CentroidTracker) SetBuildCentroid(centroid []float32, buildCount int64) {
	ct.mu.Lock()
	ct.buildCentroid = make([]float32, ct.dim)
	copy(ct.buildCentroid, centroid)
	ct.buildCount = buildCount
	ct.insertsSince = ct.count - buildCount
	if ct.insertsSince < 0 {
		ct.insertsSince = 0
	}
	ct.mu.Unlock()
}
