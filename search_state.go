// Reusable search state to minimize allocations during graph traversal.
package horosvec

import (
	"math"
	"sync"
)

// searchState is a reusable search context that eliminates per-query allocations.
// It uses a bitset for visited nodes, a hand-rolled typed min-heap (no interface boxing),
// and a pre-allocated sorted best list.
type searchState struct {
	visited  []uint64 // bitset: bit i = node i visited
	capacity int64    // bitset capacity in bits

	// Typed min-heap (no interface{} boxing from container/heap).
	heap    []searchCandidate
	heapLen int

	// Sorted best-L list.
	best []searchCandidate

	// Pre-allocated query centering buffer (reused across queries).
	queryCentered []float64
}

var searchPool = sync.Pool{
	New: func() any {
		return &searchState{}
	},
}

// acquireSearchState gets a searchState from the pool and resets it for the given parameters.
func acquireSearchState(maxNodes int64, beamWidth int, dim int) *searchState {
	s := searchPool.Get().(*searchState)
	s.reset(maxNodes, beamWidth, dim)
	return s
}

// releaseSearchState returns a searchState to the pool.
func releaseSearchState(s *searchState) {
	searchPool.Put(s)
}

// reset prepares the searchState for a new query.
func (s *searchState) reset(maxNodes int64, beamWidth int, dim int) {
	// Ensure bitset capacity
	needed := int((maxNodes + 63) / 64)
	if len(s.visited) < needed {
		s.visited = make([]uint64, needed)
	} else {
		clear(s.visited[:needed])
	}
	s.capacity = maxNodes

	// Reset heap
	if cap(s.heap) < beamWidth*2 {
		s.heap = make([]searchCandidate, 0, beamWidth*2)
	} else {
		s.heap = s.heap[:0]
	}
	s.heapLen = 0

	// Reset best list
	if cap(s.best) < beamWidth+1 {
		s.best = make([]searchCandidate, 0, beamWidth+1)
	} else {
		s.best = s.best[:0]
	}

	// Reset query centering buffer
	if cap(s.queryCentered) < dim {
		s.queryCentered = make([]float64, dim)
	} else {
		s.queryCentered = s.queryCentered[:dim]
	}
}

// visit marks a node as visited. Returns true if newly visited, false if already visited.
func (s *searchState) visit(id int64) bool {
	if id < 0 || id >= s.capacity {
		return false
	}
	word := id / 64
	bit := uint(id % 64)
	if s.visited[word]&(1<<bit) != 0 {
		return false
	}
	s.visited[word] |= 1 << bit
	return true
}

// --- typed min-heap (eliminates interface{} boxing from container/heap) ---

func (s *searchState) pushHeap(c searchCandidate) {
	s.heap = append(s.heap, c)
	s.heapLen++
	// sift up
	i := s.heapLen - 1
	for i > 0 {
		parent := (i - 1) / 2
		if s.heap[parent].dist <= s.heap[i].dist {
			break
		}
		s.heap[parent], s.heap[i] = s.heap[i], s.heap[parent]
		i = parent
	}
}

func (s *searchState) popHeap() searchCandidate {
	top := s.heap[0]
	s.heapLen--
	if s.heapLen > 0 {
		s.heap[0] = s.heap[s.heapLen]
		s.siftDown(0)
	}
	s.heap = s.heap[:s.heapLen]
	return top
}

func (s *searchState) siftDown(i int) {
	for {
		smallest := i
		left := 2*i + 1
		right := 2*i + 2
		if left < s.heapLen && s.heap[left].dist < s.heap[smallest].dist {
			smallest = left
		}
		if right < s.heapLen && s.heap[right].dist < s.heap[smallest].dist {
			smallest = right
		}
		if smallest == i {
			break
		}
		s.heap[smallest], s.heap[i] = s.heap[i], s.heap[smallest]
		i = smallest
	}
}

// --- sorted best list ---

// insertBest inserts a candidate into the sorted best list, maintaining at most beamWidth entries.
func (s *searchState) insertBest(c searchCandidate, beamWidth int) {
	// Binary search for insertion point
	lo, hi := 0, len(s.best)
	for lo < hi {
		mid := (lo + hi) / 2
		if s.best[mid].dist < c.dist {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	// Insert at position lo
	s.best = append(s.best, searchCandidate{})
	copy(s.best[lo+1:], s.best[lo:])
	s.best[lo] = c
	if len(s.best) > beamWidth {
		s.best = s.best[:beamWidth]
	}
}

func (s *searchState) worstBestDist() float64 {
	if len(s.best) == 0 {
		return math.MaxFloat64
	}
	return s.best[len(s.best)-1].dist
}
