package horosvec

import (
	"container/heap"
	"context"
	"math"
	"math/rand/v2"
)

// graphNode represents a node in the Vamana graph during construction.
type graphNode struct {
	id        int32
	extID     []byte
	vec       []float32
	neighbors []int32
	code      []byte
	sqNorm    float64
	l1Norm    float64
}

// l2DistanceSquared computes squared L2 distance between two float32 vectors.
// Unrolled 8x for performance.
func l2DistanceSquared(a, b []float32) float64 {
	var sum float64
	n := len(a)
	i := 0
	// Unrolled loop: 8 elements per iteration
	for ; i+8 <= n; i += 8 {
		d0 := float64(a[i]) - float64(b[i])
		d1 := float64(a[i+1]) - float64(b[i+1])
		d2 := float64(a[i+2]) - float64(b[i+2])
		d3 := float64(a[i+3]) - float64(b[i+3])
		d4 := float64(a[i+4]) - float64(b[i+4])
		d5 := float64(a[i+5]) - float64(b[i+5])
		d6 := float64(a[i+6]) - float64(b[i+6])
		d7 := float64(a[i+7]) - float64(b[i+7])
		sum += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7
	}
	for ; i < n; i++ {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return sum
}

// cosineSimilarity computes cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// --- priority queue for greedy search ---

type searchCandidate struct {
	nodeID int32
	dist   float64
}

type candidateHeap []searchCandidate

func (h candidateHeap) Len() int            { return len(h) }
func (h candidateHeap) Less(i, j int) bool   { return h[i].dist < h[j].dist }
func (h candidateHeap) Swap(i, j int)        { h[i], h[j] = h[j], h[i] }
func (h *candidateHeap) Push(x interface{}) { *h = append(*h, x.(searchCandidate)) }
func (h *candidateHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// findMedoid finds the node closest to the centroid of all vectors.
func findMedoid(nodes []graphNode) int32 {
	if len(nodes) == 0 {
		return 0
	}
	dim := len(nodes[0].vec)
	centroid := make([]float64, dim)
	for _, n := range nodes {
		for j, v := range n.vec {
			centroid[j] += float64(v)
		}
	}
	invN := 1.0 / float64(len(nodes))
	centroidF32 := make([]float32, dim)
	for j := range dim {
		centroidF32[j] = float32(centroid[j] * invN)
	}

	bestID := nodes[0].id
	bestDist := math.MaxFloat64
	for _, n := range nodes {
		d := l2DistanceSquared(centroidF32, n.vec)
		if d < bestDist {
			bestDist = d
			bestID = n.id
		}
	}
	return bestID
}

// greedySearch performs a beam search on the Vamana graph starting from the medoid.
// It returns the top-L closest candidates and the set of all visited nodes.
// getVec returns the vector for a given nodeID.
// getNeighbors returns the neighbor list for a given nodeID.
func greedySearch(
	query []float32,
	start int32,
	L int,
	getVec func(int32) []float32,
	getNeighbors func(int32) []int32,
) (candidates []searchCandidate, visited map[int32]bool) {
	visited = make(map[int32]bool)

	// Initialize with start node
	startVec := getVec(start)
	if startVec == nil {
		return nil, visited
	}
	startDist := l2DistanceSquared(query, startVec)
	visited[start] = true

	// Use a min-heap for candidates
	h := &candidateHeap{{nodeID: start, dist: startDist}}
	heap.Init(h)

	// best holds the top-L results (sorted by distance)
	best := make([]searchCandidate, 0, L+1)
	best = append(best, searchCandidate{nodeID: start, dist: startDist})

	worstBest := startDist

	for h.Len() > 0 {
		cur := heap.Pop(h).(searchCandidate)

		// If this candidate is worse than our worst L-th result, we're done
		if len(best) >= L && cur.dist > worstBest {
			break
		}

		neighbors := getNeighbors(cur.nodeID)
		for _, nbr := range neighbors {
			if visited[nbr] {
				continue
			}
			visited[nbr] = true

			nbrVec := getVec(nbr)
			if nbrVec == nil {
				continue
			}
			d := l2DistanceSquared(query, nbrVec)

			// Add to candidates if potentially useful
			if len(best) < L || d < worstBest {
				heap.Push(h, searchCandidate{nodeID: nbr, dist: d})

				// Insert into best list maintaining sort
				best = insertSorted(best, searchCandidate{nodeID: nbr, dist: d})
				if len(best) > L {
					best = best[:L]
				}
				worstBest = best[len(best)-1].dist
			}
		}
	}

	return best, visited
}

// insertSorted inserts a candidate into a sorted slice.
func insertSorted(sorted []searchCandidate, c searchCandidate) []searchCandidate {
	// Binary search for insertion point
	lo, hi := 0, len(sorted)
	for lo < hi {
		mid := (lo + hi) / 2
		if sorted[mid].dist < c.dist {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	// Insert at position lo
	sorted = append(sorted, searchCandidate{})
	copy(sorted[lo+1:], sorted[lo:])
	sorted[lo] = c
	return sorted
}

// robustPrune selects up to R neighbors for a node using the α-RNG rule.
// candidates should include the node's current neighbors and new candidates.
// alpha > 1 promotes longer edges for better graph connectivity.
func robustPrune(
	nodeID int32,
	candidates []searchCandidate,
	alpha float64,
	R int,
	getVec func(int32) []float32,
) []int32 {
	// Remove self and duplicates
	seen := map[int32]bool{nodeID: true}
	filtered := make([]searchCandidate, 0, len(candidates))
	for _, c := range candidates {
		if seen[c.nodeID] {
			continue
		}
		seen[c.nodeID] = true
		filtered = append(filtered, c)
	}

	// Sort by distance
	sortCandidates(filtered)

	result := make([]int32, 0, R)
	nodeVec := getVec(nodeID)
	if nodeVec == nil {
		return result
	}

	for len(filtered) > 0 && len(result) < R {
		// Pick closest candidate
		best := filtered[0]
		filtered = filtered[1:]
		result = append(result, best.nodeID)

		bestVec := getVec(best.nodeID)
		if bestVec == nil {
			continue
		}

		// Filter remaining: remove candidates that are closer to 'best' than to nodeID
		// (scaled by alpha) — the α-RNG rule
		kept := filtered[:0]
		for _, c := range filtered {
			cVec := getVec(c.nodeID)
			if cVec == nil {
				continue
			}
			distToBest := l2DistanceSquared(bestVec, cVec)
			if alpha*distToBest > c.dist {
				kept = append(kept, c)
			}
		}
		filtered = kept
	}

	return result
}

// sortCandidates sorts candidates by distance (insertion sort for small slices).
func sortCandidates(candidates []searchCandidate) {
	for i := 1; i < len(candidates); i++ {
		key := candidates[i]
		j := i - 1
		for j >= 0 && candidates[j].dist > key.dist {
			candidates[j+1] = candidates[j]
			j--
		}
		candidates[j+1] = key
	}
}

// buildGraph builds a Vamana graph from the given nodes.
// Nodes must have sequential IDs 0..len(nodes)-1 for slice-based access.
// It performs the specified number of passes, each time iterating over all nodes
// in random order, doing greedy search + robust prune.
func buildGraph(
	ctx context.Context,
	nodes []graphNode,
	medoid int32,
	R int,
	L int,
	alpha float64,
	passes int,
) {
	n := len(nodes)
	if n == 0 {
		return
	}

	// Slice-based access for O(1) lookup (nodes have IDs 0..n-1)
	getVec := func(id int32) []float32 {
		if id >= 0 && int(id) < n {
			return nodes[id].vec
		}
		return nil
	}
	getNeighbors := func(id int32) []int32 {
		if id >= 0 && int(id) < n {
			return nodes[id].neighbors
		}
		return nil
	}

	rng := rand.New(rand.NewPCG(42, 0))

	// Initialize with random neighbors using Fisher-Yates partial shuffle
	nNeighbors := R
	if nNeighbors > n-1 {
		nNeighbors = n - 1
	}
	// Reusable buffer for sampling random neighbors
	pool := make([]int32, n)
	for i := range n {
		pool[i] = int32(i)
	}
	for i := range n {
		node := &nodes[i]
		if len(node.neighbors) > 0 {
			continue
		}
		// Partial Fisher-Yates: swap node.id to the end, then pick nNeighbors from the rest
		// First, swap this node's ID to the end to exclude it
		myIdx := int(node.id)
		pool[myIdx], pool[n-1] = pool[n-1], pool[myIdx]

		// Pick nNeighbors from pool[0..n-2]
		node.neighbors = make([]int32, nNeighbors)
		for j := range nNeighbors {
			ri := rng.IntN(n - 1 - j)
			node.neighbors[j] = pool[ri]
			pool[ri], pool[n-2-j] = pool[n-2-j], pool[ri]
		}

		// Restore pool
		pool[myIdx], pool[n-1] = pool[n-1], pool[myIdx]
		for j := range nNeighbors {
			// Restore swapped elements (approximate — full restore not needed
			// since we only need distinct random IDs, not a pristine pool)
			_ = j
		}
		// Re-initialize pool for next node (simple and correct)
		for j := range n {
			pool[j] = int32(j)
		}
	}

	// Build order buffer (reused across passes)
	order := make([]int, n)
	for i := range n {
		order[i] = i
	}

	for pass := range passes {
		_ = pass
		if ctx.Err() != nil {
			return
		}

		// Fisher-Yates shuffle of order
		for i := n - 1; i > 0; i-- {
			j := rng.IntN(i + 1)
			order[i], order[j] = order[j], order[i]
		}

		for _, oi := range order {
			if ctx.Err() != nil {
				return
			}

			node := &nodes[oi]

			// Greedy search from medoid to this node's vector
			candidates, _ := greedySearch(node.vec, medoid, L, getVec, getNeighbors)

			// Add current neighbors to candidate set
			for _, nbr := range node.neighbors {
				nbrVec := getVec(nbr)
				if nbrVec != nil {
					d := l2DistanceSquared(node.vec, nbrVec)
					candidates = append(candidates, searchCandidate{nodeID: nbr, dist: d})
				}
			}

			// Robust prune to select new neighbors
			newNeighbors := robustPrune(node.id, candidates, alpha, R, getVec)
			node.neighbors = newNeighbors

			// Add reverse edges (simplified: just append, don't prune unless over 2R)
			for _, nbr := range newNeighbors {
				if int(nbr) < 0 || int(nbr) >= n {
					continue
				}
				nbrNode := &nodes[nbr]
				found := false
				for _, nn := range nbrNode.neighbors {
					if nn == node.id {
						found = true
						break
					}
				}
				if !found {
					nbrNode.neighbors = append(nbrNode.neighbors, node.id)
					// Only prune if well over capacity (2R threshold avoids constant re-pruning)
					if len(nbrNode.neighbors) > 2*R {
						cands := make([]searchCandidate, len(nbrNode.neighbors))
						for ci, nn := range nbrNode.neighbors {
							nnVec := getVec(nn)
							if nnVec != nil {
								cands[ci] = searchCandidate{
									nodeID: nn,
									dist:   l2DistanceSquared(nbrNode.vec, nnVec),
								}
							}
						}
						nbrNode.neighbors = robustPrune(nbr, cands, alpha, R, getVec)
					}
				}
			}
		}
	}

	// Final pass: prune any over-capacity neighborhoods
	for i := range nodes {
		if len(nodes[i].neighbors) > R {
			node := &nodes[i]
			cands := make([]searchCandidate, len(node.neighbors))
			for ci, nn := range node.neighbors {
				nnVec := getVec(nn)
				if nnVec != nil {
					cands[ci] = searchCandidate{
						nodeID: nn,
						dist:   l2DistanceSquared(node.vec, nnVec),
					}
				}
			}
			node.neighbors = robustPrune(node.id, cands, alpha, R, getVec)
		}
	}
}

// insertNode inserts a new node into an existing Vamana graph by:
// 1. Greedy searching for its nearest neighbors
// 2. Robust pruning to select edges
// 3. Adding reverse edges
func insertNode(
	newNode *graphNode,
	medoid int32,
	R int,
	L int,
	alpha float64,
	getVec func(int32) []float32,
	getNeighbors func(int32) []int32,
	setNeighbors func(int32, []int32),
) {
	// Greedy search from medoid
	candidates, _ := greedySearch(newNode.vec, medoid, L, getVec, getNeighbors)

	// Robust prune to select neighbors
	newNode.neighbors = robustPrune(newNode.id, candidates, alpha, R, getVec)
	setNeighbors(newNode.id, newNode.neighbors)

	// Add reverse edges
	for _, nbr := range newNode.neighbors {
		nbrNeighbors := getNeighbors(nbr)
		if nbrNeighbors == nil {
			continue
		}

		// Check if reverse edge exists
		found := false
		for _, nn := range nbrNeighbors {
			if nn == newNode.id {
				found = true
				break
			}
		}
		if found {
			continue
		}

		if len(nbrNeighbors) < R {
			setNeighbors(nbr, append(nbrNeighbors, newNode.id))
		} else {
			// Need to prune
			cands := make([]searchCandidate, 0, len(nbrNeighbors)+1)
			nbrVec := getVec(nbr)
			for _, nn := range nbrNeighbors {
				nnVec := getVec(nn)
				if nnVec != nil && nbrVec != nil {
					cands = append(cands, searchCandidate{
						nodeID: nn,
						dist:   l2DistanceSquared(nbrVec, nnVec),
					})
				}
			}
			if nbrVec != nil {
				cands = append(cands, searchCandidate{
					nodeID: newNode.id,
					dist:   l2DistanceSquared(nbrVec, newNode.vec),
				})
			}
			setNeighbors(nbr, robustPrune(nbr, cands, alpha, R, getVec))
		}
	}
}
