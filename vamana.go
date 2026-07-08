package horosvec

import (
	"container/heap"
	"context"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
	"sync/atomic"
)

// graphNode represents a node in the Vamana graph during construction.
type graphNode struct {
	id        int64
	extID     []byte
	vec       []float32
	neighbors []int64
	code      []byte
	sqNorm    float64
	l1Norm    float64
}

// Slots de tampon d'un vecSource. Ils nomment les vecteurs qui peuvent être
// SIMULTANÉMENT vivants dans une même passe de construction (par worker), afin qu'une
// implémentation à décodage à la demande (arène) n'écrase jamais un vecteur encore
// référencé : slotSelf = le nœud traité (tenu pendant tout processNode), slotBest = le
// meilleur candidat de robustPrune (tenu pendant la boucle interne), slotProbe = le
// vecteur transitoire comparé immédiatement puis oublié.
const (
	slotSelf  = 0
	slotBest  = 1
	slotProbe = 2
)

// vecSource fournit les vecteurs fp32 des nœuds à la construction du graphe. Deux
// implémentations : memVecSource (slices stables en mémoire, zéro copie, comportement
// legacy strictement inchangé) et arenaVecSource (décodage fp16→fp32 à la demande depuis
// l'arène mmap, dans des tampons par worker réutilisés). Le paramètre slot n'a d'effet que
// sur l'implémentation arène (choix du tampon) ; memVecSource l'ignore et renvoie toujours
// la slice stable du nœud.
type vecSource interface {
	// vec renvoie le vecteur du nœud id dans le tampon slot, ou nil si id est hors bornes.
	// Le contenu est valide jusqu'au prochain vec(_, slot) sur le MÊME slot de la MÊME
	// instance ; deux slots distincts ne s'écrasent jamais entre eux.
	vec(id int64, slot int) []float32
	// clone renvoie une instance indépendante (tampons propres) pour un worker donné.
	clone() vecSource
}

// memVecSource renvoie les slices vecteur stables détenues par les graphNode. Sans état
// mutable : clone se renvoie lui-même et l'accès concurrent est sûr (lecture seule).
type memVecSource struct{ nodes []graphNode }

func (m *memVecSource) vec(id int64, _ int) []float32 {
	if id >= 0 && int(id) < len(m.nodes) {
		return m.nodes[id].vec
	}
	return nil
}

func (m *memVecSource) clone() vecSource { return m }

// funcVecSource adapte une fonction d'accès vecteur (chemin Insert : cache LRU + overlay SQL)
// en vecSource. Les slices renvoyées sont détenues par le cache (stables) : le slot est ignoré
// et clone se renvoie lui-même (lecture seule, sûr en concurrence sur ce chemin).
type funcVecSource func(int64) []float32

func (f funcVecSource) vec(id int64, _ int) []float32 { return f(id) }

func (f funcVecSource) clone() vecSource { return f }

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

// --- priority queue for greedy search ---

type searchCandidate struct {
	nodeID int64
	dist   float64
}

type candidateHeap []searchCandidate

func (h candidateHeap) Len() int            { return len(h) }
func (h candidateHeap) Less(i, j int) bool  { return h[i].dist < h[j].dist }
func (h candidateHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *candidateHeap) Push(x interface{}) { *h = append(*h, x.(searchCandidate)) } // heap.Interface contract
func (h *candidateHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// findMedoid finds the node closest to the centroid of all vectors.
func findMedoid(nodes []graphNode) int64 {
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

// findMedoidSource trouve le nœud le plus proche du centroïde en lisant les vecteurs à la
// demande depuis vs (chemin arène : node.vec est nil). Séquentiel : un seul tampon slotProbe
// suffit, aucun vecteur n'est tenu au-delà de son usage immédiat.
func findMedoidSource(n int, dim int, vs vecSource) int64 {
	if n == 0 {
		return 0
	}
	centroid := make([]float64, dim)
	for i := 0; i < n; i++ {
		v := vs.vec(int64(i), slotProbe)
		for j := 0; j < dim; j++ {
			centroid[j] += float64(v[j])
		}
	}
	invN := 1.0 / float64(n)
	centroidF32 := make([]float32, dim)
	for j := range dim {
		centroidF32[j] = float32(centroid[j] * invN)
	}
	var bestID int64
	bestDist := math.MaxFloat64
	for i := 0; i < n; i++ {
		d := l2DistanceSquared(centroidF32, vs.vec(int64(i), slotProbe))
		if d < bestDist {
			bestDist = d
			bestID = int64(i)
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
	start int64,
	beamWidth int,
	vs vecSource,
	getNeighbors func(int64) []int64,
) (candidates []searchCandidate, visited map[int64]bool) {
	visited = make(map[int64]bool)

	// query est fourni par l'appelant (tampon slotSelf). greedySearch ne tient jamais deux
	// vecteurs récupérés simultanément : un seul tampon transitoire (slotProbe) suffit, il
	// n'écrase jamais query.
	startVec := vs.vec(start, slotProbe)
	if startVec == nil {
		return nil, visited
	}
	startDist := l2DistanceSquared(query, startVec)
	visited[start] = true

	// Use a min-heap for candidates
	h := &candidateHeap{{nodeID: start, dist: startDist}}
	heap.Init(h)

	// best holds the top-beamWidth results (sorted by distance)
	best := make([]searchCandidate, 0, beamWidth+1)
	best = append(best, searchCandidate{nodeID: start, dist: startDist})

	worstBest := startDist

	for h.Len() > 0 {
		cur := heap.Pop(h).(searchCandidate) // heap.Interface contract

		// If this candidate is worse than our worst beamWidth-th result, we're done
		if len(best) >= beamWidth && cur.dist > worstBest {
			break
		}

		neighbors := getNeighbors(cur.nodeID)
		for _, nbr := range neighbors {
			if visited[nbr] {
				continue
			}
			visited[nbr] = true

			nbrVec := vs.vec(nbr, slotProbe)
			if nbrVec == nil {
				continue
			}
			d := l2DistanceSquared(query, nbrVec)

			// Add to candidates if potentially useful
			if len(best) < beamWidth || d < worstBest {
				heap.Push(h, searchCandidate{nodeID: nbr, dist: d})

				// Insert into best list maintaining sort
				best = insertSorted(best, searchCandidate{nodeID: nbr, dist: d})
				if len(best) > beamWidth {
					best = best[:beamWidth]
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
	nodeID int64,
	candidates []searchCandidate,
	alpha float64,
	maxDegree int,
	vs vecSource,
) []int64 {
	// Remove self and duplicates
	seen := map[int64]bool{nodeID: true}
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

	result := make([]int64, 0, maxDegree)
	// Trois vecteurs peuvent être simultanément vivants : nodeVec (tenu toute la fonction),
	// bestVec (tenu pendant la boucle interne), cVec (transitoire). Trois slots distincts
	// (slotSelf/slotBest/slotProbe) garantissent qu'aucun n'écrase l'autre côté arène.
	// robustPrune exclut nodeID de filtered (seen), donc best != nodeID et cVec != nodeVec :
	// aucun alias entre slots ne fausse la distance.
	nodeVec := vs.vec(nodeID, slotSelf)
	if nodeVec == nil {
		return result
	}

	for len(filtered) > 0 && len(result) < maxDegree {
		// Pick closest candidate
		best := filtered[0]
		filtered = filtered[1:]
		result = append(result, best.nodeID)

		bestVec := vs.vec(best.nodeID, slotBest)
		if bestVec == nil {
			continue
		}

		// Filter remaining: remove candidates that are closer to 'best' than to nodeID
		// (scaled by alpha) — the α-RNG rule
		kept := filtered[:0]
		for _, c := range filtered {
			cVec := vs.vec(c.nodeID, slotProbe)
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

const graphMutexShards = 256

type graphShardMutexes struct {
	shards [graphMutexShards]sync.Mutex
}

func (g *graphShardMutexes) lock(id int64) {
	g.shards[id&(graphMutexShards-1)].Lock()
}

func (g *graphShardMutexes) unlock(id int64) {
	g.shards[id&(graphMutexShards-1)].Unlock()
}

// initRandomNeighbors seeds each node with random neighbors using a deterministic PCG RNG.
// Returns neighbor slices pre-drawn sequentially for deterministic consumption.
func initRandomNeighbors(rng *rand.Rand, nodes []graphNode, maxDegree int) {
	n := len(nodes)
	nNeighbors := maxDegree
	if nNeighbors > n-1 {
		nNeighbors = n - 1
	}
	if nNeighbors <= 0 {
		return
	}

	pool := make([]int64, n)
	for i := range n {
		pool[i] = int64(i)
	}
	pickBuf := make([]int, nNeighbors)

	for i := range n {
		node := &nodes[i]
		if len(node.neighbors) > 0 {
			continue
		}
		myIdx := int(node.id)
		pool[myIdx], pool[n-1] = pool[n-1], pool[myIdx]

		node.neighbors = make([]int64, nNeighbors)
		for j := range nNeighbors {
			ri := rng.IntN(n - 1 - j)
			node.neighbors[j] = pool[ri]
			pickBuf[j] = ri
			pool[ri], pool[n-2-j] = pool[n-2-j], pool[ri]
		}

		for j := nNeighbors - 1; j >= 0; j-- {
			ri := pickBuf[j]
			pool[ri], pool[n-2-j] = pool[n-2-j], pool[ri]
		}
		pool[myIdx], pool[n-1] = pool[n-1], pool[myIdx]
	}
}

// shuffleOrders pre-draws a random visit order for each pass (sequential PCG).
func shuffleOrders(rng *rand.Rand, n, passes int) [][]int {
	orders := make([][]int, passes)
	order := make([]int, n)
	for i := range n {
		order[i] = i
	}
	for pass := range passes {
		perm := make([]int, n)
		copy(perm, order)
		for i := n - 1; i > 0; i-- {
			j := rng.IntN(i + 1)
			perm[i], perm[j] = perm[j], perm[i]
		}
		orders[pass] = perm
	}
	return orders
}

// buildGraph builds a Vamana graph from the given nodes.
// Nodes must have sequential IDs 0..len(nodes)-1 for slice-based access.
// buildWorkers==1 uses the legacy sequential path (bit-identical determinism).
// buildWorkers>1 parallelizes each pass; interleaving is non-deterministic but quality-equivalent.
func buildGraph(
	ctx context.Context,
	nodes []graphNode,
	medoid int64,
	maxDegree int,
	beamWidth int,
	alpha float64,
	passes int,
	buildWorkers int,
) error {
	if len(nodes) == 0 {
		return nil
	}
	// Chemin en mémoire : les vecteurs sont les slices stables des graphNode (zéro copie),
	// comportement legacy strictement inchangé. Le chemin arène passe par buildGraphSource.
	return buildGraphSource(ctx, nodes, &memVecSource{nodes: nodes}, medoid, maxDegree, beamWidth, alpha, passes, buildWorkers)
}

// buildGraphSource construit le graphe Vamana en lisant les vecteurs via vs (mémoire ou
// arène). Les voisinages sont portés par nodes[i].neighbors ; vs ne fournit que les
// vecteurs. buildWorkers==1 emprunte le chemin séquentiel déterministe.
func buildGraphSource(
	ctx context.Context,
	nodes []graphNode,
	vs vecSource,
	medoid int64,
	maxDegree int,
	beamWidth int,
	alpha float64,
	passes int,
	buildWorkers int,
) error {
	if len(nodes) == 0 {
		return nil
	}
	workers := buildWorkers
	if workers == 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers == 1 {
		return buildGraphSequential(ctx, nodes, vs, medoid, maxDegree, beamWidth, alpha, passes)
	}
	return buildGraphParallel(ctx, nodes, vs, medoid, maxDegree, beamWidth, alpha, passes, workers)
}

// buildGraphSequential is the legacy single-threaded Vamana build (BuildWorkers=1).
func buildGraphSequential(
	ctx context.Context,
	nodes []graphNode,
	vs vecSource,
	medoid int64,
	maxDegree int,
	beamWidth int,
	alpha float64,
	passes int,
) error {
	n := len(nodes)

	getNeighbors := func(id int64) []int64 {
		if id >= 0 && int(id) < n {
			return nodes[id].neighbors
		}
		return nil
	}

	rng := rand.New(rand.NewPCG(42, 0))
	initRandomNeighbors(rng, nodes, maxDegree)

	order := make([]int, n)
	for i := range n {
		order[i] = i
	}

	for pass := range passes {
		_ = pass
		if err := ctx.Err(); err != nil {
			return err
		}

		for i := n - 1; i > 0; i-- {
			j := rng.IntN(i + 1)
			order[i], order[j] = order[j], order[i]
		}

		for _, oi := range order {
			if err := ctx.Err(); err != nil {
				return err
			}

			node := &nodes[oi]
			selfVec := vs.vec(node.id, slotSelf)

			candidates, _ := greedySearch(selfVec, medoid, beamWidth, vs, getNeighbors)

			for _, nbr := range node.neighbors {
				nbrVec := vs.vec(nbr, slotProbe)
				if nbrVec != nil {
					d := l2DistanceSquared(selfVec, nbrVec)
					candidates = append(candidates, searchCandidate{nodeID: nbr, dist: d})
				}
			}

			newNeighbors := robustPrune(node.id, candidates, alpha, maxDegree, vs)
			node.neighbors = newNeighbors

			for _, nbr := range newNeighbors {
				if nbr < 0 || int(nbr) >= n {
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
					if len(nbrNode.neighbors) > 2*maxDegree {
						// nbrVecB (slotBest) tenu pendant toute la boucle ; nnVec (slotProbe)
						// transitoire. robustPrune(nbr) réécrit ensuite ces slots, mais cands
						// est déjà matérialisé (distances calculées).
						nbrVecB := vs.vec(nbr, slotBest)
						cands := make([]searchCandidate, len(nbrNode.neighbors))
						for ci, nn := range nbrNode.neighbors {
							nnVec := vs.vec(nn, slotProbe)
							if nnVec != nil {
								cands[ci] = searchCandidate{
									nodeID: nn,
									dist:   l2DistanceSquared(nbrVecB, nnVec),
								}
							}
						}
						nbrNode.neighbors = robustPrune(nbr, cands, alpha, maxDegree, vs)
					}
				}
			}
		}
	}

	for i := range nodes {
		if len(nodes[i].neighbors) > maxDegree {
			node := &nodes[i]
			nodeVecB := vs.vec(node.id, slotBest)
			cands := make([]searchCandidate, len(node.neighbors))
			for ci, nn := range node.neighbors {
				nnVec := vs.vec(nn, slotProbe)
				if nnVec != nil {
					cands[ci] = searchCandidate{
						nodeID: nn,
						dist:   l2DistanceSquared(nodeVecB, nnVec),
					}
				}
			}
			node.neighbors = robustPrune(node.id, cands, alpha, maxDegree, vs)
		}
	}
	return nil
}

// buildGraphParallel parallelizes each Vamana pass with a worker pool.
// Neighbor slices are replaced atomically (never mutated in-place) for race-free reads during greedySearch.
func buildGraphParallel(
	ctx context.Context,
	nodes []graphNode,
	vs vecSource,
	medoid int64,
	maxDegree int,
	beamWidth int,
	alpha float64,
	passes int,
	workers int,
) error {
	n := len(nodes)

	rng := rand.New(rand.NewPCG(42, 0))
	initRandomNeighbors(rng, nodes, maxDegree)
	orders := shuffleOrders(rng, n, passes)

	neighborStore := make([]atomic.Pointer[[]int64], n)
	for i := range n {
		stored := append([]int64(nil), nodes[i].neighbors...)
		neighborStore[i].Store(&stored)
	}

	getNeighbors := func(id int64) []int64 {
		if id < 0 || int(id) >= n {
			return nil
		}
		p := neighborStore[id].Load()
		if p == nil {
			return nil
		}
		return *p
	}
	setNeighbors := func(id int64, nb []int64) {
		stored := append([]int64(nil), nb...)
		neighborStore[id].Store(&stored)
	}

	var locks graphShardMutexes

	// processNode reçoit le vecSource propre au worker (wvs) : ses tampons de décodage
	// arène ne sont jamais partagés entre goroutines.
	processNode := func(oi int, wvs vecSource) {
		node := &nodes[oi]
		selfVec := wvs.vec(node.id, slotSelf)

		candidates, _ := greedySearch(selfVec, medoid, beamWidth, wvs, getNeighbors)

		for _, nbr := range getNeighbors(node.id) {
			nbrVec := wvs.vec(nbr, slotProbe)
			if nbrVec != nil {
				d := l2DistanceSquared(selfVec, nbrVec)
				candidates = append(candidates, searchCandidate{nodeID: nbr, dist: d})
			}
		}

		newNeighbors := robustPrune(node.id, candidates, alpha, maxDegree, wvs)
		locks.lock(node.id)
		setNeighbors(node.id, newNeighbors)
		locks.unlock(node.id)

		for _, nbr := range newNeighbors {
			if nbr < 0 || int(nbr) >= n {
				continue
			}
			locks.lock(nbr)
			cur := getNeighbors(nbr)
			found := false
			for _, nn := range cur {
				if nn == node.id {
					found = true
					break
				}
			}
			if !found {
				updated := append(append([]int64(nil), cur...), node.id)
				if len(updated) > 2*maxDegree {
					// nbrVecB (slotBest) tenu pendant toute la boucle ; nnVec (slotProbe)
					// transitoire ; selfVec (slotSelf) n'est plus utilisé ici.
					nbrVecB := wvs.vec(nbr, slotBest)
					cands := make([]searchCandidate, len(updated))
					for ci, nn := range updated {
						nnVec := wvs.vec(nn, slotProbe)
						if nnVec != nil {
							cands[ci] = searchCandidate{
								nodeID: nn,
								dist:   l2DistanceSquared(nbrVecB, nnVec),
							}
						}
					}
					updated = robustPrune(nbr, cands, alpha, maxDegree, wvs)
				}
				setNeighbors(nbr, updated)
			}
			locks.unlock(nbr)
		}
	}

	runPass := func(order []int) error {
		var wg sync.WaitGroup
		var firstErr atomic.Pointer[error]
		batchSize := (n + workers - 1) / workers
		for batchStart := 0; batchStart < n; batchStart += batchSize {
			if firstErr.Load() != nil {
				break
			}
			if err := ctx.Err(); err != nil {
				firstErr.Store(&err)
				break
			}
			batchEnd := batchStart + batchSize
			if batchEnd > n {
				batchEnd = n
			}
			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()
				wvs := vs.clone()
				for bi := start; bi < end; bi++ {
					if firstErr.Load() != nil {
						return
					}
					if err := ctx.Err(); err != nil {
						firstErr.CompareAndSwap(nil, &err)
						return
					}
					processNode(order[bi], wvs)
				}
			}(batchStart, batchEnd)
		}
		wg.Wait()
		if err := firstErr.Load(); err != nil {
			return *err
		}
		return ctx.Err()
	}

	for pass := range passes {
		_ = pass
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := runPass(orders[pass]); err != nil {
			return err
		}
	}

	// Final pass: prune over-capacity neighborhoods (parallel).
	var wg sync.WaitGroup
	var firstErr atomic.Pointer[error]
	batchSize := (n + workers - 1) / workers
	for batchStart := 0; batchStart < n; batchStart += batchSize {
		if firstErr.Load() != nil {
			break
		}
		if err := ctx.Err(); err != nil {
			firstErr.Store(&err)
			break
		}
		batchEnd := batchStart + batchSize
		if batchEnd > n {
			batchEnd = n
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			wvs := vs.clone()
			for i := start; i < end; i++ {
				if firstErr.Load() != nil {
					return
				}
				if err := ctx.Err(); err != nil {
					firstErr.CompareAndSwap(nil, &err)
					return
				}
				if len(getNeighbors(int64(i))) <= maxDegree {
					continue
				}
				cur := getNeighbors(int64(i))
				// nodeVecB (slotBest) tenu pendant la boucle ; nnVec (slotProbe) transitoire.
				nodeVecB := wvs.vec(int64(i), slotBest)
				cands := make([]searchCandidate, len(cur))
				for ci, nn := range cur {
					nnVec := wvs.vec(nn, slotProbe)
					if nnVec != nil {
						cands[ci] = searchCandidate{
							nodeID: nn,
							dist:   l2DistanceSquared(nodeVecB, nnVec),
						}
					}
				}
				setNeighbors(int64(i), robustPrune(int64(i), cands, alpha, maxDegree, wvs))
			}
		}(batchStart, batchEnd)
	}
	wg.Wait()
	if err := firstErr.Load(); err != nil {
		return *err
	}

	for i := range n {
		p := neighborStore[i].Load()
		if p != nil {
			nodes[i].neighbors = *p
		}
	}
	return ctx.Err()
}
