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

// flatNeighborStore porte l'adjacence de CONSTRUCTION en arène plate : un unique []int32
// global de capacité n×stride, indexé par node_id, plus un compteur de degré par nœud —
// à la place des tranches []int64 par nœud du build historique (en-têtes de slice, arrondis
// d'allocateur, pointeurs scannés par le GC, et churn d'allocation à chaque échange de
// voisinage). C'est le patron d'arène plate de hotPlane reproduit côté build. int32 (au lieu
// d'int64) suffit : les node_id de construction sont denses 0..n-1 avec n < 2^31.
//
// Sûreté concurrente : contrairement au build parallèle antérieur (échange lock-free d'un
// pointeur de slice immuable), une arène plate mutable ne peut offrir de lecture cohérente
// sans verrou. Toute lecture ET toute écriture d'un voisinage passent donc par le mutex
// shardé du nœud (locks != nil en parallèle ; nil en séquentiel, aucun verrou). loadInto
// copie les voisins dans un tampon détenu par l'appelant : la valeur rendue reste valide
// après unlock (c'est une copie, jamais un aliasing de nbrs).
type flatNeighborStore struct {
	n      int
	stride int
	nbrs   []int32
	deg    []int32
	locks  *graphShardMutexes // nil => chemin séquentiel, aucun verrou
}

// newFlatNeighborStore alloue l'arène plate. stride = 2*maxDegree+1 couvre le débordement
// transitoire du build : un voisinage retour croît d'une arête à la fois et n'est re-élagué
// que lorsqu'il dépasse 2*maxDegree (sémantique identique au build antérieur), atteignant au
// pire 2*maxDegree+1 avant élagage. parallel==true arme les verrous shardés.
func newFlatNeighborStore(n, maxDegree int, parallel bool) *flatNeighborStore {
	stride := 2*maxDegree + 1
	s := &flatNeighborStore{
		n:      n,
		stride: stride,
		nbrs:   make([]int32, n*stride),
		deg:    make([]int32, n),
	}
	if parallel {
		s.locks = &graphShardMutexes{}
	}
	return s
}

func (s *flatNeighborStore) lock(id int64) {
	if s.locks != nil {
		s.locks.lock(id)
	}
}

func (s *flatNeighborStore) unlock(id int64) {
	if s.locks != nil {
		s.locks.unlock(id)
	}
}

func (s *flatNeighborStore) degree(id int64) int { return int(s.deg[id]) }

// loadInto ajoute les voisins du nœud id à dst (typiquement dst[:0], détenu par l'appelant)
// et renvoie la slice remplie. L'appelant DOIT détenir lock(id) sur le chemin parallèle.
func (s *flatNeighborStore) loadInto(id int64, dst []int64) []int64 {
	base := int(id) * s.stride
	d := int(s.deg[id])
	for i := 0; i < d; i++ {
		dst = append(dst, int64(s.nbrs[base+i]))
	}
	return dst
}

// set remplace le voisinage du nœud id (au plus stride entrées). L'appelant DOIT détenir
// lock(id) sur le chemin parallèle. Les valeurs sont supposées < 2^31 (node_id denses).
func (s *flatNeighborStore) set(id int64, nbrs []int64) {
	base := int(id) * s.stride
	d := len(nbrs)
	if d > s.stride {
		d = s.stride
	}
	for i := 0; i < d; i++ {
		s.nbrs[base+i] = int32(nbrs[i])
	}
	s.deg[id] = int32(d)
}

// initRandomNeighborsStore ensemence chaque nœud de voisins aléatoires via un PCG
// déterministe, en écrivant dans l'arène plate. La séquence de tirage (mêmes IntN dans le
// même ordre) est conçue pour rester déterministe et reproductible d'un build à l'autre.
// nid == i (node_id denses 0..n-1).
func initRandomNeighborsStore(rng *rand.Rand, store *flatNeighborStore, n, maxDegree int) {
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
	drawn := make([]int64, nNeighbors)

	for i := range n {
		if store.degree(int64(i)) > 0 {
			continue
		}
		myIdx := i
		pool[myIdx], pool[n-1] = pool[n-1], pool[myIdx]

		for j := range nNeighbors {
			ri := rng.IntN(n - 1 - j)
			drawn[j] = pool[ri]
			pickBuf[j] = ri
			pool[ri], pool[n-2-j] = pool[n-2-j], pool[ri]
		}
		store.set(int64(i), drawn)

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

// buildGraph builds a Vamana graph from the given nodes (chemin en mémoire legacy).
// Nodes must have sequential IDs 0..len(nodes)-1. buildWorkers==1 uses the legacy sequential
// path (bit-identical determinism) ; buildWorkers>1 parallelizes each pass. L'adjacence est
// construite dans une arène plate ; à la fin, nodes[i].neighbors est renseigné depuis le store
// pour que la persistance legacy (saveGraph, blob vecteur inclus) reste strictement inchangée.
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
	// Chemin en mémoire : les vecteurs sont les slices stables des graphNode (zéro copie).
	store, err := buildGraphStore(ctx, len(nodes), &memVecSource{nodes: nodes}, medoid, maxDegree, beamWidth, alpha, passes, buildWorkers)
	if err != nil {
		return err
	}
	buf := make([]int64, 0, store.stride)
	for i := range nodes {
		nodes[i].neighbors = append([]int64(nil), store.loadInto(int64(i), buf[:0])...)
	}
	return nil
}

// buildGraphStore construit le graphe Vamana dans une arène plate d'adjacence
// (flatNeighborStore), en lisant les vecteurs via vs (mémoire ou arène). Renvoie le store —
// l'appelant en extrait les voisinages (legacy : recopie dans les graphNode ; arène :
// persistance streaming directe). Aucune tranche de voisins par nœud n'est matérialisée
// pendant la construction. buildWorkers==1 emprunte le chemin séquentiel déterministe.
func buildGraphStore(
	ctx context.Context,
	n int,
	vs vecSource,
	medoid int64,
	maxDegree int,
	beamWidth int,
	alpha float64,
	passes int,
	buildWorkers int,
) (*flatNeighborStore, error) {
	if n == 0 {
		return newFlatNeighborStore(0, maxDegree, false), nil
	}
	workers := buildWorkers
	if workers == 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers == 1 {
		return buildGraphSequentialStore(ctx, n, vs, medoid, maxDegree, beamWidth, alpha, passes)
	}
	return buildGraphParallelStore(ctx, n, vs, medoid, maxDegree, beamWidth, alpha, passes, workers)
}

// buildGraphSequentialStore : build Vamana séquentiel (BuildWorkers=1), déterministe et
// reproductible (mêmes tirages PCG, même ordre de candidats d'un build à l'autre).
// L'adjacence vit dans l'arène plate (store), aucun verrou.
func buildGraphSequentialStore(
	ctx context.Context,
	n int,
	vs vecSource,
	medoid int64,
	maxDegree int,
	beamWidth int,
	alpha float64,
	passes int,
) (*flatNeighborStore, error) {
	store := newFlatNeighborStore(n, maxDegree, false)

	rng := rand.New(rand.NewPCG(42, 0))
	initRandomNeighborsStore(rng, store, n, maxDegree)

	// nbrBuf sert les lectures de greedySearch et de la boucle candidats ; rmwBuf sert la
	// lecture-modification-écriture des arêtes retour. Deux tampons distincts : leurs usages
	// ne se chevauchent jamais dans le temps (cap = stride, aucune réallocation).
	nbrBuf := make([]int64, 0, store.stride)
	rmwBuf := make([]int64, 0, store.stride)
	finalBuf := make([]int64, 0, store.stride)
	getNeighbors := func(id int64) []int64 {
		if id < 0 || int(id) >= n {
			return nil
		}
		return store.loadInto(id, nbrBuf[:0])
	}

	order := make([]int, n)
	for i := range n {
		order[i] = i
	}

	for pass := range passes {
		_ = pass
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		for i := n - 1; i > 0; i-- {
			j := rng.IntN(i + 1)
			order[i], order[j] = order[j], order[i]
		}

		for _, oi := range order {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			nid := int64(oi)
			selfVec := vs.vec(nid, slotSelf)

			candidates, _ := greedySearch(selfVec, medoid, beamWidth, vs, getNeighbors)

			for _, nbr := range getNeighbors(nid) {
				nbrVec := vs.vec(nbr, slotProbe)
				if nbrVec != nil {
					d := l2DistanceSquared(selfVec, nbrVec)
					candidates = append(candidates, searchCandidate{nodeID: nbr, dist: d})
				}
			}

			newNeighbors := robustPrune(nid, candidates, alpha, maxDegree, vs)
			store.set(nid, newNeighbors)

			for _, nbr := range newNeighbors {
				if nbr < 0 || int(nbr) >= n {
					continue
				}
				cur := store.loadInto(nbr, rmwBuf[:0])
				found := false
				for _, nn := range cur {
					if nn == nid {
						found = true
						break
					}
				}
				if found {
					continue
				}
				updated := append(cur, nid)
				if len(updated) > 2*maxDegree {
					// nbrVecB (slotBest) tenu pendant la boucle ; nnVec (slotProbe) transitoire.
					nbrVecB := vs.vec(nbr, slotBest)
					cands := make([]searchCandidate, len(updated))
					for ci, nn := range updated {
						nnVec := vs.vec(nn, slotProbe)
						if nnVec != nil {
							cands[ci] = searchCandidate{nodeID: nn, dist: l2DistanceSquared(nbrVecB, nnVec)}
						}
					}
					updated = robustPrune(nbr, cands, alpha, maxDegree, vs)
				}
				store.set(nbr, updated)
			}
		}
	}

	for i := 0; i < n; i++ {
		if store.degree(int64(i)) <= maxDegree {
			continue
		}
		cur := store.loadInto(int64(i), finalBuf[:0])
		nodeVecB := vs.vec(int64(i), slotBest)
		cands := make([]searchCandidate, len(cur))
		for ci, nn := range cur {
			nnVec := vs.vec(nn, slotProbe)
			if nnVec != nil {
				cands[ci] = searchCandidate{nodeID: nn, dist: l2DistanceSquared(nodeVecB, nnVec)}
			}
		}
		store.set(int64(i), robustPrune(int64(i), cands, alpha, maxDegree, vs))
	}
	return store, nil
}

// buildGraphParallelStore : build Vamana parallèle sur l'arène plate. Le swap lock-free d'un
// pointeur de slice immuable du build antérieur est remplacé par un mécanisme équivalent SÛR
// sous -race : toute lecture ET écriture d'un voisinage passe par le mutex shardé du nœud
// (store.lock/unlock). Une lecture rend une COPIE (loadInto dans un tampon détenu par le
// worker), donc valide après unlock. Un seul verrou de nœud est jamais tenu à la fois (aucune
// imbrication) : pas d'interblocage.
func buildGraphParallelStore(
	ctx context.Context,
	n int,
	vs vecSource,
	medoid int64,
	maxDegree int,
	beamWidth int,
	alpha float64,
	passes int,
	workers int,
) (*flatNeighborStore, error) {
	store := newFlatNeighborStore(n, maxDegree, true)

	rng := rand.New(rand.NewPCG(42, 0))
	initRandomNeighborsStore(rng, store, n, maxDegree)
	orders := shuffleOrders(rng, n, passes)

	// processNode reçoit le vecSource propre au worker (wvs) et deux tampons de voisinage
	// (nbrBuf pour les lectures greedySearch/candidats, rmwBuf pour la boucle arêtes retour) :
	// jamais partagés entre goroutines.
	processNode := func(oi int, wvs vecSource, nbrBuf, rmwBuf []int64) {
		nid := int64(oi)
		getNeighbors := func(id int64) []int64 {
			if id < 0 || int(id) >= n {
				return nil
			}
			store.lock(id)
			r := store.loadInto(id, nbrBuf[:0])
			store.unlock(id)
			return r
		}
		selfVec := wvs.vec(nid, slotSelf)

		candidates, _ := greedySearch(selfVec, medoid, beamWidth, wvs, getNeighbors)

		for _, nbr := range getNeighbors(nid) {
			nbrVec := wvs.vec(nbr, slotProbe)
			if nbrVec != nil {
				d := l2DistanceSquared(selfVec, nbrVec)
				candidates = append(candidates, searchCandidate{nodeID: nbr, dist: d})
			}
		}

		newNeighbors := robustPrune(nid, candidates, alpha, maxDegree, wvs)
		store.lock(nid)
		store.set(nid, newNeighbors)
		store.unlock(nid)

		for _, nbr := range newNeighbors {
			if nbr < 0 || int(nbr) >= n {
				continue
			}
			store.lock(nbr)
			cur := store.loadInto(nbr, rmwBuf[:0])
			found := false
			for _, nn := range cur {
				if nn == nid {
					found = true
					break
				}
			}
			if !found {
				updated := append(cur, nid)
				if len(updated) > 2*maxDegree {
					// nbrVecB (slotBest) tenu pendant la boucle ; nnVec (slotProbe) transitoire.
					nbrVecB := wvs.vec(nbr, slotBest)
					cands := make([]searchCandidate, len(updated))
					for ci, nn := range updated {
						nnVec := wvs.vec(nn, slotProbe)
						if nnVec != nil {
							cands[ci] = searchCandidate{nodeID: nn, dist: l2DistanceSquared(nbrVecB, nnVec)}
						}
					}
					updated = robustPrune(nbr, cands, alpha, maxDegree, wvs)
				}
				store.set(nbr, updated)
			}
			store.unlock(nbr)
		}
	}

	runBatches := func(iter func(i int, wvs vecSource, nbrBuf, rmwBuf []int64)) error {
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
				nbrBuf := make([]int64, 0, store.stride)
				rmwBuf := make([]int64, 0, store.stride)
				for bi := start; bi < end; bi++ {
					if firstErr.Load() != nil {
						return
					}
					if err := ctx.Err(); err != nil {
						firstErr.CompareAndSwap(nil, &err)
						return
					}
					iter(bi, wvs, nbrBuf, rmwBuf)
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
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		order := orders[pass]
		if err := runBatches(func(bi int, wvs vecSource, nbrBuf, rmwBuf []int64) {
			processNode(order[bi], wvs, nbrBuf, rmwBuf)
		}); err != nil {
			return nil, err
		}
	}

	// Passe finale : élague les voisinages en surcapacité (parallèle).
	if err := runBatches(func(i int, wvs vecSource, nbrBuf, _ []int64) {
		id := int64(i)
		store.lock(id)
		d := store.degree(id)
		var cur []int64
		if d > maxDegree {
			cur = store.loadInto(id, nbrBuf[:0])
		}
		store.unlock(id)
		if d <= maxDegree {
			return
		}
		// nodeVecB (slotBest) tenu pendant la boucle ; nnVec (slotProbe) transitoire.
		nodeVecB := wvs.vec(id, slotBest)
		cands := make([]searchCandidate, len(cur))
		for ci, nn := range cur {
			nnVec := wvs.vec(nn, slotProbe)
			if nnVec != nil {
				cands[ci] = searchCandidate{nodeID: nn, dist: l2DistanceSquared(nodeVecB, nnVec)}
			}
		}
		pruned := robustPrune(id, cands, alpha, maxDegree, wvs)
		store.lock(id)
		store.set(id, pruned)
		store.unlock(id)
	}); err != nil {
		return nil, err
	}

	return store, ctx.Err()
}
