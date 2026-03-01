# horosvec -- Technical Schema

**Pure Go ANN (Approximate Nearest Neighbor) vector search library using Vamana (DiskANN) graph + RaBitQ 1-bit quantization, backed by SQLite.**

Module: `github.com/hazyhaar/horosvec`
Go: 1.24.0 | Single external dep: `modernc.org/sqlite` v1.46.1 | CGO_ENABLED=0
No binaries (cmd/) -- library-first, imported by siftrag and HORAG.

## File Layout

```
horosvec/
├── horosvec.go        Public API: Index, Config, Result, VectorIterator, Search, Build, Insert, RebuildAsync
├── rabitq.go          RaBitQ 1-bit encoder + asymmetric/symmetric distance functions
├── vamana.go          Vamana graph: buildGraph, greedySearch, robustPrune, insertNode, l2/cosine
├── cache.go           Thread-safe LRU cache (doubly-linked list + map)
├── schema.go          SQLite DDL, node CRUD, metadata, swap, warmCache
├── serial.go          Little-endian binary serialization (int32, float32, float64, int64)
├── centroid.go        CentroidTracker: running average + drift/insert-ratio rebuild detection
├── go.mod             Module definition
├── go.sum             Dependency checksums
├── CLAUDE.md          Project manifest
├── METAVEC.md         Meta-vectorisation concept doc (stratified search)
├── horosvec_test.go   Core tests: recall, insert, persistence, concurrency, rebuild
├── rabitq_test.go     RaBitQ unit tests: order preservation, consistency, correlation, benchmarks
└── audit_test.go      Algorithmic audit tests: correction factor, connectivity, degradation
```

## Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          CONSUMER (siftrag / HORAG)                        ║
║                                                                            ║
║   db, _ := sql.Open("sqlite", path)                                       ║
║   idx, _ := horosvec.New(db, horosvec.DefaultConfig())                     ║
║   idx.Build(ctx, iterator)                                                 ║
║   results, _ := idx.Search(queryVec, topK)                                 ║
║   idx.Insert(newVecs, newIDs)                                              ║
║   if idx.NeedsRebuild() { idx.RebuildAsync(ctx, iter) }                    ║
╚════════════════════════════════╤═════════════════════════════════════════════╝
                                 │
                                 ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                           horosvec.Index                                   ║
║                                                                            ║
║  Fields:                                                                   ║
║    db        *sql.DB           -- SQLite connection (caller-owned)          ║
║    cfg       Config            -- all tuning parameters                     ║
║    cache     *nodeCache        -- LRU cache for graph nodes                 ║
║    encoder   *Encoder          -- RaBitQ encoder (holds centroid)           ║
║    centroid  *CentroidTracker  -- running avg + drift detection             ║
║    medoid    int32             -- entry point node (closest to centroid)    ║
║    dim       int               -- vector dimensionality                    ║
║    nextID    int32             -- next node_id for inserts                  ║
║    built     bool              -- true after Build/load                     ║
║    mu        sync.RWMutex     -- readers (Search) vs writers (Build/Swap)  ║
║    rebuildMu sync.Mutex       -- serializes concurrent rebuilds            ║
╚════════════════════════════════╤═════════════════════════════════════════════╝
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
╔══════════════════╗  ╔════════════════════╗  ╔═══════════════════╗
║   RaBitQ Layer   ║  ║   Vamana Layer     ║  ║   Cache Layer     ║
║  (rabitq.go)     ║  ║  (vamana.go)       ║  ║  (cache.go)       ║
║                  ║  ║                    ║  ║                   ║
║  Encoder         ║  ║  buildGraph()      ║  ║  nodeCache        ║
║   .Encode()      ║  ║  greedySearch()    ║  ║   .get(nodeID)    ║
║                  ║  ║  robustPrune()     ║  ║   .put(node)      ║
║  rabitqDist*()   ║  ║  insertNode()      ║  ║   .clear()        ║
║   AsymPrecomp    ║  ║  findMedoid()      ║  ║   .size()         ║
║   Asym           ║  ║  l2DistSquared()   ║  ║                   ║
║   Symmetric      ║  ║  cosineSimilarity()║  ║  LRU eviction     ║
╚══════════════════╝  ╚════════════════════╝  ║  doubly-linked    ║
                                              ║  list + map       ║
          ┌───────────────────────┐            ╚═══════════════════╝
          ▼                       ▼
╔══════════════════╗  ╔════════════════════╗
║  Schema Layer    ║  ║  Centroid Layer    ║
║  (schema.go)     ║  ║  (centroid.go)     ║
║                  ║  ║                    ║
║  initSchema()    ║  ║  CentroidTracker   ║
║  saveNode()      ║  ║   .Add(vec)        ║
║  loadNode()      ║  ║   .AddBatch(vecs)  ║
║  saveGraph()     ║  ║   .SnapshotBuild() ║
║  loadIndex()     ║  ║   .DriftRatio()    ║
║  swapIndex()     ║  ║   .NeedsRebuild()  ║
║  warmCache()     ║  ║   .SetCentroid()   ║
║  updateNeighbors ║  ║   .SetBuildCentroid║
╚════════╤═════════╝  ╚════════════════════╝
         │
         ▼
╔══════════════════════════════════════════╗
║              SQLite Database             ║
║  (modernc.org/sqlite, pure Go)           ║
║                                          ║
║  WAL mode + busy_timeout + synchronous   ║
╚══════════════════════════════════════════╝
```

## SQLite Database Schema

```
╔══════════════════════════════════════════════════════════════════════╗
║  TABLE: vec_nodes                                                   ║
╠═══════════════╤═══════════╤═════════════════════════════════════════╣
║  Column       │ Type      │ Description                             ║
╠═══════════════╪═══════════╪═════════════════════════════════════════╣
║  node_id      │ INTEGER   │ PRIMARY KEY, internal sequential ID     ║
║  ext_id       │ BLOB      │ NOT NULL UNIQUE, caller-provided UUID   ║
║  neighbors    │ BLOB      │ NOT NULL, serialized []int32 (LE)       ║
║  vector       │ BLOB      │ NOT NULL, serialized []float32 (LE)     ║
║  rabitq       │ BLOB      │ NOT NULL, 1-bit code (dim/8 bytes)      ║
║  sq_norm      │ REAL      │ NOT NULL, ||vec - centroid||^2          ║
║  l1_norm      │ REAL      │ NOT NULL DEFAULT 0, L1 of centered vec  ║
╠═══════════════╧═══════════╧═════════════════════════════════════════╣
║  INDEX: idx_vec_nodes_ext ON vec_nodes(ext_id)                      ║
╚═════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║  TABLE: vec_meta                                                    ║
╠═══════════════╤═══════════╤═════════════════════════════════════════╣
║  Column       │ Type      │ Description                             ║
╠═══════════════╪═══════════╪═════════════════════════════════════════╣
║  key          │ TEXT      │ PRIMARY KEY                              ║
║  value        │ BLOB      │ NOT NULL                                 ║
╠═══════════════╧═══════════╧═════════════════════════════════════════╣
║  Stored keys:                                                       ║
║    "medoid"           int64 LE   -- entry node for graph traversal  ║
║    "dimension"        int64 LE   -- vector dimensionality           ║
║    "max_degree"       int64 LE   -- R parameter at build time       ║
║    "node_count"       int64 LE   -- total vectors indexed           ║
║    "centroid"         []float32  -- centroid used for RaBitQ         ║
║    "built_at"         RFC3339    -- timestamp of last build          ║
║    "vectors_at_build" int64 LE   -- node count at last build        ║
╚═════════════════════════════════════════════════════════════════════╝

Staging tables for async rebuild (same schema, "_new" suffix):
  vec_nodes_new  +  idx_vec_nodes_new_ext
  vec_meta_new

Swap procedure (atomic in single tx):
  vec_nodes -> vec_nodes_old -> DROP
  vec_nodes_new -> vec_nodes
  vec_meta  -> vec_meta_old  -> DROP
  vec_meta_new  -> vec_meta
  Recreate idx_vec_nodes_ext
```

## Storage Size Per Node

```
For dim=1024:
  vector:    1024 * 4 = 4096 bytes (float32)
  rabitq:    1024 / 8 =  128 bytes (1-bit code)
  neighbors:   64 * 4 =  256 bytes (max, int32)
  sq_norm:               8 bytes
  l1_norm:               8 bytes
  ext_id:               16 bytes (UUID)
  ─────────────────────────────
  Total per node:     ~4.5 KB

For dim=128:
  vector:     128 * 4 =  512 bytes
  rabitq:     128 / 8 =   16 bytes
  neighbors:   64 * 4 =  256 bytes
  Total per node:     ~0.8 KB
```

## Config Parameters

```
╔═══════════════════════╤═════════╤═══════════════════════════════════════════════╗
║  Parameter            │ Default │ Description                                   ║
╠═══════════════════════╪═════════╪═══════════════════════════════════════════════╣
║  MaxDegree            │    64   │ R: max edges per node in Vamana graph         ║
║  SearchListSize       │   128   │ L: beam width during graph construction       ║
║  BuildPasses          │     2   │ Number of graph construction passes           ║
║  EfSearch             │   128   │ Beam width during search (auto-expanded)      ║
║  RerankTopN           │   500   │ Candidates re-ranked with exact L2            ║
║  CacheCapacity        │ 100000  │ LRU cache size (nodes)                        ║
║  BruteForceThreshold  │  50000  │ Below: brute-force O(N). Above: Vamana+RaBitQ ║
║  Alpha                │   1.2   │ alpha-RNG pruning (>1 = longer edges)          ║
║  DriftThreshold       │  0.05   │ Centroid drift ratio to trigger rebuild       ║
║  InsertRatioThreshold │  0.30   │ inserts/buildCount ratio to trigger rebuild   ║
╚═══════════════════════╧═════════╧═══════════════════════════════════════════════╝
```

## Search Data Flow

```
Search(query []float32, topK int) -> []Result
│
├── idx.built == false ?  ──────────────────────────────→  ERROR "index not built"
├── len(query) != idx.dim ?  ───────────────────────────→  ERROR "dim mismatch"
│
├── nextID <= BruteForceThreshold ?
│   │
│   YES ──→ bruteForceSearch(query, topK)
│   │       ╔═══════════════════════════════════════════════╗
│   │       ║  SELECT ext_id, vector FROM vec_nodes         ║
│   │       ║  For each row:                                ║
│   │       ║    vec = deserializeFloat32s(vectorBlob)       ║
│   │       ║    d = l2DistanceSquared(query, vec)           ║
│   │       ║    maintain sorted top-K by insertion sort     ║
│   │       ║  Return top-K results                         ║
│   │       ║  Recall: 100%  |  Complexity: O(N * dim)      ║
│   │       ╚═══════════════════════════════════════════════╝
│   │
│   NO ──→ vamanaSearch(query, topK)
│          │
│          │  1. Compute rerankN = max(RerankTopN, topK*3)
│          │     efSearch = max(EfSearch, rerankN)
│          │
│          │  2. STAGE 1: rabitqGreedySearch(query, efSearch)
│          │     ╔═══════════════════════════════════════════════════════╗
│          │     ║  Pre-compute query centering once:                    ║
│          │     ║    queryCentered[i] = query[i] - centroid[i]          ║
│          │     ║    querySqNorm = sum(queryCentered[i]^2)              ║
│          │     ║                                                       ║
│          │     ║  Start at medoid node                                 ║
│          │     ║  Min-heap + sorted best-list (beam search)            ║
│          │     ║                                                       ║
│          │     ║  For each candidate popped from heap:                 ║
│          │     ║    if dist > worstBest and |best| >= L: STOP          ║
│          │     ║    For each neighbor:                                 ║
│          │     ║      loadNode(db, cache, nbrID)                       ║
│          │     ║      d = rabitqDistanceAsymPrecomp(                   ║
│          │     ║            queryCentered, querySqNorm,                ║
│          │     ║            node.code, node.sqNorm, node.l1Norm)       ║
│          │     ║      Insert into heap + best-list if promising        ║
│          │     ║                                                       ║
│          │     ║  Return sorted candidates (by approx distance)        ║
│          │     ╚═══════════════════════════════════════════════════════╝
│          │
│          │  3. STAGE 2: L2 rerank
│          │     ╔═══════════════════════════════════════════════════════╗
│          │     ║  Take top rerankN candidates from Stage 1             ║
│          │     ║  For each:                                            ║
│          │     ║    loadNode(db, cache, nodeID)                        ║
│          │     ║    dist = l2DistanceSquared(query, node.vec)          ║
│          │     ║  Sort by exact L2, take top-K                        ║
│          │     ║  Lookup ext_id for each result                       ║
│          │     ╚═══════════════════════════════════════════════════════╝
│
└── Return []Result{ ID []byte, Score float64 }
```

## Build Data Flow

```
Build(ctx, VectorIterator) -> error
│
│  1. Drain iterator into allVecs[], allIDs[]
│  2. Compute centroid = mean(allVecs)
│  3. Create Encoder(centroid)
│  4. Initialize CentroidTracker, AddBatch, SnapshotBuild
│  5. For each vector:
│       code, sqNorm, l1Norm = encoder.Encode(vec)
│       Create graphNode{id, extID, vec, code, sqNorm, l1Norm}
│  6. medoid = findMedoid(nodes)  -- closest node to centroid
│  7. buildGraph(ctx, nodes, medoid, R, L, alpha, passes)
│       ╔══════════════════════════════════════════════════════╗
│       ║  a. Init random neighbors (Fisher-Yates partial)     ║
│       ║  b. For each pass:                                   ║
│       ║       Shuffle node order                             ║
│       ║       For each node:                                 ║
│       ║         greedySearch(node.vec, medoid, L)            ║
│       ║         Merge with current neighbors                 ║
│       ║         robustPrune(alpha-RNG rule, keep R)          ║
│       ║         Add reverse edges to selected neighbors      ║
│       ║         If nbr.degree > 2R: prune nbr too           ║
│       ║  c. Final pass: prune all nodes to <= R              ║
│       ╚══════════════════════════════════════════════════════╝
│  8. BEGIN transaction
│       DELETE FROM vec_nodes
│       DELETE FROM vec_meta
│       saveGraph(tx, "vec_nodes", nodes, metadata)
│     COMMIT
│  9. Populate LRU cache with all nodes
│ 10. Set built=true, nextID=len(nodes)
```

## Insert Data Flow

```
Insert(vecs [][]float32, ids [][]byte) -> error
│
│  Requires: idx.built == true, dim match
│  Acquires: idx.mu.Lock() (exclusive)
│
│  BEGIN transaction
│  For each vector:
│    1. nodeID = nextID++
│    2. code, sqNorm, l1Norm = encoder.Encode(vec)
│    3. saveNode(tx, nodeID, extID, nil, vec, code, sqNorm, l1Norm)
│    4. rabitqGreedySearch(vec, SearchListSize) -> candidates
│    5. neighbors = top MaxDegree candidates
│    6. setNeighbors(nodeID, neighbors) in DB + cache
│    7. For each neighbor: add reverse edge if room (< MaxDegree)
│    8. cache.put(new cachedNode)
│    9. centroid.Add(vec) -- update running average
│  COMMIT
│  Update node_count in vec_meta
```

## Async Rebuild Data Flow

```
RebuildAsync(ctx, VectorIterator)
│
│  Acquires: rebuildMu.Lock() (serializes rebuilds)
│  Runs in goroutine:
│
│  1. initSchemaNew(db) -- create vec_nodes_new, vec_meta_new
│  2. Drain iterator, compute centroid, encode all
│  3. buildGraph() on new nodes (full rebuild)
│  4. BEGIN tx
│       saveGraph(tx, "vec_nodes_new", ...)
│     COMMIT
│  5. Acquire idx.mu.Lock() (blocks all searches)
│  6. swapIndex(db):
│       RENAME vec_nodes -> vec_nodes_old
│       RENAME vec_nodes_new -> vec_nodes
│       RENAME vec_meta -> vec_meta_old
│       RENAME vec_meta_new -> vec_meta
│       DROP old tables
│       Recreate ext_id index
│  7. Update idx fields (medoid, dim, encoder, centroid, nextID)
│  8. cache.clear() + warmCache(medoid, depth=2)
│  9. Release mu.Lock() -- searches resume
```

## RaBitQ Encoding Detail

```
Encode(vec []float32) -> (code []byte, sqNorm float64, l1Norm float64)
│
│  For each dimension i:
│    centered = vec[i] - centroid[i]
│    sqNorm += centered^2
│    if centered >= 0:
│      l1Norm += centered
│      code[i/8] |= 1 << (i%8)     -- set bit = positive
│    else:
│      l1Norm -= centered           -- |centered| for negative
│
│  Output:
│    code:   dim/8 bytes, 1 bit per dimension (sign of centered value)
│    sqNorm: ||vec - centroid||^2  (float64)
│    l1Norm: L1 norm of centered vector (float64)
│
│  Storage compression ratio:
│    float32 vector: dim * 4 bytes
│    RaBitQ code:    dim / 8 bytes
│    Ratio: 32x compression
```

## RaBitQ Distance Functions

```
1. rabitqDistanceAsymPrecomp(queryCentered, querySqNorm, code, sqNorm, l1Norm)
   ── Used in search (Stage 1 beam traversal)
   ── Query centered ONCE, reused for all nodes
   ── signDot = sum( queryCentered[i] * sign_from_code[i] )
   ── dist = querySqNorm + sqNorm - 2*sqNorm*signDot/l1Norm

2. rabitqDistanceAsym(query, centroid, code, sqNorm, l1Norm)
   ── Used in tests, centers query internally
   ── Same formula, computes queryCentered inline

3. rabitqDistance(queryCode, storedCode, querySqNorm, storedSqNorm)
   ── Symmetric (code vs code), uses POPCOUNT
   ── XOR codes, count differing bits
   ── cosEst = 2*agreement/totalBits - 1
   ── dist = qSq + sSq - 2*sqrt(qSq*sSq)*cosEst
   ── POPCOUNT on uint64 blocks (8 bytes at a time)
```

## LRU Cache Architecture

```
╔════════════════════════════════════════════════════════════╗
║  nodeCache                                                 ║
║                                                            ║
║  mu: sync.RWMutex (get=RLock+promote, put/clear=Lock)     ║
║  capacity: int (default 100,000 nodes)                     ║
║  items: map[int32]*cachedNode   -- O(1) lookup             ║
║                                                            ║
║  Doubly-linked list (most-recent at head):                 ║
║                                                            ║
║  head ──→ [node_A] ←→ [node_B] ←→ [node_C] ←── tail      ║
║           (MRU)                        (LRU)               ║
║                                                            ║
║  get(id):  RLock lookup, then Lock to moveToHead           ║
║  put(node): Lock, update-or-add, addToHead, evictTail      ║
║  clear():  Lock, reset map + head/tail                     ║
║                                                            ║
║  cachedNode fields:                                        ║
║    nodeID    int32                                          ║
║    neighbors []int32                                       ║
║    vec       []float32   -- raw vector for L2 rerank       ║
║    code      []byte      -- RaBitQ 1-bit code              ║
║    sqNorm    float64                                       ║
║    l1Norm    float64                                       ║
║    prev/next *cachedNode -- linked list pointers            ║
╚════════════════════════════════════════════════════════════╝

Cache warming at startup:
  warmCache(db, cache, medoid, depth=2)
  BFS from medoid, 2 hops deep
  Pre-loads medoid + all nodes within 2 graph hops
```

## CentroidTracker Detail

```
╔════════════════════════════════════════════════════════════╗
║  CentroidTracker                                           ║
║                                                            ║
║  current[]      -- running average (Welford-style)         ║
║  buildCentroid[] -- snapshot at last build                  ║
║  count          -- total vectors seen                      ║
║  buildCount     -- vectors at last build                   ║
║  insertsSince   -- inserts since last build                ║
║  driftThreshold -- default 0.05                            ║
║  insertRatio    -- default 0.30                            ║
║                                                            ║
║  Add(vec):                                                 ║
║    count++; insertsSince++                                 ║
║    current[i] += (vec[i] - current[i]) / count             ║
║                                                            ║
║  DriftRatio():                                             ║
║    L2(current, buildCentroid) / L2norm(buildCentroid)      ║
║                                                            ║
║  NeedsRebuild():                                           ║
║    DriftRatio() > 0.05  OR  insertsSince/buildCount > 0.30 ║
╚════════════════════════════════════════════════════════════╝
```

## Concurrency Model

```
╔══════════════════════════════════════════════════════════════════╗
║  sync.RWMutex (idx.mu)                                          ║
║                                                                  ║
║  RLock: Search(), NeedsRebuild(), Count()                       ║
║         Multiple concurrent readers allowed                      ║
║                                                                  ║
║  Lock:  Build(), Insert(), swapIndex (inside rebuildInternal)    ║
║         Exclusive -- blocks all readers                          ║
║                                                                  ║
║  sync.Mutex (idx.rebuildMu)                                      ║
║         Serializes RebuildAsync calls                             ║
║         Held for entire rebuild duration (goroutine)             ║
║         idx.mu.Lock only acquired at swap moment                 ║
║                                                                  ║
║  Timeline during RebuildAsync:                                   ║
║                                                                  ║
║  ─────────────────────────────────────────────────────────────   ║
║  │ rebuildMu.Lock()                                          │   ║
║  │ Build new graph in _new tables  (searches continue)       │   ║
║  │ ....................................................      │   ║
║  │ mu.Lock()  ← brief exclusive lock                         │   ║
║  │   swapIndex + update fields + clear cache                 │   ║
║  │ mu.Unlock()                                               │   ║
║  │ rebuildMu.Unlock()                                        │   ║
║  ─────────────────────────────────────────────────────────────   ║
╚══════════════════════════════════════════════════════════════════╝
```

## Serialization Format (serial.go)

```
All serialization is little-endian (LE).

int32 slice:   each int32 -> 4 bytes LE (uint32 cast)
float32 slice: each float32 -> 4 bytes LE (Float32bits)
float64:       8 bytes LE (Float64bits)
int64:         8 bytes LE (uint64 cast)

Used for:
  neighbors BLOB: []int32 -> serializeInt32s
  vector BLOB:    []float32 -> serializeFloat32s
  meta values:    int64 -> serializeInt64, float32 centroid -> serializeFloat32s
```

## Key Public Types

```
╔═══════════════════════════════════════════════════════════════╗
║  type Config struct {                                         ║
║      MaxDegree, SearchListSize, BuildPasses int               ║
║      EfSearch, RerankTopN, CacheCapacity   int               ║
║      BruteForceThreshold                   int               ║
║      Alpha, DriftThreshold, InsertRatioThreshold float64     ║
║  }                                                            ║
╠═══════════════════════════════════════════════════════════════╣
║  type VectorIterator interface {                              ║
║      Next() (id []byte, vec []float32, ok bool)              ║
║      Reset() error                                            ║
║  }                                                            ║
╠═══════════════════════════════════════════════════════════════╣
║  type Result struct {                                         ║
║      ID    []byte   -- caller-provided external ID            ║
║      Score float64  -- L2 distance squared (lower = closer)   ║
║  }                                                            ║
╠═══════════════════════════════════════════════════════════════╣
║  type Index struct { ... }  -- opaque, thread-safe            ║
║    New(db, cfg) -> (*Index, error)                            ║
║    Build(ctx, iter) -> error                                  ║
║    Search(query, topK) -> ([]Result, error)                   ║
║    SearchWithRerank(query, topK, reranker) -> ([]Result, err) ║
║    Insert(vecs, ids) -> error                                 ║
║    NeedsRebuild() -> bool                                     ║
║    RebuildAsync(ctx, iter)                                    ║
║    Count() -> int                                             ║
║    Close() -> error                                           ║
╚═══════════════════════════════════════════════════════════════╝
```

## Key Internal Types

```
╔═══════════════════════════════════════════════════════════════╗
║  graphNode (vamana.go) -- used during build only (in-memory)  ║
║    id        int32                                            ║
║    extID     []byte                                           ║
║    vec       []float32                                        ║
║    neighbors []int32                                          ║
║    code      []byte     -- RaBitQ                             ║
║    sqNorm    float64                                          ║
║    l1Norm    float64                                          ║
╠═══════════════════════════════════════════════════════════════╣
║  cachedNode (cache.go) -- in LRU cache at runtime             ║
║    nodeID    int32                                            ║
║    neighbors []int32                                          ║
║    vec       []float32                                        ║
║    code      []byte                                           ║
║    sqNorm    float64                                          ║
║    l1Norm    float64                                          ║
║    prev/next *cachedNode                                      ║
╠═══════════════════════════════════════════════════════════════╣
║  searchCandidate (vamana.go)                                  ║
║    nodeID    int32                                            ║
║    dist      float64                                          ║
╠═══════════════════════════════════════════════════════════════╣
║  candidateHeap (vamana.go) -- min-heap on dist                ║
║    implements container/heap.Interface                         ║
╠═══════════════════════════════════════════════════════════════╣
║  Encoder (rabitq.go)                                          ║
║    dim      int                                               ║
║    centroid []float32                                          ║
╠═══════════════════════════════════════════════════════════════╣
║  CentroidTracker (centroid.go)                                ║
║    dim, current[], buildCentroid[]                             ║
║    count, buildCount, insertsSince                            ║
║    driftThreshold, insertRatio                                ║
╚═══════════════════════════════════════════════════════════════╝
```

## Vamana Graph Algorithm Detail

```
buildGraph(ctx, nodes, medoid, R, L, alpha, passes)
│
│  Phase 1: Random initialization
│    For each node without neighbors:
│      Fisher-Yates partial shuffle to pick R random neighbors
│      (excluding self)
│
│  Phase 2: Iterative refinement (repeat for each pass)
│    Shuffle node order (Fisher-Yates)
│    For each node p:
│      candidates = greedySearch(p.vec, medoid, L) + current neighbors
│      p.neighbors = robustPrune(p, candidates, alpha, R)
│      For each new neighbor q of p:
│        if p not in q.neighbors:
│          q.neighbors += p   (reverse edge)
│          if |q.neighbors| > 2R:
│            q.neighbors = robustPrune(q, q.neighbors, alpha, R)
│
│  Phase 3: Final pruning
│    For each node with degree > R:
│      robustPrune to exactly R neighbors

robustPrune(nodeID, candidates, alpha, R):
│  Remove self + duplicates
│  Sort by distance (ascending)
│  result = []
│  While candidates remain and |result| < R:
│    best = closest remaining candidate
│    result += best
│    Filter remaining: keep c only if alpha * dist(best, c) > dist(node, c)
│    (alpha-RNG rule: removes candidates "covered" by best)
│  Return result

greedySearch(query, start, L):
│  Min-heap initialized with start node
│  best = sorted list of top-L candidates
│  While heap not empty:
│    cur = pop min
│    if cur.dist > worst(best) and |best| >= L: break
│    For each neighbor of cur:
│      d = l2DistanceSquared(query, neighbor.vec)
│      if |best| < L or d < worst(best):
│        push to heap, insertSorted into best
│        trim best to L
│  Return best, visited
```

## Performance Characteristics

```
╔═══════════════════════════╤══════════════════════════════════════╗
║  Operation                │ Complexity / Latency                 ║
╠═══════════════════════════╪══════════════════════════════════════╣
║  bruteForceSearch         │ O(N * dim)          ~1ms @10K d128   ║
║  vamanaSearch (2-stage)   │ O(L * R * dim/8)    ~3.3ms @10K d128 ║
║  RaBitQ Encode            │ O(dim)              ~1.1us @d1024    ║
║  rabitqDistAsymPrecomp    │ O(dim)              ~1.08us @d1024   ║
║  l2DistanceSquared        │ O(dim), 8x unrolled ~482ns @d1024    ║
║  POPCOUNT (128 bytes)     │ O(dim/64)           ~48ns            ║
║  Build 10K                │ O(N * passes * L)   ~24s @d128       ║
║  Insert (single)          │ O(L * R)            amortized        ║
║  Cache lookup             │ O(1) map + list ops                  ║
╠═══════════════════════════╧══════════════════════════════════════╣
║  Recall@10 Results:                                              ║
║    Brute-force (<=50K):  100%                                    ║
║    Vamana+RaBitQ @10K:    98.2%                                  ║
║    Vamana+RaBitQ @5K:     99.6%                                  ║
║    Vamana+RaBitQ @1K:    100%                                    ║
║    After +50% inserts:   ~80% (triggers NeedsRebuild)            ║
║                                                                  ║
║  Crossover brute-force / Vamana: ~50K vectors                    ║
║  Memory: all cached nodes in RAM (100K * node_size)              ║
║  Build: ALL vectors loaded in RAM (45M*1024d = 180GB)            ║
╚══════════════════════════════════════════════════════════════════╝
```

## Dependencies

```
Direct:
  modernc.org/sqlite v1.46.1    -- Pure Go SQLite (CGO_ENABLED=0)

Indirect (via modernc.org/sqlite):
  github.com/dustin/go-humanize
  github.com/google/uuid
  github.com/mattn/go-isatty
  github.com/ncruces/go-strftime
  github.com/remyoudompheng/bigfft
  golang.org/x/exp
  golang.org/x/sys
  modernc.org/libc
  modernc.org/mathutil
  modernc.org/memory

Stdlib packages used:
  container/heap, context, database/sql, encoding/binary, fmt,
  math, math/bits, math/rand/v2, sync
```

## Consumers

```
╔═══════════════════════════════════════════════════════════════╗
║  siftrag (Front Office SaaS RAG)                              ║
║    imports horosvec as vector search engine per shard          ║
║    One Index per user-tenant SQLite shard                      ║
║                                                               ║
║  HORAG (Vectorisation Pipeline)                               ║
║    imports horosvec for indexing embeddings                    ║
║    buffer .md -> chunk -> embed -> horosvec.Build()            ║
╚═══════════════════════════════════════════════════════════════╝
```

## SearchWithRerank (External Reranker)

```
SearchWithRerank(query, topK, reranker func([][]byte)([][]float32, error))
│
│  1. Search(query, rerankN)  -- get wide candidate set
│  2. Collect candidate IDs
│  3. reranker(ids) -> exact vectors from external source
│  4. L2 rerank with external vectors
│  5. Return top-K
│
│  Fallback: if reranker is nil or errors, return Search results
│  Use case: cross-index reranking with full-precision vectors
│            stored outside horosvec
```

## Known Limitations

```
1. Build loads ALL vectors into RAM
   45M vectors * 1024 dims * 4 bytes = 180 GB
   Mitigation: sharded builds (siftrag/HORAG split by tenant)

2. Insert degradation: -20% recall after +50% inserts without rebuild
   Mitigation: NeedsRebuild() triggers at 30% insert ratio or 5% drift

3. No loadNodeLight: graph traversal loads full vector (8KB @dim1024)
   even when only RaBitQ code (128B) is needed for Stage 1
   Optimization target for >1M vector indices

4. No rotation in RaBitQ (centering + sign only)
   Spearman rho ~ 0.82-0.85 (vs ~0.95 with random rotation)
   Acceptable for current scale, Vamana graph compensates

5. Simplified RaBitQ correction: ||o'||^2/L1_o vs paper formula
   Gap: rho difference ~ 0.016 vs full paper formula
```

## Sub-schema Files

```
None -- this is a library-only project with no cmd/ binaries.
All code is in a single package (horosvec).
This file is the sole schema document.
```
