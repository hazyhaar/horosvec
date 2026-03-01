> **Schema technique** : voir [`horosvec_schem.md`](horosvec_schem.md) — lecture prioritaire avant tout code source.

# CLAUDE.md — horosvec

> **Règle n°1** — Un bug trouvé en audit mais pas par un test est d'abord une faille de test. Écrire le test rouge, puis fixer. Pas de fix sans test.

## Description

Bibliothèque Go standalone de recherche vectorielle ANN (Approximate Nearest Neighbor) utilisant :
- **Vamana** (DiskANN) : graphe de navigation plat pour la recherche
- **RaBitQ** : compression 1-bit des vecteurs (paper formula avec correction <ô_bar,ô>)
- **SQLite** : stockage persistant du graphe, vecteurs bruts et codes compressés

## Build & Test

```bash
cd /data/HOROS\ SYSTEM\ DEV\ AREA/horosvec
CGO_ENABLED=0 go build ./...
go test -v -count=1 ./...
go test -bench=. -benchmem ./...
go test -race ./...
```

## Principes

- Pure Go, `CGO_ENABLED=0`, aucune dépendance C
- SQLite via `modernc.org/sqlite` uniquement
- Seule dépendance externe : modernc.org/sqlite — tout le reste = stdlib
- Pattern "library-first" : siftrag et HORAG importent horosvec comme une librairie
- **HORAG auto-index** (mars 2026) : HORAG intègre un `indexmgr` qui appelle `Build()`, `Insert()`, `NeedsRebuild()`, `RebuildAsync()` automatiquement après chaque insertion dans `rag_vectors`. Le CLI `build-indexes` reste pour rattrapage/maintenance uniquement.

## Architecture

### Fichiers

| Fichier | Rôle |
|---------|------|
| `horosvec.go` | API publique : Index, Config, Result, VectorIterator, Search (2-stage + brute-force) |
| `rabitq.go` | Encoder RaBitQ, Encode(), rabitqDistanceAsym/Precomp(), rabitqDistance() POPCOUNT |
| `vamana.go` | Graphe Vamana : buildGraph, greedySearch, robustPrune, insertNode |
| `cache.go` | LRU cache pour les nœuds + vecteurs bruts (map + doubly-linked list) |
| `schema.go` | SQLite DDL + persistence (save/load/swap) |
| `serial.go` | Sérialisation binaire int32/float32 ↔ []byte |
| `centroid.go` | CentroidTracker : running average + drift detection |

### Schema SQLite

```sql
CREATE TABLE vec_nodes (
    node_id   INTEGER PRIMARY KEY,
    ext_id    BLOB NOT NULL UNIQUE,
    neighbors BLOB NOT NULL,
    vector    BLOB NOT NULL,
    rabitq    BLOB NOT NULL,
    sq_norm   REAL NOT NULL,
    l1_norm   REAL NOT NULL DEFAULT 0
);
CREATE TABLE vec_meta (key TEXT PRIMARY KEY, value BLOB NOT NULL);
```

### Pipeline de recherche (2-stage + brute-force dynamique)

```
query
  │
  ├─ count ≤ BruteForceThreshold (50K) ──→ bruteForceSearch()
  │     SELECT ext_id, vector FROM vec_nodes  →  L2 exact  →  100% recall
  │
  └─ count > BruteForceThreshold ──→ vamanaSearch() (2-stage)
        Stage 1: RaBitQ beam search (rabitqDistanceAsymPrecomp, ~1µs/paire)
            → query centré une fois, réutilisé pour tous les hops
            → traverse le graphe Vamana via distances approximatives
        Stage 2: L2 rerank (l2DistanceSquared, ~480ns/paire)
            → re-rank top RerankTopN candidats avec vecteurs float32 exacts
            → tri final, retourne topK
```

### Config

| Paramètre | Défaut | Rôle |
|-----------|--------|------|
| `MaxDegree` | 64 | Arêtes max par noeud (R) |
| `SearchListSize` | 128 | Beam width au build (L) |
| `BuildPasses` | 2 | Passes de construction du graphe |
| `EfSearch` | 128 | Beam width au search |
| `RerankTopN` | 500 | Candidats re-rankés en L2 exact (auto-expand: max(RerankTopN, topK×3)) |
| `BruteForceThreshold` | 50000 | Sous ce seuil → brute-force O(N), 100% recall |
| `CacheCapacity` | 100000 | Taille LRU (noeuds) |
| `Alpha` | 1.2 | α-RNG pruning (>1 = arêtes longues) |
| `DriftThreshold` | 0.05 | Ratio drift centroïde pour trigger rebuild |
| `InsertRatioThreshold` | 0.30 | Ratio inserts/build pour trigger rebuild |

### Décisions

- **Pas de rotation** pour RaBitQ (centrage + signe uniquement) — Spearman ρ ≈ 0.82-0.85
- **RaBitQ actif au search** : Stage 1 traverse le graphe via `rabitqDistanceAsymPrecomp()`, pré-calcul query centering
- **Vecteurs bruts stockés** dans SQLite pour L2 exact en Stage 2 (re-rank) et brute-force
- RaBitQ distance corrigée avec facteur simplifié `||o'||²/L1_o` (gap ρ=0.016 vs paper formula)
- **Seuil dynamique** : brute-force pour petits shards (<50K), Vamana+RaBitQ pour grands
- Cache LRU obligatoire : vecteurs + graphe en mémoire (sans cache = ~100-300 queries SQLite par search)
- Warm cache au démarrage : medoid + BFS 2 hops
- Insert incrémental via greedy search + ajout reverse edges
- Rebuild async : build dans tables `_new`, swap atomique, cache clear
- Concurrency : sync.RWMutex (lectures parallèles, lock exclusif au swap)

## Résultats (i9-14900K, dim 128)

### Recall

| Mode | Scale | Recall@10 |
|------|-------|-----------|
| Brute-force (≤50K) | 10K | **100%** |
| Vamana+RaBitQ (>50K) | 10K | **98.2%** |
| Vamana+RaBitQ | 5K | 99.6% |
| Vamana+RaBitQ | 1K | 100% |

### Latence

| Opération | Latence |
|-----------|---------|
| Brute-force 10K | 1 ms |
| Vamana+RaBitQ 10K | 3.3 ms |
| RaBitQ Encode (1024d) | 1.1 µs |
| RaBitQ AsymDist (1024d) | 1.08 µs |
| L2 exact (1024d) | 482 ns |
| POPCOUNT 128 bytes | 48 ns |
| Build 10K | ~24s |

### Crossover brute-force ↔ Vamana

~50K vecteurs. En dessous : brute-force plus rapide ET 100% recall. Au dessus : Vamana+RaBitQ nécessaire.

## Limites connues

- **Build en mémoire** : charge tous les vecteurs float32. 45M × 1024d = 180 GB → build shardé obligatoire
- **Insert degradation** : -20% recall après +50% insertions sans rebuild. `NeedsRebuild()` trigger à 30%
- **Pas de loadNodeLight** : traversée charge le vecteur complet (8KB @dim1024) même si seul le code RaBitQ (128B) est utilisé. Optimisation future pour >1M vecteurs
- Aucune data race (vérifié `go test -race`)
