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
- Pattern "library-first" : horos47 importera horosvec comme une librairie

## Architecture

### Fichiers

| Fichier | Rôle |
|---------|------|
| `horosvec.go` | API publique : Index, Config, Result, VectorIterator |
| `rabitq.go` | Encoder RaBitQ, Encode(), rabitqDistanceAsym() avec correction, rabitqDistance() POPCOUNT |
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

### Décisions

- **Pas de rotation** pour RaBitQ (centrage + signe uniquement) — ~70% brute-force recall@10
- **Vecteurs bruts stockés** dans SQLite pour navigation exacte L2 du graphe (standard DiskANN)
- RaBitQ codes stockés pour pré-filtrage futur (IVF, scan rapide)
- RaBitQ distance corrigée avec facteur <ô_bar,ô> = L1/(sqrt(D)*L2) par vecteur
- Cache LRU obligatoire : vecteurs + graphe en mémoire (sans cache = ~100-300 queries SQLite par search)
- Warm cache au démarrage : medoid + BFS 2 hops
- Insert incrémental via greedy search + ajout reverse edges
- Rebuild async : build dans tables `_new`, swap atomique, cache clear
- Concurrency : sync.RWMutex (lectures parallèles, lock exclusif au swap)

## Résultats

- recall@10 = **99.2%** sur 10K vecteurs dim 128 (cible ≥ 85%)
- POPCOUNT 192 bytes = **47ns** (cible < 200ns)
- Search 10K vecteurs = **388μs** (cible < 1ms)
- Build 10K vecteurs dim 128 = ~24s
- Aucune data race
