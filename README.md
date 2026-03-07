# horosvec

Bibliothèque Go standalone de recherche vectorielle ANN (Approximate Nearest Neighbor).

- **Vamana** (DiskANN) — graphe de navigation plat
- **RaBitQ** — compression 1-bit des vecteurs
- **SQLite** — stockage persistant (graphe, vecteurs bruts, codes compressés)

## Prérequis

- Go 1.24+

## Build

```bash
CGO_ENABLED=0 go build ./...
```

## Test

```bash
go test -race -count=1 ./...
go test -bench=. -benchmem ./...
```

## Dépendances

- `modernc.org/sqlite` — seule dépendance externe

## Performance

| Mode | Scale | Recall@10 | Latence |
|------|-------|-----------|---------|
| Brute-force (≤50K) | 10K | 100% | 1 ms |
| Vamana+RaBitQ (>50K) | 10K | 98.2% | 3.3 ms |

Seuil dynamique à 50K vecteurs : brute-force en dessous, Vamana+RaBitQ au-dessus.

## Usage

```go
idx, _ := horosvec.Open(db, horosvec.DefaultConfig())
results, _ := idx.Search(queryVec, topK)
```

Pattern library-first : importé par siftrag et HORAG comme bibliothèque.
