# Fix — rerank db-blob via flatVecs sans verrou (Job 019f4b7e)

Object : proj_horosvec_incremental. Module autonome github.com/hazyhaar/horosvec
(hors go.work). Gates : `env GOWORK=off CGO_ENABLED=0`.

## Contrat de réutilisation

- **Réutilisé tel quel** : le miroir plat fp32 `flatVecs []float32` + `flatIDs [][]byte`
  (horosvec.go:219-221), déjà entretenu par l'Insert après commit (horosvec.go:1292-1295,
  `append(idx.flatVecs, pendingFlatVecs...)`). Aucun second miroir, aucune mécanique de
  synchronisation neuve.
- **Correspondance node_id → offset établie au corps** : `loadFlatVectors` (horosvec.go:376)
  remplit `flatVecs` par `SELECT ext_id, vector FROM vindex_nodes ORDER BY node_id`, donc
  dense et ordonné par node_id. L'Insert assigne `nodeID := next; next++` (dense 0..n-1) et
  appende dans le même ordre. Aucune méthode de suppression/purge n'existe (pas de trous).
  L'offset du candidat est donc `int(nodeID) * idx.dim`, borné par
  `off >= 0 && off+idx.dim <= len(idx.flatVecs)` (fail-soft → repli SQL, jamais de panic).
- **Surface neuve** : une branche de rerank (horosvec.go:~850) et le test
  `rerank_flatvecs_test.go`. Assertion d'un test existant corrigée (arena_test.go:288).
- **Compteurs réutilisés** : `RerankSQLLoads()` comme oracle du non-recours au SQL.

## Critères

### C0 — Golden STRATE 0 (avant édition)
`env GOWORK=off CGO_ENABLED=0 go build ./...` → OK. `go test ./...` →
`ok github.com/hazyhaar/horosvec 43.514s`. **VERT.**

### C2 — Chargement flatVecs à toute échelle en db-blob
Condition horosvec.go:348 relâchée de `nodeCount <= cfg.BruteForceThreshold && cfg.ArenaPath == ""`
à `cfg.ArenaPath == ""`. En mode arène (ArenaPath posé) flatVecs reste NON chargé (SQLite
vector-less → flat vide → panic évitée). **VERT.**

### C3 — Rerank câblé sans verrou
Branche ajoutée après la branche arène (inchangée, prioritaire) : quand `idx.flatVecs != nil`,
lecture directe `idx.flatVecs[off:off+idx.dim]` puis `l2DistanceSquared`, sans mu.RLock, sans
cache LRU, sans SQL. Ordre : arène → flatVecs → loadNodeReadOnly. Repli SQL seulement si
flatVecs nil ou offset hors bornes. **VERT.**

### C4 — Parité de rappel (test au sol)
`rerank_flatvecs_test.go` : index Vamana (BruteForceThreshold=0), 1200 vecteurs dim 64, 50
requêtes. Passe 1 via flatVecs, passe 2 avec `idx.flatVecs = nil` forçant le repli SQL.
Comparaison stricte ext_id + ordre + score (`fr[i].Score != sqlRes[i].Score`). Delta
RerankSQLLoads = 0 en passe flatVecs, > 0 en passe SQL (repli réellement exercé).
`go test -run TestRerankFlatVecsParity` → PASS (0.70s). **VERT.**

### C5 — Banc end-to-end (MESURÉ, base1m.fvecs n=1M dim=512, k=10, ef=64, appareillé au
golden 2026-07-09, même protocole)

| concurrence | golden db-blob 2026-07-09 (rerank SQL, pré-fix) | db-blob POST-fix (flatVecs) | golden arène e2e 2026-07-09 |
|---|---|---|---|
| 8  | 16 453 qps | **17 062 qps** | 14 790 qps |
| 16 | (non collecté) | **28 775 qps** | (non collecté) |
| 32 | 38 240 qps | **40 619 qps** | 39 599 qps |

recall_mean post-fix = 0,951 (ef=64). Chiffres collés depuis
`scratchpad/bench_dbblob64.out` (build_s=1252,6).

**Lecture honnête.** À conc 32, le fix porte le db-blob de 38 240 à **40 619 qps** (+6,2 %),
et le db-blob **dépasse désormais l'arène e2e** (39 599). Le gain e2e est réel mais modeste,
et non le facteur 9× du micro-banc isolé cité au brief : ce 9× (source plate contiguë ~77k)
mesurait la boucle de rerank EN ISOLATION ; dans la recherche complète, le coût est dominé
par la marche Vamana + RaBitQ et le tri, non par le verrou du rerank. Le fix supprime bien la
contention mesurée (verrou franchi 128×/recherche, tas dispersé) — attesté par
RerankSQLLoads=0 — mais son effet relatif au niveau bout-en-bout est borné par les autres
étages. Le plafond e2e (~40k @ conc 32 sur ce dataset) est désormais atteint par le db-blob
sans verrou, à parité/au-dessus de l'arène. Mesure additionnelle à ef=128 (recall 0,978) :
conc8=9 185, conc16=16 343, conc32=23 828 qps (`scratchpad/bench_dbblob.out`).

### C6 — Gates vs golden
`env GOWORK=off CGO_ENABLED=0 go build ./...` OK. `go test ./...` →
`ok github.com/hazyhaar/horosvec 44.350s` (0 nouvel échec vs golden). `gofmt -l .` vide.
`go vet ./...` propre. **VERT.**
Oracle `RerankSQLLoads()==0` sur le chemin flatVecs prouvé par C4. **VERT.**

## NEGCRITERIA
- N0 : 0 fichier hors /devhoros/horosvec (+ ce ledger). Banc horosvec-bench lancé, jamais édité.
- N1 : 0 dépendance ajoutée. N2 : mode arène et contrat transactionnel de l'Insert inchangés.
- N3 : flatVecs reste fp32. N4 : 0 test affaibli (l'assertion arena_test.go:288 a été
  CORRIGÉE, pas affaiblie : l'ancien oracle « db-blob → SQL » contredit désormais le contrat ;
  le nouvel oracle est plus fort — flatVecs chargé ET RerankSQLLoads==0). N5 : commits locaux.

## Auto-audit adversarial (subagent auditeur, dedup-redundancy + secu-deep)
6 lentilles VERTES, 0 finding (hard ni soft). Parité non tautologique confirmée (passe SQL
réellement exercée par compteur), correspondance node_id→offset correcte et bornée, densité
garantie (aucune méthode Delete), lecture sans verrou, arène intacte, RerankSQLLoads=0.
