# Audit horosvec — état 2026-07-13

Date : 2026-07-13. Périmètre : `/devhoros/horosvec` (module autonome). Mode : revue architecte (lecture au corps + gates + synthèse des audits 2026-07-10 + vérification des invariants). Aucun code modifié. Gates exécutés par l'auditeur.

## Protocole suivi (CLAUDE.md dossier)
- Lecture horosvec/CLAUDE.md (claims, invariants, gates).
- Grep CLAUDE:* sur *.go : aucun marqueur (toute la doctrine vit dans CLAUDE.md + doc.go + docs/).
- Grep ^func sur les sources cibles (horosvec.go, schema.go, hotplane.go, arena*.go, rabitq.go, binary_format.go, vamana.go, ...).
- Lecture par tranches (offset/limit) des fichiers pivots ; jamais lecture pleine en première passe.
- Exécution des gates stricts du module : `env GOWORK=off CGO_ENABLED=0 go build ./... && go vet ./... && gofmt -l . && go test -count=1 -timeout=90s ./...`

## Gates (exécutés au sol)
- BUILD : OK
- VET : OK
- Gofmt : propre (aucun fichier listé)
- TEST : `ok  github.com/hazyhaar/horosvec  86.818s` (exit 0). Suite complète incluant hardening, fixes oracle, arena roundtrips, hotplane, budget flatVecs, recall planchers, parité rerank, insert commit, rebuilds, cancellation, medoid, etc.

Tous verts sous la contrainte du module (hors workspace, CGO off).

## Ce que horosvec EST (vérifié)
Bibliothèque Go pure embarquée d'index ANN (Vamana + RaBitQ 1-bit + rotation Hadamard randomisée persistée). 

CLAIM:: horosvec est une lib en-process, aucun daemon, aucun port, aucun import horos55/jurhoros.
CLAIM-NEG:: horosvec n'est ni siftrag ni composant du bus ; il est consommé par import (github.com/hazyhaar/horosvec).

Stockage : une SQLite (tables vindex_nodes + vindex_meta) + optionnellement arène fp16 sidecar mmap (HVARENA1). 

Frontières : dépendance unique modernc.org/sqlite ; tout le reste stdlib. Aucune connaissance du bus, rules.db, idgen, etc.

## Carte fonctionnelle principale (ancrée au corps)
- New(db, cfg) : configure PRAGMA, initSchema, loadIndex (avec réconciliation node_count vs COUNT(*)), loadRotationMeta, getMaxNodeID+1 (A2 fail-loud), checkAndRefreshMedoid, warmCache, loadFlat (avec FlatVecsMaxBytes guard), rebuildPlaneLocked, open/validate arena (dim/count match).
- Build(ctx, iter) : deux chemins — db-blob (matérialise + encode + parbuild) ou streamingArena (pass1 : stream fp16, pass2 : buildFromOpenArena sans buffer fp32 complet).
- Insert(ctx, vecs, ids) : Lock, refuse arena, tx, saveNode initial, rabitqGreedySearchInternal sur overlay (visibilité intra-batch), setNeighbors + robustPrune arêtes inverses, node_count dans tx, commit puis effets mémoire + extendPlaneAfterInsert.
- Search(ctx, q, k) : RLock ; si <= BruteForceThreshold → brute (flat/arena/SQL) ; sinon vamana (rabitqGreedy + rerank L2 exact via arena ou flat ou loadNodeReadOnly).
- SearchWithRerank : Search puis callback reranker (dégradation gracieuse sur erreur reranker documentée).
- ExportBinary / ImportBinary : format HVEC v2 durci (magic, versions, bornes maxPreallocBlob 64MiB / nodes 1M, longueurs préfixées validées).
- ImportAdjacency / ExportAdjacency / BuildFromArena : apport de graphe externe (CAGRA etc.).
- Observabilité (atomics, jamais nil) : RerankSQLLoads, DegradedNeighborLoads, PlaneDegraded, MalformedVectorSkips, FlatVecsBudgetSkips.
- NeedsRebuild / RebuildAsync / recomputeMedoid : refonte et médoïde délégués à l'appelant.
- Close : rebuildMu + mu, close arena, clear cache.

## Invariants durs vérifiés au corps
- Transactionnalité Insert : effets mémoire (cache, flat, nextID, centroid, plane) seulement post-commit (horosvec.go ~1284). node_count écrit dans la tx.
- Contexte : partout sur chemins chauds ; cancellation → erreur wrappée, jamais résultat vide silencieux. Brute-force sonde ~tous les 4096.
- Fail-loud sur corruption/limites : médoïde illisible = erreur ; offset int32 plane = checkInt32Offset (max ~33M nœuds degré 64) ; dims hétérogènes = erreur ; arena count/dim mismatch = erreur.
- Arena vs incrémental : Insert refuse arena (fail-loud) ; rebuild aussi.
- Rotation seed persistée et rejouée.
- Plane chaud pointer-free : tout le graphe (pas delta) ; buildHotPlane full SELECT ORDER BY node_id.
- Repli gracieux : degraded counts, flat budget skip → SQL/cache, reranker fail → topK approx sans erreur.
- Plateformes : arena mmap unix seulement (stub sur autres) ; DB-blob partout.

## Qualité & tests (observés)
- Couverture dense des angles critiques : commit failure injection, corrupt blobs, OOM guards, int32 offset, medoid staleness, plane degradation, cancellation mid-search/greedy, arena parity, flatVecs budget à New/Build/Insert, recall planchers déterministes, hotplane recall invariance, parité rerank flat/arena/SQL.
- Pas de TODO/FIXME/HACK dans *.go.
- Un panic défensif seulement (rotate dst too small).
- Récupération : recover présent dans RebuildAsync (pour goroutine).
- Pas d'import uuid/google direct (seulement transitif via sqlite).
- Pas de state global mutable ; tout derrière *Index.

## Frontières & écosystème
- Aucun import horos55/jurhoros/siftrag dans le code horosvec (vérifié).
- Consommateurs dans horos55 (go.mod v0.2.1, usage réel) : internal/silo_vec_indexer/builder.go (New + Build sur :memory:), runbooks, silo_retrieval, codemap embedder, similar_cmd.
- horosvec n'a aucune dépendance inverse sur le workspace.
- Module autonome : gates toujours GOWORK=off.

## Dette / observations résiduelles (pas de blockers)
- Pas de compaction/réindexation automatique : la greffe incrémentale (robustPrune local) peut dériver ; l'appelant interroge NeedsRebuild (centroid drift ou ratio inserts). OBSERVÉ dans horosvec.go et hotplane.
- Pas d'API Delete : rebuild seulement.
- Requêtes meta/load sans context (loadMeta, loadIndex, getMaxNodeID, getNodeCount, certains dans medoid.go) : n'apparaissent que sur chemins New / Count / recompute (New n'a pas de ctx dans sa signature). Les chemins runtime chauds utilisent Context.
- Arena : figée post-build ; pas de "growing segmented" dans ce dépôt (prototype externe dans horosvec-bench).
- Count() sous RLock peut rendre un compte légèrement stale si Insert concurrent (commentaire au corps).
- 1-bit RaBitQ rend le rerank obligatoire pour les planchers publiés ; Extended RaBitQ (4-5+ bits) est dans la veille externe, pas implémenté ici.

## Synthèse des audits antérieurs (2026-07-10) recoupés
- cdc-reverse insertion : chemin greffe Vamana incrémental confirmé ; pas de "tampon chaud" séparé ; plan = miroir du graphe entier.
- Architecture disruptif + fix rerank/flatVecs/garde RAM : flatVecs désormais utilisé pour rerank db-blob à toute échelle (sans verrou), garde budget avant load/append, parité + perf mesurée (db-blob post-fix dépasse légèrement l'arène à conc 32). Le rerank est un artefact du choix 1-bit.
- Hotpath/dbblob, medoid, plane, arena import : garde int32, réconciliation méta, dégradations comptées, validations arena.

Aucun des angles critiques listés dans les audits du 10 n'est revenu rouge dans la passe du 13.

## Verdict
Le module est sain, rigoureusement testé, durci (bornes, fail-loud, transactionalité, observabilité), et en production à 26,7 M vecteurs (HNbook). Les invariants du CLAUDE.md tiennent au corps. Les frontières sont étanches. Les gates sont verts.

Dettes nommées (non bloquantes) : délégation du rebuild à l'appelant, absence de delete, limitation plateforme de l'arène, queries meta sans ctx sur chemins d'init.

Aucune vulnérabilité, course, ou trou de couverture majeur détecté dans le périmètre audité. Prêt pour usage comme dépendance.

## Angles couverts dans cette passe
- Machine à états load/persist/méta/restart (récupération nextID, node_count réconcilié, medoid refresh, plane rebuild).
- Chemin chaud (plane, arena/flat/sql rerank, verrous, ctx).
- Hardening & corruption (binary, arena, budget, degraded counts).
- Écosystème consommateur (imports réels dans horos55).
- Qualité (gates + tests au sol).
- Pas de recodage / duplication interne.
- Conformité à la doctrine du module (pure Go, interface côté consommateur, etc.).

(Grille d'audit 2026-07-09 appliquée ; les angles non couverts ici — ex. perf sous très haute échelle live ou Extended RaBitQ — sont hors périmètre ou dans la veille externe.)