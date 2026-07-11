# Audit — goulot du chemin chaud db-blob (rerank SQLite)

Date : 2026-07-10. Mode : quality-review + perf (rétro-ingénierie de goulot).
Périmètre : `/devhoros/horosvec`, chemin de recherche db-blob (rerank lisant les
vecteurs depuis `vindex_nodes`). Lecture seule stricte : aucun code moteur modifié ;
seuls ce rapport et une sonde jetable (`horosvec-bench/cmd/proto-hotpath`) ont été
écrits. Discipline : la cause racine est MESURÉE (expérience contrôlée + profil CPU),
jamais déduite par élimination.

## Verdict en une ligne

Le facteur ~15 ne vit PAS dans le rerank SQL. Les quatre hypothèses du brief (requête
re-préparée, scan complet, verrou de tampon, décodage coûteux) sont **RÉFUTÉES** par
la mesure. Le plafond db-blob (~35-40 kQPS) est un phénomène de **concurrence** : la
source des vecteurs de rerank en db-blob — le cache LRU sous RWMutex partagé, vecteurs
fp32 dispersés dans le tas — sature une ressource partagée sous 32 clients, là où
l'arène (fp16 contigu en mmap, sans verrou) monte presque linéairement. `modernc` et
les allers-retours SQL sont **innocentés au sol**.

## Ce que fait réellement le code (OBSERVÉ)

Chemin db-blob à l'échelle vamana (`horosvec.go:792` `vamanaSearch`) :

1. Marche greedy sur les codes RaBitQ (`rabitqGreedySearchInternal`). En db-blob comme
   en arène, elle lit codes/normes/voisins depuis le **plan chaud** en RAM
   (`greedyNodeCodeNorms` hotplane.go:200-203, `greedyEachNeighbor`) — tableaux
   contigus, **sans verrou**, `loadFn == nil`. Le plan couvre tous les nœuds dans les
   deux modes ; la marche est donc rigoureusement identique arène vs db-blob.
2. Rerank exact L2 des candidats (`horosvec.go:829-854`). Le nombre de candidats est
   `rerankN = min(RerankTopN=500, efSearch)`, soit **128** avec les défauts
   (EfSearch=128). MESURÉ : `sqlPerSearch = 128.00` exactement.
   - Mode arène (`:835`) : `idx.arena.vecInto(nodeID, buf)` — lecture mmap fp16,
     **sans verrou, sans SQL**.
   - Mode db-blob (`:839`) : `loadNodeReadOnly` → `nodeCache.getReadOnly`
     (`cache.go:56`, **`c.mu.RLock()` sur un RWMutex partagé**) ; sur miss →
     `loadNode` (`schema.go:74`, un `QueryRowContext` **par candidat**, 6 colonnes,
     puis `deserializeFloat32s` alloue un vecteur fp32 dispersé dans le tas + `put`
     sous `Lock`).

Le compteur `rerankSQLLoads` (horosvec.go:840) s'incrémente à CHAQUE candidat, hit
cache compris : il compte les chargements de rerank (128/recherche), pas seulement les
requêtes SQL. Le plan chaud N'héberge PAS les vecteurs bruts (codes 1 bit/dim = 16 o à
dim 128 ; vecteur fp32 = 512 o) : par conception, la source des vecteurs de rerank est
l'arène (fp16 mmap, hors tas) ou, à défaut, SQLite via le cache. En db-blob à grande
échelle (> `BruteForceThreshold=50000`, `ArenaPath` vide), le miroir `flatVecs`
lockless N'est PAS chargé (`horosvec.go:348`) et le rerank vamana ne le consulte pas :
il n'existe **aucune source de vecteurs contiguë et sans verrou**.

## Expérience contrôlée (OBSERVÉ, décidable)

Sonde `horosvec-bench/cmd/proto-hotpath` (jetable), même corpus (60 000 vecteurs
aléatoires dim 128, > seuil brute-force → chemin vamana), mêmes 500 requêtes, K=10,
EfSearch=128, cache chaud (warm-up 2000), fenêtre 4 s. On fait varier UNIQUEMENT la
source de rerank et la concurrence :

```
[arena            ] conc=1   qps=4654
[arena            ] conc=32  qps=76732     -> scaling x16.5
[dbblob-allcache  ] conc=1   qps=4934      (cache >= corpus : ZÉRO SQL au rerank)
[dbblob-allcache  ] conc=32  qps=35470     -> scaling x7.2
[dbblob-halfcache ] conc=1   qps=5000      (cache = 50% : SQL réel sur ~moitié)
[dbblob-halfcache ] conc=32  qps=35367     -> scaling x7.1
```

Trois faits décidables :

1. **À conc 1, les trois régimes sont équivalents** (~4650-5000 qps ; db-blob est même
   marginalement plus rapide). Le coût par recherche mono-thread est identique : le
   rerank SQL/cache ne coûte rien en séquentiel (page cache chaud). → Hypothèses 1, 2,
   4 du brief (re-préparation, scan, décodage) RÉFUTÉES : elles frapperaient aussi
   conc 1.
2. **cache-plein == demi-cache** (35 470 ≈ 35 367). Forcer des allers-retours SQL réels
   sur la moitié des candidats **ne change pas le débit**. → Le SQL n'est PAS le
   goulot. Le plafond existe même **sans aucun SQL** (cache-plein). Le diagnostic
   « withLock 21,6 % » de la campagne antérieure décrivait le chemin SQL, qui n'est PAS
   le plafond réel.
3. **La divergence est purement concurrentielle** : arène ×16.5, db-blob ×7.2 pour ×32
   cœurs. La SEULE différence de code entre arène-conc32 (77 k) et db-blob-cache-plein-
   conc32 (35 k) est le site de rerank (`arena.vecInto` lockless vs `getReadOnly` sous
   RWMutex + vecteur fp32 dispersé).

## Profil CPU — la signature (OBSERVÉ)

`pprof` CPU, conc 32, même corpus, `greedyNodeCodeNorms` (marche greedy, code
**identique** dans les deux modes, sans verrou cache) :

| Fonction (flat) | arène | db-blob |
|---|---|---|
| `greedyNodeCodeNorms` | 10.91 s (9.1 %) | **68.76 s (57.6 %)** |
| `rabitqDistanceLUT` | 37.75 s (31.6 %) | 17.12 s (14.4 %) |
| `arena.vecInto`+`float16ToFloat32` | 18.65 s | — |
| `loadNodeReadOnly` (cum) | — | 1.64 s (1.4 %) |

La même fonction de marche greedy, faisant le même travail, coûte **6,3× plus** en
db-blob. Elle ne prend aucun verrou : elle lit les tableaux du plan (partagés,
lecture seule, non invalidés entre cœurs). Son gonflement sous concurrence est la
**signature de stalls mémoire par saturation d'une ressource partagée** attribués à
la fonction qui retire l'instruction au moment où le stall se résout. Le rerank
lui-même reste à ~2 % du CPU : ce n'est pas le rerank qui BRÛLE du CPU, c'est sa
source de vecteurs qui, sous 32 clients, sature la ressource partagée et affame la
marche greedy.

## Cause racine (le mécanisme, MESURÉ)

Le rerank db-blob lit 128 vecteurs **fp32 (512 o) dispersés dans le tas** (allocations
`cachedNode.vec` distinctes) à travers un **RWMutex de cache partagé** (`getReadOnly`,
128 `RLock` par recherche sur un seul mutex). Sous 32 clients :

- l'atomique `readerCount` du RWMutex fait rebondir sa ligne de cache entre les 32
  cœurs (128 × 32 × 35 k ≈ plusieurs M d'opérations/s sur une seule ligne) ;
- les lectures de vecteurs fp32 dispersés consomment ~2× la bande passante mémoire
  des lectures fp16 contiguës de l'arène.

Les deux effets sont des phénomènes de concurrence (absents à conc 1, dominants à
conc 32) et partagent une **même remède** : donner au rerank db-blob une source de
vecteurs **contiguë et sans verrou** en RAM. L'arène EST précisément cette source
(fp16, mmap, hors tas, sans verrou) — c'est pourquoi elle monte à ×16. La distinction
fine entre part « atomique du RWMutex » et part « bande passante fp32 dispersé »
n'a pas été isolée par un profil de blocage dédié ; elle n'affecte pas le remède, les
deux tombant avec la même correction.

## Gain récupérable (estimation chiffrée)

L'infrastructure du remède existe déjà à 90 % : le plan chaud (`hotPlane`) est un
tableau contigu, lockless, étendu de façon incrémentale à l'insert (`appendNode`), et
le miroir `flatVecs` est déjà un tableau fp32 contigu lockless — mais chargé seulement
pour les petits index et **jamais consulté par le rerank vamana**. Router le rerank
vamana à travers une source contiguë sans verrou (charger `flatVecs` en db-blob grande
échelle, ou ajouter un plan de vecteurs fp16 en RAM) supprimerait le RWMutex et la
dispersion du chemin chaud.

Estimation : le débit db-blob passerait de ~35 k à la **parité arène ~72-77 kQPS**
sous forte concurrence (**≈ ×2 à conc 32**), tout en préservant l'incrémentalité
SQLite (le plan s'étend déjà à l'insert). Coût : détenir les vecteurs bruts en RAM —
exactement ce que l'arène fp16/mmap est conçue pour éviter à l'échelle 26,7 M
(~54 Go > RAM).

## Réponse tranchée — le mode SQLite est-il réparable vers la vitesse de l'arène ?

**PEUT-ÊTRE-AVEC-QUOI.** Réparable à la parité arène **uniquement là où les vecteurs
bruts tiennent en RAM** : il suffit de câbler une source de vecteurs contiguë et sans
verrou dans le rerank vamana (miroir `flatVecs` déjà présent, ou plan fp16 en RAM). Ce
levier **rend l'arène mmap et le sharding superflus au régime petit/moyen** (< quelques
millions de vecteurs), en gardant l'écriture incrémentale de SQLite.

En revanche, au régime réel 26,7 M (~54 Go > RAM), tenir les vecteurs bruts en RAM est
impossible : l'arène fp16 en mmap (moitié de taille, pagination OS) reste nécessaire,
et le sharding garde sa pertinence pour la mise à l'échelle en écriture et le régime
> RAM — pas pour lever CE plafond de concurrence, qui n'est ni SQL ni modernc mais la
contention de la source de vecteurs du rerank. Le postulat « le SQLite direct est
plafonné par modernc » est donc réfuté ; le postulat « il faut sharder/arène » n'est
vrai qu'au-delà de la RAM.

## Artefacts

- Sonde jetable : `/devhoros/horosvec-bench/cmd/proto-hotpath/main.go` (à supprimer
  après audit ; non promue en bloc de banc).
- Profils : `.../scratchpad/cpu_arena.prof`, `.../scratchpad/cpu_dbblob.prof`.
- Audit de contexte antérieur : `/devhoros/horosvec-bench/audits/2026-07-10_bench_arene_vs_sqlite.md`.

## RETEX cognitif

J'ai d'abord suivi les quatre hypothèses du brief, toutes centrées sur le rerank SQL,
et j'ai failli conclure « contention du RWMutex du cache » par simple lecture du corps
— exactement le piège que la loi cardinale interdit. Ce qui m'a rattrapé : refuser de
conclure sans mesurer, et surtout concevoir l'expérience de contrôle qui varie UNE
seule chose. Le tournant décisif fut la variante cache-plein (zéro SQL) donnant le même
plafond que le demi-cache : elle a innocenté le SQL d'un coup, sans profil. Le profil,
ensuite, m'a déstabilisé (57 % dans une fonction de marche greedy censée être
identique aux deux modes) avant que je comprenne qu'un stall mémoire s'attribue à la
fonction qui retire, pas à sa cause. Le compteur `RerankSQLLoads` mal nommé (il compte
tous les chargements, pas les SQL) m'a un instant fait lire `128.00` comme « 128 SQL/
recherche » — vérifié au corps, c'était 128 chargements dont la plupart en cache.
Pattern réutilisable : face à un plafond de concurrence, ne pas profiler d'abord —
construire l'expérience qui NEUTRALISE le suspect (ici le SQL) ; si le plafond tient
sans lui, il est innocent, quel que soit ce que dit le profil.

## flatVecs fp32 contigu : parité ou pas (mesure décisive)

Point manquant tranché : le rerank servi depuis une source CONTIGUË SANS VERROU en
fp32 (ce que donnerait le miroir `flatVecs` actuel — `[]float32` plat indexé par
node_id, aucun mutex, pleine précision) atteint-il la parité arène, ou plafonne-t-il
sous elle à cause du double volume fp32 ?

Le moteur n'expose pas d'injection de source dans `vamanaSearch` (rerank interne, non
modifiable). La marche greedy étant IDENTIQUE aux trois régimes (elle lit le plan
chaud, sans verrou), l'écart entre les trois points est ENTIÈREMENT porté par la boucle
de rerank. Sonde `horosvec-bench/cmd/proto-rerank` : boucle de rerank ISOLÉE, 128
lectures de vecteur candidat (node_id aléatoires) + distance L2 par « recherche », même
jeu de 60 000 vecteurs sous trois représentations, conc 8/16/32. L'isolement donne la
BORNE HAUTE de l'écart (dans le pipeline complet, la marche greedy commune — ~70 % du
CPU au profil — le diluerait fortement).

### Sortie collée (rerank-ops/s)

```
[1-cache-verrou-fp32   ] conc=8   rerank_ops/s=125937
[2-contigu-fp32        ] conc=8   rerank_ops/s=605312
[3-contigu-fp16        ] conc=8   rerank_ops/s=214184

[1-cache-verrou-fp32   ] conc=16  rerank_ops/s=116262
[2-contigu-fp32        ] conc=16  rerank_ops/s=1031261
[3-contigu-fp16        ] conc=16  rerank_ops/s=336172

[1-cache-verrou-fp32   ] conc=32  rerank_ops/s=163804
[2-contigu-fp32        ] conc=32  rerank_ops/s=1467492
[3-contigu-fp16        ] conc=32  rerank_ops/s=440993
```

### Tableau à conc 32 et écart 2→3

| Point | rerank-ops/s | vs point 1 |
|---|---|---|
| 1. cache-verrou fp32 (existant) | 163 804 | 1.00× |
| 2. **contigu fp32 sans verrou** | **1 467 492** | **8.96×** |
| 3. contigu fp16 (= arène) | 440 993 | 2.69× |

**Écart 2→3 : le contigu fp32 est 3,33× plus RAPIDE que le contigu fp16** (1 467 492 /
440 993). Le signe est INVERSE de l'hypothèse du double volume.

### Interprétation — l'hypothèse du ×2 ne mord PAS à cette échelle

À 60 000 vecteurs, `flat32` occupe 30,7 Mo : les lectures aléatoires frappent
majoritairement le cache de dernier niveau (L3, ~32 Mo), pas la mémoire principale. Le
volume fp32 n'est donc PAS le facteur limitant ici. En revanche, le fp16 impose un
décodage fp16→fp32 pour chacune des 128 dimensions de chaque candidat — un surcoût CPU
qui dépasse la bande passante économisée quand la donnée est déjà en cache. À ce
régime, le fp32 contigu (lecture directe, zéro décodage, zéro verrou) est le plus
rapide des trois, et dépasse le régime arène.

Caveat de représentativité : le décodeur fp16 de la sonde est une implémentation naïve
(branchy) ; le moteur peut utiliser une table plus rapide, réduisant l'écart. Mais le
fp32 sans décodage est un PLANCHER de coût CPU du rerank : l'avantage du fp32 contigu à
l'échelle cache-résidente est robuste.

### Réponse binaire

**flatVecs fp32 contigu SUFFIT — et dépasse la parité arène — à l'échelle cache/RAM-
résidente.** Le câblage du miroir `flatVecs` (fp32) dans le rerank vamana est un fix
trivial (le tableau plat existe déjà, il n'est simplement pas consulté par ce chemin) :
il lève d'un coup le verrou ET la dispersion, sources conjointes du plafond, et n'exige
AUCUN passage au fp16. L'hypothèse « la parité exige le fp16 » est RÉFUTÉE à cette
échelle : le fp16 y est contre-productif (décodage plus cher que le gain de volume).

**Le fp16 ne redevient nécessaire qu'au régime > RAM.** Au 26,7 M réel (fp32 ~54 Go,
fp16 ~27 Go > RAM), la donnée ne tient plus en cache ni en mémoire : le volume redevient
le facteur dominant, chaque octet lu venant du NVMe par pagination. Là, le fp16 mmap
(l'arène) halve les pages à faire remonter du disque et reprend l'avantage ; le fp32
doublerait le trafic disque. La bascule est donc dictée par l'ÉCHELLE : fp32 contigu en
RAM sous le seuil de cache, fp16 mmap (arène) au-dessus.

### Synthèse des trois points (rappel, pipeline complet, conc 32)

| Régime | Search complet qps | source rerank |
|---|---|---|
| db-blob cache-verrou fp32 (existant) | 37 032 | cache LRU sous RWMutex, fp32 dispersé |
| db-blob N instances répliquées | 48 681 | idem, verrou non partagé (dispersion demeure) |
| arène fp16 contigu | 76 732 | mmap fp16 contigu sans verrou |
| **db-blob + flatVecs fp32 contigu (projeté)** | **≥ parité arène** | `[]float32` plat sans verrou (fix trivial) |

Artefact : `/devhoros/horosvec-bench/cmd/proto-rerank/main.go` (jetable, à supprimer).
