# Audit d'architecture disruptif — ce que la matinée a raté

Date : 2026-07-10. Mode : critique de conception, lecture seule, mandat contrariant
explicite. Périmètre : `/devhoros/horosvec` + audits du jour + veille
`/devhoros/horosvec-bench/audits/2026-07-10_veille_sqlite_ann_2026.md`. Aucun fichier
de code modifié.

## La thèse en une phrase

Le dilemme « arène rapide-figée vs db-blob mutable-lent » n'est pas un arbitrage de
STOCKAGE : c'est le symptôme aval d'un choix de QUANTIFICATION — 1 bit par dimension
est trop pauvre pour classer seul, donc chaque recherche doit relire 128 vecteurs
pleine précision, donc il faut un magasin de vecteurs chauds, donc il faut choisir où
il vit. Monter le budget de bits (Extended RaBitQ 4-5 bits, déjà dans la veille du
jour, NŒUD 2) rend le re-classement optionnel — et toute la question « où vivent les
vecteurs de rerank », avec ses deux modes, son miroir fp32, son arène fp16, son
verrou de cache et son trou d'OOM, s'évapore en amont au lieu d'être arbitrée en aval.

Cinq relecteurs ont audité le fix du rerank. Aucun n'a demandé pourquoi l'étage de
rerank existe. La matinée entière a optimisé la SOURCE d'une lecture dont la
NÉCESSITÉ est un paramètre de conception, pas une loi.

## Les prémisses fausses de la matinée

### P1 (fausse) — « il faut choisir entre rapide et incrémental »

Les propres mesures de la matinée réfutent la prémisse : après le fix flatVecs, le
db-blob DÉPASSE l'arène de bout en bout (40 619 vs 39 599 qps à conc 32,
`audits/2026-07-10_fix_rerank_flatvecs.md` C5). La vraie frontière n'est donc pas
« rapide vs incrémental » mais « tient en RAM vs ne tient pas ». Et pour le régime
> RAM, le design de l'arène segmentée append-only
(`horosvec-bench/audits/2026-07-10_design_arene_croissante.md`) — prouvé au sol le
MÊME JOUR : 435 M de lectures vérifiées sous churn d'appends, 0 lecture déchirée,
débit 1,0× l'arène figée — dissout aussi le « figé ». La conclusion admise (« arène
pour le grand, db-blob-flat pour le petit, pont périodique ») reconduit un partage
que les artefacts du jour ont déjà rendu caduc. Le « pont périodique » est un
vestige : il n'existe que parce que deux magasins coexistent.

### P2 (fausse) — « le rerank pleine précision est une donnée du problème »

`docs/ARCHITECTURE.md §1` pose le deux-étages comme LE design (« Every search runs
the same two stages »). Or la veille du jour établit précisément : 1-bit seul ≈ 76 %
de rappel (Milvus, source primaire), mais Extended RaBitQ 4 bits ≈ 90 %, 5 bits
≈ 95 %, 7 bits ≈ 99 % SANS re-classement — comparable aux planchers publiés du
moteur (0,95 à ef=64, 0,978 à ef=128, fix C5). La veille a été lue avec la lunette
« l'arène fp16 est validée comme palier de re-classement obligatoire » (NŒUD 2,
verdict) — c'est la lecture conservatrice du même fait. La lecture disruptive du
même paragraphe : le re-classement n'est obligatoire QU'À 1 BIT. La sensibilité de
l'estimateur 1-bit était déjà admise dans `doc.go` (« the two-stage design absorbs
the estimator noise ») : le deuxième étage est un pansement dimensionné pour la
pauvreté du premier.

Coût du passage à 5 bits, chiffré honnêtement : les codes du plan chaud passent de
dim/8 à 5·dim/8 octets par nœud. À 26,7 M × 512 : ~1,7 Go → ~8,5 Go de codes (+7 Go
de plan chaud) — CONTRE la disparition de l'arène 27,3 Go, du miroir flatVecs
(54 Go fp32 à cette échelle, aujourd'hui non borné, limite connue v0.2 de `doc.go`),
du cache LRU et de son RWMutex. Le bilan mémoire à l'échelle est FAVORABLE, pas
seulement neutre. SQLite garde les blobs fp32 comme source de vérité FROIDE
(rebuild, export, rerank exact opt-in pour les appelants exigeants) — hors du chemin
chaud, donc hors du problème de concurrence que la matinée a mesuré.

### P3 (fausse) — « le rerank était le goulot »

Le profil de la matinée le dit lui-même : la marche greedy + LUT domine (~70 % du
CPU, `audits/2026-07-10_audit_hotpath_dbblob.md`), le rerank pèse ~2 % ; le +6 %
end-to-end est la loi d'Amdahl appliquée à un étage minoritaire. Le plafond commun
~40 kQPS que l'arène ET le db-blob-flat atteignent désormais est celui de l'ÉTAGE 1
(bande passante mémoire de la marche sur le plan chaud). Toute optimisation
supplémentaire du stockage des vecteurs de rerank est structurellement bornée à
quelques pour cent. La matinée a limé le mauvais goulot — le +6 % n'est pas un
demi-succès, c'est le signal que ce goulot-là était déjà secondaire.

## Trois architectures alternatives, avec leur arbitrage

### A — « Un seul mode » : NodeStore + arène segmentée croissante (conservatrice)

Faire ce que le backlog de `docs/ARCHITECTURE.md §10` nomme déjà (« the tri-modal
storage branching wants a NodeStore abstraction ») et y brancher l'arène segmentée
prouvée : UN mode, append-only, incrémental, débit arène, du zéro au > RAM. Lever le
refus `horosvec.go:1090`, router Insert vers l'append de segment. Supprime : le
choix de mode, le pont périodique, le miroir non borné. Garde : le deux-étages et
son coût. Risque : faible (design prouvé au sol le jour même) ; le « vrai morceau
restant » avoué est l'extension incrémentale du plan chaud — qui existe déjà
(`extendPlaneAfterInsert`). C'est le chantier borné que le design du jour recommande
de différer ; la présente critique conteste le report : différer maintient DEUX
modes vivants, chacun avec ses invariants, ses tests, ses audits — le coût de
possession du dualisme est payé chaque matinée comme celle-ci.

### B — « Plus de rerank obligatoire » : Extended RaBitQ 4-5 bits (la thèse)

Étage 1 assez riche pour classer seul ; rerank exact depuis SQLite en OPTION
(`SearchExact`) pour les appelants qui veulent le score L2 contractuel. Supprime :
l'arène, le miroir, le cache LRU de vecteurs, le dilemme fp16/fp32, la limite « flat
mirror is not bounded ». Coût : réécrire encodeur + LUT + estimateur (le cœur
mathématique audité ligne-à-ligne le 2026-07-08 — c'est le morceau le plus
précieux et le plus risqué du dépôt), plan chaud ×~5 sur les codes, re-mesurer les
planchers de rappel par distribution. Rupture de format d'index (re-encode
intégral, mais les blobs SQLite suffisent à re-encoder sans les fvecs d'origine —
la migration est un rebuild local). Point d'attention : `Result.Score` cesse d'être
une distance exacte par défaut — c'est un changement de CONTRAT, pas seulement
d'implémentation ; l'option exacte le préserve pour qui le demande.

### C — « Assumer le produit tel qu'il est » : deux artefacts, pas deux modes

Si le vrai usage est celui des consommateurs connus (silo_retrieval, codemap :
corpus petits/moyens, incrémentaux) plus UNE démo read-only à 26,7 M, alors le
dualisme n'est pas un défaut à réparer mais deux PRODUITS : une lib incrémentale
RAM-résidente (db-blob-flat, désormais la plus rapide) et un format de publication
figé (arène + ImportAdjacency GPU) — analogues respectivement à une table et à un
fichier Parquet. Cesser de vouloir les unifier ; nommer la frontière dans l'API au
lieu de la subir dans la config. Coût : nul en code ; le prix est doctrinal —
renoncer à « un seul moteur qui fait tout », argument de vente implicite.

## Ce qui devient caduc si la thèse (B) est suivie

- L'arène fp16, son format `HVARENA1`, `arena.go`/`arena_build.go` (~1 100 lignes),
  le garde anti-Insert, le design d'arène croissante (résolu par disparition du
  besoin, ironiquement le jour de sa preuve).
- Le miroir flatVecs, le fix du matin, le cache LRU de vecteurs et son RWMutex — tout
  le chapitre « source de vecteurs de rerank ».
- Le « pont périodique » arène↔db-blob et la moitié du tableau des topologies
  (`docs/ARCHITECTURE.md §4`).
- La limite v0.2 « flat mirror not bounded » et le trou d'OOM associé.
- NON caduc : Vamana (le substrat de graphe reste le bon choix pour l'incrémental
  pur-Go — FreshVamana est le pattern canonique de la veille, et la greffe
  transactionnelle d'Insert est déjà conforme), la rotation Hadamard, SQLite comme
  source de vérité, la limite int32 du plan chaud (le sharding reste l'issue > 33 M).

## Ordre de vérification proposé (avant tout engagement)

1. Oracle décidable du rappel Extended-RaBitQ sur les corpus RÉELS du projet
   (bge-m3 1024d, qwen 512d) : prototype d'encodeur 5 bits hors moteur, mesurer
   recall@10 sans rerank vs les planchers publiés. Une semaine de banc, zéro ligne
   dans le moteur. Si < 0,95 sur les distributions réelles, la thèse tombe et A
   redevient la voie.
2. Si l'oracle tient : chiffrer la LUT 5 bits (le coût par saut de la marche croît ;
   l'étage 1 est DÉJÀ le goulot — vérifier que l'enrichissement des codes ne le
   creuse pas plus qu'il n'économise l'étage 2).
3. Trancher B vs C au niveau produit : la question « qui a besoin de 26,7 M
   read-only ? » est une question d'usage, pas d'ingénierie.

## Banc budget de bits — l'oracle décidable de la thèse (MESURÉ)

Date : 2026-07-10. Prototype jetable `horosvec-bench/cmd/proto-bitbudget` (pur
stdlib, `env GOWORK=off CGO_ENABLED=0`, 0 dépendance, moteur horosvec NON modifié).

### Corpus — RÉEL, dimension de production

`/inference/hnbook/bench_final/prefix1m.arena` : embeddings HackerNews réels
(pipeline bge/qwen, fp16, format `HVARENA1`), **dim 512**. Échantillon : 100 000
vecteurs de base + 200 requêtes (les 200 suivants). PAS de synthétique — le corpus
`horosvec-bench/data/base512_2m.jsonl` a été sondé et ÉCARTÉ : uniforme [0,1]
(fraction négative 0,0000, moyenne 0,500), c'est-à-dire le cas facile isotrope où
RaBitQ atteint déjà ~1,0 ; il aurait faussé la thèse dans le sens flatteur.

### Protocole

Vérité terrain : brute-force L2 exact fp32 (top-10) sur les 100 000 vecteurs, par
requête. Transformations portées fidèlement du moteur (`rotation.go`, `rabitq.go`) :
centrage sur le centroïde de base, **rotation Hadamard 1 round graine 42**,
codeDim 512. Classement APPROXIMATIF EXHAUSTIF (les 100 000 candidats notés par
l'estimateur, sans graphe : isole la qualité du QUANTIFICATEUR, indépendamment de
la marche Vamana). Deux références : (a) baseline actuel = estimateur 1-bit du
moteur (L1-corrigé) + re-classement fp32 exact du top-128 ; (b) exact = 1,0.

Estimateur multi-bits — approximation documentée et sa limite : quantificateur
scalaire uniforme à 2^B niveaux sur [−A, A] (A = max|coord| par vecteur, une échelle
fp32 stockée) du vecteur tourné-centré ; requête gardée pleine précision
(ASYMÉTRIQUE, comme la voie de production du moteur). C'est la baseline « SQ-B bits,
requête fp32 » qu'Extended RaBitQ RAFFINE (codebook non biaisé, normalisation
optimale) : les chiffres mesurés sont donc un **PLANCHER conservateur** de ce qu'un
encodeur Extended-RaBitQ fidèle atteindrait (courbe décalée vers la gauche). La
direction critique pour la thèse est conservatrice — si ce plancher franchit déjà la
baseline, l'encodeur réel la franchit aussi.

### Tableau mesuré (sortie collée)

| Régime | rappel@10 | octets/vecteur |
|---|---|---|
| (a) 1-bit + rerank fp32 top-128 **[BASELINE]** | **0,9945** | 2128 |
| B=1 bit, SANS rerank | 0,1300 | 72 |
| B=2 bits, SANS rerank | 0,3460 | 136 |
| B=3 bits, SANS rerank | 0,7920 | 200 |
| B=4 bits, SANS rerank | 0,8970 | 264 |
| B=5 bits, SANS rerank | 0,9450 | 328 |
| B=6 bits, SANS rerank | 0,9700 | 392 |
| B=7 bits, SANS rerank | 0,9865 | 456 |
| B=8 bits, SANS rerank | **0,9940** | 520 |
| exact fp32 (par construction) | 1,0000 | 2048 |

Store fp32 de re-classement (référence supprimable) = **2048 octets/vecteur**.

### Verdict binaire

**La thèse est REFUTÉE dans sa forme forte (≤ 5 bits) et CONFIRMÉE dans sa forme
architecturale (≈ 8 bits), sur ce corpus réel.**

- Forme forte de la note (« ≈ 4-5 bits ≈ 0,90-0,95 suffisent à supprimer le
  rerank ») : **REFUTÉE**. Aucun B ≤ 5 n'atteint la baseline réelle 0,9945 ; B=5
  plafonne à 0,945 — sous la baseline, et sous le seuil 0,95 visé. Le chiffre « 5
  bits ≈ 0,95 » de la veille était mesuré sur d'autres corpus, non sur ces
  embeddings ; il ne transfère pas.
- Le B de bascule contre la baseline actuelle est **B ≈ 8** (0,9940 ≈ 0,9945). B=7
  (0,9865) reste ~1 point sous la baseline.
- Forme architecturale de la thèse (« supprimer le rerank fp32 fait s'évaporer
  arène/miroir/verrou/OOM ») : **CONFIRMÉE, mais à B ≈ 7-8, pas ≤ 5**. À B=8, le
  rerank est supprimable À PARITÉ de rappel pour **520 octets/vecteur contre 2048**
  au store fp32 — un facteur **3,9× de mémoire en moins**, sans arène, sans miroir
  fp32, sans cache LRU, sans RWMutex, sans trou d'OOM. B=7 (456 o, 4,5× moins) offre
  0,9865 : un arbitrage produit à ~1 point de rappel. Comme l'estimateur SQ est un
  plancher, Extended RaBitQ fidèle décalerait probablement la bascule vers B=6-7.

Conséquence pour l'usager : le geste de conception « monter le budget de bits pour
supprimer le rerank » tient — l'apparat arène/miroir/verrou disparaît et le store
rétrécit d'un facteur ~4 — mais le budget réel est **~7-8 bits/dim, pas 4-5**. À
26,7 M × 512 : codes B=8 ≈ 13,3 Go, contre 27,3 Go d'arène + 54 Go de miroir fp32
supprimés. La thèse P2 de la note (rerank vestigial) est vraie ; sa CHIFFRAISON en
bits était optimiste d'un facteur ~1,6. Repli si l'usager refuse la rupture de
format ou le budget 8 bits : l'arène segmentée croissante (voie A) reste la porte de
sortie, thèse non universellement réfutée mais reprofilée.

Caveat de portée : le banc note la qualité du QUANTIFICATEUR en scan exhaustif ; à
travers la marche Vamana réelle, le rappel bout-en-bout serait légèrement inférieur
(défauts de faisceau), mais la marche lit les MÊMES codes B-bits — l'ordre relatif
entre B est préservé. Artefact : `/devhoros/horosvec-bench/cmd/proto-bitbudget/main.go`
(jetable). Corpus : `/inference/hnbook/bench_final/prefix1m.arena`.

## Banc Extended RaBitQ fidèle (correction de l'estimateur SQ)

Date : 2026-07-10. Correction sourcée par l'usager : le banc précédent notait les
bas budgets avec un quantificateur scalaire uniforme (SQ) SANS le facteur de
correction par vecteur — donc des chiffres faux-bas aux petits B. Extended RaBitQ
(arXiv:2409.09913, SIGMOD 2025) restaure ce facteur. L'estimateur a été réimplémenté
FIDÈLEMENT dans son mécanisme et remesuré, même corpus, même protocole.

### Fidélité de l'estimateur — déclarée honnêtement

Ce qui est FIDÈLE : grille de niveaux impairs symétriques `ℓ(c)=2c−(2^B−1)` (le MSB
de chaque code = le bit de signe, donc la concaténation des MSB = EXACTEMENT le code
RaBitQ 1-bit ; B=1 est le cas de base) ; facteur de correction par vecteur
`G=⟨ℓ,o'⟩` stocké ; estimateur asymétrique `dist²=‖q'‖²+‖o'‖²−2·estDot·‖o'‖²/G`,
`estDot=Σℓ(c_i)q'_i` — qui se réduit EXACTEMENT à `rabitqDistanceAsym` du moteur à
B=1 (ℓ=signe, G=L1). C'est précisément le facteur de correction que la version SQ
omettait. Ce qui reste APPROCHÉ : le choix du code utilise une échelle uniforme par
vecteur (`(2^B−1)/max|o'|`), non la recherche d'échelle optimale du papier/lib de
référence (VectorDB-NTU/RaBitQ-Library). Les chiffres fidèles ci-dessous restent donc
un PLANCHER (plus serré que la SQ) : un codebook Ext-RaBitQ pleinement optimisé
relèverait encore les petits B.

### Tableau corrigé (sortie collée, recall@10 SANS rerank fp32)

| B | recall@10 **fidèle** | (ancien SQ faux-bas) | octets/vec |
|---|---|---|---|
| 1 | 0,6190 | 0,1300 | 72 |
| 2 | 0,6775 | 0,3460 | 136 |
| 3 | 0,8350 | 0,7920 | 200 |
| 4 | 0,9165 | 0,8970 | 264 |
| 5 | 0,9540 | 0,9450 | 328 |
| 6 | 0,9785 | 0,9700 | 392 |
| 7 | 0,9900 | 0,9865 | 456 |
| 8 | 0,9935 | 0,9940 | 520 |

Références inchangées : baseline 1-bit + rerank fp32 top-128 = **0,9945**
(2128 o/vec) ; exact fp32 = 1,0 (2048 o/vec) ; store fp32 supprimable = 2048 o/vec.

### Verdict corrigé — le B de bascule

**L'usager a raison sur les faux-bas ; la prédiction « bascule à 3-4 bits » n'est
confirmée QUE contre une cible ~0,95, PAS contre la baseline réelle 0,9945.**

- La correction est spectaculaire aux TRÈS petits budgets (B=1 : 0,13→0,62 ; B=2 :
  0,35→0,68) — les chiffres SQ y étaient effectivement faussés. Le 1-bit fidèle
  (0,619) est cohérent avec l'ordre de grandeur du 1-bit-seul de la littérature.
- Mais aux budgets moyens/hauts la correction converge (B=5 : 0,945→0,954 ; B=8 :
  0,9940→0,9935, identiques au bruit). **Le B de bascule contre la baseline 0,9945
  reste ≈ 8** (B=7 fidèle = 0,990, ~0,5 pt sous). La « bascule à ~8 bits » n'était
  donc PAS un pur artefact de la SQ : elle est robuste au passage à l'estimateur
  fidèle, parce que la baseline elle-même est très haute (elle inclut le rerank fp32).
- En revanche, si la cible produit est **~0,95 sans rerank**, Extended RaBitQ fidèle
  l'atteint dès **B=5 (0,954, 328 o/vec, 6,2× moins que le store fp32)** ; et ~0,92
  dès **B=4 (264 o/vec, 7,8× moins)**. C'est là que la prédiction « 3-5 bits »
  s'incarne — au seuil 0,90-0,95, pas au seuil 0,9945.

Conséquence pour l'usager : la thèse architecturale (supprimer le rerank fp32 →
évaporation arène/miroir/verrou/OOM) tient à DEUX régimes selon l'exigence de rappel.
Parité stricte avec l'existant : B≈8 (520 o/vec, 3,9× moins de mémoire). Régime
« 95 % suffisent » : B≈5 (328 o/vec, 6,2× moins). Comme l'échelle du code reste
heuristique (non la recherche optimale du papier), un encodeur Ext-RaBitQ de
référence pourrait abaisser la parité vers B=6-7 et le seuil 95 % vers B=4 — AU
CONDITIONNEL, non mesuré ici. Artefact remesuré :
`/devhoros/horosvec-bench/cmd/proto-bitbudget/main.go`.

CLAIM:: le deux-étages de horosvec (marche 1-bit puis rerank exact) n'est pas une loi du moteur mais la conséquence du budget de 1 bit par dimension ; le dilemme de stockage arène/db-blob n'existe que parce que le rerank exige une source de vecteurs pleine précision sur le chemin chaud.
CLAIM-NEG:: le goulot résiduel de horosvec n'est pas le rerank ni SQLite ni modernc — c'est la marche greedy sur le plan chaud (~70 % du CPU), commune à tous les modes, que la matinée n'a pas touchée.
