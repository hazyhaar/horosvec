# Audit cdc-reverse — chemin d'insertion incrémentale de horosvec

Date : 2026-07-10. Mode : cdc-reverse. Périmètre (lecture seule) : `/devhoros/horosvec`
(moteur) + `/devhoros/horosvec-bench/cmd/hnbook-serve` (mode servi de la démo).
Aucun fichier de code modifié ; seul ce rapport est écrit.

Couches d'oracle disponibles : horosvec est une **bibliothèque en process**, sans
`rules.db`/`audit.db` (cf. `CLAUDE.md`, `CLAIM::`). Seule la couche **Réalisé** (code, tests
exécutés au sol) existe. Tout besoin reste **INFÉRÉ**.

## Reformulation liminaire — ce que le « plan chaud » EST (correction du modèle du brief)

CLAIM:: le plan chaud (`hotPlane`) de horosvec n'est pas un tampon des seuls items insérés
fusionné à la recherche ; c'est un miroir plat, contigu et sans pointeurs de TOUT le graphe
persisté (nœuds denses `0..n-1`), utilisé comme représentation chaude de l'adjacence Vamana.
CLAIM-NEG:: il n'existe aucune « base immuable » distincte d'un « tampon chaud » balayé
linéairement ; l'insertion greffe le nouveau nœud directement dans le graphe Vamana.

OBSERVÉ : `hotplane.go:10-23` (« vue contiguë et pointer-free des nœuds persistés, indexée
par nodeID dense 0..n-1 ») ; `buildHotPlane` charge `SELECT ... FROM vindex_nodes ORDER BY
node_id` (hotplane.go:45-49) — donc l'intégralité des nœuds, pas un delta. Cette correction
gouverne toutes les réponses ci-dessous : il n'y a **pas de coût de fusion** au sens du brief.

## C0 — Carte du chemin

```
Index.Insert(ctx, vecs, ids)                    horosvec.go:1067
  ├─ mu.Lock()                                  :1079  (single-writer)
  ├─ refus si idx.arena != nil (fail-loud)      :1090-1092
  ├─ BeginTx ; defer tx.Rollback()              :1099-1103
  ├─ pour chaque vecteur :
  │    ├─ Rotate + Encode (RaBitQ)              :1137-1139
  │    ├─ saveNode(tx, ...) dans vindex_nodes   :1141
  │    ├─ rabitqGreedySearchInternal            :1172  (trouve les voisins via le graphe)
  │    ├─ setNeighbors (updateNeighbors tx)     :1159-1168
  │    ├─ arêtes inverses + robustPrune         :1199-1240 (greffe réelle dans le graphe)
  │    └─ pendingNodes[nodeID] = ...            :1242 (overlay in-tx)
  ├─ écrit node_count DANS la tx                :1265-1274 (atomicité données/méta)
  ├─ tx.Commit()                                :1280
  ├─ effets mémoire APRÈS commit :              :1284-1300
  │    cache.put, flatVecs, nextID, centroid,
  │    extendPlaneAfterInsert(...)              :1300 -> hotplane.go:246
  └─ shouldRefreshMedoid ? recomputeMedoid      :1307-1315

Index.Search(ctx, query, topK)                  horosvec.go:629
  ├─ mu.RLock()                                 :634
  ├─ si nextID <= BruteForceThreshold : bruteForceSearch  :651-652 (exact, O(N))
  └─ sinon vamanaSearch                         :655 -> :792
       ├─ étage 1 : rabitqGreedySearch (marche greedy sur le graphe/plan)  :814
       ├─ étage 2 : re-rank L2 exact (arène fp16 ou loadNode)              :829-854
       └─ résolution ext_id via planeExtID (plan chaud) sinon SQL          :861-874
```

Stockage d'un item inséré : **persisté** (blob + code + voisins dans `vindex_nodes`, dans la
tx) ET reflété en mémoire (cache LRU, miroir flat si actif, et `hotPlane` étendu en place par
`appendNode`, hotplane.go:132-137). La recherche ne « fusionne » pas deux structures : elle
parcourt un unique graphe Vamana dont le nœud inséré fait désormais partie (voisins directs +
arêtes inverses élaguées). **Coût de fusion à la recherche : nul** — pas de balayage linéaire
d'un tampon ; le surcoût est payé à l'insertion (greffe O(EfSearch·degré) par item).

Verdict : **confirmé, avec reformulation** du modèle mental du brief. OBSERVÉ.

## C1 — Croissance bornée / compaction

Il n'existe pas de « tampon chaud » à compacter : le plan chaud est le graphe entier, étendu
nœud par nœud (hotplane.go:246-263). Sa croissance est bornée fail-loud par la **capacité des
offsets int32** : `checkInt32Offset` refuse tout cumul N×degré ou octets d'ext_ids au-delà de
2^31-1, soit ~33 M nœuds à degré 64 (hotplane.go:29-42, 108-125). Au-delà : erreur dure
« rebuild with a sharded index » — pas de troncature silencieuse.

Refonte disponible : `RebuildAsync` → `rebuildInternal` reconstruit tout le graphe depuis un
itérateur fourni par l'appelant (horosvec.go:1351-1410). Elle n'est PAS déclenchée
automatiquement par `Insert` : l'appelant doit interroger `NeedsRebuild()` (drift de
centroïde, horosvec.go:1339-1346) et décider. Mitigation partielle intégrée : rafraîchissement
du médoïde quand trop de nœuds ont été ajoutés (`shouldRefreshMedoid`, seuil périodique ou
>50 % de nœuds neufs, horosvec.go:1322-1335).

Dette OBSERVÉE : aucune compaction/refonte **automatique** du graphe incrémental n'est câblée ;
la qualité de la greffe incrémentale (robustPrune local) peut dériver sur de longues séries
d'insertions sans rebuild. INFÉRÉ (non prouvé au sol dans ce dépôt) : cette dérive dégrade le
rappel bien avant la borne des 33 M nœuds. La décision de rebuild est déléguée au consommateur.

Verdict : **croissance bornée fail-loud (OBSERVÉ) ; refonte manuelle, jamais automatique
(OBSERVÉ) ; dérive de qualité incrémentale plausible mais non mesurée ici (INFÉRÉ)**.

## C2 — Rappel sous charge (post-insertion)

Le nœud inséré est cherché exactement comme un nœud de base (même graphe, même marche greedy) :
la recherche dans le plan chaud n'est ni « exacte exhaustive » ni « approchée à part » — c'est
l'ANN Vamana+RaBitQ standard. Oracles exécutés au sol (`env GOWORK=off CGO_ENABLED=0 go test`) :

- `TestInsertAndFind` : taux de retrouvage des 200 vecteurs insérés = **100,00 % (200/200)**
  (horosvec_test.go:208).
- `TestHotPlane_InsertExtendsPlaneAndPatch` : retrouvage via le chemin du plan = **100,00 %
  (40/40)** (hotplane_test.go:169).
- `TestHotPlane_RecallClustersUnchanged` : le plan chaud ne change pas les résultats, plancher
  de rappel tenu (hotplane_test.go:175-190).
- `TestRecallMeasure_VamanaRabitq` : rappel@10 moyen 0,93 (uniforme) / 0,68 (clusters
  gaussiens) — plancher de référence du moteur, hors insertion (recall_measure_test.go:37-44).

OBSERVÉ (petite échelle, ≤ quelques milliers) : l'insertion préserve le rappel de retrouvage.
INFÉRÉ / non couvert ici : la non-dégradation du rappel après de très grandes séries
d'insertions sans rebuild ; `CLAUDE.md` indique que la non-dégradation à l'échelle est prouvée
hors dépôt (bancs 1 M / 26,7 M), pas au gate de commit. Verdict : **rappel post-insertion tenu
à petite échelle (OBSERVÉ) ; à grande échelle sous insertions massives, AU CONDITIONNEL
(INFÉRÉ, hors périmètre testable ici)**.

## C3 — Exposé par le service `hnbook-serve` (le plus concret)

Réponse tranchée : **le service servi N'ACCEPTE PAS l'insertion.** `openIndex`
(main.go:175-188) ouvre la base puis pose `cfg.ArenaPath = arenaPath` avant `horosvec.New`.
`New` charge alors l'arène fp16 et affecte `idx.arena` (horosvec.go:357-368). Or `Insert`
refuse fail-loud tout index adossé à une arène : `if idx.arena != nil { return ... "incremental
Insert is not supported on an arena-backed index; rebuild instead" }` (horosvec.go:1090-1092).
Symétriquement `RebuildAsync` est refusé (horosvec.go:1360-1363).

Nuance sur le mode d'ouverture : la base SQLite est ouverte en lecture/écriture ordinaire
(`sql.Open("sqlite", indexPath)`, main.go:176 — aucun pragma `immutable`/`mode=ro`) ; ce n'est
donc pas un verrou disque qui interdit l'écriture. C'est le **garde arène applicatif** qui rend
`Insert` mécaniquement inopérant. De plus, l'API HTTP n'expose que la recherche (documentée
« serveur sans état », main.go:8) : aucune route d'écriture. L'objet `Index` en mémoire
refuserait `Insert` même s'il était appelé directement.

Verdict : **le mode servi est en lecture seule effective par le garde arène (OBSERVÉ,
main.go:181 + horosvec.go:1090). L'insertion incrémentale y est structurellement exclue.**

## C4 — Persistance & concurrence

Persistance : un item inséré est écrit dans `vindex_nodes` DANS la transaction, `node_count`
compris (horosvec.go:1141, 1265-1274), commit à :1280 ; les effets mémoire ne s'appliquent
qu'après commit (:1284-1300). L'insertion **survit donc au redémarrage** — hors mode arène, où
`Insert` est refusé (donc sans objet). Le plan chaud lui-même est volatile mais reconstruit au
chargement depuis SQLite (`rebuildPlaneLocked` → `buildHotPlane`, horosvec.go:352,
hotplane.go:154-170) : sa perte en RAM est sans conséquence, la source de vérité reste la base.
OBSERVÉ.

Concurrence : `Index.mu` est un `sync.RWMutex` (horosvec.go:223, « protects searches vs.
structural changes »). `Search` prend `RLock` (horosvec.go:634), `Insert` prend `Lock`
(horosvec.go:1079), `rebuildInternal` prend `Lock` sur toute sa durée (horosvec.go:1384-1385).
Modèle single-writer : insertion et recherche concurrentes sont sérialisées, jamais une course
sur le plan/le graphe. Les compteurs d'observabilité sont des `atomic.Int64`
(horosvec.go:259). Oracle de course `-race` : non exécuté (profil CGO requis, hors gate
`CGO_ENABLED=0` par doctrine `CLAUDE.md`) ; verdict rendu par **lecture des verrous**.
`TestInsertDuringRebuildNoDataLoss` passe (insertion pendant rebuild sans perte,
horosvec_test.go). OBSERVÉ (verrous + test) ; l'absence de course fine reste INFÉRÉE faute de
`-race`.

Verdict : **persistance transactionnelle atomique (OBSERVÉ) ; concurrence protégée par
RWMutex single-writer (OBSERVÉ) ; oracle -race non disponible dans le périmètre (limite
déclarée).**

## C5 — Synthèse

Verdict global : **tenu à jour FAISABLE tel quel pour un index en mode SQLite (non-arène), à
petite/moyenne échelle ; INFAISABLE en l'état sur le mode servi (arène), qui exige une refonte
par `Build`.** L'insertion incrémentale est une greffe de graphe réelle, transactionnelle,
persistée et concurrente-safe (OBSERVÉ). Le mode servi de la démo est délibérément en lecture
seule (garde arène).

Dettes chiffrées :

| Dette | Nature | OBSERVÉ/INFÉRÉ | Effort |
|---|---|---|---|
| Aucune refonte automatique du graphe incrémental (dérive de qualité sur longues séries d'inserts sans rebuild) ; déclenchement délégué au consommateur | qualité de rappel long terme | OBSERVÉ (absence de trigger) + INFÉRÉ (ampleur de dérive) | M |
| Mode arène incapable d'insertion : tout ajout impose un `Build` complet (26,7 M vecteurs → coût de reconstruction) | frontière fonctionnelle du mode servi | OBSERVÉ (horosvec.go:1090, 1360) | L |
| Oracle de course `-race` non exécutable au gate (CGO off) : la sûreté concurrente repose sur la lecture des verrous | couverture de test | OBSERVÉ (contrainte doctrine) | S |
| Non-dégradation du rappel à grande échelle sous insertions massives non prouvée dans ce dépôt (bancs hors dépôt) | couverture de bench | INFÉRÉ | M |

## Lentilles

- **Croissance bornée** : bornée fail-loud à ~33 M nœuds par les offsets int32
  (hotplane.go:36) ; pas de compaction automatique en deçà.
- **Rappel sous charge** : 100 % de retrouvage des insérés à petite échelle (OBSERVÉ) ;
  grande échelle au conditionnel.
- **Exposition par le service** : `hnbook-serve` ouvre en mode arène → `Insert` refusé
  (lecture seule effective, OBSERVÉ).

## Critique de complétude (ce qui n'a pas été lu / oracle manquant)

Non lus au corps : `rabitqGreedySearchInternal` (909-985) dans le détail de la marche greedy
sous overlay ; `robustPrune`/`vamana.go` (qualité de l'élagage) ; `recomputeMedoid`/`medoid.go`.
Oracle manquant le plus signifiant : **aucune mesure de rappel APRÈS de grandes séries
d'insertions sans rebuild** (le harnais de recall teste un index construit, ou une insertion de
quelques centaines d'items) — la dérive de qualité incrémentale reste INFÉRÉE. Oracle `-race`
absent (CGO). Négatifs bornés : « pas de compaction automatique » vaut pour les symboles
sondés du paquet `horosvec` ; aucune énumération exhaustive d'un éventuel déclencheur externe
côté consommateur n'a été faite (hors périmètre).

## RETEX cognitif

L'attaque a démarré au corps de `Insert`, et le premier réflexe payant a été de refuser le
modèle mental du brief : « plan chaud = tampon d'insérés fusionné à la recherche ». La lecture
de `hotplane.go:10-23` puis du `SELECT ... FROM vindex_nodes ORDER BY node_id` de
`buildHotPlane` a renversé ce cadre — le plan chaud est le miroir du graphe entier, pas un
delta. Sans cette bascule, C0 aurait décrit un coût de fusion inexistant et C2 aurait cherché
un balayage linéaire qui n'existe pas. Le CLAUDE.md du dépôt, dense et honnête (invariants A1,
A5, B3 cités en commentaire au bon endroit), a considérablement accéléré : les commentaires
ici ne mentaient pas, cas rare, mais j'ai quand même vérifié le garde arène au sol
(`main.go:181` → `horosvec.go:1090`) plutôt que de croire l'en-tête « lecture seule ». Le
piège évité de justesse : conclure C3 « lecture seule car SQLite ouvert en read-only » — faux,
la base est ouverte en R/W, c'est le garde applicatif arène qui bloque. Exécuter les tests
plutôt que citer leurs noms a transformé C2 d'une inférence en un fait à 100 %. Pattern
réutilisable : quand un brief impose un modèle, le premier oracle à chercher est celui qui
pourrait le RÉFUTER.
