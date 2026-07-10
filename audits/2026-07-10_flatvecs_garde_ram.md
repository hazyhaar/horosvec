# Ledger — garde budget flatVecs étendue au Build et au runtime (trous A/B, audits externes)

Module autonome `github.com/hazyhaar/horosvec` (hors go.work). Gates : `env GOWORK=off
CGO_ENABLED=0`. Ce ledger documente la fermeture de deux constats convergents remontés par
deux audits externes (grok-composer, grok-build) sur le fix précédent
(`2026-07-10_fix_rerank_flatvecs.md`) : la garde `Config.FlatVecsMaxBytes` protégeait le seul
rechargement (`New()`), pas le premier `Build()` ni la croissance runtime par `Insert()` —
et ce ledger lui-même était absent du dépôt.

## Contrat de réutilisation

- **Réutilisé tel quel** : le champ `Config.FlatVecsMaxBytes` (horosvec.go:49-56), le compteur
  `flatVecsBudgetSkips atomic.Int64` et son accesseur `FlatVecsBudgetSkips()` (horosvec.go:273-
  308), la formule d'estimation `nodeCount * dim * 4` déjà posée au chemin `New()`
  (horosvec.go:381-391).
- **Surface neuve** : un unique point d'entrée `flatVecsExceedsBudget(holdNodeCount int) bool`
  (méthode sur `*Index`, horosvec.go, juste avant `loadFlatVectors`) qui centralise la formule
  d'estimation — Build, le rebuild async et Insert l'appellent tous, au lieu de trois
  implémentations divergentes de la même règle.
- **Zéro nouvelle dépendance, zéro nouveau fichier hors `audits/` et `flatvecs_budget_test.go`.**

## État AVANT ce fix (constat des deux audits)

Seul un site consultait le budget : horosvec.go:381-391, dans le bloc `New()` exécuté
uniquement au rechargement d'un index déjà persisté. Trois sites peuplaient/étendaient
`flatVecs` **sans jamais consulter `Config.FlatVecsMaxBytes`** :

1. `Build()` (ex-ligne 580-586) — premier build à grande échelle, alloue
   `make([]float32, len(allVecs)*dim)` inconditionnellement.
2. Le rebuild async déclenché par dérive (`InsertRatioThreshold`/`DriftThreshold`, ex-ligne
   1574-1580) — même allocation inconditionnelle.
3. `Insert()` (ex-ligne 1345-1348) — `append(idx.flatVecs, pendingFlatVecs...)`
   inconditionnel dès que `flatVecs != nil`, sans jamais revérifier le budget projeté.

Conséquence : un index ouvert la première fois via `Build()` (jamais rechargé) ou grossi par
Insert au fil de l'eau pouvait dépasser `FlatVecsMaxBytes` et OOMer, contournant intégralement
la garde posée au fix précédent — celle-ci ne couvrait qu'un sous-ensemble des trajectoires de
vie de l'index (ouverture d'un index déjà bâti).

## Fix — chemins gardés désormais

| Chemin qui fait croître flatVecs | Avant | Après |
|---|---|---|
| `New()` (rechargement) | Gardé | Gardé (inchangé) |
| `Build()` (première construction) | **Non gardé** | **Gardé** — `flatVecsExceedsBudget(len(allVecs))` avant l'allocation ; si dépassé, `flatVecs` reste `nil`, `flatVecsBudgetSkips` incrémenté, `slog.Info` |
| Rebuild async (dérive centroïde/ratio insert) | **Non gardé** | **Gardé** — même garde, même formule |
| `Insert()` (croissance incrémentale) | **Non gardé** | **Gardé (gel)** — voir sémantique ci-dessous |

### Sémantique runtime choisie pour l'Insert : GEL, pas contrat figé au chargement

Deux voies étaient possibles : (a) le budget est un contrat de CHARGEMENT — une fois flatVecs
peuplé sous le budget initial, l'Insert continue de l'étendre sans revérification (le budget ne
gouverne que l'ouverture) ; (b) le budget est un contrat de TAILLE MAXIMALE — l'Insert
revérifie la projection à chaque batch et **gèle** la croissance dès que l'ajout dépasserait le
budget.

**Voie (b) retenue** — la voie la plus sûre : un budget qui ne protège que l'instant du
chargement laisse un index ouvert petit (sous le budget) grossir sans borne au fil des
inserts jusqu'à l'OOM, exactement le trou signalé par les deux audits pour Build. Geler la
croissance à l'Insert applique la même garantie « jamais d'OOM » de façon uniforme sur tout le
cycle de vie de l'index, pas seulement à l'ouverture.

Implémentation (horosvec.go, bloc post-commit de `Insert()`) : avant l'`append`, projection de
la taille finale `len(idx.flatVecs)/idx.dim + len(pendingFlatIDs)` ; si
`flatVecsExceedsBudget(projected)`, l'append est sauté (flatVecs reste à sa taille courante,
miroir **partiel** couvrant les node_id 0..k-1 les plus anciens), `flatVecsBudgetSkips`
incrémenté, `slog.Info`. Le gel est définitif pour la durée de vie du process (le budget est
une constante de `Config`, non réévaluée à la baisse) — cohérent avec le fait qu'un rechargement
ultérieur (`New()`) réévalue tout depuis zéro.

## Cohérence node_id → offset avec un flatVecs partiel

Deux lecteurs de `flatVecs` existent, avec des garanties différentes face à un miroir partiel :

1. **Branche rerank de `Search`** (horosvec.go:~885, chemin Vamana grande échelle) : borne déjà
   `off >= 0 && off+idx.dim <= len(idx.flatVecs)` avant de lire — un node_id au-delà du gel
   déborde la borne, retombe automatiquement sur `loadNodeReadOnly` (SQL + cache LRU). **Aucune
   modification nécessaire ici** : le bornage préexistant absorbe déjà un miroir partiel sans
   désynchronisation ni troncature, car les node_id sont denses et attribués dans l'ordre
   d'insertion — le préfixe couvert par `flatVecs` correspond exactement aux node_id
   `0..len(flatVecs)/dim-1`.
2. **`bruteForceFlat`** (horosvec.go:~758, chemin petit corpus sous `BruteForceThreshold`) :
   scanne INDISTINCTEMENT tout `idx.flatIDs`, sans borne par offset — un miroir partiel y
   produirait un déficit de rappel silencieux (nœuds au-delà du gel simplement absents du
   résultat), jamais un crash mais une régression de justesse. **Fix appliqué** :
   `bruteForceSearch` n'emprunte plus le chemin flat que si la couverture est complète
   (`idx.flatVecs != nil && int64(len(idx.flatIDs)) == idx.nextID`) ; sinon repli sur
   `bruteForceSQLite`, qui scanne exhaustivement la table et garde 100 % de rappel exact.

## Tests au sol (assertions décidables)

Fichier `flatvecs_budget_test.go`, trois fonctions :

- `TestFlatVecsBudgetGuard` (préexistant, inchangé) — garde au rechargement (`New()`).
- `TestFlatVecsBudgetGuardBuild` (neuf) — 1200 vecteurs dim 64 (empreinte 307 200 octets),
  budget 1024 octets, `Build()` direct (jamais de rechargement). Assertions : `flatVecs == nil`
  après Build, `FlatVecsBudgetSkips() == 1`, recherche fonctionnelle
  (`RerankSQLLoads()` progresse — preuve décidable du repli SQL), 0 panic, 0 résultat vide.
- `TestFlatVecsBudgetGuardInsertRuntime` (neuf) — Build initial de 400 vecteurs dim 64
  (102 400 octets, sous un budget de 150 000 → flatVecs chargé), puis `Insert` de 400 vecteurs
  supplémentaires (projection 800 nœuds = 204 800 octets > budget). Assertions : longueur de
  `flatVecs` inchangée après l'Insert (gel constaté), `FlatVecsBudgetSkips()` incrémenté,
  recherche fonctionnelle après le gel (`RerankSQLLoads()` progresse pour les nœuds hors
  miroir), 0 panic, 0 résultat vide.

Sortie constatée :
```
=== RUN   TestFlatVecsBudgetGuard
--- PASS: TestFlatVecsBudgetGuard (0.77s)
=== RUN   TestFlatVecsBudgetGuardBuild
--- PASS: TestFlatVecsBudgetGuardBuild (0.72s)
=== RUN   TestFlatVecsBudgetGuardInsertRuntime
--- PASS: TestFlatVecsBudgetGuardInsertRuntime (1.41s)
PASS
ok  	github.com/hazyhaar/horosvec	2.907s
```

## Golden et gates

**STRATE 0 (avant édition)** : `cd /devhoros/horosvec && env GOWORK=off CGO_ENABLED=0 go build
./...` OK, `env GOWORK=off CGO_ENABLED=0 go test ./...` → `ok github.com/hazyhaar/horosvec
(cached)`. VERT.

**Gate final** :
```
env GOWORK=off CGO_ENABLED=0 go build ./...   → OK
env GOWORK=off CGO_ENABLED=0 go test ./...    → ok  github.com/hazyhaar/horosvec  49.031s
gofmt -l .                                    → (vide)
env GOWORK=off CGO_ENABLED=0 go vet ./...     → (vide)
```
0 nouvel échec, 0 test affaibli, parité de rappel préservée (suite complète verte, incluant les
bancs `recall_measure_test.go`).

## Négacritères vérifiés

- N0 : 0 fichier hors `/devhoros/horosvec` touché.
- N1 : 0 nouvelle dépendance (`go.mod`/`go.sum` inchangés).
- N2 : arène fp16 et contrat transactionnel de l'Insert intacts — les branches modifiées sont
  postérieures au commit SQLite, la garde budget n'altère que l'état en mémoire.
- N3 : `flatVecs` reste `[]float32` (fp32), aucun changement de format.
- N4 : 0 test affaibli — les trois tests de garde budget sont additifs, la suite préexistante
  passe à l'identique.
- N5 : 0 push effectué (commit local uniquement).
