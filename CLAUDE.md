# horosvec — briefing

## CLAIM:: ce que horosvec EST

horosvec est une **bibliothèque Go embarquée** d'index vectoriel à plus proches voisins
approché (ANN), adossée à un unique fichier SQLite. C'est du code de bibliothèque pur, sans
démon, sans serveur, sans port réseau : l'appelant ouvre un `*sql.DB`, construit un `*Index`
et interroge en process.

CLAIM:: horosvec est une bibliothèque en process, pas un service — aucun daemon, aucun port, aucune API réseau.
CLAIM-NEG:: horosvec n'est ni siftrag-engine (le moteur de données RAG servi en HTTPS `:8470`)
ni un pôle interne de horos55 ; c'est un module Git autonome, publiable, consommé par import.

## Module et frontières

- Module : `github.com/hazyhaar/horosvec` (module Git indépendant, `go 1.26`).
- Dépendance unique hors stdlib : `modernc.org/sqlite` (pur Go, `CGO_ENABLED=0`, binaires
  statiques). Aucune autre dépendance externe n'est tolérée — c'est un argument de vente.
- Zone : `/devhoros/horosvec` exclusivement. Ce dépôt n'importe RIEN de horos55/jurhoros et
  n'a aucune connaissance du bus, des Objects, du catalog rules.db. Il est en amont de tout.
- Consommateurs connus dans horos55 (import descendant, jamais l'inverse) :
  `internal/silo_retrieval/retriever.go`, `internal/codemap/embedder/horosvec_writer.go`,
  `cmd/horos55-codemap/similar_cmd.go`. Le bump de version dans `horos55/go.mod` est une
  passe séparée, jamais faite depuis ce dépôt.

## Architecture en deux briques (vocabulaire anglais)

1. **Vamana** — graphe de proximité (DiskANN : marche greedy, élagage robuste alpha-RNG)
   pour la navigation.
2. **RaBitQ** — quantification binaire (un bit de signe par dimension + normes par vecteur)
   avec rotation de Hadamard randomisée (graine persistée) pour des distances approchées bon
   marché pendant la marche.

Recherche en deux étages : présélection par faisceau sur codes RaBitQ, puis re-classement
exact L2 sur les vrais vecteurs float32. Sous `Config.BruteForceThreshold`, le graphe est
court-circuité et le scan est exact (rappel parfait sur petit shard).

À l'échelle : **arène fp16** (`HVARENA1`, fichier sidecar mmap) — le SQLite devient
vector-less (le graphe et les codes vivent en base, les vecteurs bruts en fichier plat lu à
la demande). Éprouvé à 26,7 M vecteurs. Build en flux (empreinte mémoire bornée) et import
d'adjacence externe (`ImportAdjacency`, p. ex. graphe GPU CAGRA).

## Invariants durs — opposables à toute modification

- **Pure Go, CGO off.** `env GOWORK=off CGO_ENABLED=0 go build/test ./...` en pré-condition
  de tout commit. `-race` exige CGO et n'est donc pas au gate par défaut (cf. doctrine
  workspace). Les gates s'exécutent en `GOWORK=off` (module autonome, hors workspace).
- **Interface côté consommateur.** L'appelant fournit son `*sql.DB` et ses itérateurs ;
  horosvec n'impose aucun schéma applicatif au-delà de ses propres tables `vindex_*`.
- **Transactionnalité de l'Insert.** Les effets en mémoire (cache, miroir flat, compteur
  d'id, centroïde, node_count) ne sont appliqués qu'après un commit SQLite réussi. Le
  `node_count` de `vindex_meta` est écrit DANS la transaction — données et métadonnée sont
  atomiques. Au chargement, `node_count` est réconcilié avec `COUNT(*)` (la méta n'est jamais
  crue sur parole).
- **`context.Context` partout.** L'annulation est une erreur wrappée, jamais un résultat vide
  silencieux — y compris sur les scans brute-force (sonde toutes les ~4096 itérations).
- **Fail-loud sur corruption.** Un médoïde illisible fait échouer la recherche (jamais un `[]`
  indistinct d'un corpus vide). Un blob vecteur de longueur non conforme est sauté, jamais
  déréférencé hors borne. Un nœud voisin illisible est une dégradation TOLÉRÉE (comptée via
  `DegradedNeighborLoads()`), pas une erreur dure.
- **Limite int32 du plan chaud.** Les offsets du plan chaud (voisins, ext_ids) sont des
  int32 : le cumul N×degré et le cumul d'octets d'ext_ids doivent rester sous 2^31 (~33 M
  nœuds à degré 64). Au-delà, build/chargement/import échouent fail-loud (`checkInt32Offset`)
  plutôt que de tronquer un offset. Au-delà, sharder.
- **Génération d'ids.** Les ids de nœuds sont des entiers séquentiels internes (`node_id`
  dense 0..n-1) ; les `ext_id` sont fournis par l'appelant. horosvec ne génère pas d'UUID et
  n'a aucun lien avec `pkg/idgen` — cette règle du workspace ne s'applique pas ici.

## Compteurs d'observabilité (fail-soft, rétro-compatibles)

Méthodes de sonde exposées, jamais nil, sûres en concurrence :
`RerankSQLLoads()`, `DegradedNeighborLoads()`, `PlaneDegraded()`, `MalformedVectorSkips()`.
Un compteur > 0 signale un incident toléré (corruption locale, plan invalidé, blob non
conforme) sans rompre la recherche.

## Tests et gates

- `env GOWORK=off CGO_ENABLED=0 go test ./...` — suite complète (paquet unique
  `github.com/hazyhaar/horosvec`). `gofmt -l .` vide, `go vet ./...` propre.
- Bancs de mesure déterministes : `recall_measure_test.go`, `recall_real_test.go`
  (opt-in `HOROSVEC_REAL_VECS`). Les planchers de rappel publiés ne se dégradent pas.
- La non-dégradation à l'échelle (bancs 1M, validation 26,7M) est prouvée hors dépôt par le
  harnais de bench public ; elle n'est pas au gate de commit.

## Documentation de référence

- `doc.go` — doc de package (architecture, écarts au papier RaBitQ, limites connues).
- `docs/ARCHITECTURE.md` — conception détaillée.
- `docs/BENCHMARK-2026-07.md`, `docs/RETROSPECTIVE-2026-07.md` — mesures et leçons.
- `README.md` — prise en main.

## Convention rédaction

3e personne, langue soutenue, vocabulaire structurel en anglais, pas d'emoji, aucune
première personne (cf. doctrine workspace `/devhoros/CLAUDE.md`).
