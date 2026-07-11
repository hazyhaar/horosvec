# Veille sourcée — Extended RaBitQ multi-bit : préserve-t-il le débit du 1-bit ?

Question d'oracle : un index à codes multi-bits (Extended RaBitQ, 2-9 bits/dim)
préserve-t-il le débit de requête du schéma 1-bit, ou paie-t-il une pénalité de
vitesse — dernier arbitrage avant de remplacer le re-classement fp32 par du
multi-bit dans un moteur ANN Go pur, CGO off (donc sans SIMD/assembleur).

## Q1 — Le schéma MSB-first préserve-t-il le chemin rapide ?

**FAIT SOURCÉ.** Le papier Extended RaBitQ (arXiv:2409.09913, SIGMOD 2025)
décrit explicitement un schéma d'affinage incrémental par bits de poids fort :

> « During querying, we first estimate a distance by accessing only the most
> significant bits... If the estimated distance is sufficiently accurate to
> decide that a data vector cannot be the NN, then we drop it. Otherwise, we
> access the remaining bits and incrementally estimate a distance with higher
> accuracy. »
(https://arxiv.org/html/2409.09913)

Donc oui : la présélection peut s'arrêter au premier palier de bits (équivalent
du code 1-bit) et n'accéder aux bits suivants que pour les candidats non
déjà tranchés — un chemin rapide analogue au 1-bit est préservé, avec
accès incrémental proportionnel au nombre de candidats encore ambigus, pas à
l'ensemble du corpus. Le papier ne fournit cependant, dans l'extrait consulté,
aucun chiffre isolé de gain de ce mécanisme seul (accès direct au tableau
5.2.2 du PDF non résolu par le fetch HTML) — **la magnitude du gain reste
INFÉRÉE**, seule l'existence du mécanisme est FAIT SOURCÉ.

SymphonyQG (arXiv:2411.12229) applique un principe voisin côté graphe :
« RaBitQ [est incorporé] to the scheme of searching graph-based indices with
FastScan... stores the quantization codes of a vertex's neighbors compactly
... and uses a SIMD-based implementation named FastScan to efficiently
estimate distances based on the quantization codes in batch for guiding the
searching process. » (https://arxiv.org/abs/2411.12229). Ici la présélection
de graphe opère sur les codes quantifiés en lot via FastScan — un chemin
distinct du MSB-first d'Extended RaBitQ, mais qui confirme la même
architecture générale : la traversée rapide reste sur des codes compacts,
jamais sur le vecteur fp32.

## Q2 — Pénalité de débit mesurée du multi-bit vs 1-bit

**PARTIELLEMENT SOURCÉ, chiffres fragmentaires.** Aucune des sources
consultées (RaBitQ-Library README, blogs Elastic BBQ, VectorChord, LanceDB)
ne publie une comparaison directe et chiffrée QPS(1-bit) vs QPS(multi-bit)
à rappel comparable dans l'extrait récupéré :

- Le README RaBitQ-Library indique seulement des seuils de rappel par bit :
  « 4-bit, 5-bit and 7-bit quantization usually suffices to produce 90%, 95%
  and 99% recall respectively without reranking. »
  (https://github.com/VectorDB-NTU/RaBitQ-Library)
- Elastic BBQ (blog technique, comparaison BBQ 1-bit vs PQ, pas vs son propre
  multi-bit) : « BBQ achieved 11ms brute-force latency versus PQ's 20ms »
  (e5small), « 1776ms versus PQ's 5790ms » (CohereV3), « 2-4x faster at
  querying than PQ » en HNSW.
  (https://www.elastic.co/search-labs/blog/bit-vectors-elasticsearch-bbq-vs-pq)
  — comparaison contre PQ, pas contre un binaire strict ; n'informe pas
  directement Q2.
- VectorChord (résumé WebSearch, non vérifié en primaire) : « calculs
  compressés RaBitQ jusqu'à 100x plus rapides que le calcul de distance
  traditionnel [fp32] », avec « la plupart des comparaisons menées sur les
  vecteurs compressés, le calcul pleine précision réservé à une phase de
  reclassement adaptatif sur un sous-ensemble réduit. »
  (https://blog.vectorchord.ai — contenu résumé, pas cité verbatim, à
  reconfirmer par fetch direct si arbitrage engage du code).

**INFÉRÉ (non trouvé sourcé explicitement)** : le coût marginal du passage
2-bit → 5-bit reste faible tant que le raffinement porte sur un petit
ensemble de candidats déjà présélectionnés par les MSB (cf. Q1) — c'est
l'argument structurel du papier, mais aucune source consultée ne donne un
delta QPS chiffré isolant le seul effet du nombre de bits à rappel fixé.
**Ce point reste le trou de la veille** ; il faudrait rouvrir le PDF complet
section 5.2.2 (tableau time-accuracy) pour un chiffre dur.

Sur la question « remplace ou s'ajoute » : les trois systèmes de production
consultés (Elastic BBQ, VectorChord) **conservent un étage de reclassement en
précision supérieure après le scan quantifié** — Elastic rerank contre le
fp32 stocké, VectorChord réserve « full-precision calculations […] to an
adaptive reranking phase applied to a smaller subset ». Le multi-bit ne
remplace donc PAS le rerank dans ces déploiements ; il réduit la taille du
candidate-set et/ou la marge d'erreur qui arrive au rerank, mais ne
l'élimine pas structurellement dans les systèmes observés — **contredit
partiellement l'énoncé Q4** (voir plus bas).

## Q3 — FastScan / SIMD : viabilité en Go pur, CGO off

**FAIT SOURCÉ, favorable à Go pur pour les largeurs génériques.** Le papier
Extended RaBitQ précise que le FastScan/SIMD n'est pas une dépendance dure
pour toutes les largeurs de bits :

> « When B equals to 4 or 8, the implementations in existing systems...can be
> directly applied. Other settings of B's can be implemented by splitting a
> B-bit unsigned integer vector into several parts, where each part has the
> size of the power of 2. »
(https://arxiv.org/html/2409.09913)

C'est-à-dire : les largeurs B=4 ou B=8 s'alignent nativement sur des
implémentations FastScan/SIMD (AVX2/AVX-512) existantes, mais **toute autre
largeur reste implémentable par décomposition algorithmique en parties de
taille puissance de 2**, sans dépendre du SIMD. Ceci est compatible avec un
moteur Go pur CGO-off : la décomposition en opérations bit à bit / additions
d'entiers reste exprimable en Go stdlib pur (accumulation par paliers de
bits), au prix de ne pas bénéficier de l'accélération FastScan/AVX propre
aux implémentations C/C++ (SymphonyQG, RaBitQ-Library en C++, Elastic/Lucene
en Java+Panama-SIMD). La perte relative précise (perte de facteur XxN
d'absence de SIMD) **n'est pas chiffrée dans les sources consultées** —
INFÉRÉ que la perte est substantielle (FastScan/SIMD est justement l'axe de
gain revendiqué par SymphonyQG et RaBitQ-Library), mais sans un delta
mesuré isolé pour Go pur spécifiquement (aucune des sources ne teste un
portage Go/no-SIMD).

## Q4 — Le multi-bit rend-il le rerank fp32 superflu en production ?

**FAIT SOURCÉ, réponse nuancée : NON dans les systèmes observés, sauf
régime spécifique.** Deux signaux contradictoires selon le point de
compression :

- Extended RaBitQ (papier) : « we target the setting where the raw vectors
  cannot be accessed during querying so as to save main memory consumption »
  — à des taux de compression modérés (4-8x, soit grosso modo 4 à 8 bits/dim),
  le papier vise explicitement une élimination du besoin de stocker/accéder
  au vecteur brut, donc plus de rerank fp32 possible par construction (le
  fp32 n'est simplement plus en mémoire). C'est un FAIT du papier, dans le
  RÉGIME haute précision (proche 8 bits, ~99% recall).
- RaBitQ-Library (README) confirme ce même seuil : « 7-bit quantization
  usually suffices to produce 99% recall respectively without reranking. »
- MAIS Elastic BBQ et VectorChord, en PRODUCTION à des taux de compression
  plus agressifs (1-bit stocké côté BBQ, code compact côté VectorChord),
  gardent un étage de rerank contre le fp32 stocké séparément — cf. Q2.

**Verdict combiné** : la thèse « le multi-bit rend le rerank fp32 superflu »
est vraie côté théorique/papier au régime haut (proche 7-8 bits, où le
vecteur brut n'est plus stocké du tout), mais **fausse en pratique** dans les
deux systèmes de production consultés qui opèrent à des compressions plus
faibles (1-bit ou code compact) et gardent un rerank fp32 comme dernier
étage — parce qu'ils visent une compression mémoire maximale et acceptent la
dépendance au vecteur brut stocké ailleurs.

## Synthèse pour arbitrage

| Axe | Verdict | Statut |
|---|---|---|
| MSB-first préserve un chemin rapide analogue au 1-bit | Oui, mécanisme confirmé | FAIT (magnitude INFÉRÉE) |
| Pénalité QPS multi-bit vs 1-bit à rappel égal | Non chiffrée dans les sources trouvées | TROU DE VEILLE |
| Multi-bit remplace le rerank fp32 en prod | Non en général (BBQ, VectorChord le gardent) ; oui seulement au régime haut-bit sans stockage du vecteur brut (papier Extended RaBitQ) | FAIT nuancé |
| Viabilité Go pur / CGO off | Décomposable sans SIMD pour B≠4,8 selon le papier ; perte relative non chiffrée | FAIT (perte INFÉRÉE substantielle, non mesurée) |

## Sources

- https://arxiv.org/abs/2409.09913 et https://arxiv.org/html/2409.09913 — Extended RaBitQ (SIGMOD 2025)
- https://github.com/VectorDB-NTU/RaBitQ-Library — README, bornes de rappel par bit
- https://github.com/VectorDB-NTU/Extended-RaBitQ
- https://arxiv.org/abs/2411.12229 — SymphonyQG (FastScan + graphe)
- https://www.elastic.co/search-labs/blog/bit-vectors-elasticsearch-bbq-vs-pq — BBQ vs PQ, chiffres latence
- https://www.elastic.co/search-labs/blog/rabitq-explainer-101
- https://www.elastic.co/search-labs/blog/bbq-implementation-into-use-case
- https://blog.vectorchord.ai/vectorchord-store-400k-vectors-for-1-in-postgresql (résumé WebSearch, non vérifié en primaire — à refetch si l'arbitrage en dépend directement)
- https://www.lancedb.com/blog/feature-rabitq-quantization (non consulté en primaire, cité par recherche)
