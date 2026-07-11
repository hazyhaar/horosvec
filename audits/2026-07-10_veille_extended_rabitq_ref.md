# Veille — Extended RaBitQ (multi-bit), spécification de l'estimateur de référence

Date : 2026-07-10. Source primaire lue en texte intégral (PDF, pages 1-10) :
arXiv:2409.09913, "Practical and Asymptotically Optimal Quantization of
High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor
Search", Gao, Gou, Xu, Yang, Long, Wong (SIGMOD/ACM Conference'17 format,
soumis 16 sept 2024). Dépôt de référence cité par le papier :
github.com/VectorDB-NTU/RaBitQ-Library (Apache-2.0) — README consulté, code
source C++ non parcouru fichier par fichier (page GitHub "tree/main/src" a
renvoyé 404 ; le README suffit à corroborer les chiffres). Papier 1-bit
arXiv:2405.12497 non refetché séparément — ses résultats (Lemma 2.1 / Theorem
3.2) sont cités et repris tels quels par le papier étendu (Section 2.2 le
nomme "[27]").

Toute affirmation ci-dessous porte le tag **[SOURCÉ §x.y / Eq n]** (lu
directement dans le PDF) ou **[INFÉRÉ]** (déduction non lue telle quelle dans
le texte).

## 1. Construction du code B-bit

**[SOURCÉ §3.2.1, Eq 7-8]** La grille de niveaux utilisée par le papier est :

```
G := { -(2^B - 1)/2 + u | u = 0, 1, 2, ..., 2^B - 1 }^D
```

C'est une grille de `2^B` niveaux entiers ou demi-entiers centrés sur zéro,
espacés de 1 — équivalente à `ℓ(c) = (2c - (2^B - 1)) / 2` pour `c = u`. La
grille `ℓ(c) = 2c - (2^B - 1)` du prototype heuristique est donc la MÊME
famille de grille (niveaux impairs symétriques), à un facteur d'échelle
global de 2 près — sans conséquence car le vecteur de grille est ensuite
**normalisé** par sa propre norme (`y / ‖y‖`, Eq 8) avant rotation : un
facteur d'échelle constant sur la grille s'annule dans la normalisation.
**Verdict : la grille du prototype n'est pas la faute** — elle est
topologiquement correcte.

**[SOURCÉ §4.2, texte + Fig 2]** Propriété MSB = code 1-bit confirmée
littéralement : *"the concatenation of the most significant bits of all the
dimensions of ȳ_u exactly equals the quantization code x̄_b of the original
RaBitQ."* Le code B-bit se décompose en `ȳ_u = 2^{B-1}·ȳ_0 + ȳ_last`
(Eq 13), où `ȳ_0` est exactement le code 1-bit RaBitQ.

## 2. Le facteur d'échelle / scaling optimal par vecteur — LA pièce probable ratée par le prototype

**[SOURCÉ §3.2.2, Algorithm 1, Lemma 3.1]** Ce n'est PAS un facteur d'échelle
en forme close (une formule à évaluer une fois). Le papier pose le problème
comme une **recherche du point de grille `ȳ ∈ G` qui maximise la similarité
cosinus** avec le vecteur de données tourné `o' = P⁻¹o` (Eq 9-10) :

```
ȳ = argmax_{y∈G} ⟨ y/‖y‖ , o' ⟩
```

Résolu par **énumération exacte des "valeurs critiques"** d'un facteur de
re-scaling `t` (Algorithm 1, "Quantize") : en balayant les `t` croissants,
chaque dimension change de niveau arrondi à des points précis
(`t = (x+0.5)/o'[i]`), et l'algorithme maintient incrémentalement
`⟨y_cur, o'⟩` et `‖y_cur‖` via un min-heap, en `O(2^{B-1}·D·log D)`. Le point
`t_max` qui maximise `⟨y_cur,o'⟩/‖y_cur‖` donne le code optimal.

**Écart avec un prototype heuristique probable** : un prototype qui fixe un
facteur d'échelle unique (ex. la norme du vecteur brut, ou une formule
fermée par dimension) au lieu de **rechercher exhaustivement le meilleur
point de la grille par produit scalaire cosinus** perd l'optimalité de
l'assignation codeword-par-codeword. C'est l'étape qui porte l'optimalité
asymptotique revendiquée par le papier (Theorem 3.2) : sans cette recherche,
l'estimateur reste non biaisé (la construction codebook/rotation aléatoire
suffit à ça) mais l'erreur ne décroît plus au taux optimal en fonction de B.
**[INFÉRÉ]** L'ampleur du gain (en bits économisés pour un recall cible) ne
peut être quantifiée sans reproduire l'algo — le papier ne donne pas de
comparaison directe "grille optimale vs facteur d'échelle naïf", seulement
"RaBitQ (ext) vs RaBitQ (pad)" (extension triviale par padding de zéros, pas
la même faute que "mauvais scale"). Ne pas présenter de chiffre de gain en
bits sans le mesurer.

## 3. Estimateur de distance / produit scalaire

**[SOURCÉ §3.3, Eq 11-12]**

```
⟨ō, q⟩ = (1/‖ȳ‖) · ⟨ȳ/‖ȳ‖, P⁻¹q⟩
       = (1/‖ȳ‖) · ( ⟨ȳ_u, q'⟩ - (2^B-1)/2 · Σ_{i=1}^{D} q'[i] )
```

où `ȳ_u = ȳ + (2^B-1)/2 · 1_D` est le code stocké en entiers non signés
(Eq 9 du §3.2.1), `q' = P⁻¹q`, et `‖ȳ‖` est pré-calculé à l'indexation. Le
distance carrée s'obtient ensuite via Eq 1-2 (loi des cosinus reformulée à
partir de `⟨o,q⟩`), reprise identique à RaBitQ 1-bit.

**Réduction à B=1** **[SOURCÉ §3.4, texte]** : *"When B equals 1 (the case
of the original RaBitQ), RaBitQ's implementation can be directly applied."*
À B=1, `ȳ_u = x̄_b` (code binaire {0,1}), et Eq 12 se réduit exactement à
Eq 6 du papier 1-bit (Lemma 2.1 restaurée) : `⟨ō,q⟩ = (1/‖ō_0‖)·(2/√D·⟨q',x̄_b⟩ − 1/√D·Σq'[i])`
— la forme B-bit et la forme 1-bit coïncident structurellement, le préfacteur
`(2^B-1)/2` remplaçant le `1` du cas binaire.

## 4. Affinage incrémental MSB → bits bas

**[SOURCÉ §4.2, Eq 13-14, texte]** À la requête, pour chaque candidat :

1. Calculer `⟨ȳ_0, q'⟩` (MSB seuls = code RaBitQ 1-bit original, via FastScan
   SIMD batché) et en déduire une distance estimée **avec la borne d'erreur
   exacte de Lemma 2.1** (papier 1-bit, Theorem 3.2).
2. Si cette borne suffit à établir que le candidat ne peut pas être plus
   proche que le meilleur NN déjà trouvé → **élaguer sans lire les bits
   bas** (condition d'arrêt/pruning — texte renvoie à un "recent study" [26]
   pour ce principe générique de borne inférieure d'estimation).
3. Sinon, lire `ȳ_last` (bits restants), calculer `⟨ȳ_last,q'⟩`, recomposer
   `⟨ȳ_u,q'⟩ = 2^{B-1}·⟨ȳ_0,q'⟩ + ⟨ȳ_last,q'⟩` (Eq 13-14) pour l'estimation
   pleine précision.

## 5. Chiffres publiés (recall / bits)

**[SOURCÉ §5.2.2, texte + Fig 4]** *"4-bit, 5-bit and 7-bit quantization
usually suffices to produce 90%, 95% and 99% recall respectively without
re-ranking"* — cohérent sur les 6 corpus testés (MSong, Youtube, OpenAI-1536,
OpenAI-3072, Word2Vec, GIST, tous >100 K à 1 M vecteurs, IVF avec 4096
clusters). Sur GIST spécifiquement, le texte relève même *"3-bit
quantization suffices to produce > 95% recall"* (dataset "robuste").

**[SOURCÉ §3.4, "Remark (Empirical Formula)"]** Borne d'erreur empirique à
>99,9% de confiance : `ε < 2^{-B} · c_ε / √D`, avec `c_ε = 5.75` mesuré
expérimentalement (pas dérivé analytiquement).

**Calibrage vs le prototype** (0,954 recall à B=5) : le chiffre publié pour
B=5 est ~95% de recall — **le prototype heuristique est déjà dans l'ordre
de grandeur annoncé par le papier de référence à B=5**, pas manifestement en
retard. Le gain attendu de la construction exacte porterait plutôt sur
l'**erreur maximale** (queue de distribution, "Maximum Relative Error", Fig 3)
où le papier revendique un facteur 1,3×-3,1× meilleur que SQ/LVQ à B>6, et
sur la **stabilité inter-datasets**, pas nécessairement sur le recall moyen à
B=5 seul. **[INFÉRÉ]** — non mesuré ici.

## 6. Portabilité Go pur (CGO off)

**[SOURCÉ §3.2.2 Algorithm 1, §3.3 Eq 11-12, §4.2 texte]** L'algorithme de
construction du code (Algorithm 1 : boucle + min-heap sur `O(2^{B-1}·D)`
valeurs critiques) et l'estimateur (Eq 11-14 : produits scalaires, sommes,
divisions) sont de l'**arithmétique scalaire pure** — aucune primitive
n'exige de SIMD pour être CORRECTE. Le texte est explicite sur le rôle du
SIMD : *"the distance estimation based on the most significant bits can be
realized with a rather efficient SIMD-based implementation called FastScan
[4]"* — FastScan (référence externe, Guo et al.) est cité comme optimisation
de VITESSE (traiter un batch de candidats avec des instructions AVX512 dans
l'implémentation du dépôt de référence), jamais comme condition de
correction. Le README du dépôt confirme un existant en C++/AVX512
("optimized with the SIMD instructions till AVX512", §5.1) mais l'algorithme
mathématique n'en dépend pas. **Un portage Go pur (boucles scalaires) est
donc fidèle à la référence** — seule la vitesse de traitement par lot en
pâtira face à l'implémentation C++/SIMD du dépôt de référence.

## Sources

- arXiv:2409.09913 — https://arxiv.org/abs/2409.09913 (PDF lu intégralement
  pages 1-10, texte + équations + Algorithm 1 + Figures 1-5, Table 1)
- github.com/VectorDB-NTU/RaBitQ-Library — https://github.com/VectorDB-NTU/RaBitQ-Library
  (README fetché, confirme "1-bit + multi-bit implementations", chiffres
  90/95/99% recall à 4/5/7 bits, AVX512)
- Papier 1-bit RaBitQ (référencé [27] dans le papier étendu, non refetché
  séparément) — arXiv:2405.12497, cité pour Lemma 2.1 / Theorem 3.2 dont le
  papier étendu reprend l'énoncé littéral en §2.2.
