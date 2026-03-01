# Méta-vectorisation par recherches stratifiées

## Concept

La recherche vectorielle classique opère sur des embeddings figés : un texte → un vecteur → une recherche L2. Le résultat est limité par ce que l'embedding encode.

La méta-vectorisation inverse le paradigme : **la recherche vectorielle devient un outil de construction de vecteurs**. On lance des batteries de recherches spécialisées sur un objet, et les résultats de ces recherches deviennent de nouvelles dimensions. Le métavecteur enrichi capture des relations structurelles invisibles au L2 brut.

## Principe

```
Objet X → embedding brut (dim D)

Stratum 1 : recherche spécialisée A → scores/ranks → S₁ dims
Stratum 2 : recherche spécialisée B → scores/ranks → S₂ dims
Stratum N : recherche spécialisée N → scores/ranks → Sₙ dims

Métavecteur X : [embedding_D | S₁ | S₂ | ... | Sₙ]  →  dim D + ΣSᵢ
```

Chaque stratum est un **capteur sémantique** qui mesure l'objet sous un angle différent. L'ensemble des mesures est la représentation riche.

## Exemple : corpus législatif

Le vecteur brut "Législateur X" est un embedding de sa biographie — dim 128. Pauvre.

| Stratum | Source interrogée | Ce qu'il capture | Dims ajoutées |
|---------|-------------------|------------------|---------------|
| Jurisprudence | Index des textes de loi | Quels textes il a rédigés/votés, quels domaines juridiques | 64 |
| Carrière | Index des postes publics | Trajectoire : commissions, ministères, postes, évolution temporelle | 64 |
| Réseau social | Index des posts/tweets/déclarations | Ton, opinions affichées, sujets récurrents, polarisation | 64 |
| Votes croisés | Index des scrutins | Avec qui il vote, contre qui, coalitions implicites | 64 |
| Financement | Index des déclarations patrimoniales / lobbys | Sources de financement, liens économiques | 64 |

Métavecteur enrichi : **dim 128 + 320 = 448 dims**.

Résultat : une recherche "qui ressemble au Législateur X" dans le méta-espace trouve un législateur Y qui a un parcours similaire, vote pareil sur le fond, a le même réseau, les mêmes financeurs — même si leurs biographies textuelles n'ont rien en commun.

## Propriétés

### Sur-dimensionnement sémantique

Chaque stratum ajoute des dimensions qui encodent une **relation structurelle** plutôt qu'un contenu textuel. Le méta-espace a plus de dimensions que l'embedding de base, mais chaque dimension supplémentaire porte du sens.

### Résultats impossibles autrement

Deux objets distants en L2 dans l'espace brut (bios différentes, vocabulaire différent) peuvent être proches dans le méta-espace (mêmes votes, même réseau, même trajectoire). Cette information est structurelle — aucun embedding texte ne la capture.

### Construction dynamique

Les strata ne sont pas figés. On peut :
- Ajouter un stratum quand une nouvelle source de données apparaît
- Pondérer les strata différemment selon le contexte de recherche
- Construire des strata de niveau 2 (recherche sur les métavecteurs de niveau 1) → **récursion**

### Arbres de recherche

La récursion produit des arbres :

```
Niveau 0 : embedding brut
Niveau 1 : enrichi par recherches sur indices spécialisés
Niveau 2 : enrichi par recherches sur les métavecteurs de niveau 1
Niveau K : capture des relations d'ordre K
```

Chaque niveau capture des corrélations d'ordre supérieur. Le niveau 2 trouve des objets qui "répondent pareil aux mêmes questions sur les mêmes sujets" — une méta-similarité.

## Généralisation

| Domaine | Objet | Strata possibles |
|---------|-------|------------------|
| Médical | Médicament | Essais cliniques, effets secondaires, molécules proches, prescriptions réelles, interactions |
| Finance | Entreprise | Brevets, fournisseurs, sentiment marché, concurrents, régulateurs |
| Code | Fonction | Callers, tests, bugs associés, patterns similaires, historique git |
| Musique | Morceau | Genre, BPM/structure, playlists contenant, artistes similaires, paroles |
| Juridique | Décision de justice | Textes cités, juridiction, juge, parties, domaine de droit |

## Implémentation dans horosvec

L'idée se map naturellement sur l'architecture existante :

- **Un `Index` par stratum** : chaque source spécialisée a son propre index Vamana/SQLite
- **Phase d'enrichissement** : pour chaque objet, lancer les recherches stratifiées, concaténer les scores en métavecteur
- **Un `Index` méta** : indexer les métavecteurs enrichis pour la recherche finale
- **Pondération** : coefficients par stratum, ajustables au query-time
- **Rebuild incrémental** : quand un stratum change, ne re-enrichir que les objets affectés

La concurrence de horosvec (sync.RWMutex, recherches parallèles) permet de lancer les N recherches stratifiées en parallèle pour chaque objet.
