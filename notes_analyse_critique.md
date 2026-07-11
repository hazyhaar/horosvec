# Notes d'analyse critique et proactives — horosvec (Juillet 2026)

Ce document compile les conclusions d'une revue d'architecture indépendante et proactive réalisée sur la bibliothèque [horosvec](file:///devhoros/horosvec/README.md). Il synthétise les points forts du système, ses faiblesses structurelles actuelles et propose des chantiers d'évolution prioritaires.

---

## 1. Forces de l'architecture actuelle

* **Discipline mémoire et gestion du GC** : L'utilisation d'une structure plate et pointer-free en mémoire vive ([hotPlane](file:///devhoros/horosvec/hotplane.go)) limite la charge de marquage du ramasse-miettes (GC) de Go à un coût de $O(1)$.
* **Stockage hybride vector-less** : Le couplage d'une base SQLite pour la métastructure et d'une arène vectorielle compacte `fp16` cartographiée hors-tas via `mmap` ([arena.go](file:///devhoros/horosvec/arena.go)) permet d'héberger des millions de vecteurs (jusqu'à 26,7 M validés sur le corpus HackerNews) sur des configurations mémoire modestes.
* **Double indexation de recherche** : La structure en deux étapes — recherche gloutonne sur le graphe de proximité (Vamana) au moyen de l'estimateur binaire RaBitQ ([rabitq.go](file:///devhoros/horosvec/rabitq.go)), puis ré-ordonnancement exact L2 sur les vrais vecteurs — absorbe le bruit de quantification tout en offrant une latence de recherche basse (p50 de ~7,8 ms sur SSD NVMe).

---

## 2. Limites critiques et faiblesses structurelles

* **Incohérence documentaire sur la rotation** : Le fichier [doc.go](file:///devhoros/horosvec/doc.go#L40) indique à tort que la rotation aléatoire est omise de l'implémentation, alors que [rotation.go](file:///devhoros/horosvec/rotation.go) applique désormais des rounds de transformée rapide de Walsh-Hadamard (FHT) pour redresser les distributions anisotropes (Sift).
* **Dépendance système directe (`syscall`)** : L'utilisation directe de `syscall.Mmap` et `syscall.Munmap` ([arena.go#L60](file:///devhoros/horosvec/arena.go#L60)) brise la portabilité multi-plateforme en empêchant la compilation native sur Windows sans adaptation, ce qui nuit à l'argument "pure Go, zéro dépendance".
* **Refus des écritures incrémentales à grande échelle** : Le mode arène (`ArenaPath` actif) fige l'index en lecture seule. La fonction [Insert](file:///devhoros/horosvec/horosvec.go#L1152) refuse d'insérer des vecteurs si une arène est configurée, limitant l'usage en temps réel dès que le corpus dépasse la RAM physique.
* **Noyau L2 non vectorisé** : La fonction [l2DistanceSquared](file:///devhoros/horosvec/vamana.go#L75) est le point chaud de l'étape de ré-ordonnancement exact. Écrite en Go pur, elle n'exploite pas les instructions SIMD matérielles du processeur (AVX2/AVX-512 ou NEON).
* **Absence de filtrage sémantique à la marche** : Le parcours glouton ne prend pas en charge de prédicats de filtrage de métadonnées, ce qui pose des problèmes pour les applications de RAG nécessitant des droits d'accès ou des filtrages temporels.

---

## 3. Levier du Sharding Temporel et Intégration Multicouche

Les arbitrages de conception récents s'orientent vers le patron d'architecture **`semantic_cpu`** :

* **Le Sharding Temporel comme alternative au monolithe** : La construction d'un index global de 26,7 M de vecteurs sur CPU se heurte à des contraintes de temps et de mémoire insurmontables (OOM). La partition par date (shards quotidiens ou mensuels, cf. [2026-07-10_minichrono_shard.md](file:///devhoros/horosvec-bench/audits/2026-07-10_minichrono_shard.md)) permet des builds rapides (65 s pour 11 600 vecteurs) et déplace la fusion lors du *retrieval* via un partitionnement logique temporel.
* **Recherche Hybride Multicouche** : La recherche combine l'appariement lexical exact de SQLite FTS5 (BM25) pour les mots-clés rares et le filtrage thématique préalable, avec une recherche de similarité dense (vecteurs) pour le sens général, optimisant ainsi les temps de réponse CPU globaux.

---

## 4. Feuille de route proactive recommandée

### Court Terme (Rigueur et Portabilité)
1. **Mise à jour documentaire** : Corriger [doc.go](file:///devhoros/horosvec/doc.go) pour décrire fidèlement la rotation de Hadamard active et la graine PCG persistée.
2. **Abstraction OS pour Mmap** : Déplacer les appels cartographie mémoire d'arène dans des fichiers séparés par tags de build (ex. `arena_unix.go` et `arena_windows.go`) et implémenter le support Windows via `CreateFileMapping` / `MapViewOfFile` pour restaurer la compilation universelle.

### Moyen Terme (Performance et Richesse fonctionnelle)
3. **Option codes-seuls Extended RaBitQ 6-8 bits** : Implémenter un mode à budget de bits accru (B=6 à 8) permettant de supprimer l'étape de ré-ordonnancement exact (L2) tout en maintenant un rappel élevé (0,99 vs 0,9945). Ce mode résoudrait la contrainte de l'insertion en temps réel à l'échelle (plus d'arène de rerank, 4× moins de mémoire).
4. **Intégration SIMD (Assembleur Go)** : Écrire des fichiers assembleurs Go `.s` pour le calcul de distance L2 et la conversion à la volée `float16ToFloat32` afin d'exploiter AVX2/NEON sans introduire de dépendance CGO.
5. **Pré-filtrage des métadonnées (Metadata Filtering)** : Introduire un callback de type `FilterPredicate(extID []byte) bool` dans le parcours [greedySearch](file:///devhoros/horosvec/vamana.go#L180) pour écarter les nœuds invalides lors de la marche sur le graphe.
