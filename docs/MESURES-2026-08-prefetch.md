# Préchargement du lot de re-classement — mesures et choix

Journée du 10 août 2026. Machine : Intel Core i9-14900K, 62 Gio de mémoire vive,
trois SSD NVMe. Index de référence : HackerNews, **26 691 317 vecteurs**,
dimension 512, base de 19,3 Go et arène demi-précision de 27,3 Go, tous deux sur
un support distinct de celui du code.

Toutes les mesures ont été conduites sous plafond mémoire imposé par groupe de
contrôle (`systemd-run --scope -p MemoryMax=… -p MemorySwapMax=0`), de sorte
qu'un dépassement tue le seul processus de mesure et jamais la machine. Un
premier tir plafonné à 24 Go a effectivement été tué : l'ouverture de cet index
demande **34,3 Go résidents**, dont 27,2 Go de tas, pour une minute de
chargement.

## 1. Le défaut

Les identifiants des candidats à re-classer sont tous connus avant la première
lecture, mais la boucle les lisait un par un. Chaque accès à une page absente
provoquait un défaut de page servi de façon synchrone : le processus attendait le
disque autant de fois qu'il y avait de candidats, sur un support capable d'en
servir des dizaines simultanément.

Le profil processeur l'a établi sans ambiguïté : sur 9,92 s de mesure,
l'échantillonneur ne capte que **1,64 s de temps processeur, soit 16,5 %**. Les
cinq sixièmes du temps de recherche sont de l'attente.

Répartition du sixième réellement consommé :

| Poste | part |
| :--- | ---: |
| Lecture des entiers 16 bits de l'arène | 40,2 % |
| Lecture des codes et normes des voisins | 26,8 % |
| Estimation de distance approchée | 13,4 % |
| Bitset des nœuds visités | 6,1 % |
| Remise à zéro de ce bitset | 5,5 % |

Compteurs d'entrées-sorties, 500 requêtes en 10 s : **6 138 Mo lus sur le
disque, soit 12,3 Mo par requête** pour rapporter dix voisins, et 47 991 défauts
de page majeurs, soit 96 par requête à environ 200 µs chacun.

La linéarité confirme la cause. En faisant varier le nombre de candidats
re-classés, à faisceau constant :

| Candidats re-classés | latence médiane | défauts majeurs par requête |
| ---: | ---: | ---: |
| 32 | 7,60 ms | 26 |
| 64 | 13,54 ms | 50 |
| 128 | 22,41 ms | 96 |

Un accès disque par candidat, environ 0,17 ms chacun. La marche dans le graphe,
avec ses 5 712 nœuds visités par requête, ne coûte qu'environ trois
millisecondes : elle lit les codes, qui résident en mémoire, et ne touche jamais
l'arène.

## 2. Le correctif retenu

Annoncer au noyau, en une fois et avant la boucle, les plages que le
re-classement va lire (`MADV_WILLNEED`). Deux effets distincts : les lectures
sont émises en parallèle au lieu d'être enchaînées, et elles sont bornées aux
plages demandées au lieu d'étendre chaque défaut à sa fenêtre de lecture
anticipée, qui vaut jusqu'à 128 Kio pour 1 Kio utile.

| Mesure, 26,7 M vecteurs | sans | avec | rapport |
| :--- | ---: | ---: | ---: |
| Latence médiane, 128 candidats | 21,5 ms | **2,0 ms** | 10,8× |
| Latence médiane, 500 candidats | 40,2 ms | 3,4 ms | 11,8× |
| Latence médiane, 32 candidats | 8,3 ms | 2,5 ms | 3,4× |
| Défauts de page majeurs | 17 949 | **0** | — |
| Volume lu, 200 requêtes | 2 311 Mo | 107 Mo | 21× |
| Débit, 8 recherches simultanées | 261 req/s | **1 940 req/s** | 7,4× |
| Centile 99, 8 recherches simultanées | 45,2 ms | 6,4 ms | 7,1× |

Sémantique intacte, vérifiée au sol : sur 200 requêtes, les trois régimes testés
rendent un top-10 **identique**, même ordre et mêmes distances.

## 3. Ce qui a été essayé puis écarté, et pourquoi

**Déclarer l'accès aléatoire sur toute la projection** (`MADV_RANDOM` à
l'ouverture). Rend 1,45× seul, mais **rien de plus** une fois le conseil groupé
en place (1,983 contre 2,003 ms) — ce dernier borne déjà la lecture à la plage
utile. Et il coûte cher ailleurs : sur un parcours séquentiel à froid de cinq
gigaoctets, mesuré sur des régions vierges, **1 700 Mo/s sans le conseil contre
164 Mo/s avec, dix fois plus lent**. Les passes de construction, d'import et de
calcul du médoïde l'auraient payé. Écarté sans regret.

**Paralléliser par goroutines** plutôt que par conseil au noyau. Rend 3,6×
seulement : la lecture anticipée du noyau s'applique toujours, le volume lu reste
identique (2 456 Mo contre 2 311). Nettement inférieur au conseil groupé.

**Fusionner les plages adjacentes** avant l'appel, pour réduire le nombre
d'appels système. Ne rend rien : 798 µs contre 771 sans fusion. Les candidats
sont trop dispersés pour partager des pages. Complexité évitée.

**Réduire le nombre de candidats re-classés.** Mesuré à 1,66× pour 2,3 % de
recouvrement perdu, ce compromis était réel tant que le disque dominait. Il
disparaît avec le préchargement : re-classer 500 candidats au lieu de 128 ne
coûte plus que 0,6 ms au lieu de 16. On peut désormais **augmenter** la qualité
tout en restant sept fois plus rapide qu'avant.

**Réordonner l'arène pour colocaliser les voisins.** Le potentiel a été chiffré
en traçant les nœuds touchés : les 128 candidats s'étalent sur 160 pages là où,
contigus, ils en occuperaient 32 — un facteur 5. Mais ce gisement portait sur des
pages qui ne coûtent plus rien. Un chantier de réécriture de vingt-sept
gigaoctets n'a plus de justification.

**Le bloc de nœud unifié**, regroupant graphe, code quantifié et vecteur brut
dans une structure contiguë. Écarté sur analyse : la marche visite 5 712 nœuds
par requête et n'a besoin que d'environ quatre-vingts octets pour chacun ; un
bloc unifié lui en ferait toucher 1 488, dont 1 024 inutiles — dix-huit fois plus
de volume, et le basculement de la marche depuis la mémoire vers le disque dès
que l'index dépasse la mémoire disponible. La séparation actuelle n'est pas un
défaut : c'est ce qui permet à la marche de ne rien coûter.

**La pré-lecture matérielle du processeur** (`__builtin_prefetch`), proposée pour
masquer la latence du SSD. Mesurée en C, région de 128 Mo entièrement froide :
2 000 instructions émises en 0,010 ms, **zéro page chargée**, contre 32 000 pages
résidentes après des accès réels à 56 µs chacun. Cette instruction est
architecturalement incapable de provoquer un défaut de page. Elle masque des
latences de dizaines de nanosecondes, jamais une entrée-sortie.

## 4. La contrepartie assumée

Lorsque l'arène tient intégralement dans le cache de pages, les appels système ne
servent plus à rien et coûtent **4,4 %** — 741 µs contre 710 sur un index de
300 000 vecteurs au second passage, toutes pages résidentes.

C'est le seul cas défavorable trouvé. À froid, même ce petit index gagne :
moyenne de 803 µs contre 1 323, et centile 99 divisé par cinq et demi, de 7,7 à
1,4 ms. Le réglage `Config.PrefetchRerank` reste donc **actif par défaut**, et
permet de supprimer les appels sur un déploiement dont l'arène est durablement
résidente.

## 5. Ce que l'épisode dit du dispositif de mesure

Ce défaut valait un facteur huit et demi. Il a traversé plusieurs campagnes de
mesure, dont une qui déclare l'échelle de 26,7 millions « éprouvée », sans être
vu — parce que ces campagnes chronométraient la recherche sans jamais demander au
système d'où venait ce temps. Trois compteurs l'auraient révélé au premier tir.

D'où l'ajout de `IOStats` (`iostats_unix.go`) : lectures disque effectives,
octets passés par les appels système, défauts de page majeurs et mineurs,
empreinte résidente. Une mesure de performance de ce module qui ne rapporte
qu'une durée ne permet pas de distinguer un calcul lent d'une attente disque, et
laisse donc passer la classe de défauts la plus coûteuse. `IOStats` est
elle-même gardée par un test de vivacité : un instrument qui rendrait toujours
zéro passerait autrement pour un banc parfaitement sain.

## 6. Note sur les deux commits qui précèdent

Les commits `ceb6425` et `24a4333` de la même journée réécrivent deux noyaux de
calcul et annoncent un gain cumulé de 1,37×, établi à cinquante mille vecteurs.
**Ce gain ne se matérialise pas à l'échelle réelle** : mesuré sur les 26,7
millions, avant et après, l'écart va de −1,7 % à +0,5 % selon le tir et change de
signe — du bruit. À cette échelle, la recherche coûtait vingt millisecondes dont
dix-neuf d'attente disque, et les deux noyaux optimisaient 14 % d'un sixième du
temps.

Ce que ces commits gardent d'acquis reste réel mais mineur : rappel inchangé,
empreinte de l'état de recherche ramenée de 128 Kio à 320 octets par requête,
préparation par requête six fois moins chère, code plus simple. Leurs messages
survendent le résultat ; le présent document vaut rectification.

## 7. Après le préchargement : le régime a changé, les cibles aussi

Le préchargement ayant supprimé l'attente disque, la part du temps réellement
consommée par le processeur passe de **16,5 % à 87,8 %**. Le profil devient donc
pour la première fois représentatif de ce que fait la recherche, et il désigne
des postes qu'aucun des plans précédents n'avait ciblés :

| Poste | part du CPU | nature |
| :--- | ---: | :--- |
| Lecture code + normes des voisins | **31,4 %** | trois indexations, aucun calcul |
| Appels système de préchargement | **15,7 %** | un par candidat, 128 par requête |
| Recomposition d'entiers 64 bits | 12,4 % | décodage des codes dans la marche |
| Comptage de bits (marche) | 20,1 % cum | calcul |
| Distance exacte (re-classement) | 10,0 % cum | calcul |
| Bitset des visités et sa remise à zéro | 8,4 % | — |

Les deux premiers postes, près de la moitié du temps, ne calculent rien. C'est ce
constat qui a écarté définitivement l'idée d'écrire des noyaux en C : la
transpilation ne produit que du Go scalaire, et aucun des postes dominants n'est
un problème de calcul.

### Trois changements de structure

**Entrelacement du plan chaud.** Le code, la norme carrée et la norme L1 étaient
rangés dans trois tranches distinctes — 1,7 Go et deux fois 213 Mo à 26,7 M
nœuds. Ces trois grandeurs sont toujours lues ensemble, une fois par voisin
visité, soit 5 712 fois par requête : chaque voisin coûtait trois accès dans
trois régions éloignées. Elles sont désormais entrelacées à pas fixe.

**Alignement des codes sur huit octets.** Le pas d'entrelacement arrondit la
taille du code au multiple de huit supérieur, ce qui permet à la boucle de
comptage de bits de lire des mots de 64 bits sans les recomposer octet par
octet.

**Regroupement des conseils de préchargement.** `process_madvise(2)` accepte un
vecteur de plages et annonce tout le lot en **un seul appel système** au lieu de
cent vingt-huit. Vérifié utilisable sur le processus lui-même sans privilège
particulier ; repli automatique sur les appels individuels si le noyau refuse.

### Ce que ces trois changements rendent, mesuré

L'alignement tient sa promesse : la recomposition d'entiers passe de **12,4 % à
2,3 %** du temps processeur. Le regroupement des appels fait reculer leur poste
de **15,2 % à 12,7 %**, et le temps processeur total de **3 090 à 2 830 ms** pour
deux mille requêtes, soit 8,4 % de moins.

L'entrelacement, en revanche, ne rend que **1,5 à 3 %** en latence, très loin des
31 % visés. La raison est mesurable : l'enregistrement fusionné fait 80 octets et
chevauche donc systématiquement deux lignes de cache de 64. Trois accès dans
trois régions ont été remplacés par deux lignes dans une seule — le progrès est
réel mais modeste. Le poste fusionné reste le premier du profil, à 47 %.

Bilan honnête : environ 8 % de temps processeur gagné, un poste de calcul
supprimé, et une structure plus simple — trois tranches devenues une. L'espoir
de viser 60 % du temps ne s'est pas matérialisé, parce que le coût dominant
n'est ni le calcul ni le nombre d'accès, mais la latence mémoire d'accès
aléatoires dans deux gigaoctets, qu'aucune réorganisation locale ne supprime.

### Ce qui reste, et ce qu'il ne faut plus chercher

Le seul bloc encore légitimement candidat à la vectorisation est la distance
exacte du re-classement : la conversion demi-précision dispose d'une instruction
matérielle dédiée, présente sur ce processeur, et la distance qui suit se
vectorise sans permutation ni décalage — les deux primitives dont la mesure a
établi qu'elles coûtent quatre-vingt-dix cycles au lieu d'un. Mais le paquet
vectoriel de Go **n'expose aucun type demi-précision**, et le poste ne pèse que
10 % : le gain plafonnerait à huit pour cent, pour de l'assembleur écrit à la
main.

## 8. Deux hypothèses invalidées par la mesure

### L'agencement mémoire du plan chaud n'a aucune importance

L'entrelacement des trois tranches ne rendant que 1,5 à 3 %, j'avais attribué ce
faible rendement au chevauchement de lignes de cache : l'enregistrement fusionné
fait 80 octets et enjambe donc deux lignes de 64. Un micro-banc reproduisant le
motif d'accès — lecture aléatoire du code et des deux normes, sur une structure
bien plus grande que le cache — compare quatre agencements :

| Agencement | empreinte à 26,7 M | ns par accès |
| :--- | ---: | ---: |
| Trois tranches séparées (avant) | 2,14 Go | 62,4 à 67,6 |
| Fusionné à pas 80 (actuel) | 2,14 Go | 61,9 à 63,6 |
| Fusionné à pas 128, aligné | 3,42 Go | 62,3 à 63,9 |
| Code à pas 64 + normes denses | 2,14 Go | 62,3 à 64,0 |

**Les quatre sont équivalents**, les écarts changeant de signe d'un tour à
l'autre. L'agencement aligné sur 128, qui devrait gagner si les lignes de cache
étaient en cause, ne gagne pas et coûte 60 % de mémoire en plus. L'hypothèse est
donc fausse.

La vraie cause est incompressible : soixante nanosecondes, c'est la latence d'un
accès aléatoire en mémoire principale sur cette machine. Que les données soient
dans une, deux ou trois régions ne change rien — le processeur attend la DRAM une
fois par nœud quoi qu'il arrive. Regrouper trois accès en un n'aide pas quand
aucun des trois n'était prévisible.

L'entrelacement est conservé parce qu'il ne coûte rien, qu'il porte l'alignement
des codes (qui, lui, rend dix points) et qu'il remplace trois tranches par une —
non parce qu'il accélère quoi que ce soit.

### Les 5 712 visites par requête ne sont pas du gaspillage

Anatomie de la marche, mesurée sur 320 requêtes : 138,6 nœuds dépilés du tas,
7 761,7 voisins visités, 1 111 déjà vus, et **seulement 634,8 visites retenues,
soit 8,2 %**. Neuf visites sur dix semblent perdues.

Elles ne le sont pas. En faisant varier la largeur de faisceau :

| Largeur | visites | latence | recouvrement du top-10 contre 256 |
| ---: | ---: | ---: | ---: |
| 32 | 2 554 | 213 µs | 0,374 |
| 64 | 4 497 | 360 µs | 0,571 |
| 128 (défaut) | 7 762 | 609 µs | 0,783 |
| 256 | 13 527 | 1 163 µs | référence |

Les visites achètent de la qualité presque linéairement : diviser le faisceau par
deux coûte plus de vingt points de recouvrement. Les 8,2 % retenus sont le
fonctionnement normal d'une marche gloutonne, qui doit évaluer pour écarter.

**Mais ce tableau dit autre chose.** À la largeur par défaut, la recherche ne
retrouve que 78 % de ce que trouve un faisceau deux fois plus large : sur un
graphe de proximité bien construit, elle devrait avoir convergé. Que la qualité
continue de monter aussi franchement indique que le graphe force la marche à
visiter huit mille nœuds pour en retenir six cents.

Le seul levier restant sur le poste dominant n'est donc ni la disposition des
données ni le réglage de la recherche, mais la **construction du graphe** :
élagage, nombre de passes, degré. C'est le seul chemin qui réduirait les visites
sans payer en qualité. Il n'a pas été mesuré, et il suppose une reconstruction
de l'index.

## 9. Quantification multi-bits : le levier qui reste

Le rappel de l'index de référence, mesuré pour la première fois **en absolu**
contre une force brute exacte, vaut **0,470** au réglage par défaut. Les
campagnes antérieures comparaient des réglages entre eux et ne pouvaient pas le
voir. Deux causes ont été séparées par la mesure :

| Réglage | rappel@10 absolu |
| :--- | ---: |
| ef=128, re-classement borné à 500 (défaut) | 0,470 |
| ef=2048, re-classement borné à 500 | 0,758 |
| ef=2048, re-classement aligné à 2048 | 0,845 |
| ef=4096, re-classement aligné à 4096 | 0,912 |

**Le graphe n'est pas en cause** : exploré assez largement, il rend neuf voisins
sur dix. Ce qui limite est l'estimateur à un bit par dimension, qui sélectionne
mal, et le plafond de re-classement qui empêchait de rattraper.

### Qualité de sélection selon la largeur du code

Mesurée hors moteur, sur 100 000 vecteurs réels en dimension 512 : part des
vrais dix plus proches voisins présents dans les 128 meilleurs candidats classés
par l'estimateur seul.

| Bits | octets par code | rappel de la présélection |
| ---: | ---: | ---: |
| 1 | 64 | 0,582 |
| 2 | 128 | 0,774 |
| 3 | 192 | 0,876 |
| 4 | 256 | 0,888 |
| 8 | 512 | 0,902 |

### Mesure de bout en bout, à travers le graphe

Index réellement construit à chaque largeur, corpus en grappes, rappel contre
vérité terrain exacte :

| Bits | rappel@10 | écart |
| ---: | ---: | ---: |
| 1 | 0,7625 | référence |
| 2 | 0,8175 | +0,055 |
| 3 | **0,9725** | **+0,210** |
| 4 | 1,0000 | +0,238 |

### Ce que cela coûte

L'évaluation croise les B plans du code avec les cinq plans de la requête : le
coût croît linéairement, d'environ cinquante nanosecondes par bit.

| Bits | ns par distance | mémoire des codes à 26,7 M |
| ---: | ---: | ---: |
| 1 | 27,6 | 1,7 Go |
| 2 | 106 | 3,4 Go |
| 3 | 155 | 5,1 Go |
| 4 | 203 | 6,8 Go |

À un bit, un chemin spécialisé est conservé : la forme généralisée rend la même
valeur — garde par test — mais coûte 56 ns contre 27,6. Les index existants ne
paient donc rien pour une généralisation dont ils ne se servent pas.

### Choix

`Config.CodeBits`, de 1 à 8, **1 par défaut** : le comportement d'un index
existant est strictement inchangé, et la valeur est celle de la CONSTRUCTION,
persistée en métadonnée. Un index rouvert relit la largeur de ses propres codes,
jamais celle que réclame la configuration — sans quoi les codes seraient
interprétés au mauvais format.

Les plans sont rangés du poids fort au poids faible, si bien que les premiers
(dim+7)/8 octets d'un code multi-bits **sont** le code à un bit du même vecteur.
Un lecteur qui ignore les plans suivants lit l'ancien format sans le savoir, et
l'affinage incrémental décrit par la littérature — trancher sur les bits de poids
fort, ne lire la suite que pour les candidats ambigus — reste ouvert.

Trois bits paraissent le point d'équilibre : vingt et un points de rappel pour
trois fois la mémoire des codes et cinq fois le coût d'évaluation. Au-delà de
quatre, la précision est bornée par la quantification de la REQUÊTE, elle-même
sur cinq plans : élargir le code sans élargir la requête ne rapporte plus rien,
ce qu'un test grave explicitement.

### Sur le recours au C, définitivement

L'implémentation de référence, [RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library),
est en C++ — donc hors de portée d'une chaîne de transpilation qui traite le C.
Le seul transpileur C++ vers Go trouvé produit du code utilisant cgo, ce que les
invariants du module interdisent. Cela n'a aucune importance : la veille interne
du 10 juillet, en lisant le papier intégralement, avait déjà établi que
l'algorithme est de l'arithmétique scalaire pure et que le SIMD n'y est qu'une
optimisation de vitesse, jamais une condition de correction. Ce fichier le
confirme : une centaine de lignes de Go suffisent, et rendent vingt et un points
de rappel.

## 10. RECTIFICATION — le rappel de 0,470 ne concernait pas l'index de référence

La section 9 affirme que « le rappel de l'index de référence vaut 0,470 au
réglage par défaut ». **C'est faux.** Ce chiffre a été mesuré sur l'index de
300 000 vecteurs et attribué, sans vérification, à l'index de 26,7 millions.
La mesure directe, faite depuis sur l'index de référence lui-même avec les mêmes
requêtes et la même vérité terrain exacte, donne :

| Index de 26,7 M, mêmes requêtes | rappel@10 absolu | latence médiane |
| :--- | ---: | ---: |
| 1 bit (existant) | **0,9733** | 1,788 ms |
| 3 bits (ré-encodé) | 0,9817 | 2,589 ms |

Le rappel réel est donc de 0,973, non de 0,470.

### Pourquoi les deux chiffres diffèrent

Les deux corpus ne sont pas deux tailles du même jeu, mais deux natures :

| Corpus | valeur absolue moyenne | maximum |
| :--- | ---: | ---: |
| `hnbook.arena` (26,7 M) | 0,034 | 0,23 |
| `prefix300000.arena` (300 k) | 0,500 | 1,00 |

Le premier porte des plongements normalisés — 0,034 est la valeur attendue pour
des vecteurs de norme unitaire en dimension 512. Le second porte des vecteurs
uniformes aléatoires, sans structure : le cas pathologique de la recherche
approchée, où la concentration des distances rend tous les points presque
équidistants. Un rappel de 0,470 y est un fait de géométrie et non un défaut du
moteur ; la moyenne des normes L1 des deux index, 15,5 contre 117,9, disait déjà
que les corpus n'avaient rien à voir.

### Conséquence sur la quantification multi-bits

Sur le corpus réel, le passage à trois bits rapporte **0,8 point de rappel pour
45 % de latence en plus** et deux gigaoctets de mémoire résidente
supplémentaires. **Il n'est pas rentable ici**, et la recommandation de la
section 9 — trois bits par défaut pour tout nouvel index — est retirée.

Ce que le dispositif garde de valeur : il reste correct, testé, désactivé par
défaut, et il rend beaucoup là où l'estimateur à un bit décroche vraiment, soit
sur des corpus peu structurés ou de forte dimension intrinsèque. Le choix se
mesure corpus par corpus, jamais par défaut. Le ré-encodage d'un index existant
prend sept minutes pour 26,7 millions de vecteurs, sans reconstruire le graphe,
ce qui rend l'essai bon marché quand un doute existe.

### Ce que cette erreur dit de la méthode

Toute la journée a consisté à refuser les conclusions non mesurées, et cette
erreur est de la même famille que celles écartées : un chiffre obtenu sur un
objet, transporté vers un autre parce qu'ils portaient le même nom de projet. La
garde qui manquait est triviale — vérifier la nature du corpus avant de comparer
deux mesures — et elle aurait coûté une commande.
