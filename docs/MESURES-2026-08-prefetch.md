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
