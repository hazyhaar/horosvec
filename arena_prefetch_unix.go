//go:build unix

package horosvec

import (
	"os"
	"sync"
	"syscall"
	"unsafe"
)

// prefetchRerank annonce au noyau, EN UNE FOIS, les plages de l'arène que la
// boucle de re-classement va lire.
//
// Motif. Les identifiants des candidats à re-classer sont tous connus avant la
// première lecture, mais la boucle les lisait un par un : chaque accès à une
// page absente provoquait un défaut de page servi de façon synchrone, et le
// processus attendait le disque autant de fois qu'il y avait de candidats. Sur
// un support capable de servir des dizaines d'accès simultanés, cette
// sérialisation n'était imposée que par l'ordre du code.
//
// Le conseil groupé produit deux effets distincts, tous deux mesurés :
// le noyau émet les lectures en parallèle au lieu de les enchaîner, et il les
// borne aux plages demandées au lieu d'étendre chaque défaut à sa fenêtre de
// lecture anticipée, qui vaut jusqu'à 128 Kio pour 1 Kio utile.
//
// Le conseil ne change RIEN au résultat : c'est une indication d'accès, aucune
// donnée n'est interprétée. Cette propriété est gardée par
// TestPrefetchNeChangePasLesResultats.
//
// Mesures — index HackerNews, 26,7 M vecteurs, dimension 512, arène de 27,3 Go,
// SSD NVMe, empreinte résidente de 34 Go, index et arène plus grands que la
// mémoire disponible :
//
//	                        sans conseil   avec conseil
//	latence médiane           21,5 ms        2,0 ms      (10,8×)
//	à 500 candidats           40,2 ms        3,4 ms      (11,8×)
//	défauts de page majeurs   17 949         0
//	volume lu (200 requêtes)  2 311 Mo       107 Mo
//	débit à 8 requêtes ||     261 req/s      1 940 req/s (7,4×)
//	centile 99 à 8 req. ||    45,2 ms        6,4 ms
//
// Contrepartie, unique cas défavorable trouvé : lorsque l'arène tient
// intégralement dans le cache de pages, les appels système ne servent plus à
// rien et coûtent 4,4 % (710 → 741 µs sur un index de 300 000 vecteurs au
// second passage). D'où le réglage Config.PrefetchRerank, qui permet de les
// supprimer sur un déploiement dont l'arène est durablement résidente.
//
// Écarté après mesure : la fusion préalable des plages adjacentes, censée
// réduire le nombre d'appels système, ne rend rien (798 µs contre 771 sans
// elle) — les candidats sont trop dispersés pour partager des pages.
func (a *arena) prefetchRerank(candidates []searchCandidate) {
	if len(candidates) == 0 {
		return
	}
	octets := a.dim * 2
	plages := plagesPool.Get().(*[]syscall.Iovec)
	*plages = (*plages)[:0]
	defer plagesPool.Put(plages)

	for i := range candidates {
		id := candidates[i].nodeID
		if id < 0 || id >= a.count {
			// Hors couverture : le candidat sera servi par la voie de repli, il n'y
			// a pas de plage d'arène à annoncer.
			continue
		}
		off := arenaHeaderSize + int(id)*octets
		debut := off &^ (pageTailleOctets - 1)
		fin := (off + octets + pageTailleOctets - 1) &^ (pageTailleOctets - 1)
		if fin > len(a.data) {
			fin = len(a.data)
		}
		*plages = append(*plages, syscall.Iovec{
			Base: &a.data[debut],
			Len:  uint64(fin - debut),
		})
	}
	if len(*plages) == 0 {
		return
	}

	// Un seul appel système pour tout le lot quand le noyau le permet. Mesuré :
	// les appels individuels pesaient 15,2 % du temps processeur une fois
	// l'attente disque supprimée — un par candidat, cent vingt-huit par requête.
	if groupePlages(*plages) {
		return
	}
	// Repli : noyau sans appel groupé, ou refus. Le conseil reste utile plage par
	// plage ; seul son coût d'émission augmente.
	for _, v := range *plages {
		_ = syscall.Madvise(unsafe.Slice(v.Base, v.Len), syscall.MADV_WILLNEED)
	}
}

// plagesPool réutilise les tableaux de plages d'une requête à l'autre : le lot a
// une taille bornée par le nombre de candidats, et le chemin est chaud.
var plagesPool = sync.Pool{New: func() any {
	s := make([]syscall.Iovec, 0, 512)
	return &s
}}

// pidfdCourant est le descripteur du processus lui-même, ouvert une seule fois.
// Zéro signifie « pas encore tenté », valeur négative « indisponible ».
var (
	pidfdUne     sync.Once
	pidfdCourant int
)

// groupePlages annonce tout le lot en un seul appel système (process_madvise).
// Rend false si le noyau ne le permet pas, auquel cas l'appelant se replie sur
// les appels individuels.
func groupePlages(plages []syscall.Iovec) bool {
	pidfdUne.Do(func() {
		fd, _, errno := syscall.Syscall(unix_SYS_pidfd_open, uintptr(os.Getpid()), 0, 0)
		if errno != 0 {
			pidfdCourant = -1
			return
		}
		pidfdCourant = int(fd)
	})
	if pidfdCourant < 0 {
		return false
	}
	_, _, errno := syscall.Syscall6(unix_SYS_process_madvise,
		uintptr(pidfdCourant),
		uintptr(unsafe.Pointer(&plages[0])),
		uintptr(len(plages)),
		uintptr(syscall.MADV_WILLNEED),
		0, 0)
	return errno == 0
}

// Numéros d'appel système, stables sur linux/amd64 et linux/arm64.
const (
	unix_SYS_pidfd_open      = 434
	unix_SYS_process_madvise = 440
)
