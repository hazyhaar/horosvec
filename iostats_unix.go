//go:build unix

package horosvec

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
)

// IOStats rapporte d'où vient le temps d'une mesure, et non seulement combien il
// y en a.
//
// Motif. Un banc qui ne rapporte que des durées ne distingue pas un calcul lent
// d'une attente disque, et laisse donc passer la classe de défauts la plus
// coûteuse. Le préchargement du lot de re-classement (cf. arena_prefetch_unix.go)
// valait un facteur 8,5 à 11,8 et a traversé plusieurs campagnes de mesure sans
// être vu, parce que ces campagnes chronométraient la recherche sans jamais
// demander au système ce qu'elle faisait faire au disque. Trois compteurs
// l'auraient révélé au premier tir.
type IOStats struct {
	// LecturesDisque : octets réellement transférés depuis le périphérique.
	LecturesDisque uint64
	// OctetsAppels : octets passés par les appels système de lecture, cache
	// compris. L'écart avec le champ précédent mesure ce que le cache a évité.
	OctetsAppels uint64
	// DefautsMajeurs : défauts de page servis par une lecture disque. C'est
	// l'oracle de l'attente : chacun coûte une centaine de microsecondes.
	DefautsMajeurs int64
	// DefautsMineurs : défauts de page servis depuis le cache, sans disque.
	DefautsMineurs int64
	// ResidentOctets : empreinte mémoire résidente du processus.
	ResidentOctets uint64
}

// ReadIOStats relève les compteurs du processus courant. Portée Unix ; sur les
// autres plateformes, la variante neutre rend une structure vide.
//
// Usage : relever avant et après la phase mesurée, puis soustraire par Sub. Les
// compteurs sont ceux du PROCESSUS entier, non d'une goroutine : isoler la phase
// à mesurer est à la charge de l'appelant.
func ReadIOStats() IOStats {
	var s IOStats
	if b, err := os.ReadFile("/proc/self/io"); err == nil {
		for _, l := range strings.Split(string(b), "\n") {
			cle, val, ok := strings.Cut(l, ": ")
			if !ok {
				continue
			}
			v, err := strconv.ParseUint(val, 10, 64)
			if err != nil {
				continue
			}
			switch cle {
			case "read_bytes":
				s.LecturesDisque = v
			case "rchar":
				s.OctetsAppels = v
			}
		}
	}
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err == nil {
		s.DefautsMineurs = ru.Minflt
		s.DefautsMajeurs = ru.Majflt
	}
	if b, err := os.ReadFile("/proc/self/statm"); err == nil {
		champs := strings.Fields(string(b))
		if len(champs) >= 2 {
			if pages, err := strconv.ParseUint(champs[1], 10, 64); err == nil {
				s.ResidentOctets = pages * uint64(os.Getpagesize())
			}
		}
	}
	return s
}

// Sub rend la consommation entre deux relevés.
func (s IOStats) Sub(avant IOStats) IOStats {
	return IOStats{
		LecturesDisque: s.LecturesDisque - avant.LecturesDisque,
		OctetsAppels:   s.OctetsAppels - avant.OctetsAppels,
		DefautsMajeurs: s.DefautsMajeurs - avant.DefautsMajeurs,
		DefautsMineurs: s.DefautsMineurs - avant.DefautsMineurs,
		ResidentOctets: s.ResidentOctets,
	}
}

// String rend un résumé d'une ligne, destiné à accompagner une latence.
func (s IOStats) String() string {
	return fmt.Sprintf("disque=%.1f Mo appels=%.1f Mo defauts_majeurs=%d mineurs=%d resident=%.2f Go",
		float64(s.LecturesDisque)/1e6, float64(s.OctetsAppels)/1e6,
		s.DefautsMajeurs, s.DefautsMineurs, float64(s.ResidentOctets)/1e9)
}
