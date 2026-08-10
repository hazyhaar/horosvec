//go:build !unix

package horosvec

// IOStats : voir iostats_unix.go. Hors Unix, les compteurs ne sont pas
// disponibles et les relevés sont vides — l'appelant obtient des zéros, jamais
// une erreur, pour que le code de mesure reste identique sur toute plateforme.
type IOStats struct {
	LecturesDisque uint64
	OctetsAppels   uint64
	DefautsMajeurs int64
	DefautsMineurs int64
	ResidentOctets uint64
}

// ReadIOStats rend un relevé vide hors Unix.
func ReadIOStats() IOStats { return IOStats{} }

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

// String rend un résumé d'une ligne.
func (s IOStats) String() string {
	return "compteurs d'entrees-sorties indisponibles sur cette plateforme"
}
