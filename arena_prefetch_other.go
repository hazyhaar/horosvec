//go:build !unix

package horosvec

// prefetchRerank est sans effet hors Unix : le mode arène y exige déjà la
// cartographie mémoire, indisponible sur ces plateformes (cf. mmap_stub.go).
// La méthode existe pour que le chemin d'appel reste unique et compile partout.
func (a *arena) prefetchRerank(candidates []searchCandidate) {
	_ = candidates
}
