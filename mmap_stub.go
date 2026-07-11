//go:build !unix

package horosvec

import "errors"

// errMmapUnsupported : le mode arène exige la cartographie mémoire, portée Unix
// uniquement (cf. doc.go §Known limits). Posture compile-only assumée sur les
// autres plateformes : le paquet compile, ouvrir ou importer une arène échoue
// fail-loud à l'exécution ; le mode DB-blob n'est pas concerné.
var errMmapUnsupported = errors.New("horosvec: arena mmap is not supported on this platform (unix only)")

func mmapRO(fd uintptr, size int) ([]byte, error) {
	_ = fd
	_ = size
	return nil, errMmapUnsupported
}

func munmapRO(data []byte) error {
	_ = data
	return errMmapUnsupported
}
