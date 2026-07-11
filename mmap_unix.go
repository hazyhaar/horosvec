//go:build unix

package horosvec

import "syscall"

// mmapRO cartographie fd en lecture seule MAP_SHARED : les pages restent dans le
// page cache noyau, jamais copiées dans le tas Go. L'fd peut être refermé après
// l'appel : la cartographie survit à la fermeture du descripteur.
func mmapRO(fd uintptr, size int) ([]byte, error) {
	return syscall.Mmap(int(fd), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
}

// munmapRO libère une cartographie produite par mmapRO.
func munmapRO(data []byte) error {
	return syscall.Munmap(data)
}
