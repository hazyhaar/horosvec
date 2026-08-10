package horosvec

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestIOStatsCompteLesLectures vérifie que le relevé n'est pas un instrument
// mort : une lecture disque réelle doit s'y voir. Sans ce contrôle, une
// structure rendant toujours zéro passerait pour un banc parfaitement sain —
// c'est exactement le mode de défaillance qui a masqué le défaut de
// préchargement pendant plusieurs campagnes.
func TestIOStatsCompteLesLectures(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("compteurs releves depuis /proc, propres a Linux")
	}
	chemin := filepath.Join(t.TempDir(), "bloc")
	donnees := make([]byte, 8<<20)
	for i := range donnees {
		donnees[i] = byte(i)
	}
	if err := os.WriteFile(chemin, donnees, 0o600); err != nil {
		t.Fatal(err)
	}

	avant := ReadIOStats()
	lu, err := os.ReadFile(chemin)
	if err != nil {
		t.Fatal(err)
	}
	delta := ReadIOStats().Sub(avant)

	if len(lu) != len(donnees) {
		t.Fatalf("lecture partielle : %d octets", len(lu))
	}
	// rchar compte les octets passés par les appels système, cache compris : il
	// doit bouger même quand le fichier vient d'être écrit et reste résident.
	if delta.OctetsAppels < uint64(len(donnees)) {
		t.Fatalf("octets d'appels systeme non comptes : %d pour %d lus", delta.OctetsAppels, len(donnees))
	}
	if delta.ResidentOctets == 0 {
		t.Fatal("empreinte residente non relevee")
	}
	t.Logf("delta releve : %s", delta)
}
