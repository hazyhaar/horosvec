package horosvec

import (
	"os"
	"path/filepath"
	"testing"
)

// TestArenaPublicRoundtrip vérifie l'écriture puis la relecture par l'API publique.
func TestArenaPublicRoundtrip(t *testing.T) {
	dim := 4
	path := filepath.Join(t.TempDir(), "a.arena")
	w, err := NewArenaWriter(path, dim)
	if err != nil {
		t.Fatalf("NewArenaWriter: %v", err)
	}
	vecs := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}
	for _, v := range vecs {
		if err := w.WriteVec(v); err != nil {
			t.Fatalf("WriteVec: %v", err)
		}
	}
	if w.Count() != int64(len(vecs)) {
		t.Fatalf("Count=%d, veut %d", w.Count(), len(vecs))
	}
	if err := w.Finalize(); err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	r, err := OpenArenaReader(path)
	if err != nil {
		t.Fatalf("OpenArenaReader: %v", err)
	}
	if r.Dim() != dim || r.Count() != int64(len(vecs)) {
		t.Fatalf("dim=%d count=%d", r.Dim(), r.Count())
	}
	dst := make([]float32, dim)
	for i, want := range vecs {
		if !r.VecInto(int64(i), dst) {
			t.Fatalf("VecInto(%d) faux", i)
		}
		for j := range want {
			if dst[j] != want[j] {
				t.Fatalf("vec[%d][%d]=%v veut %v", i, j, dst[j], want[j])
			}
		}
	}
}

// TestArenaPublicResume vérifie qu'après un checkpoint (Sync) et une interruption simulée
// (écriture de vecteurs supplémentaires NON confirmés au checkpoint, puis abandon du .tmp),
// ResumeArenaWriter tronque au count du checkpoint et reprend sans doublon ni trou.
func TestArenaPublicResume(t *testing.T) {
	dim := 3
	path := filepath.Join(t.TempDir(), "a.arena")

	// Phase 1 : écrire 2 vecteurs, checkpoint (Sync), puis 1 vecteur "partiel" non confirmé,
	// et ne PAS finaliser (simule un kill). Le .tmp survit.
	w, err := NewArenaWriter(path, dim)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.WriteVec([]float32{1, 1, 1}); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteVec([]float32{2, 2, 2}); err != nil {
		t.Fatal(err)
	}
	if err := w.Sync(); err != nil { // checkpoint durable = count 2
		t.Fatal(err)
	}
	if err := w.WriteVec([]float32{9, 9, 9}); err != nil { // écriture post-checkpoint à jeter
		t.Fatal(err)
	}
	// kill : on abandonne w sans finaliser (le .tmp reste sur disque).
	if _, err := os.Stat(path + ".tmp"); err != nil {
		t.Fatalf(".tmp attendu: %v", err)
	}

	// Phase 2 : reprise au checkpoint count=2 → le 3e vecteur partiel est tronqué.
	w2, err := ResumeArenaWriter(path, dim, 2)
	if err != nil {
		t.Fatalf("ResumeArenaWriter: %v", err)
	}
	if w2.Count() != 2 {
		t.Fatalf("count repris=%d, veut 2", w2.Count())
	}
	if err := w2.WriteVec([]float32{3, 3, 3}); err != nil {
		t.Fatal(err)
	}
	if err := w2.Finalize(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenArenaReader(path)
	if err != nil {
		t.Fatal(err)
	}
	if r.Count() != 3 {
		t.Fatalf("count final=%d, veut 3 (2 checkpoint + 1 repris, le partiel jeté)", r.Count())
	}
	dst := make([]float32, dim)
	want := [][]float32{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}
	for i, wv := range want {
		r.VecInto(int64(i), dst)
		for j := range wv {
			if dst[j] != wv[j] {
				t.Fatalf("vec[%d][%d]=%v veut %v (le vecteur partiel 9 aurait dû être tronqué)", i, j, dst[j], wv[j])
			}
		}
	}
}
