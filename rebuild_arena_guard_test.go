package horosvec

import (
	"bytes"
	"context"
	"log/slog"
	"math/rand/v2"
	"strings"
	"testing"
)

// TestRebuildAsyncRefusedOnArena vérifie que RebuildAsync refuse fail-loud sur un index
// arène (mode vector-less) : l'état persisté (node_count, medoid, centroïde) reste inchangé,
// aucun blob vecteur n'est réécrit, et le refus est journalisé. Symétrique du refus d'Insert.
func TestRebuildAsyncRefusedOnArena(t *testing.T) {
	const n, dim = 500, 16
	idx, db, _, _ := buildStreamingTestIndex(t, n, dim)
	defer idx.Close()

	if idx.arena == nil {
		t.Fatal("index de test attendu en mode arène (arena != nil)")
	}

	// État persisté AVANT le refus : lecture directe de la DB (oracle décidable).
	readState := func() (rowCount, maxVecLen int, medoid, centroid []byte) {
		if err := db.QueryRow(
			"SELECT COUNT(*), COALESCE(MAX(LENGTH(vector)),0) FROM vindex_nodes",
		).Scan(&rowCount, &maxVecLen); err != nil {
			t.Fatalf("lecture vindex_nodes: %v", err)
		}
		_ = db.QueryRow("SELECT value FROM vindex_meta WHERE key = 'medoid'").Scan(&medoid)
		_ = db.QueryRow("SELECT value FROM vindex_meta WHERE key = 'centroid'").Scan(&centroid)
		return
	}
	rowBefore, vecLenBefore, medoidBefore, centroidBefore := readState()
	nextIDBefore := idx.nextID

	// Capture du journal slog.
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelError})))
	defer slog.SetDefault(prev)

	// Itérateur de rebuild avec des données DIFFÉRENTES : s'il était exécuté, l'état changerait
	// (node_count, centroïde), rendant l'assertion d'inchangé décisive.
	rng := rand.New(rand.NewPCG(7, 0))
	newVecs, newIDs := generateVecs(rng, n/2, dim)
	idx.RebuildAsync(context.Background(), &sliceIterator{vecs: newVecs, ids: newIDs})

	// RebuildAsync refuse de façon synchrone (retour avant goroutine) ; l'état DB est relu.
	rowAfter, vecLenAfter, medoidAfter, centroidAfter := readState()

	if rowAfter != rowBefore {
		t.Fatalf("node_count modifié: %d -> %d (rebuild non refusé)", rowBefore, rowAfter)
	}
	if vecLenAfter != vecLenBefore || vecLenAfter != 0 {
		t.Fatalf("blob vecteur réécrit: maxLen %d -> %d (mode arène détruit)", vecLenBefore, vecLenAfter)
	}
	if !bytes.Equal(medoidBefore, medoidAfter) {
		t.Fatal("méta medoid modifiée par un rebuild qui aurait dû être refusé")
	}
	if !bytes.Equal(centroidBefore, centroidAfter) {
		t.Fatal("méta centroïde modifiée par un rebuild qui aurait dû être refusé")
	}
	if idx.nextID != nextIDBefore {
		t.Fatalf("idx.nextID modifié: %d -> %d", nextIDBefore, idx.nextID)
	}
	if idx.arena == nil {
		t.Fatal("idx.arena annulé par le rebuild refusé")
	}
	if !strings.Contains(buf.String(), "arena-backed index") {
		t.Fatalf("refus non journalisé; log=%q", buf.String())
	}
}
