package horosvec

import (
	"context"
	"database/sql"
	"math/rand/v2"
	"path/filepath"
	"strings"
	"testing"
)

// Les deux noyaux chauds introduits par le commit ceb6425 — arena.l2SquaredFP16
// et prepareQueryPlanes/bitProduct — supposent tous deux que la requête reçue a
// exactement la dimension de l'index, et que l'arène ouverte a exactement cette
// même dimension. Ils ne le vérifient pas eux-mêmes : sur une requête plus
// courte, le premier découpe hors borne ; sur une requête vide, le second lit
// l'élément d'indice zéro.
//
// Ce n'est pas un défaut tant que les deux contrôles d'entrée tiennent, et ils
// tiennent aujourd'hui (horosvec.go, validation de Search et ouverture d'arène).
// Mais aucun test ne les gardait : un remaniement qui déplacerait l'une de ces
// validations rendrait les deux noyaux faillibles sans qu'aucune suite ne le
// signale. Les deux tests ci-dessous gravent cette hypothèse implicite.
//
// Le choix délibéré est de NE PAS ajouter de garde défensive dans les noyaux :
// rendre false sur une requête mal dimensionnée ferait basculer l'appelant sur
// sa voie de repli et produirait une distance silencieusement fausse, alors que
// le module tient l'échec bruyant pour préférable au résultat muet.

// TestSearch_RefuseDimensionIncorrecte grave le contrôle dont dépend l'absence
// de découpage hors borne dans arena.l2SquaredFP16 et de lecture d'indice nul
// dans prepareQueryPlanes.
func TestSearch_RefuseDimensionIncorrecte(t *testing.T) {
	db := newTestDB(t)
	idx, err := New(db, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	rng := rand.New(rand.NewPCG(4, 8))
	vecs, ids := generateVecs(rng, 40, 16)
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}

	cas := []struct {
		nom   string
		query []float32
	}{
		{"plus courte", make([]float32, 8)},
		{"plus longue", make([]float32, 32)},
		{"vide", []float32{}},
		{"nil", nil},
	}
	for _, c := range cas {
		t.Run(c.nom, func(t *testing.T) {
			// Une panique ici signalerait que le contrôle d'entrée a disparu et que
			// les noyaux chauds sont désormais atteignables par une requête mal
			// dimensionnée.
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("panique au lieu d'une erreur : %v — le controle de dimension de Search a disparu", r)
				}
			}()
			res, err := idx.Search(context.Background(), c.query, 5)
			if err == nil {
				t.Fatalf("dimension %d acceptee sur un index de dimension 16, %d resultats rendus", len(c.query), len(res))
			}
			if !strings.Contains(err.Error(), "dim") {
				t.Fatalf("erreur inattendue : %v", err)
			}
		})
	}
}

// TestOpenArena_RefuseDimensionDivergente grave le second contrôle : une arène
// dont la dimension diffère de celle de l'index doit être refusée à l'ouverture.
// Sans lui, l2SquaredFP16 découperait la requête à la dimension de l'ARÈNE, qui
// pourrait excéder celle de la requête validée par Search.
func TestOpenArena_RefuseDimensionDivergente(t *testing.T) {
	dir := t.TempDir()
	arenaPath := filepath.Join(dir, "divergente.arena")

	// Arène de dimension 32, construite depuis un index de dimension 32.
	{
		db := newTestDB(t)
		idx, err := New(db, DefaultConfig())
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewPCG(9, 3))
		vecs, ids := generateVecs(rng, 20, 32)
		if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
			t.Fatal(err)
		}
		if err := idx.ExportArena(arenaPath); err != nil {
			t.Fatal(err)
		}
		idx.Close()
	}

	// Une base persistante portant un index de dimension 16, fermée puis rouverte
	// avec l'arène de dimension 32 : le contrôle se joue à la relecture.
	dbPath := filepath.Join(dir, "index16.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	idx, err := New(db, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(5, 5))
	vecs, ids := generateVecs(rng, 20, 16)
	if err := idx.Build(context.Background(), &sliceIterator{vecs: vecs, ids: ids}); err != nil {
		t.Fatal(err)
	}
	idx.Close()
	db.Close()

	db2, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()
	cfg := DefaultConfig()
	cfg.ArenaPath = arenaPath
	idx2, err := New(db2, cfg)
	if err == nil {
		idx2.Close()
		t.Fatal("arene de dimension 32 acceptee sur un index de dimension 16 : le controle a disparu, l2SquaredFP16 decouperait la requete au-dela de sa longueur")
	}
	if !strings.Contains(err.Error(), "dim") {
		t.Fatalf("erreur inattendue : %v", err)
	}
	t.Logf("refus attendu : %v", err)
}
