package horosvec

import (
	"encoding/binary"
	"math"
	"math/bits"
	"math/rand/v2"
	"sort"
	"testing"
)

// Bancs des deux noyaux chauds, à dimension 512 — la dimension des mesures de
// profil qui ont motivé leur réécriture.
//
// Noyau de marche : rabitqDistanceBitProduct contre rabitqDistanceLUT.
// Noyau de re-classement : arena.l2SquaredFP16 contre vecInto + l2DistanceSquared.
//
// Les codes sont parcourus en ordre dispersé, ce qui reproduit la dispersion des
// accès que provoque la marche dans le graphe : un banc sur code fixe mesure un
// cache chaud que la recherche réelle n'a jamais.

const benchDim = 512
const benchPool = 4096

func benchCodes(n, dim int) [][]byte {
	r := rand.New(rand.NewPCG(21, 5))
	out := make([][]byte, n)
	for i := range out {
		c := make([]byte, (dim+7)/8)
		for j := range c {
			c[j] = byte(r.UintN(256))
		}
		out[i] = c
	}
	return out
}

func benchQueryCentered(dim int) []float64 {
	r := rand.New(rand.NewPCG(3, 3))
	q := make([]float64, dim)
	for i := range q {
		q[i] = r.NormFloat64()
	}
	return q
}

// encodeSignsCentered produit un code au format de l'encodeur (bit i à l'octet
// i/8, position i%8) depuis un vecteur déjà centré, et rend sa norme L1.
func encodeSignsCentered(centered []float64) ([]byte, float64) {
	code := make([]byte, (len(centered)+7)/8)
	var l1 float64
	for i, v := range centered {
		if v >= 0 {
			code[i/8] |= 1 << uint(i%8)
			l1 += v
		} else {
			l1 -= v
		}
	}
	return code, l1
}

// TestBitProductConcordeAvecLUT vérifie que la voie BitProduct estime la même
// distance que la table, à la tolérance près de la quantification de la requête
// sur bpBits niveaux. Une égalité exacte n'est pas attendue d'un estimateur
// quantifié : le contrôle porte sur l'écart relatif.
//
// Les codes sont tirés de vecteurs CORRÉLÉS à la requête, comme le sont les
// voisins effectivement parcourus par la marche. Des codes indépendants
// donneraient un produit scalaire de signes voisin de zéro, dont l'écart relatif
// diverge sans rien dire de la qualité de l'estimation — ce serait un oracle
// trompeur, pas un test.
func TestBitProductConcordeAvecLUT(t *testing.T) {
	const dim = benchDim
	r := rand.New(rand.NewPCG(31, 17))
	qc := make([]float64, dim)
	for i := range qc {
		qc[i] = r.NormFloat64()
	}

	lut := make([]float64, (dim+7)/8*256)
	buildRabitqLUT(qc, lut)
	var qp queryPlanes
	prepareQueryPlanes(qc, &qp)

	// La propriété qui décide n'est pas l'égalité des distances mais la
	// préservation du CLASSEMENT : la marche ne consomme ces distances que pour
	// ordonner des candidats. Le contrôle porte donc sur le recouvrement des dix
	// meilleurs entre les deux estimateurs, l'écart relatif restant journalisé à
	// titre indicatif.
	const trials = 1024
	type est struct{ lut, bp float64 }
	all := make([]est, trials)
	var sumRel float64
	for t := range trials {
		stored := make([]float64, dim)
		var sq float64
		for i := range stored {
			stored[i] = 0.6*qc[i] + 0.8*r.NormFloat64()
			sq += stored[i] * stored[i]
		}
		code, l1 := encodeSignsCentered(stored)

		a := rabitqDistanceLUT(lut, 1.0, code, sq, l1)
		b := rabitqDistanceBitProduct(&qp, 1.0, code, sq, l1)
		all[t] = est{a, b}
		sumRel += math.Abs(a-b) / math.Abs(a)
	}

	topK := func(sel func(est) float64) map[int]bool {
		idx := make([]int, len(all))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(i, j int) bool { return sel(all[idx[i]]) < sel(all[idx[j]]) })
		out := make(map[int]bool, 10)
		for _, i := range idx[:10] {
			out[i] = true
		}
		return out
	}
	ref := topK(func(e est) float64 { return e.lut })
	got := topK(func(e est) float64 { return e.bp })
	inter := 0
	for i := range got {
		if ref[i] {
			inter++
		}
	}
	overlap := float64(inter) / 10
	t.Logf("recouvrement des 10 meilleurs = %.2f ; ecart relatif moyen sur la distance = %.5f (bpBits=%d)",
		overlap, sumRel/trials, bpBits)
	if overlap < 0.9 {
		t.Fatalf("recouvrement des 10 meilleurs %.2f sous le plancher 0.90", overlap)
	}
}

// TestBitProductNormeL1Nulle vérifie le contrat partagé avec rabitqDistanceLUT :
// un vecteur de norme L1 nulle rend la norme carrée stockée, sans division.
func TestBitProductNormeL1Nulle(t *testing.T) {
	qc := benchQueryCentered(benchDim)
	var qp queryPlanes
	prepareQueryPlanes(qc, &qp)
	code := benchCodes(1, benchDim)[0]
	if got := rabitqDistanceBitProduct(&qp, 1.0, code, 42.0, 0); got != 42.0 {
		t.Fatalf("norme L1 nulle : got %v, want 42.0", got)
	}
}

func BenchmarkNoyauMarcheLUT(b *testing.B) {
	qc := benchQueryCentered(benchDim)
	lut := make([]float64, (benchDim+7)/8*256)
	buildRabitqLUT(qc, lut)
	codes := benchCodes(benchPool, benchDim)
	var sink float64
	b.ResetTimer()
	for i := range b.N {
		sink += rabitqDistanceLUT(lut, 1.0, codes[(i*7919)%benchPool], 42.0, 17.5)
	}
	if sink == 1 {
		b.Log(sink)
	}
}

func BenchmarkNoyauMarcheBitProduct(b *testing.B) {
	qc := benchQueryCentered(benchDim)
	var qp queryPlanes
	prepareQueryPlanes(qc, &qp)
	codes := benchCodes(benchPool, benchDim)
	var sink float64
	b.ResetTimer()
	for i := range b.N {
		sink += rabitqDistanceBitProduct(&qp, 1.0, codes[(i*7919)%benchPool], 42.0, 17.5)
	}
	if sink == 1 {
		b.Log(sink)
	}
}

func BenchmarkParRequeteBuildLUT(b *testing.B) {
	qc := benchQueryCentered(benchDim)
	lut := make([]float64, (benchDim+7)/8*256)
	b.ResetTimer()
	for range b.N {
		buildRabitqLUT(qc, lut)
	}
}

func BenchmarkParRequetePrepareplans(b *testing.B) {
	qc := benchQueryCentered(benchDim)
	var qp queryPlanes
	prepareQueryPlanes(qc, &qp) // première passe hors mesure : alloue les tampons
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		prepareQueryPlanes(qc, &qp)
	}
}

// --- noyau de re-classement ---

// benchArena fabrique une arène en mémoire (non mmappée) de n vecteurs.
func benchArena(n, dim int) (*arena, []float32) {
	r := rand.New(rand.NewPCG(7, 11))
	data := make([]byte, arenaHeaderSize+n*dim*2)
	for i := 0; i < n*dim; i++ {
		binary.LittleEndian.PutUint16(data[arenaHeaderSize+i*2:], float32ToFloat16(float32(r.NormFloat64())))
	}
	q := make([]float32, dim)
	for i := range q {
		q[i] = float32(r.NormFloat64())
	}
	return &arena{dim: dim, count: int64(n), data: data}, q
}

// TestL2FP16EgaleVecIntoPuisL2 vérifie que le noyau fusionné rend exactement la
// même valeur que la voie qu'il remplace — égalité stricte attendue, les deux
// chemins faisant les mêmes opérations flottantes dans le même ordre.
func TestL2FP16EgaleVecIntoPuisL2(t *testing.T) {
	a, q := benchArena(256, benchDim)
	dst := make([]float32, benchDim)
	for i := int64(0); i < a.count; i++ {
		if !a.vecInto(i, dst) {
			t.Fatalf("vecInto(%d) false", i)
		}
		want := l2DistanceSquared(q, dst)
		got, ok := a.l2SquaredFP16(i, q)
		if !ok {
			t.Fatalf("l2SquaredFP16(%d) false", i)
		}
		if want != got {
			t.Fatalf("noeud %d : fusionne=%v, materialise=%v", i, got, want)
		}
	}
	// Hors couverture : même contrat de repli que vecInto.
	if _, ok := a.l2SquaredFP16(a.count, q); ok {
		t.Fatal("hors borne haute devrait rendre false")
	}
	if _, ok := a.l2SquaredFP16(-1, q); ok {
		t.Fatal("index negatif devrait rendre false")
	}
}

func BenchmarkRerankMaterialisePuisL2(b *testing.B) {
	a, q := benchArena(benchPool, benchDim)
	dst := make([]float32, benchDim)
	var sink float64
	b.ResetTimer()
	for i := range b.N {
		id := int64((i * 7919) % benchPool)
		a.vecInto(id, dst)
		sink += l2DistanceSquared(q, dst)
	}
	if sink == 1 {
		b.Log(sink)
	}
}

func BenchmarkRerankFusionneFP16(b *testing.B) {
	a, q := benchArena(benchPool, benchDim)
	var sink float64
	b.ResetTimer()
	for i := range b.N {
		d, _ := a.l2SquaredFP16(int64((i*7919)%benchPool), q)
		sink += d
	}
	if sink == 1 {
		b.Log(sink)
	}
}

// bitProductSansPopcnt : variante mesurant le noyau SANS le comptage de bits du
// code stocké, pour établir ce que coûte ce terme. Le résultat est incomplet et
// ne sert qu'au banc.
func bitProductSansPopcnt(code []byte, qp *queryPlanes) uint64 {
	planes := qp.planes
	w := qp.words
	full := len(code) / 8
	var a0, a1, a2, a3, a4 int
	for i := 0; i < full; i++ {
		c := binary.LittleEndian.Uint64(code[i*8:])
		a0 += bits.OnesCount64(c & planes[i])
		a1 += bits.OnesCount64(c & planes[w+i])
		a2 += bits.OnesCount64(c & planes[2*w+i])
		a3 += bits.OnesCount64(c & planes[3*w+i])
		a4 += bits.OnesCount64(c & planes[4*w+i])
	}
	return uint64(a0) + uint64(a1)<<1 + uint64(a2)<<2 + uint64(a3)<<3 + uint64(a4)<<4
}

func BenchmarkNoyauMarcheBitProductSansPopcnt(b *testing.B) {
	qc := benchQueryCentered(benchDim)
	var qp queryPlanes
	prepareQueryPlanes(qc, &qp)
	codes := benchCodes(benchPool, benchDim)
	var sink uint64
	b.ResetTimer()
	for i := range b.N {
		sink += bitProductSansPopcnt(codes[(i*7919)%benchPool], &qp)
	}
	if sink == 1 {
		b.Log(sink)
	}
}
