package horosvec

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

// TestMultiBitReduitAuBit verifie que la largeur 1 redonne EXACTEMENT le code
// existant, octet pour octet, et le meme facteur de correction.
func TestMultiBitReduitAuBit(t *testing.T) {
	const dim = 512
	r := rand.New(rand.NewPCG(1, 2))
	centroid := make([]float32, dim)
	for i := range centroid {
		centroid[i] = float32(r.NormFloat64())
	}
	e := NewEncoder(centroid)
	for essai := 0; essai < 200; essai++ {
		v := make([]float32, dim)
		for i := range v {
			v[i] = float32(r.NormFloat64())
		}
		code1, sq1, l1 := e.Encode(v)
		codeM, sqM, corr := e.EncodeMultiBit(v, 1)
		if len(codeM) != len(code1) {
			t.Fatalf("longueur %d != %d", len(codeM), len(code1))
		}
		for i := range code1 {
			if code1[i] != codeM[i] {
				t.Fatalf("essai %d octet %d : %08b != %08b", essai, i, code1[i], codeM[i])
			}
		}
		if sq1 != sqM {
			t.Fatalf("norme carree %v != %v", sq1, sqM)
		}
		if math.Abs(l1-corr) > 1e-9*math.Abs(l1) {
			t.Fatalf("correction %v != norme L1 %v", corr, l1)
		}
	}
}

// TestMultiBitPrefixeEstLeCode1Bit verifie la propriete de rangement : les
// premiers nBytes d'un code multi-bits sont le code a un bit.
func TestMultiBitPrefixeEstLeCode1Bit(t *testing.T) {
	const dim = 512
	r := rand.New(rand.NewPCG(3, 4))
	centroid := make([]float32, dim)
	e := NewEncoder(centroid)
	for _, b := range []int{2, 3, 4, 5, 8} {
		v := make([]float32, dim)
		for i := range v {
			v[i] = float32(r.NormFloat64())
		}
		code1, _, _ := e.Encode(v)
		codeB, _, _ := e.EncodeMultiBit(v, b)
		if len(codeB) != len(code1)*b {
			t.Fatalf("b=%d : longueur %d != %d", b, len(codeB), len(code1)*b)
		}
		for i := range code1 {
			if codeB[i] != code1[i] {
				t.Fatalf("b=%d octet %d du prefixe : %08b != %08b", b, i, codeB[i], code1[i])
			}
		}
	}
}

// TestMultiBitEstimationConverge verifie que l'erreur d'estimation DECROIT
// quand la largeur augmente — la propriete qui justifie tout le dispositif.
func TestMultiBitEstimationConverge(t *testing.T) {
	const dim = 512
	r := rand.New(rand.NewPCG(5, 6))
	centroid := make([]float32, dim)
	e := NewEncoder(centroid)

	var precedent float64
	for _, b := range []int{1, 2, 3, 4, 5} {
		var somme float64
		const essais = 300
		for range essais {
			q := make([]float32, dim)
			for i := range q {
				q[i] = float32(r.NormFloat64())
			}
			v := make([]float32, dim)
			for i := range v {
				v[i] = float32(0.6*float64(q[i]) + 0.8*r.NormFloat64())
			}
			qc := make([]float64, dim)
			var qSq float64
			for i := range q {
				qc[i] = float64(q[i])
				qSq += qc[i] * qc[i]
			}
			var qp queryPlanes
			prepareQueryPlanes(qc, &qp)

			code, sq, corr := e.EncodeMultiBit(v, b)
			got := rabitqDistanceMultiBit(&qp, qSq, code, (dim+7)/8, b, sq, corr)

			var exact float64
			for i := range v {
				d := float64(q[i]) - float64(v[i])
				exact += d * d
			}
			somme += math.Abs(got-exact) / exact
		}
		moyenne := somme / essais
		t.Logf("b=%d : erreur relative moyenne sur la distance = %.4f", b, moyenne)
		// La decroissance n'est exigee que jusqu'a quatre bits : au-dela, la
		// precision est bornee par la quantification de la REQUETE, elle-meme sur
		// bpBits plans. Le plateau observe a partir de b=4 est donc un fait de
		// construction, pas une regression — et il dit qu'elargir le code sans
		// elargir la requete ne sert a rien.
		if b > 1 && b <= 4 && moyenne > precedent {
			t.Fatalf("b=%d : erreur %.4f superieure a celle de b=%d (%.4f)", b, moyenne, b-1, precedent)
		}
		if b == 5 && moyenne > precedent*1.10 {
			t.Fatalf("b=5 : erreur %.4f nettement au-dessus du plateau de b=4 (%.4f)", moyenne, precedent)
		}
		precedent = moyenne
	}
}

// TestMultiBitQualiteSelection mesure ce qui justifie le dispositif : la part
// des vrais plus proches voisins presents dans la preselection rendue par
// l'estimateur seul, sans graphe.
func TestMultiBitQualiteSelection(t *testing.T) {
	const dim, base, nq, topK, presel = 512, 20000, 20, 10, 128
	r := rand.New(rand.NewPCG(9, 9))

	vecs := make([][]float32, base)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(r.NormFloat64())
		}
		vecs[i] = v
	}
	centroid := make([]float32, dim)
	for _, v := range vecs {
		for j, x := range v {
			centroid[j] += x / float32(base)
		}
	}
	e := NewEncoder(centroid)

	queries := make([][]float32, nq)
	for i := range queries {
		src := vecs[r.IntN(base)]
		q := make([]float32, dim)
		for j := range q {
			q[j] = float32(0.7*float64(src[j]) + 0.7*r.NormFloat64())
		}
		queries[i] = q
	}

	// Verite terrain exacte.
	verite := make([]map[int]bool, nq)
	for qi, q := range queries {
		type pd struct {
			id int
			d  float64
		}
		tous := make([]pd, base)
		for i, v := range vecs {
			var s float64
			for j := range v {
				d := float64(q[j]) - float64(v[j])
				s += d * d
			}
			tous[i] = pd{i, s}
		}
		sort.Slice(tous, func(a, b int) bool { return tous[a].d < tous[b].d })
		m := map[int]bool{}
		for _, p := range tous[:topK] {
			m[p.id] = true
		}
		verite[qi] = m
	}

	nBytes := (dim + 7) / 8
	for _, b := range []int{1, 2, 3, 4} {
		codes := make([][]byte, base)
		sqs := make([]float64, base)
		corrs := make([]float64, base)
		for i, v := range vecs {
			codes[i], sqs[i], corrs[i] = e.EncodeMultiBit(v, b)
		}
		var somme float64
		for qi, q := range queries {
			qc := make([]float64, dim)
			var qSq float64
			for i := range q {
				qc[i] = float64(q[i]) - float64(centroid[i])
				qSq += qc[i] * qc[i]
			}
			var qp queryPlanes
			prepareQueryPlanes(qc, &qp)

			type ie struct {
				id int
				d  float64
			}
			est := make([]ie, base)
			for i := range codes {
				est[i] = ie{i, rabitqDistanceMultiBit(&qp, qSq, codes[i], nBytes, b, sqs[i], corrs[i])}
			}
			sort.Slice(est, func(a, c int) bool { return est[a].d < est[c].d })
			trouve := 0
			for _, x := range est[:presel] {
				if verite[qi][x.id] {
					trouve++
				}
			}
			somme += float64(trouve) / float64(topK)
		}
		t.Logf("b=%d : rappel de la preselection@%d = %.4f  (code de %d octets)",
			b, presel, somme/float64(nq), nBytes*b)
	}
}

// Coût d'évaluation selon la largeur du code. Le comptage de bits croise les B
// plans du code avec les cinq plans de la requête : le travail croît linéairement
// avec B, ce que ce banc chiffre.
func benchLargeur(b *testing.B, nbBits int) {
	const dim = 512
	r := rand.New(rand.NewPCG(11, 13))
	centroid := make([]float32, dim)
	e := NewEncoder(centroid)
	qc := make([]float64, dim)
	for i := range qc {
		qc[i] = r.NormFloat64()
	}
	var qp queryPlanes
	prepareQueryPlanes(qc, &qp)

	const pool = 2048
	codes := make([][]byte, pool)
	sqs := make([]float64, pool)
	corrs := make([]float64, pool)
	for i := range codes {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(r.NormFloat64())
		}
		codes[i], sqs[i], corrs[i] = e.EncodeMultiBit(v, nbBits)
	}
	nBytes := (dim + 7) / 8
	var sink float64
	b.ResetTimer()
	for i := range b.N {
		j := (i * 7919) % pool
		sink += rabitqDistanceMultiBit(&qp, 1.0, codes[j], nBytes, nbBits, sqs[j], corrs[j])
	}
	if sink == 1 {
		b.Log(sink)
	}
}

func BenchmarkMultiBit1(b *testing.B) { benchLargeur(b, 1) }
func BenchmarkMultiBit2(b *testing.B) { benchLargeur(b, 2) }
func BenchmarkMultiBit3(b *testing.B) { benchLargeur(b, 3) }
func BenchmarkMultiBit4(b *testing.B) { benchLargeur(b, 4) }

// TestMultiBitCoincideAvecBitProduct garde l'aiguillage : à un bit, la forme
// généralisée et le chemin spécialisé doivent rendre la MÊME valeur. Sans cette
// garde, l'aiguillage introduirait une différence de résultat invisible.
func TestMultiBitCoincideAvecBitProduct(t *testing.T) {
	const dim = 512
	r := rand.New(rand.NewPCG(21, 22))
	centroid := make([]float32, dim)
	e := NewEncoder(centroid)
	qc := make([]float64, dim)
	for i := range qc {
		qc[i] = r.NormFloat64()
	}
	var qp queryPlanes
	prepareQueryPlanes(qc, &qp)

	nBytes := (dim + 7) / 8
	for essai := 0; essai < 300; essai++ {
		v := make([]float32, dim)
		for i := range v {
			v[i] = float32(r.NormFloat64())
		}
		code, sq, corr := e.EncodeMultiBit(v, 1)
		specialise := rabitqDistanceBitProduct(&qp, 1.0, code, sq, corr)
		generalise := rabitqDistanceMultiBitGenerique(&qp, 1.0, code, nBytes, 1, sq, corr)
		if math.Abs(specialise-generalise) > 1e-9*math.Abs(specialise) {
			t.Fatalf("essai %d : specialise=%v generalise=%v", essai, specialise, generalise)
		}
	}
}
