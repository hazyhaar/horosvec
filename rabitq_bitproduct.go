package horosvec

import (
	"encoding/binary"
	"math"
	"math/bits"
)

// Voie BitProduct : estimation de la distance RaBitQ asymétrique SANS table de
// correspondance par octet.
//
// Principe. La table remplacée coûtait un accès indirect par octet de code dans
// une table de (dim+7)/8 × 256 flottants — 128 Kio à dimension 512 — reconstruite
// à chaque requête. La voie retenue quantifie la requête sur bpBits niveaux et la
// transpose en bpBits plans de bits ; l'estimation devient une somme pondérée de
// comptages de bits sur les mots du code stocké, sans aucune table.
//
//	Σ b_i q_i ≈ vMin·popcount(code) + delta·Σ_k 2^k·popcount(code & plan_k)
//	Σ s_i q_i = 2·Σ b_i q_i − Σ q_i          (s_i = ±1, b_i ∈ {0,1})
//
// Mesures au sol DANS CE MODULE (dimension 512, i9-14900K, bancs de
// noyaux_bench_test.go) :
//   - évaluation par distance : 28,4 ns contre 30,9 ns pour la table, soit 1,09× ;
//     dont 3,0 ns pour le comptage des bits du code stocké, terme constant par
//     nœud qui serait supprimable en le précalculant à l'encodage — sans lui, la
//     mesure tombe à 25,4 ns, soit 1,22×.
//   - préparation par requête : 2 630 ns contre 17 490 ns, soit 6,6×, sans
//     allocation.
//
// Les bancs de primitives menés hors module donnaient 1,41× sur l'évaluation :
// ils comparaient des boucles nues sur des mots déjà décodés, sans le comptage
// du code ni la lecture des octets. Les chiffres retenus ici sont ceux du module.
//   - rappel@10 sur trois tranches de 500 requêtes (arène de 50 000 vecteurs,
//     dimension 512) : écarts de −0,0090, −0,0006 et +0,0034 face à la table.
//
// Le choix de bpBits = 5 vient de ces mesures : à 4 bits le rappel perd de 0,011
// à 0,013, à 8 bits l'estimation ne gagne rien et l'évaluation devient plus lente
// que la table.
const bpBits = 5

// queryPlanes porte ce qui se calcule une fois par requête, en remplacement de
// buildRabitqLUT. Les tampons vivent dans searchState et sont réutilisés.
type queryPlanes struct {
	planes []uint64 // bpBits plans consécutifs de words mots chacun
	words  int      // mots de 64 bits par plan
	vMin   float64  // borne basse de quantification
	delta  float64  // pas de quantification
	sumQ   float64  // Σ q_i, constante du changement de repère signe → bit
}

// prepareQueryPlanes quantifie la requête centrée et la transpose en plans de
// bits. dst.planes doit avoir une capacité d'au moins bpBits × ((dim+63)/64).
func prepareQueryPlanes(queryCentered []float64, dst *queryPlanes) {
	dim := len(queryCentered)
	words := (dim + 63) / 64
	need := bpBits * words
	if cap(dst.planes) < need {
		dst.planes = make([]uint64, need)
	} else {
		dst.planes = dst.planes[:need]
		clear(dst.planes)
	}
	dst.words = words

	vMin, vMax := queryCentered[0], queryCentered[0]
	var sumQ float64
	for _, v := range queryCentered {
		if v < vMin {
			vMin = v
		}
		if v > vMax {
			vMax = v
		}
		sumQ += v
	}
	levels := float64(int(1)<<bpBits - 1)
	delta := (vMax - vMin) / levels
	if delta == 0 {
		// Requête centrée constante : tout niveau vaut 0, le pas est arbitraire.
		delta = 1
	}
	dst.vMin, dst.delta, dst.sumQ = vMin, delta, sumQ

	for i, v := range queryCentered {
		q := uint64(math.Round((v - vMin) / delta))
		if q > uint64(levels) {
			q = uint64(levels)
		}
		w, bit := i/64, uint(i%64)
		for k := 0; k < bpBits; k++ {
			if (q>>uint(k))&1 == 1 {
				dst.planes[k*words+w] |= 1 << bit
			}
		}
	}
}

// bitProduct rend Σ_k 2^k·popcount(code & plan_k) et le nombre de bits à 1 du
// code. Le code est lu en little-endian, convention de packing de rabitq.go :
// la dimension i occupe le bit (i%8) de l'octet i/8, donc le bit (i%64) du mot
// little-endian i/64.
//
// La boucle est déballée sur cinq accumulateurs indépendants : le processeur
// enchaîne les comptages sans dépendance de données, forme la plus rapide
// obtenue en Go pur (les variantes vectorisées mesurées sont toutes plus lentes,
// cf. RESULTAT_NOYAU).
func bitProduct(code []byte, qp *queryPlanes) (uint64, int) {
	planes := qp.planes
	w := qp.words
	full := len(code) / 8

	var a0, a1, a2, a3, a4 int
	var pc int
	for i := 0; i < full; i++ {
		c := binary.LittleEndian.Uint64(code[i*8:])
		pc += bits.OnesCount64(c)
		a0 += bits.OnesCount64(c & planes[i])
		a1 += bits.OnesCount64(c & planes[w+i])
		a2 += bits.OnesCount64(c & planes[2*w+i])
		a3 += bits.OnesCount64(c & planes[3*w+i])
		a4 += bits.OnesCount64(c & planes[4*w+i])
	}
	// Reste : les octets du code qui ne complètent pas un mot de 64 bits.
	if rem := len(code) - full*8; rem > 0 {
		var c uint64
		for j := 0; j < rem; j++ {
			c |= uint64(code[full*8+j]) << (8 * uint(j))
		}
		pc += bits.OnesCount64(c)
		a0 += bits.OnesCount64(c & planes[full])
		a1 += bits.OnesCount64(c & planes[w+full])
		a2 += bits.OnesCount64(c & planes[2*w+full])
		a3 += bits.OnesCount64(c & planes[3*w+full])
		a4 += bits.OnesCount64(c & planes[4*w+full])
	}
	acc := uint64(a0) + uint64(a1)<<1 + uint64(a2)<<2 + uint64(a3)<<3 + uint64(a4)<<4
	return acc, pc
}

// rabitqDistanceBitProduct évalue la distance RaBitQ asymétrique par la voie
// BitProduct. Contrat identique à rabitqDistanceLUT, qu'elle remplace sur le
// chemin chaud : même signature de retour, même traitement du cas storedL1Norm
// nul.
func rabitqDistanceBitProduct(qp *queryPlanes, querySqNorm float64, storedCode []byte, storedSqNorm float64, storedL1Norm float64) float64 {
	if storedL1Norm == 0 {
		return storedSqNorm
	}
	bp, pc := bitProduct(storedCode, qp)
	sumSelected := qp.vMin*float64(pc) + qp.delta*float64(bp)
	signDot := 2*sumSelected - qp.sumQ
	return querySqNorm + storedSqNorm - 2.0*storedSqNorm*signDot/storedL1Norm
}
