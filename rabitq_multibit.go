package horosvec

import (
	"encoding/binary"
	"math"
	"math/bits"
)

// Quantification multi-bits — RaBitQ étendu.
//
// Motif, mesuré. Le code à un bit par dimension sélectionne mal les candidats :
// sur un échantillon de 100 000 vecteurs en dimension 512, la présélection des
// 128 meilleurs par l'estimateur ne contient que 58 % des vrais dix plus proches
// voisins. À trois bits, elle en contient 88 %. C'est ce qui borne le rappel de
// bout en bout — mesuré à 0,47 au réglage par défaut sur l'index de référence,
// alors que le graphe, lui, se révèle navigable (0,91 quand on l'explore assez).
//
// Grille. Les niveaux sont symétriques et impairs, ℓ(c) = 2c − (2^B − 1) pour
// c dans [0, 2^B − 1] : à B = 1 elle vaut exactement {−1, +1}, c'est-à-dire le
// signe, et le schéma se réduit alors trait pour trait au code existant. Cette
// propriété est vérifiée par test, pas seulement affirmée.
//
// Rangement. Les B plans de bits sont rangés du POIDS FORT au poids faible.
// Le bit de poids fort de c vaut 1 exactement quand ℓ(c) est positif : les
// premiers (dim+7)/8 octets d'un code multi-bits sont donc, octet pour octet,
// le code à un bit du même vecteur. Un lecteur qui ignore les plans suivants lit
// l'ancien format sans le savoir, et l'affinage incrémental décrit par la
// littérature — trancher sur les bits de poids fort, ne lire la suite que pour
// les candidats ambigus — reste ouvert.
//
// Normalisation. Le facteur de correction stocké est N = Σ ℓ(c_i)·centré_i, qui
// vaut la norme L1 du vecteur centré quand B = 1, puisque ℓ y est le signe. La
// formule de distance est donc inchangée, et l'estimation converge vers la
// distance exacte à mesure que B croît : si ℓ reproduisait le vecteur centré,
// le terme correctif se simplifierait exactement.

// codeBitsMax borne la largeur d'un code. Au-delà de huit bits par dimension,
// le gain de sélection mesuré est nul et la mémoire double encore.
const codeBitsMax = 8

// normaliseCodeBits ramène une largeur demandée dans les bornes admises. Zéro,
// valeur d'une configuration qui ignore ce réglage, vaut un bit — le schéma
// d'origine.
func normaliseCodeBits(n int) int {
	if n < 1 {
		return 1
	}
	if n > codeBitsMax {
		return codeBitsMax
	}
	return n
}

// EncodeMultiBit quantifie un vecteur sur bits niveaux par dimension.
//
// Rend les B plans concaténés (poids fort d'abord), la norme carrée du vecteur
// centré et le facteur de correction N. Avec bits == 1, la sortie est
// rigoureusement celle de Encode.
func (e *Encoder) EncodeMultiBit(vec []float32, nbBits int) (code []byte, sqNorm float64, correction float64) {
	if nbBits < 1 {
		nbBits = 1
	}
	if nbBits > codeBitsMax {
		nbBits = codeBitsMax
	}
	d := e.dim
	nBytes := (d + 7) / 8
	code = make([]byte, nBytes*nbBits)

	centre := make([]float64, d)
	var ampli float64
	for i := range d {
		c := float64(vec[i]) - float64(e.centroid[i])
		centre[i] = c
		sqNorm += c * c
		if a := math.Abs(c); a > ampli {
			ampli = a
		}
	}

	niveauMax := float64(int(1)<<nbBits - 1) // ℓ maximal, impair
	echelle := ampli / niveauMax
	if echelle == 0 {
		// Vecteur nul après centrage : tout niveau vaut le même, la correction
		// sera nulle et la distance se réduira à la norme stockée.
		echelle = 1
	}

	for i := 0; i < d; i++ {
		// Niveau impair le plus proche de centre[i]/echelle.
		l := math.Round((centre[i]/echelle + niveauMax) / 2)
		if l < 0 {
			l = 0
		}
		if l > niveauMax {
			l = niveauMax
		}
		c := int(l)
		correction += float64(2*c-int(niveauMax)) * centre[i]
		// Plans du poids fort vers le poids faible : le plan 0 porte le bit
		// nbBits-1, qui est le signe.
		for p := 0; p < nbBits; p++ {
			if c>>(uint(nbBits-1-p))&1 == 1 {
				code[p*nBytes+i/8] |= 1 << uint(i%8)
			}
		}
	}
	return code, sqNorm, correction
}

// multiBitPlans décrit un code multi-bits déjà rangé en mémoire.
type multiBitPlans struct {
	code   []byte // nbBits plans de nBytes, poids fort d'abord
	nBytes int
	nbBits int
}

// sommeNiveaux rend Σ c_i, précalculable une fois par vecteur et indépendant de
// la requête. Il sert à retrancher le décalage de la grille.
func (m multiBitPlans) sommeNiveaux() int {
	var somme int
	for p := 0; p < m.nbBits; p++ {
		poids := 1 << uint(m.nbBits-1-p)
		plan := m.code[p*m.nBytes : (p+1)*m.nBytes]
		var pc int
		i := 0
		for ; i+8 <= len(plan); i += 8 {
			pc += bits.OnesCount64(binary.LittleEndian.Uint64(plan[i:]))
		}
		for ; i < len(plan); i++ {
			pc += bits.OnesCount8(plan[i])
		}
		somme += poids * pc
	}
	return somme
}

// produitCroise rend Σ_i c_i·d_i, où c est le niveau du code et d celui de la
// requête quantifiée : la somme se décompose en B × Bq comptages de bits sur
// l'intersection des plans, chacun pondéré par le produit des poids.
func produitCroise(m multiBitPlans, qp *queryPlanes) uint64 {
	var acc uint64
	mots := m.nBytes / 8
	for p := 0; p < m.nbBits; p++ {
		poidsC := uint(m.nbBits - 1 - p)
		plan := m.code[p*m.nBytes : (p+1)*m.nBytes]
		for k := 0; k < bpBits; k++ {
			pq := qp.planes[k*qp.words:]
			var pc int
			i := 0
			for ; i < mots; i++ {
				pc += bits.OnesCount64(binary.LittleEndian.Uint64(plan[i*8:]) & pq[i])
			}
			// Reste d'octets quand la dimension n'est pas multiple de 64.
			if reste := len(plan) - mots*8; reste > 0 {
				var mot uint64
				for j := 0; j < reste; j++ {
					mot |= uint64(plan[mots*8+j]) << (8 * uint(j))
				}
				pc += bits.OnesCount64(mot & pq[mots])
			}
			acc += uint64(pc) << (poidsC + uint(k))
		}
	}
	return acc
}

// rabitqDistanceMultiBit évalue la distance approchée depuis un code de largeur
// quelconque. Contrat identique à rabitqDistanceBitProduct, qu'elle généralise :
// à nbBits == 1 les deux rendent la même valeur, ce que garde un test.
func rabitqDistanceMultiBit(qp *queryPlanes, querySqNorm float64, code []byte, nBytes, nbBits int,
	storedSqNorm, correction float64) float64 {
	if correction == 0 {
		return storedSqNorm
	}
	if nbBits == 1 {
		// Chemin spécialisé : à un bit, la forme généralisée rend exactement la
		// même valeur (garde : TestMultiBitCoincideAvecBitProduct) mais coûte le
		// double — 56 ns contre 28 mesurées. Les index existants ne doivent rien
		// payer pour une généralisation dont ils ne se servent pas.
		return rabitqDistanceBitProduct(qp, querySqNorm, code, storedSqNorm, correction)
	}
	return rabitqDistanceMultiBitGenerique(qp, querySqNorm, code, nBytes, nbBits, storedSqNorm, correction)
}

// rabitqDistanceMultiBitGenerique est la forme générale, sans aiguillage : elle
// traite la largeur 1 comme les autres. Utilisée telle quelle par le test qui
// garde la coïncidence des deux voies.
func rabitqDistanceMultiBitGenerique(qp *queryPlanes, querySqNorm float64, code []byte, nBytes, nbBits int,
	storedSqNorm, correction float64) float64 {
	if correction == 0 {
		return storedSqNorm
	}
	m := multiBitPlans{code: code, nBytes: nBytes, nbBits: nbBits}

	// ⟨ℓ, q⟩ = 2·Σ c_i q_i − (2^B − 1)·Σ q_i, où q est la requête RECONSTRUITE
	// depuis ses propres plans : q_i ≈ vMin + delta·d_i.
	// D'où Σ c_i q_i = vMin·Σ c_i + delta·Σ c_i d_i.
	sommeC := float64(m.sommeNiveaux())
	croise := float64(produitCroise(m, qp))
	sommeCQ := qp.vMin*sommeC + qp.delta*croise

	niveauMax := float64(int(1)<<nbBits - 1)
	produit := 2*sommeCQ - niveauMax*qp.sumQ

	return querySqNorm + storedSqNorm - 2.0*storedSqNorm*produit/correction
}
