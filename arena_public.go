package horosvec

// API publique minimale de l'arène plate fp16 (format HVARENA1). Ajout rétro-compatible
// pur : aucune signature existante n'est modifiée. Ces types exposent l'écrivain au fil de
// l'eau (déjà utilisé en interne par le build streaming) et le lecteur, pour permettre à un
// pipeline externe (ex. hnbook-embed) de produire une arène checkpointée puis de la relire,
// sans jamais réimplémenter le format binaire ni la conversion fp16.

// ArenaWriter est l'écrivain public d'arène fp16, capable de checkpoint et de reprise.
//
// Cycle de vie : NewArenaWriter (ou ResumeArenaWriter pour reprendre) → WriteVec* →
// Sync (durcit un checkpoint) → Finalize (réécrit le header avec le count exact, fsync,
// rename atomique du .tmp vers le chemin final). Un seul écrivain à la fois (arène
// append-only, offset = rang du vecteur × dim × 2).
//
// Modèle de crash-consistance recommandé pour un pipeline checkpointé :
//  1. WriteVec des vecteurs d'une tranche (dans l'ordre de rang) ;
//  2. Sync (les octets sont durables dans le .tmp) ;
//  3. écrire atomiquement le manifest externe avec le nouveau count.
//
// Un kill entre 2 et 3 laisse le .tmp plus long que le count du manifest ; ResumeArenaWriter
// tronque au count du manifest — ni doublon ni trou. Le manifest n'avance jamais avant le
// Sync, donc son count ne dépasse jamais les octets durables.
type ArenaWriter struct {
	w *arenaWriter
}

// NewArenaWriter crée une nouvelle arène (fichier temporaire path+".tmp", header réservé).
func NewArenaWriter(path string, dim int) (*ArenaWriter, error) {
	w, err := newArenaWriter(path, dim)
	if err != nil {
		return nil, err
	}
	return &ArenaWriter{w: w}, nil
}

// ResumeArenaWriter rouvre l'arène en cours (path+".tmp") au rang count issu d'un checkpoint
// durable, en tronquant tout octet partiel au-delà. Voir resumeArenaWriter.
func ResumeArenaWriter(path string, dim int, count int64) (*ArenaWriter, error) {
	w, err := resumeArenaWriter(path, dim, count)
	if err != nil {
		return nil, err
	}
	return &ArenaWriter{w: w}, nil
}

// WriteVec encode et append un vecteur (fp32→fp16) au rang courant. La longueur doit valoir
// dim.
func (a *ArenaWriter) WriteVec(vec []float32) error { return a.w.writeVec(vec) }

// Count retourne le nombre de vecteurs écrits jusqu'ici (rang du prochain vecteur).
func (a *ArenaWriter) Count() int64 { return a.w.count }

// Sync durcit le payload courant sur disque (fsync), sans réécrire le header ni renommer.
// Point de durabilité d'un checkpoint.
func (a *ArenaWriter) Sync() error { return a.w.sync() }

// Finalize réécrit le header avec le count exact, fsync, puis renomme atomiquement le .tmp
// vers le chemin final. Après Finalize, l'arène est lisible par OpenArenaReader / horosvec.New.
func (a *ArenaWriter) Finalize() error { return a.w.finalize() }

// Abort ferme et supprime le fichier temporaire (aucune arène finale produite).
func (a *ArenaWriter) Abort() { a.w.abort() }

// ArenaReader est une vue publique en lecture seule d'une arène fp16 finalisée. Sûr en
// accès concurrent (immuable après ouverture).
type ArenaReader struct {
	a *arena
}

// OpenArenaReader lit et valide une arène finalisée (header + longueur exacte).
func OpenArenaReader(path string) (*ArenaReader, error) {
	a, err := openArena(path)
	if err != nil {
		return nil, err
	}
	return &ArenaReader{a: a}, nil
}

// Dim retourne la dimension des vecteurs stockés.
func (r *ArenaReader) Dim() int { return r.a.dim }

// Count retourne le nombre de vecteurs stockés.
func (r *ArenaReader) Count() int64 { return r.a.count }

// VecInto décode le vecteur de rang i (fp16→fp32) vers dst (len == Dim). Retourne false si
// i est hors bornes.
func (r *ArenaReader) VecInto(i int64, dst []float32) bool { return r.a.vecInto(i, dst) }
