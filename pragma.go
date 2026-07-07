// configureSQLite utilise db.Exec(multi-statement) qui ne touche qu'une connexion du pool.
//
//	Consommateurs horos48 DOIVENT ouvrir le *sql.DB avec les _pragma= dans le DSN (dbopen.Open).
//	configureSQLite reste un fallback de sécurité mais ne remplace pas les pragmas DSN.
package horosvec

import "database/sql"

// configureSQLite applies performance pragmas to a SQLite connection.
// Must be called once at connection open, before any queries.
// NOTE: consumers using dbopen.Open should include _pragma= in the DSN instead.
func configureSQLite(db *sql.DB) error {
	pragmas := `
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -65536;
PRAGMA mmap_size = 268435456;
PRAGMA page_size = 4096;
`
	_, err := db.Exec(pragmas)
	return err
}
