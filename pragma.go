// CLAUDE:SUMMARY SQLite connection configuration with WAL mode, cache, mmap, and page size pragmas.
package horosvec

import "database/sql"

// configureSQLite applies performance pragmas to a SQLite connection.
// Must be called once at connection open, before any queries.
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
