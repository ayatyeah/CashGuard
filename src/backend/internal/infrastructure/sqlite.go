package infrastructure

import (
	"cashguard/internal/domain"
	"database/sql"
	"encoding/json"
	"errors"
	_ "modernc.org/sqlite"
	"os"
	"path/filepath"
)

type SQLite struct{ db *sql.DB }

func NewSQLite(path string) (*SQLite, error) {
	db, e := sql.Open("sqlite", path)
	if e != nil {
		return nil, e
	}
	db.SetMaxOpenConns(1)
	schema := `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000; PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,email TEXT NOT NULL UNIQUE,password_hash TEXT NOT NULL,theme TEXT NOT NULL DEFAULT 'light',language TEXT NOT NULL DEFAULT 'ru');
CREATE TABLE IF NOT EXISTS sessions(hash TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,expires INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,owner_id TEXT NOT NULL,path TEXT NOT NULL,result TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS jobs_owner_created ON jobs(owner_id,created_at DESC);
PRAGMA user_version=1;`
	if _, e = db.Exec(schema); e != nil {
		db.Close()
		return nil, e
	}
	return &SQLite{db: db}, nil
}
func (s *SQLite) Close() error { return s.db.Close() }
func (s *SQLite) Save(j domain.Job) error {
	b, e := json.Marshal(j)
	if e != nil {
		return e
	}
	_, e = s.db.Exec(`INSERT INTO jobs(id,owner_id,path,result,created_at) VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET result=excluded.result,path=excluded.path WHERE jobs.owner_id=excluded.owner_id`, j.ID, j.OwnerID, j.Path, string(b), j.Created)
	return e
}
func scanJob(row interface{ Scan(...any) error }) (j domain.Job, e error) {
	var data, path, owner string
	e = row.Scan(&data, &path, &owner)
	if errors.Is(e, sql.ErrNoRows) {
		return j, domain.ErrNotFound
	}
	if e != nil {
		return
	}
	e = json.Unmarshal([]byte(data), &j)
	j.Path = path
	j.OwnerID = owner
	return
}
func (s *SQLite) Get(id string) (domain.Job, error) {
	return scanJob(s.db.QueryRow(`SELECT result,path,owner_id FROM jobs WHERE id=?`, id))
}
func (s *SQLite) List() ([]domain.Job, error) {
	return s.list(`SELECT result,path,owner_id FROM jobs ORDER BY created_at DESC`)
}
func (s *SQLite) ListForOwner(owner string) ([]domain.Job, error) {
	return s.list(`SELECT result,path,owner_id FROM jobs WHERE owner_id=? ORDER BY created_at DESC`, owner)
}
func (s *SQLite) list(query string, args ...any) ([]domain.Job, error) {
	rows, e := s.db.Query(query, args...)
	if e != nil {
		return nil, e
	}
	defer rows.Close()
	out := []domain.Job{}
	for rows.Next() {
		j, e := scanJob(rows)
		if e != nil {
			return nil, e
		}
		out = append(out, j)
	}
	return out, rows.Err()
}
func (s *SQLite) CreateUser(u domain.User) error {
	result, e := s.db.Exec(`INSERT INTO users(id,email,password_hash,theme,language) VALUES(?,?,?,?,?) ON CONFLICT(email) DO NOTHING`, u.ID, u.Email, u.PasswordHash, u.Theme, u.Language)
	if e != nil {
		return e
	}
	n, e := result.RowsAffected()
	if e == nil && n == 0 {
		return domain.ErrAccountExists
	}
	return e
}
func scanUser(row interface{ Scan(...any) error }) (u domain.User, e error) {
	e = row.Scan(&u.ID, &u.Email, &u.PasswordHash, &u.Theme, &u.Language)
	if errors.Is(e, sql.ErrNoRows) {
		e = domain.ErrNotFound
	}
	return
}
func (s *SQLite) UserByEmail(email string) (domain.User, error) {
	return scanUser(s.db.QueryRow(`SELECT id,email,password_hash,theme,language FROM users WHERE email=?`, email))
}
func (s *SQLite) UserByID(id string) (domain.User, error) {
	return scanUser(s.db.QueryRow(`SELECT id,email,password_hash,theme,language FROM users WHERE id=?`, id))
}
func (s *SQLite) SaveSession(hash, userID string, expires int64) error {
	_, e := s.db.Exec(`INSERT INTO sessions(hash,user_id,expires) VALUES(?,?,?)`, hash, userID, expires)
	return e
}
func (s *SQLite) SessionUser(hash string, now int64) (domain.User, error) {
	return scanUser(s.db.QueryRow(`SELECT u.id,u.email,u.password_hash,u.theme,u.language FROM users u JOIN sessions s ON s.user_id=u.id WHERE s.hash=? AND s.expires>?`, hash, now))
}
func (s *SQLite) DeleteSession(hash string) error {
	_, e := s.db.Exec(`DELETE FROM sessions WHERE hash=?`, hash)
	return e
}
func (s *SQLite) Preferences(id, theme, language string) error {
	_, e := s.db.Exec(`UPDATE users SET theme=?,language=? WHERE id=?`, theme, language, id)
	return e
}

// Existing shared results belong only to the jury API workspace, never a new user.
func (s *SQLite) ImportLegacy(dir, uploads string) error {
	paths, e := filepath.Glob(filepath.Join(dir, "*.json"))
	if e != nil {
		return e
	}
	for _, path := range paths {
		b, e := os.ReadFile(path)
		if e != nil {
			return e
		}
		var j domain.Job
		if e = json.Unmarshal(b, &j); e != nil {
			return e
		}
		if _, e = s.Get(j.ID); e == nil {
			continue
		} else if !errors.Is(e, domain.ErrNotFound) {
			return e
		}
		j.OwnerID = "jury"
		for _, ext := range []string{".mp4", ".mov", ".webm"} {
			video := filepath.Join(uploads, j.ID+ext)
			if _, e = os.Stat(video); e == nil {
				j.Path = video
				break
			}
		}
		if e = s.Save(j); e != nil {
			return e
		}
	}
	return nil
}
