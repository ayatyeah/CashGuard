package infrastructure

import (
	"cashguard/internal/domain"
	"context"
	"database/sql"
	_ "embed"
	"encoding/json"
	"errors"
	_ "github.com/jackc/pgx/v5/stdlib"
	"os"
	"path/filepath"
	"time"
)

//go:embed migrations/001_initial.sql
var postgresSchema string

type Postgres struct{ db *sql.DB }

func NewPostgres(dsn string) (*Postgres, error) {
	db, e := sql.Open("pgx", dsn)
	if e != nil {
		return nil, e
	}
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(30 * time.Minute)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if e = db.PingContext(ctx); e != nil {
		db.Close()
		return nil, e
	}
	tx, e := db.BeginTx(ctx, nil)
	if e != nil {
		db.Close()
		return nil, e
	}
	defer tx.Rollback()
	if _, e = tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(786154230)"); e == nil {
		_, e = tx.ExecContext(ctx, postgresSchema)
	}
	if e == nil {
		e = tx.Commit()
	}
	if e != nil {
		db.Close()
		return nil, e
	}
	return &Postgres{db: db}, nil
}
func (s *Postgres) Ping(ctx context.Context) error { return s.db.PingContext(ctx) }
func (s *Postgres) Close() error                   { return s.db.Close() }
func (s *Postgres) Save(j domain.Job) error {
	b, e := json.Marshal(j)
	if e != nil {
		return e
	}
	_, e = s.db.Exec(`INSERT INTO jobs(id,owner_id,path,result,created_at) VALUES($1,$2,$3,$4,$5) ON CONFLICT(id) DO UPDATE SET result=excluded.result,path=excluded.path WHERE jobs.owner_id=excluded.owner_id`, j.ID, j.OwnerID, j.Path, string(b), j.Created)
	return e
}
func (s *Postgres) Get(id string) (domain.Job, error) {
	return scanJob(s.db.QueryRow(`SELECT result,path,owner_id FROM jobs WHERE id=$1`, id))
}
func (s *Postgres) List() ([]domain.Job, error) {
	return s.list(`SELECT result,path,owner_id FROM jobs ORDER BY created_at DESC`)
}
func (s *Postgres) ListForOwner(owner string) ([]domain.Job, error) {
	return s.list(`SELECT result,path,owner_id FROM jobs WHERE owner_id=$1 ORDER BY created_at DESC`, owner)
}
func (s *Postgres) list(query string, args ...any) ([]domain.Job, error) {
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
func (s *Postgres) CreateUser(u domain.User) error {
	result, e := s.db.Exec(`INSERT INTO users(id,email,password_hash,theme,language) VALUES($1,$2,$3,$4,$5) ON CONFLICT(email) DO NOTHING`, u.ID, u.Email, u.PasswordHash, u.Theme, u.Language)
	if e != nil {
		return e
	}
	n, e := result.RowsAffected()
	if e == nil && n == 0 {
		return domain.ErrAccountExists
	}
	return e
}
func (s *Postgres) UserByEmail(email string) (domain.User, error) {
	return scanUser(s.db.QueryRow(`SELECT id,email,password_hash,theme,language FROM users WHERE email=$1`, email))
}
func (s *Postgres) UserByID(id string) (domain.User, error) {
	return scanUser(s.db.QueryRow(`SELECT id,email,password_hash,theme,language FROM users WHERE id=$1`, id))
}
func (s *Postgres) SaveSession(hash, userID string, expires int64) error {
	_, e := s.db.Exec(`INSERT INTO sessions(hash,user_id,expires) VALUES($1,$2,$3)`, hash, userID, expires)
	return e
}
func (s *Postgres) SessionUser(hash string, now int64) (domain.User, error) {
	return scanUser(s.db.QueryRow(`SELECT u.id,u.email,u.password_hash,u.theme,u.language FROM users u JOIN sessions s ON s.user_id=u.id WHERE s.hash=$1 AND s.expires>$2`, hash, now))
}
func (s *Postgres) DeleteSession(hash string) error {
	_, e := s.db.Exec(`DELETE FROM sessions WHERE hash=$1`, hash)
	return e
}
func (s *Postgres) Preferences(id, theme, language string) error {
	_, e := s.db.Exec(`UPDATE users SET theme=$1,language=$2 WHERE id=$3`, theme, language, id)
	return e
}

// Existing shared results belong only to the jury API workspace, never a new user.
func (s *Postgres) ImportLegacy(dir, uploads string) error {
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
