package infrastructure

import (
	"cashguard/internal/domain"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// MigrateSQLite imports a consistent snapshot into an empty PostgreSQL workspace.
// Existing source files are never changed. Retry is safe after a failed import.
func (p *Postgres) MigrateSQLite(source, destination string) (count int, err error) {
	if _, err = os.Stat(source); err != nil {
		return
	}
	legacy, err := sql.Open("sqlite", source+"?mode=ro")
	if err != nil {
		return
	}
	defer legacy.Close()
	legacy.SetMaxOpenConns(1)
	sourceTx, err := legacy.Begin()
	if err != nil {
		return
	}
	defer sourceTx.Rollback()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	targetTx, err := p.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return
	}
	defer targetTx.Rollback()
	if _, err = targetTx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(786154230)"); err != nil {
		return
	}
	var existing int
	if err = targetTx.QueryRowContext(ctx, "SELECT (SELECT count(*) FROM users WHERE id <> 'jury') + (SELECT count(*) FROM jobs)").Scan(&existing); err != nil {
		return
	}
	if existing > 0 {
		return 0, fmt.Errorf("target PostgreSQL must be empty; existing accounts and jobs will not be overwritten")
	}
	users, err := sourceTx.Query("SELECT id,email,password_hash,theme,language FROM users")
	if err != nil {
		return
	}
	for users.Next() {
		var u domain.User
		if err = users.Scan(&u.ID, &u.Email, &u.PasswordHash, &u.Theme, &u.Language); err != nil {
			users.Close()
			return
		}
		if _, err = targetTx.ExecContext(ctx, "INSERT INTO users(id,email,password_hash,theme,language) VALUES($1,$2,$3,$4,$5)", u.ID, u.Email, u.PasswordHash, u.Theme, u.Language); err != nil {
			users.Close()
			return
		}
	}
	err = users.Err()
	users.Close()
	if err != nil {
		return
	}
	sessions, err := sourceTx.Query("SELECT hash,user_id,expires FROM sessions")
	if err != nil {
		return
	}
	for sessions.Next() {
		var hash, user string
		var expires int64
		if err = sessions.Scan(&hash, &user, &expires); err != nil {
			sessions.Close()
			return
		}
		if _, err = targetTx.ExecContext(ctx, "INSERT INTO sessions(hash,user_id,expires) VALUES($1,$2,$3)", hash, user, expires); err != nil {
			sessions.Close()
			return
		}
	}
	err = sessions.Err()
	sessions.Close()
	if err != nil {
		return
	}
	uploads := filepath.Join(destination, "uploads")
	if err = os.MkdirAll(uploads, 0700); err != nil {
		return
	}
	created := []string{}
	defer func() {
		if err != nil {
			for _, path := range created {
				os.Remove(path)
			}
		}
	}()
	jobs, err := sourceTx.Query("SELECT result,path,owner_id FROM jobs")
	if err != nil {
		return
	}
	defer jobs.Close()
	for jobs.Next() {
		var result, oldPath, owner string
		if e := jobs.Scan(&result, &oldPath, &owner); e != nil {
			return count, e
		}
		var j domain.Job
		if e := json.Unmarshal([]byte(result), &j); e != nil {
			return count, e
		}
		j.Path = oldPath
		j.OwnerID = owner
		ext := filepath.Ext(j.Path)
		targetPath := ""
		if j.Path != "" {
			if ext != ".mp4" && ext != ".webm" && ext != ".mov" {
				return count, fmt.Errorf("unsupported legacy file extension for job %s", j.ID)
			}
			if len(j.ID) != 32 {
				return count, fmt.Errorf("invalid legacy job id")
			}
			for _, ch := range j.ID {
				if !((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')) {
					return count, fmt.Errorf("invalid legacy job id")
				}
			}
			inputPath := filepath.Join(filepath.Dir(source), "uploads", j.ID+ext)
			targetPath = filepath.Join(uploads, j.ID+ext)
			wasCreated, e := copyMigrationFile(inputPath, targetPath)
			if wasCreated {
				created = append(created, targetPath)
			}
			if e != nil {
				return count, e
			}
		}
		if _, e := targetTx.ExecContext(ctx, "INSERT INTO jobs(id,owner_id,path,result,created_at) VALUES($1,$2,$3,$4,$5)", j.ID, j.OwnerID, targetPath, result, j.Created); e != nil {
			return count, e
		}
		count++
	}
	if err = jobs.Err(); err != nil {
		return
	}
	err = targetTx.Commit()
	return
}
func copyMigrationFile(source, target string) (created bool, err error) {
	input, err := os.Open(source)
	if err != nil {
		return false, err
	}
	defer input.Close()
	output, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if os.IsExist(err) {
		existing, e := os.Open(target)
		if e != nil {
			return false, e
		}
		defer existing.Close()
		a, b := sha256.New(), sha256.New()
		if _, e = io.Copy(a, input); e != nil {
			return false, e
		}
		if _, e = io.Copy(b, existing); e != nil {
			return false, e
		}
		if string(a.Sum(nil)) != string(b.Sum(nil)) {
			return false, fmt.Errorf("destination file differs: %s", filepath.Base(target))
		}
		return false, nil
	}
	if err != nil {
		return false, err
	}
	_, err = io.Copy(output, input)
	closeErr := output.Close()
	if err == nil {
		err = closeErr
	}
	return true, err
}
