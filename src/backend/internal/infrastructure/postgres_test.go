package infrastructure

import (
	"cashguard/internal/domain"
	"database/sql"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func postgresTest(t *testing.T) (*Postgres, string) {
	t.Helper()
	dsn := os.Getenv("TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
	}
	admin, e := sql.Open("pgx", dsn)
	if e != nil {
		t.Fatal(e)
	}
	schema := fmt.Sprintf("cashguard_test_%d", time.Now().UnixNano())
	if _, e = admin.Exec("CREATE SCHEMA " + schema); e != nil {
		admin.Close()
		t.Fatal(e)
	}
	t.Cleanup(func() { admin.Exec("DROP SCHEMA " + schema + " CASCADE"); admin.Close() })
	parsed, e := url.Parse(dsn)
	if e != nil {
		t.Fatal(e)
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	db, e := NewPostgres(parsed.String())
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { db.Close() })
	return db, parsed.String()
}
func TestPostgresPersistence(t *testing.T) {
	db, dsn := postgresTest(t)
	u := domain.User{ID: "user-a", Email: "a@example.com", PasswordHash: "hash", Theme: "light", Language: "ru"}
	if e := db.CreateUser(u); e != nil {
		t.Fatal(e)
	}
	if e := db.CreateUser(u); e != domain.ErrAccountExists {
		t.Fatalf("duplicate user: %v", e)
	}
	j := domain.Job{ID: "job-a", OwnerID: u.ID, Filename: "video.mp4", Path: "/data/uploads/video.mp4", Status: "done", Verdict: "no_violation", Violations: []domain.Violation{}, Created: time.Now().UTC().Format(time.RFC3339Nano)}
	if e := db.Save(j); e != nil {
		t.Fatal(e)
	}
	if e := db.Preferences(u.ID, "dark", "kk"); e != nil {
		t.Fatal(e)
	}
	if e := db.SaveSession("session", u.ID, time.Now().Add(time.Hour).Unix()); e != nil {
		t.Fatal(e)
	}
	other, e := db.ListForOwner("other")
	if e != nil || len(other) != 0 {
		t.Fatal("owner isolation failed")
	}
	db.Close()
	reopened, e := NewPostgres(dsn)
	if e != nil {
		t.Fatal(e)
	}
	defer reopened.Close()
	saved, e := reopened.Get(j.ID)
	if e != nil || saved.OwnerID != u.ID || saved.Path != j.Path {
		t.Fatal("job persistence failed", e)
	}
	user, e := reopened.SessionUser("session", time.Now().Unix())
	if e != nil || user.Theme != "dark" || user.Language != "kk" {
		t.Fatal("session/preferences persistence failed", e)
	}
	if e = reopened.DeleteSession("session"); e != nil {
		t.Fatal(e)
	}
	if _, e = reopened.SessionUser("session", time.Now().Unix()); e != domain.ErrNotFound {
		t.Fatal("session not revoked")
	}
}
func TestPostgresMigration(t *testing.T) {
	pg, _ := postgresTest(t)
	dir := t.TempDir()
	source := filepath.Join(dir, "cashguard.db")
	legacy, e := NewSQLite(source)
	if e != nil {
		t.Fatal(e)
	}
	defer legacy.Close()
	u := domain.User{ID: "legacy-user", Email: "legacy@example.com", PasswordHash: "existing-hash", Theme: "dark", Language: "en"}
	if e = legacy.CreateUser(u); e != nil {
		t.Fatal(e)
	}
	id := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	uploads := filepath.Join(dir, "uploads")
	os.MkdirAll(uploads, 0700)
	path := filepath.Join(uploads, id+".mp4")
	os.WriteFile(path, []byte("private recording"), 0600)
	if e = legacy.Save(domain.Job{ID: id, OwnerID: u.ID, Path: path, Filename: "recording.mp4", Status: "done", Created: "2026-01-01"}); e != nil {
		t.Fatal(e)
	}
	dest := t.TempDir()
	count, e := pg.MigrateSQLite(source, dest)
	if e != nil || count != 1 {
		t.Fatalf("migration: %d %v", count, e)
	}
	saved, e := pg.Get(id)
	if e != nil || saved.Path != filepath.Join(dest, "uploads", id+".mp4") {
		t.Fatal("migrated path incorrect", e)
	}
	content, e := os.ReadFile(saved.Path)
	if e != nil || string(content) != "private recording" {
		t.Fatal("video copy failed")
	}
	restored, e := pg.UserByID(u.ID)
	if e != nil || restored.PasswordHash != u.PasswordHash || restored.Theme != "dark" {
		t.Fatal("account lost")
	}
	if _, e = pg.MigrateSQLite(source, dest); e == nil {
		t.Fatal("nonempty target should not be overwritten")
	}
	if _, e = os.Stat(path); e != nil {
		t.Fatal("source changed")
	}
}
