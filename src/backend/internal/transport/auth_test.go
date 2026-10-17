package transport

import (
	"cashguard/internal/domain"
	"cashguard/internal/infrastructure"
	"cashguard/internal/usecase"
	"net/http/httptest"
	"path/filepath"
	"testing"
)

func TestAPIKeyOnly(t *testing.T) {
	repo, e := infrastructure.NewSQLite(filepath.Join(t.TempDir(), "test.db"))
	if e != nil {
		t.Fatal(e)
	}
	defer repo.Close()
	r := Router(usecase.New(repo, analyzerStub{}, false), "test-key", t.TempDir())
	for _, tc := range []struct {
		method, path, header, value string
		status                      int
	}{
		{"GET", "/api/jobs", "X-API-Key", "test-key", 200},
		{"GET", "/api/jobs", "X-API-Key", "wrong", 401},
		{"GET", "/api/jobs", "Authorization", "Bearer test-key", 401},
		{"GET", "/api/jobs", "", "", 401},
		{"POST", "/api/auth/register", "", "", 404},
		{"POST", "/api/auth/login", "", "", 404},
	} {
		req := httptest.NewRequest(tc.method, tc.path, nil)
		if tc.header != "" {
			req.Header.Set(tc.header, tc.value)
		}
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		if w.Code != tc.status {
			t.Errorf("%s %s: got %d want %d", tc.method, tc.path, w.Code, tc.status)
		}
	}
}
func TestSQLiteReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "persist.db")
	db, e := infrastructure.NewSQLite(path)
	if e != nil {
		t.Fatal(e)
	}
	a := &usecase.Auth{Store: db}
	u, token, e := a.Register("persist@example.com", "a-long-password")
	if e != nil {
		t.Fatal(e)
	}
	if e = a.Preferences(u.ID, "dark", "en"); e != nil {
		t.Fatal(e)
	}
	if e = db.Save(domain.Job{ID: "persist", OwnerID: u.ID, Status: "done", Filename: "persist.mp4"}); e != nil {
		t.Fatal(e)
	}
	db.Close()
	db, e = infrastructure.NewSQLite(path)
	if e != nil {
		t.Fatal(e)
	}
	defer db.Close()
	a = &usecase.Auth{Store: db}
	restored, e := a.Authenticate(token)
	if e != nil || restored.Theme != "dark" || restored.Language != "en" {
		t.Fatalf("session/preferences lost: %+v %v", restored, e)
	}
	jobs, e := db.ListForOwner(u.ID)
	if e != nil || len(jobs) != 1 {
		t.Fatal("jobs not persistent")
	}
}
