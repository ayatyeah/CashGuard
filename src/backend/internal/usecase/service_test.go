package usecase

import (
	"cashguard/internal/domain"
	"cashguard/internal/infrastructure"
	"testing"
)

func TestRecoverInterruptedJob(t *testing.T) {
	repo, e := infrastructure.NewRepository(t.TempDir())
	if e != nil {
		t.Fatal(e)
	}
	if e = repo.Save(domain.Job{ID: "abc", Status: "processing"}); e != nil {
		t.Fatal(e)
	}
	s := &Service{Repo: repo}
	if e = s.Recover(); e != nil {
		t.Fatal(e)
	}
	j, e := repo.Get("abc")
	if e != nil || j.Status != "failed" || j.Error == "" {
		t.Fatalf("unexpected recovered job: %+v %v", j, e)
	}
}
