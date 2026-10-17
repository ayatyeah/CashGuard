package infrastructure

import (
	"cashguard/internal/domain"
	"context"
	"testing"
)

func TestMissingAnalyzerFailsExplicitly(t *testing.T) {
	_, err := (CommandAnalyzer{}).Analyze(context.Background(), domain.Job{})
	if err == nil {
		t.Fatal("missing analyzer must fail")
	}
}
func TestDemoHonorsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := (DemoAnalyzer{}).Analyze(ctx, domain.Job{})
	if err == nil {
		t.Fatal("canceled job must fail")
	}
}
func TestRepositoryUpdatePersists(t *testing.T) {
	dir := t.TempDir()
	repo, e := NewRepository(dir)
	if e != nil {
		t.Fatal(e)
	}
	j := domain.Job{ID: "test", Status: "queued"}
	if e = repo.Save(j); e != nil {
		t.Fatal(e)
	}
	j.Status = "done"
	j.Verdict = "no_violation"
	if e = repo.Save(j); e != nil {
		t.Fatal(e)
	}
	reopened, e := NewRepository(dir)
	if e != nil {
		t.Fatal(e)
	}
	actual, e := reopened.Get("test")
	if e != nil || actual.Status != "done" {
		t.Fatalf("persisted result: %+v %v", actual, e)
	}
}
