package infrastructure

import (
	"cashguard/internal/domain"
	"context"
	"encoding/json"
	"os"
	"testing"
)

// The test binary doubles as an external analyzer, including on Windows.
func TestMain(m *testing.M) {
	if os.Getenv("CASHGUARD_TEST_ANALYZER") == "1" {
		_, _ = os.Stdout.WriteString(`{"status":"done","verdict":"no_violation","violations":[],"warnings":["partial analysis"]}`)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestAnalyzerWarningsSurvivePersistence(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("CASHGUARD_TEST_ANALYZER", "1")
	j, err := (CommandAnalyzer{Command: executable}).Analyze(context.Background(), domain.Job{ID: "warnings"})
	if err != nil {
		t.Fatal(err)
	}
	repo, err := NewRepository(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err = repo.Save(j); err != nil {
		t.Fatal(err)
	}
	saved, err := repo.Get(j.ID)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(saved)
	if err != nil {
		t.Fatal(err)
	}
	var response map[string]any
	if err = json.Unmarshal(data, &response); err != nil {
		t.Fatal(err)
	}
	warnings, ok := response["warnings"].([]any)
	if !ok || len(warnings) != 1 || warnings[0] != "partial analysis" {
		t.Fatalf("warnings lost between analyzer and API JSON: %s", data)
	}
}

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
