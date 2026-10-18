package usecase

import (
	"cashguard/internal/domain"
	"cashguard/internal/infrastructure"
	"context"
	"os"
	"strconv"
	"testing"
	"time"
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

// deadlineRecorder records how much time each analysis was given.
type deadlineRecorder struct{ got chan time.Duration }

func (d deadlineRecorder) Analyze(ctx context.Context, j domain.Job) (domain.Job, error) {
	dl, _ := ctx.Deadline()
	d.got <- time.Until(dl)
	j.Verdict = "no_violation"
	return j, nil
}

// Jobs uploaded together share the remaining time; a lone job keeps its whole deadline.
func TestQueuedJobsShareDeadline(t *testing.T) {
	repo, e := infrastructure.NewRepository(t.TempDir())
	if e != nil {
		t.Fatal(e)
	}
	rec := deadlineRecorder{got: make(chan time.Duration, 4)}
	s := &Service{Repo: repo, Analyzer: rec, queue: make(chan domain.Job, 20), Deadline: 280 * time.Second}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	for _, id := range []string{"a", "b", "c", "d"} {
		s.queue <- domain.Job{ID: id, Created: now, Status: "queued"}
	}
	close(s.queue)
	s.worker()
	first := <-rec.got
	if first > 75*time.Second || first < 60*time.Second {
		t.Fatalf("first of 4 queued jobs should get ~1/4 of 280 s, got %v", first)
	}
	for i := 0; i < 3; i++ {
		if d := <-rec.got; d < MinBudget {
			t.Fatalf("job %d starved: %v", i+2, d)
		}
	}
}

// arrivalRecorder enqueues more jobs while the first one runs and reports the deadline the analyzer would see.
type arrivalRecorder struct {
	s    *Service
	seen chan float64
}

func (a arrivalRecorder) Analyze(ctx context.Context, j domain.Job) (domain.Job, error) {
	if j.ID == "first" {
		now := time.Now().UTC().Format(time.RFC3339Nano)
		for _, id := range []string{"b", "c", "d"} { // the rest of the batch lands while "first" is analysed
			a.s.queue <- domain.Job{ID: id, Created: now, Status: "queued"}
		}
		time.Sleep(1500 * time.Millisecond)
		b, _ := os.ReadFile(j.DeadlineFile)
		v, _ := strconv.ParseFloat(string(b), 64)
		a.seen <- v - float64(time.Now().UnixNano())/1e9
	}
	j.Verdict = "no_violation"
	return j, nil
}

func TestRunningJobShrinksWhenBatchArrives(t *testing.T) {
	repo, e := infrastructure.NewRepository(t.TempDir())
	if e != nil {
		t.Fatal(e)
	}
	s := &Service{Repo: repo, queue: make(chan domain.Job, 20), Deadline: 280 * time.Second}
	rec := arrivalRecorder{s: s, seen: make(chan float64, 1)}
	s.Analyzer = rec
	s.queue <- domain.Job{ID: "first", Created: time.Now().UTC().Format(time.RFC3339Nano), Status: "queued"}
	go s.worker()
	left := <-rec.seen
	if left > 80 || left < 55 { // 280 s shared by 4 jobs ≈ 70 s
		t.Fatalf("running job should shrink to ~1/4 of its deadline once 3 more jobs wait, has %.0f s", left)
	}
}
