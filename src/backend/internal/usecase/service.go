package usecase

import (
	"cashguard/internal/domain"
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

type Service struct {
	Repo     domain.Repository
	Analyzer domain.Analyzer
	queue    chan domain.Job
	Demo     bool
	// Deadline is counted from upload time: the jury waits at most 5 minutes per video (ТЗ §10).
	Deadline time.Duration
}

// MinBudget is the least time worth starting an analysis with; below it the job fails explicitly.
// A short, low-frame-rate result is better than "unprocessed", so it is kept small.
const MinBudget = 22 * time.Second

func writeDeadline(path string, t time.Time) {
	_ = os.WriteFile(path, []byte(strconv.FormatFloat(float64(t.UnixNano())/1e9, 'f', 3, 64)), 0600)
}

// shrinkWhileRunning lowers the running job's deadline when videos arrive during its analysis (uploads of a batch
// land seconds apart), so every job of the batch keeps a share of its own 5 minutes. It never extends the deadline.
func (s *Service) shrinkWhileRunning(path string, full, cur time.Time, stop <-chan struct{}) {
	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	for {
		select {
		case <-stop:
			return
		case <-tick.C:
			if w := len(s.queue); w > 0 {
				if share := time.Now().Add(time.Until(full) / time.Duration(1+w)); share.Before(cur) {
					cur = share
					writeDeadline(path, cur)
				}
			}
		}
	}
}

func New(repo domain.Repository, a domain.Analyzer, demo bool) *Service {
	s := &Service{Repo: repo, Analyzer: a, queue: make(chan domain.Job, 20), Demo: demo, Deadline: 285 * time.Second}
	go s.worker()
	return s
}
func NewID() (string, error) {
	b := make([]byte, 16)
	_, e := rand.Read(b)
	return hex.EncodeToString(b), e
}
func (s *Service) Submit(id, name, path string, owners ...string) (domain.Job, error) {
	owner := "jury"
	if len(owners) > 0 {
		owner = owners[0]
	}
	j := domain.Job{OwnerID: owner, ID: id, Filename: name, Path: path, Created: time.Now().UTC().Format(time.RFC3339Nano), Status: "queued", Violations: []domain.Violation{}, Demo: s.Demo}
	if e := s.Repo.Save(j); e != nil {
		return j, e
	}
	select {
	case s.queue <- j:
		return j, nil
	default:
		j.Status = "failed"
		j.Error = "Очередь заполнена. Попробуйте позже."
		_ = s.Repo.Save(j)
		return j, fmt.Errorf("queue full")
	}
}
func (s *Service) worker() {
	for j := range s.queue {
		j.Status = "processing"
		if e := s.Repo.Save(j); e != nil {
			log.Printf("save processing: %v", e)
			continue
		}
		deadline := time.Now().Add(s.Deadline)
		if created, e := time.Parse(time.RFC3339Nano, j.Created); e == nil {
			deadline = created.Add(s.Deadline)
		}
		// Fair share: jobs uploaded together must all finish within their own 5 minutes, so the current job
		// may use only its share of the remaining time (sequential uploads keep the full deadline).
		if waiting := len(s.queue); waiting > 0 {
			if share := time.Now().Add(time.Until(deadline) / time.Duration(1+waiting)); share.Before(deadline) {
				deadline = share
			}
		}
		full := deadline
		if created, e := time.Parse(time.RFC3339Nano, j.Created); e == nil {
			full = created.Add(s.Deadline)
		}
		j.DeadlineFile = filepath.Join(os.TempDir(), "cashguard-"+j.ID+".deadline")
		writeDeadline(j.DeadlineFile, deadline)
		stop := make(chan struct{})
		go s.shrinkWhileRunning(j.DeadlineFile, full, deadline, stop)
		var result domain.Job
		var e error
		if time.Until(deadline) < MinBudget {
			e = fmt.Errorf("не хватило времени на анализ: видео слишком долго ждало в очереди")
		} else {
			ctx, cancel := context.WithDeadline(context.Background(), deadline)
			result, e = s.Analyzer.Analyze(ctx, j)
			if ctx.Err() == context.DeadlineExceeded {
				e = fmt.Errorf("анализ не уложился в отведённое время")
			}
			cancel()
		}
		close(stop)
		_ = os.Remove(j.DeadlineFile)
		j.DeadlineFile = ""
		if e != nil {
			log.Printf("job %s failed: %v", j.ID, e)
			j.Status = "failed"
			j.Error = e.Error()
		} else {
			j = result
			j.Status = "done"
		}
		if e = s.Repo.Save(j); e != nil {
			log.Printf("save result: %v", e)
		}
	}
}

// Interrupted jobs are explicit failures rather than remaining in processing forever.
func (s *Service) Recover() error {
	jobs, e := s.Repo.List()
	if e != nil {
		return e
	}
	for _, j := range jobs {
		if j.Status == "queued" || j.Status == "processing" {
			j.Status = "failed"
			j.Error = "Обработка прервана перезапуском сервера. Загрузите видео повторно."
			if e = s.Repo.Save(j); e != nil {
				return e
			}
		}
	}
	return nil
}
