package usecase

import (
	"cashguard/internal/domain"
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"
	"time"
)

type Service struct {
	Repo     domain.Repository
	Analyzer domain.Analyzer
	queue    chan domain.Job
	Demo     bool
}

func New(repo domain.Repository, a domain.Analyzer, demo bool) *Service {
	s := &Service{Repo: repo, Analyzer: a, queue: make(chan domain.Job, 20), Demo: demo}
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
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
		result, e := s.Analyzer.Analyze(ctx, j)
		cancel()
		if e != nil {
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
