package domain

import (
	"context"
	"errors"
)

var ErrNotFound = errors.New("job not found")

type Frame struct {
	Timestamp float64 `json:"timestamp_sec"`
	Image     string  `json:"image_base64"`
}
type Rule struct {
	Title   string `json:"title"`
	Section string `json:"section"`
	URL     string `json:"url"`
}
type Violation struct {
	Type        string  `json:"type"`
	Timecode    float64 `json:"timecode_sec"`
	Description string  `json:"description"`
	Frames      []Frame `json:"evidence_frames"`
	Rule        *Rule   `json:"rule,omitempty"`
}
type Job struct {
	OwnerID string `json:"-"`
	// DeadlineFile holds the job's current absolute deadline (unix seconds); the worker lowers it while the job runs
	// when more videos arrive, and the analyzer re-reads it (fair share of the 5-minute limit). Not persisted.
	DeadlineFile string      `json:"-"`
	ID           string      `json:"job_id"`
	Status       string      `json:"status"`
	Verdict      string      `json:"verdict,omitempty"`
	Violations   []Violation `json:"violations"`
	Warnings     []string    `json:"warnings,omitempty"`
	Filename     string      `json:"filename"`
	Created      string      `json:"created_at"`
	Error        string      `json:"error,omitempty"`
	Demo         bool        `json:"demo"`
	Path         string      `json:"-"`
}
type Repository interface {
	Save(Job) error
	Get(string) (Job, error)
	List() ([]Job, error)
	ListForOwner(string) ([]Job, error)
}
type Analyzer interface {
	Analyze(context.Context, Job) (Job, error)
}
