package infrastructure

import (
	"cashguard/internal/domain"
	"encoding/json"

	"os"
	"path/filepath"
	"sort"
	"sync"
)

type Repository struct {
	dir string
	mu  sync.RWMutex
}

func NewRepository(dir string) (*Repository, error) {
	err := os.MkdirAll(dir, 0700)
	return &Repository{dir: dir}, err
}
func (r *Repository) Save(j domain.Job) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	b, e := json.Marshal(j)
	if e != nil {
		return e
	}
	p := filepath.Join(r.dir, j.ID+".json")
	if e = os.WriteFile(p+".tmp", b, 0600); e != nil {
		return e
	}
	return os.Rename(p+".tmp", p)
}
func (r *Repository) Get(id string) (domain.Job, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.read(id)
}
func (r *Repository) read(id string) (j domain.Job, e error) {
	b, e := os.ReadFile(filepath.Join(r.dir, id+".json"))
	if os.IsNotExist(e) {
		return j, domain.ErrNotFound
	}
	if e != nil {
		return j, e
	}
	e = json.Unmarshal(b, &j)
	return
}
func (r *Repository) List() ([]domain.Job, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	files, e := filepath.Glob(filepath.Join(r.dir, "*.json"))
	if e != nil {
		return nil, e
	}
	out := []domain.Job{}
	for _, p := range files {
		b, e := os.ReadFile(p)
		if e != nil {
			return nil, e
		}
		var j domain.Job
		if e = json.Unmarshal(b, &j); e != nil {
			return nil, e
		}
		out = append(out, j)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Created > out[j].Created })
	return out, nil
}
func (r *Repository) ListForOwner(owner string) ([]domain.Job, error) {
	jobs, e := r.List()
	if e != nil {
		return nil, e
	}
	out := []domain.Job{}
	for _, j := range jobs {
		if j.OwnerID == owner || (j.OwnerID == "" && owner == "jury") {
			out = append(out, j)
		}
	}
	return out, nil
}
