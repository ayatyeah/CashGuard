package transport

import (
	"bytes"
	"cashguard/internal/domain"
	"cashguard/internal/infrastructure"
	"cashguard/internal/usecase"
	"context"
	"encoding/json"
	"mime/multipart"

	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type analyzerStub struct{}

func (analyzerStub) Analyze(_ context.Context, j domain.Job) (domain.Job, error) {
	j.Verdict = "no_violation"
	return j, nil
}
func TestAPI(t *testing.T) {
	dir := t.TempDir()
	repo, e := infrastructure.NewRepository(filepath.Join(dir, "jobs"))
	if e != nil {
		t.Fatal(e)
	}
	s := usecase.New(repo, analyzerStub{}, false)
	r := Router(s, "test-key-123456789", dir)
	request := func(method, path string, body *bytes.Buffer, contentType, key string) *httptest.ResponseRecorder {
		t.Helper()
		if body == nil {
			body = &bytes.Buffer{}
		}
		req := httptest.NewRequest(method, path, body)
		req.Header.Set("X-API-Key", key)
		req.Header.Set("Content-Type", contentType)
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		return w
	}
	if w := request("GET", "/api/jobs", nil, "", ""); w.Code != 401 {
		t.Fatalf("unauthorized: %d", w.Code)
	}
	if w := request("GET", "/api/jobs/not-an-id", nil, "", "test-key-123456789"); w.Code != 404 {
		t.Fatalf("bad id: %d", w.Code)
	}
	for _, tc := range []struct {
		name   string
		data   []byte
		status int
	}{{"bad.mp4", []byte("not a video"), 400}, {"bad.exe", []byte("not a video"), 400}, {"empty.mp4", nil, 400}, {"valid.mp4", append([]byte{0, 0, 0, 24, 'f', 't', 'y', 'p', 'i', 's', 'o', 'm'}, make([]byte, 500)...), 202}} {
		t.Run(tc.name, func(t *testing.T) {
			var body bytes.Buffer
			writer := multipart.NewWriter(&body)
			part, _ := writer.CreateFormFile("video", tc.name)
			_, _ = part.Write(tc.data)
			_ = writer.Close()
			w := request("POST", "/api/jobs", &body, writer.FormDataContentType(), "test-key-123456789")
			if w.Code != tc.status {
				t.Fatalf("status %d: %s", w.Code, w.Body.String())
			}
			if tc.status == 202 {
				var job domain.Job
				_ = json.Unmarshal(w.Body.Bytes(), &job)
				deadline := time.Now().Add(2 * time.Second)
				for time.Now().Before(deadline) {
					saved, e := repo.Get(job.ID)
					if e != nil {
						t.Fatal(e)
					}
					if saved.Status == "done" {
						if saved.Verdict != "no_violation" {
							t.Fatal(saved)
						}
						return
					}
					time.Sleep(10 * time.Millisecond)
				}
				t.Fatal("job did not finish")
			}
		})
	}
	jobs, e := repo.List()
	if e != nil || len(jobs) != 1 {
		t.Fatalf("history: %v %v", jobs, e)
	}
	if _, e = os.Stat(filepath.Join(dir, jobs[0].ID+".mp4")); e != nil {
		t.Fatal(e)
	}

}
