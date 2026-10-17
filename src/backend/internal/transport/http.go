package transport

import (
	"cashguard/internal/domain"
	"cashguard/internal/usecase"
	"crypto/subtle"
	"errors"
	"github.com/gin-gonic/gin"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var idPattern = regexp.MustCompile(`^[a-f0-9]{32}$`)

func Router(s *usecase.Service, key, uploads string) *gin.Engine {
	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())
	_ = r.SetTrustedProxies(nil)
	r.GET("/api/health", func(c *gin.Context) { c.JSON(200, gin.H{"status": "ok", "demo": s.Demo}) })
	api := r.Group("/api")
	api.Use(func(c *gin.Context) {
		c.Header("Cache-Control", "no-store")
		if supplied := c.GetHeader("X-API-Key"); supplied != "" && subtle.ConstantTimeCompare([]byte(supplied), []byte(key)) == 1 {
			c.Set("owner", "jury")
			c.Next()
			return
		}
		c.AbortWithStatusJSON(401, gin.H{"error": "invalid_api_key"})
	})
	api.GET("/jobs", func(c *gin.Context) {
		jobs, e := s.Repo.ListForOwner(c.GetString("owner"))
		if e != nil {
			c.JSON(500, gin.H{"error": "Не удалось прочитать историю"})
			return
		}
		c.JSON(200, jobs)
	})
	api.GET("/jobs/:id", func(c *gin.Context) {
		id := c.Param("id")
		if !idPattern.MatchString(id) {
			c.JSON(404, gin.H{"error": "Видео не найдено"})
			return
		}
		j, e := s.Repo.Get(id)
		if errors.Is(e, domain.ErrNotFound) {
			c.JSON(404, gin.H{"error": "Видео не найдено"})
			return
		}
		if e != nil {
			c.JSON(500, gin.H{"error": "Ошибка чтения"})
			return
		}
		if j.OwnerID != c.GetString("owner") && !(j.OwnerID == "" && c.GetString("owner") == "jury") {
			c.JSON(404, gin.H{"error": "Видео не найдено"})
			return
		}
		c.JSON(200, j)
	})
	api.POST("/jobs", func(c *gin.Context) {
		upload(c, s, uploads, "video", func(c *gin.Context, j domain.Job) { c.JSON(202, j) })
	})
	// Jury API (ТЗ §10): POST /api/v1/analyze (field file) → job_id; GET /api/v1/jobs/{id} → status and result.
	api.POST("/v1/analyze", func(c *gin.Context) {
		upload(c, s, uploads, "file", func(c *gin.Context, j domain.Job) {
			c.JSON(200, gin.H{"job_id": j.ID, "status": "processing"})
		})
	})
	api.GET("/v1/jobs/:id", func(c *gin.Context) {
		id := c.Param("id")
		if !idPattern.MatchString(id) {
			c.JSON(404, gin.H{"error": "job_not_found"})
			return
		}
		j, e := s.Repo.Get(id)
		if errors.Is(e, domain.ErrNotFound) || (e == nil && j.OwnerID != c.GetString("owner") && j.OwnerID != "") {
			c.JSON(404, gin.H{"error": "job_not_found"})
			return
		}
		if e != nil {
			c.JSON(500, gin.H{"error": "server_error"})
			return
		}
		c.JSON(200, juryResult(j))
	})
	api.GET("/jobs/:id/video", func(c *gin.Context) {
		id := c.Param("id")
		if !idPattern.MatchString(id) {
			c.Status(404)
			return
		}
		j, e := s.Repo.Get(id)
		if e != nil || j.OwnerID != c.GetString("owner") || j.Path == "" {
			c.Status(404)
			return
		}
		c.Header("X-Content-Type-Options", "nosniff")
		c.File(j.Path)
	})
	return r
}

// upload stores a multipart video from the given form field, queues it and hands the job to done.
func upload(c *gin.Context, s *usecase.Service, uploads, field string, done func(*gin.Context, domain.Job)) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 501<<20)
	if e := c.Request.ParseMultipartForm(8 << 20); e != nil {
		c.JSON(400, gin.H{"error": "Некорректная загрузка или файл больше 500 МБ"})
		return
	}
	defer c.Request.MultipartForm.RemoveAll()
	f, h, e := c.Request.FormFile(field)
	if e != nil {
		c.JSON(400, gin.H{"error": "Поле " + field + " обязательно"})
		return
	}
	defer f.Close()
	ext := strings.ToLower(filepath.Ext(h.Filename))
	if ext != ".mp4" && ext != ".webm" && ext != ".mov" {
		c.JSON(400, gin.H{"error": "Поддерживаются MP4, MOV и WebM"})
		return
	}
	if h.Size == 0 || h.Size > 500<<20 {
		c.JSON(400, gin.H{"error": "Размер видео должен быть от 1 байта до 500 МБ"})
		return
	}
	header := make([]byte, 512)
	n, _ := f.Read(header)
	mime := http.DetectContentType(header[:n])
	if !strings.HasPrefix(mime, "video/") && !(n > 12 && string(header[4:8]) == "ftyp") {
		c.JSON(400, gin.H{"error": "Файл не является поддерживаемым видео"})
		return
	}
	if _, e = f.Seek(0, 0); e != nil {
		c.JSON(500, gin.H{"error": "Ошибка чтения видео"})
		return
	}
	id, e := usecase.NewID()
	if e != nil {
		c.JSON(500, gin.H{"error": "Ошибка создания задания"})
		return
	}
	path := filepath.Join(uploads, id+ext)
	out, e := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		c.JSON(500, gin.H{"error": "Ошибка сохранения видео"})
		return
	}
	_, e = io.Copy(out, f)
	closeErr := out.Close()
	if e != nil || closeErr != nil {
		os.Remove(path)
		c.JSON(500, gin.H{"error": "Ошибка сохранения видео"})
		return
	}
	j, e := s.Submit(id, filepath.Base(h.Filename), path, c.GetString("owner"))
	if e != nil {
		os.Remove(path)
		c.JSON(503, gin.H{"error": "Не удалось поставить видео в очередь"})
		return
	}
	done(c, j)
}

// juryResult is the response schema from ТЗ §10: only processing, done and failed are exposed.
func juryResult(j domain.Job) gin.H {
	out := gin.H{"job_id": j.ID, "status": j.Status}
	switch j.Status {
	case "done":
		out["verdict"] = j.Verdict
		out["violations"] = j.Violations
	case "failed":
		out["error"] = j.Error
	default:
		out["status"] = "processing"
	}
	return out
}
