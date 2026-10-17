package main

import (
	"cashguard/internal/domain"
	"cashguard/internal/infrastructure"
	"cashguard/internal/transport"
	"cashguard/internal/usecase"
	"context"
	"github.com/gin-gonic/gin"
	_ "image/jpeg"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

func main() {
	key := os.Getenv("API_KEY")
	if len(key) < 16 {
		log.Fatal("API_KEY must contain at least 16 characters")
	}
	dir := os.Getenv("DATA_DIR")
	if dir == "" {
		dir = "data"
	}
	dir, e := filepath.Abs(dir)
	if e != nil {
		log.Fatal(e)
	}
	uploads := filepath.Join(dir, "uploads")
	if e = os.MkdirAll(uploads, 0700); e != nil {
		log.Fatal(e)
	}
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		log.Fatal("DATABASE_URL is required (PostgreSQL)")
	}
	repo, e := infrastructure.NewPostgres(dsn)
	if e != nil {
		log.Fatal(e)
	}
	defer repo.Close()
	if e = repo.ImportLegacy(filepath.Join(dir, "jobs"), uploads); e != nil {
		log.Fatal(e)
	}
	demo := os.Getenv("DEMO_MODE") == "true"
	var a domain.Analyzer = infrastructure.CommandAnalyzer{Command: os.Getenv("ANALYZER_COMMAND")}
	if demo {
		a = infrastructure.DemoAnalyzer{}
	}
	s := usecase.New(repo, a, demo)
	if e = s.Recover(); e != nil {
		log.Fatal(e)
	}
	addr := os.Getenv("ADDR")
	if addr == "" {
		addr = "127.0.0.1:8080"
	}
	router := transport.Router(s, key, uploads)
	if static := os.Getenv("STATIC_DIR"); static != "" { // single-container deploy: React build served by Go
		router.NoRoute(func(c *gin.Context) {
			p := c.Request.URL.Path
			if len(p) >= 4 && p[:4] == "/api" {
				c.JSON(404, gin.H{"error": "not found"})
				return
			}
			full := filepath.Join(static, filepath.Clean("/"+p))
			if st, err := os.Stat(full); err == nil && !st.IsDir() {
				c.File(full)
				return
			}
			c.File(filepath.Join(static, "index.html"))
		})
	}
	router.GET("/api/ready", func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), 2*time.Second)
		defer cancel()
		if repo.Ping(ctx) != nil {
			c.JSON(503, gin.H{"status": "unavailable"})
			return
		}
		c.JSON(200, gin.H{"status": "ready"})
	})
	server := &http.Server{Addr: addr, Handler: router, ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 5 * time.Minute, WriteTimeout: 5 * time.Minute, IdleTimeout: time.Minute}
	log.Printf("CashGuard listening at %s; demo=%t", addr, demo)
	log.Fatal(server.ListenAndServe())
}
