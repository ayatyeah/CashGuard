package transport

import (
	"cashguard/internal/domain"
	"cashguard/internal/usecase"
	"errors"
	"github.com/gin-gonic/gin"
	"net/http"
	"strings"
	"sync"
	"time"
)

func bearer(c *gin.Context) string {
	return strings.TrimPrefix(c.GetHeader("Authorization"), "Bearer ")
}
func authRoutes(r *gin.Engine, a *usecase.Auth) {
	type attempt struct {
		count int
		until time.Time
	}
	var mu sync.Mutex
	attempts := map[string]attempt{}
	limit := func(c *gin.Context) {
		mu.Lock()
		now := time.Now()
		for ip, v := range attempts {
			if now.After(v.until) {
				delete(attempts, ip)
			}
		}
		ip := c.ClientIP()
		v := attempts[ip]
		if v.until.IsZero() {
			v.until = now.Add(time.Minute)
		}
		v.count++
		attempts[ip] = v
		mu.Unlock()
		if v.count > 20 {
			c.AbortWithStatusJSON(429, gin.H{"error": "too_many_attempts"})
			return
		}
		c.Next()
	}
	auth := r.Group("/api/auth")
	auth.Use(limit)
	for _, mode := range []string{"register", "login"} {
		mode := mode
		auth.POST("/"+mode, func(c *gin.Context) {
			c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 4096)
			var data struct {
				Email    string `json:"email"`
				Password string `json:"password"`
			}
			if c.ShouldBindJSON(&data) != nil {
				c.JSON(400, gin.H{"error": "invalid_registration"})
				return
			}
			var u domain.User
			var token string
			var e error
			if mode == "register" {
				u, token, e = a.Register(data.Email, data.Password)
			} else {
				u, token, e = a.Login(data.Email, data.Password)
			}
			if e != nil {
				status := 500
				message := "server_error"
				switch {
				case errors.Is(e, domain.ErrCredentials):
					status = 401
					message = "invalid_credentials"
				case errors.Is(e, domain.ErrAccountExists):
					status = 409
					message = "account_exists"
				case e.Error() == "invalid_registration":
					status = 400
					message = "invalid_registration"
				}
				c.JSON(status, gin.H{"error": message})
				return
			}
			c.Header("Cache-Control", "no-store")
			c.JSON(200, gin.H{"user": u, "token": token})
		})
	}
	protected := r.Group("/api/auth")
	protected.Use(func(c *gin.Context) {
		u, e := a.Authenticate(bearer(c))
		if e != nil {
			c.AbortWithStatusJSON(401, gin.H{"error": "session_expired"})
			return
		}
		c.Set("user", u)
		c.Header("Cache-Control", "no-store")
		c.Next()
	})
	protected.GET("/me", func(c *gin.Context) { c.JSON(200, c.MustGet("user")) })
	protected.POST("/logout", func(c *gin.Context) {
		if a.Logout(bearer(c)) != nil {
			c.JSON(500, gin.H{"error": "server_error"})
			return
		}
		c.Status(204)
	})
	protected.PATCH("/preferences", func(c *gin.Context) {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 1024)
		var data struct {
			Theme    string `json:"theme"`
			Language string `json:"language"`
		}
		if c.ShouldBindJSON(&data) != nil {
			c.JSON(400, gin.H{"error": "invalid_preferences"})
			return
		}
		u := c.MustGet("user").(domain.User)
		if e := a.Preferences(u.ID, data.Theme, data.Language); e != nil {
			status := 500
			if e.Error() == "invalid_preferences" {
				status = 400
			}
			c.JSON(status, gin.H{"error": "preferences_failed"})
			return
		}
		u.Theme = data.Theme
		u.Language = data.Language
		c.JSON(200, u)
	})
}
