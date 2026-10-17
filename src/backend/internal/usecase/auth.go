package usecase

import (
	"cashguard/internal/domain"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"golang.org/x/crypto/bcrypt"
	"net/mail"
	"strings"
	"time"
	"unicode/utf8"
)

type Auth struct{ Store domain.Accounts }

func SessionHash(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}
func (a *Auth) Register(email, password string) (domain.User, string, error) {
	email = strings.ToLower(strings.TrimSpace(email))
	address, e := mail.ParseAddress(email)
	if e != nil || address.Address != email || len(email) > 254 || utf8.RuneCountInString(password) < 8 || len(password) > 72 {
		return domain.User{}, "", errors.New("invalid_registration")
	}
	hash, e := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if e != nil {
		return domain.User{}, "", e
	}
	id, e := NewID()
	if e != nil {
		return domain.User{}, "", e
	}
	u := domain.User{ID: id, Email: email, PasswordHash: string(hash), Theme: "light", Language: "ru"}
	if e = a.Store.CreateUser(u); e != nil {
		return domain.User{}, "", e
	}
	return a.session(u)
}
func (a *Auth) Login(email, password string) (domain.User, string, error) {
	if len(password) > 72 {
		return domain.User{}, "", domain.ErrCredentials
	}
	u, e := a.Store.UserByEmail(strings.ToLower(strings.TrimSpace(email)))
	if e != nil {
		if !errors.Is(e, domain.ErrNotFound) {
			return domain.User{}, "", e
		}
		_ = bcrypt.CompareHashAndPassword([]byte("$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy"), []byte(password))
		return domain.User{}, "", domain.ErrCredentials
	}
	if bcrypt.CompareHashAndPassword([]byte(u.PasswordHash), []byte(password)) != nil {
		return domain.User{}, "", domain.ErrCredentials
	}
	return a.session(u)
}
func (a *Auth) session(u domain.User) (domain.User, string, error) {
	one, e := NewID()
	if e != nil {
		return u, "", e
	}
	two, e := NewID()
	if e != nil {
		return u, "", e
	}
	token := one + two
	e = a.Store.SaveSession(SessionHash(token), u.ID, time.Now().Add(7*24*time.Hour).Unix())
	return u, token, e
}
func (a *Auth) Authenticate(token string) (domain.User, error) {
	if len(token) != 64 {
		return domain.User{}, domain.ErrCredentials
	}
	return a.Store.SessionUser(SessionHash(token), time.Now().Unix())
}
func (a *Auth) Logout(token string) error { return a.Store.DeleteSession(SessionHash(token)) }
func (a *Auth) Preferences(id, theme, language string) error {
	if (theme != "light" && theme != "dark") || (language != "ru" && language != "en" && language != "kk") {
		return errors.New("invalid_preferences")
	}
	return a.Store.Preferences(id, theme, language)
}
