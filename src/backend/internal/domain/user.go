package domain

import "errors"

var ErrCredentials = errors.New("invalid_credentials")
var ErrAccountExists = errors.New("account_exists")

type User struct {
	ID           string `json:"id"`
	Email        string `json:"email"`
	PasswordHash string `json:"-"`
	Theme        string `json:"theme"`
	Language     string `json:"language"`
}
type Accounts interface {
	CreateUser(User) error
	UserByEmail(string) (User, error)
	UserByID(string) (User, error)
	SaveSession(hash, userID string, expires int64) error
	SessionUser(hash string, now int64) (User, error)
	DeleteSession(hash string) error
	Preferences(id, theme, language string) error
}
