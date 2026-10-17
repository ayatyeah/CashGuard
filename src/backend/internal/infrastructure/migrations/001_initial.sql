CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
CREATE TABLE IF NOT EXISTS users (
 id TEXT PRIMARY KEY,
 email TEXT NOT NULL UNIQUE,
 password_hash TEXT NOT NULL,
 theme TEXT NOT NULL DEFAULT 'light' CHECK(theme IN ('light','dark')),
 language TEXT NOT NULL DEFAULT 'ru' CHECK(language IN ('ru','en','kk'))
);
INSERT INTO users(id,email,password_hash) VALUES('jury','jury@internal.invalid','!disabled') ON CONFLICT(id) DO NOTHING;
CREATE TABLE IF NOT EXISTS sessions (
 hash TEXT PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 expires BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires);
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY,
 owner_id TEXT NOT NULL REFERENCES users(id),
 path TEXT NOT NULL,
 result JSONB NOT NULL,
 created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_owner_created ON jobs(owner_id,created_at DESC);
INSERT INTO schema_migrations(version) VALUES(1) ON CONFLICT(version) DO NOTHING;
