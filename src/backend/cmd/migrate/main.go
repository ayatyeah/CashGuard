package main

import (
	"cashguard/internal/infrastructure"
	"log"
	"os"
)

func main() {
	dsn := os.Getenv("DATABASE_URL")
	source := os.Getenv("SQLITE_PATH")
	destination := os.Getenv("DATA_DIR")
	if dsn == "" || source == "" || destination == "" {
		log.Fatal("DATABASE_URL, SQLITE_PATH and DATA_DIR are required")
	}
	repo, e := infrastructure.NewPostgres(dsn)
	if e != nil {
		log.Fatal(e)
	}
	defer repo.Close()
	count, e := repo.MigrateSQLite(source, destination)
	if e != nil {
		log.Fatal(e)
	}
	log.Printf("Migration complete: accounts, sessions and %d jobs imported; source unchanged", count)
}
