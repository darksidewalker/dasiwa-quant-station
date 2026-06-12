package main

import (
	"log"

	"dasiwa-quant-station/internal/app"
)

func main() {
	server, err := app.NewServer()
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("DaSiWa Quant Station listening on %s", server.Addr())
	log.Fatal(server.ListenAndServe())
}
