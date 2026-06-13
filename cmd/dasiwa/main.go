package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"time"

	"dasiwa-quant-station/internal/app"
)

func main() {
	server, err := app.NewServer()
	if err != nil {
		log.Fatal(err)
	}
	addr := server.Addr()
	log.Printf("DaSiWa Quant Station listening on %s", addr)

	errCh := make(chan error, 1)
	go func() {
		errCh <- server.ListenAndServe()
	}()

	if os.Getenv("DASIWA_NO_BROWSER") == "" {
		if err := openBrowserWhenReady(addr, errCh); err != nil {
			if startErr, ok := err.(serverStartError); ok {
				log.Fatal(startErr.err)
			}
			log.Printf("Browser did not open automatically: %v", err)
			log.Printf("Open %s in your browser.", addr)
		}
	}

	log.Fatal(<-errCh)
}

func openBrowserWhenReady(url string, errCh <-chan error) error {
	client := http.Client{Timeout: 500 * time.Millisecond}
	deadline := time.After(5 * time.Second)
	ticker := time.NewTicker(150 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case err := <-errCh:
			return serverStartError{err: err}
		case <-deadline:
			return fmt.Errorf("server did not become ready in time")
		case <-ticker.C:
			resp, err := client.Get(url)
			if err == nil {
				_ = resp.Body.Close()
				return openBrowser(url)
			}
		}
	}
}

type serverStartError struct {
	err error
}

func (e serverStartError) Error() string {
	return e.err.Error()
}

func openBrowser(url string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		if os.Getenv("DISPLAY") == "" && os.Getenv("WAYLAND_DISPLAY") == "" {
			return fmt.Errorf("no desktop session detected")
		}
		cmd = exec.Command("xdg-open", url)
	}
	return cmd.Start()
}
