package app

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestW6QuantizeFailureAndCancellationCleanStage(t *testing.T) {
	for _, cancelJob := range []bool{false, true} {
		name := "failure"
		if cancelJob {
			name = "cancel"
		}
		t.Run(name, func(t *testing.T) {
			real, err := NewServer()
			if err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			if err := os.Mkdir(filepath.Join(dir, "scripts"), 0700); err != nil {
				t.Fatal(err)
			}
			script := `import os,json,time,sys
stage=os.environ['DASIWA_H3_STAGE_DIR']
open(os.path.join(stage,'payload'),'w').write('owned')
print(json.dumps({'type':'status','status':'ready'}),flush=True)
`
			if cancelJob {
				script += "time.sleep(60)\n"
			} else {
				script += "sys.exit(1)\n"
			}
			if err := os.WriteFile(filepath.Join(dir, "scripts", "go_bridge.py"), []byte(script), 0600); err != nil {
				t.Fatal(err)
			}
			foreign := filepath.Join(dir, ".h3_job_foreign")
			if err := os.Mkdir(foreign, 0700); err != nil {
				t.Fatal(err)
			}
			s := &Server{rootDir: dir, modelsDir: dir, python: real.python}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			job := &Job{Events: make(chan Event, 512)}
			go s.runQuantizeJob(ctx, job, QuantizeRequest{OutputDir: dir, ModelName: "test", Formats: []string{"W6A8"}})
			timer := time.NewTimer(15 * time.Second)
			defer timer.Stop()
			terminal := ""
			for terminal == "" {
				select {
				case ev, ok := <-job.Events:
					if !ok {
						t.Fatal("closed without terminal event")
					}
					if ev.Status == "ready" && cancelJob {
						cancel()
					}
					if ev.Type == "done" {
						terminal = ev.Status
					}
				case <-timer.C:
					t.Fatal("job did not finish")
				}
			}
			want := "failed"
			if cancelJob {
				want = "stopped"
			}
			if terminal != want {
				t.Fatalf("status %q want %q", terminal, want)
			}
			// Closing the event stream happens after owned staging cleanup.
			for range job.Events {
			}
			entries, err := os.ReadDir(dir)
			if err != nil {
				t.Fatal(err)
			}
			for _, entry := range entries {
				if entry.Name() != "scripts" && entry.Name() != ".h3_job_foreign" {
					t.Fatalf("leftover %s", entry.Name())
				}
			}
			if _, err := os.Stat(foreign); err != nil {
				t.Fatal("foreign stage removed")
			}
		})
	}
}
