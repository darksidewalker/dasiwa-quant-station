package app

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// IdleShutdown tracks browser connections and shuts down the server
// after a grace period with zero active connections.
type IdleShutdown struct {
	mu          sync.Mutex
	active      int
	grace       time.Duration
	shutdownAt  time.Time
	running     bool
	shutdownFn  func()
	shutdownCtx context.Context
	shutdownCancel context.CancelFunc
}

// NewIdleShutdown creates a tracker that calls shutdownFn when no
// connections have been active for [grace] duration.
func NewIdleShutdown(grace time.Duration, shutdownFn func()) *IdleShutdown {
	ctx, cancel := context.WithCancel(context.Background())
	return &IdleShutdown{
		grace:          grace,
		shutdownFn:     shutdownFn,
		shutdownCtx:    ctx,
		shutdownCancel: cancel,
		running:        true,
	}
}

// Enter is called at the start of each HTTP request.
func (i *IdleShutdown) Enter() {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.active++
	i.shutdownAt = time.Time{}
}

// Leave is called when an HTTP request completes.
func (i *IdleShutdown) Leave() {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.active--
	if i.active <= 0 {
		i.active = 0
		if i.shutdownAt.IsZero() {
			i.shutdownAt = time.Now().Add(i.grace)
			go i.watch()
		}
	}
}

func (i *IdleShutdown) watch() {
	remaining := time.Until(i.shutdownAt)
	if remaining <= 0 {
		i.trigger()
		return
	}
	select {
	case <-time.After(remaining):
		i.mu.Lock()
		// Double-check: a new request may have reset the timer.
		if i.active == 0 && !i.shutdownAt.IsZero() {
			i.mu.Unlock()
			i.trigger()
		} else {
			i.mu.Unlock()
		}
	case <-i.shutdownCtx.Done():
		return
	}
}

func (i *IdleShutdown) trigger() {
	i.mu.Lock()
	if !i.running {
		i.mu.Unlock()
		return
	}
	i.running = false
	i.mu.Unlock()
	i.shutdownCancel()
	log.Printf("Idle shutdown: no browser connections for %s, shutting down", i.grace)
	i.shutdownFn()
}

// Server is the HTTP server.
type Server struct {
	rootDir   string
	modelsDir string
	python    string
	http      *http.Server
	jobs      *JobStore
	idle      *IdleShutdown
}

// idleGrace is how long the server waits after the last browser
// connection drops before shutting down. 3 minutes is long enough
// to survive page reloads and brief tab-switching, short enough to
// avoid leaving the process running forever.
const idleGrace = 3 * time.Minute

func NewServer() (*Server, error) {
	root, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	models := filepath.Join(root, "models")
	_ = os.MkdirAll(models, 0o755)
	_ = os.MkdirAll(filepath.Join(root, "logs"), 0o755)

	py := filepath.Join(root, ".venv", "bin", "python")
	if _, err := os.Stat(py); err != nil {
		py = "python3"
	}

	s := &Server{
		rootDir:   root,
		modelsDir: models,
		python:    py,
		jobs:      NewJobStore(),
	}
	s.idle = NewIdleShutdown(idleGrace, func() { s.shutdownIdle() })

	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/config", s.handleConfig)
	mux.HandleFunc("GET /api/system", s.handleSystem)
	mux.HandleFunc("GET /api/browse", s.handleBrowse)
	mux.HandleFunc("GET /api/files", s.handleFiles)
	mux.HandleFunc("GET /api/inspect", s.handleInspect)
	mux.HandleFunc("GET /api/metadata-preview", s.handleMetadataPreview)
	mux.HandleFunc("POST /api/metadata/read", s.handleMetadataRead)
	mux.HandleFunc("POST /api/metadata/inject", s.handleMetadataInject)
	mux.HandleFunc("POST /api/quantize", s.handleQuantize)
	mux.HandleFunc("POST /api/lora/merge", s.handleLoraMerge)
	mux.HandleFunc("POST /api/update", s.handleUpdate)
	mux.HandleFunc("POST /api/memory/clean", s.handleMemoryClean)
	mux.HandleFunc("GET /api/jobs/{id}/events", s.handleJobEvents)
	mux.HandleFunc("POST /api/jobs/{id}/stop", s.handleJobStop)
	mux.HandleFunc("POST /api/tools/scan", s.handleScan)
	mux.HandleFunc("POST /api/tools/audit", s.handleAudit)
	mux.Handle("/", noCache(http.FileServer(http.Dir(filepath.Join(root, "web")))))

	s.http = &http.Server{
		Addr:              "127.0.0.1:7878",
		Handler:           idleTracker(s.idle, logRequests(mux)),
		ReadHeaderTimeout: 10 * time.Second,
	}
	return s, nil
}

// shutdownIdle performs a graceful server shutdown triggered by idle timeout.
// It waits up to 5 s for in-flight requests (SSE streams, etc.) to finish,
// then forces a hard close and exits the process.
func (s *Server) shutdownIdle() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.http.Shutdown(ctx); err != nil {
		log.Printf("idle shutdown forced: %v", err)
		s.http.Close()
	}
	os.Exit(0)
}

// idleTracker is middleware that reports each request entering/leaving
// to the IdleShutdown tracker so it can fire the grace timer.
func idleTracker(i *IdleShutdown, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		i.Enter()
		defer i.Leave()
		next.ServeHTTP(w, r)
	})
}

func (s *Server) Addr() string {
	return "http://" + s.http.Addr
}

func (s *Server) ListenAndServe() error {
	return s.http.ListenAndServe()
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		next.ServeHTTP(w, r)
	})
}

func noCache(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func (s *Server) handleConfig(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]any{
		"root_dir":   s.rootDir,
		"models_dir": s.modelsDir,
		"architectures": []string{
			"Not set", "WAN 2.2", "LTX-2.3", "Hunyuan Video", "Flux.2",
			"Qwen Image", "Z-Image", "Z-Image Refiner", "Anima", "Radiance",
			"Distillation Large", "Distillation Small", "NeRF Large", "NeRF Small",
			"T5-XXL", "Qwen 3.5", "Mistral", "Visual", "Generic Text",
		},
		"formats": []map[string]string{
			{"label": "FP8", "value": "FP8"},
			{"label": "NVFP4", "value": "NVFP4"},
			{"label": "INT8 Tensor-wise", "value": "INT8 Tensor-wise"},
			{"label": "INT8 Row-wise ConvRot (runtime)", "value": "INT8 Row-wise ConvRot Runtime"},
			{"label": "GGUF F32", "value": "GGUF_F32"},
			{"label": "GGUF BF16", "value": "GGUF_BF16"},
			{"label": "GGUF F16", "value": "GGUF_F16"},
			{"label": "GGUF Q8_0", "value": "GGUF_Q8_0"},
			{"label": "GGUF Q6_K", "value": "GGUF_Q6_K"},
			{"label": "GGUF Q5_K", "value": "GGUF_Q5_K"},
			{"label": "GGUF Q4_K", "value": "GGUF_Q4_K"},
			{"label": "GGUF Q3_K", "value": "GGUF_Q3_K"},
			{"label": "GGUF Q2_K", "value": "GGUF_Q2_K"},
		},
	})
}

func (s *Server) handleBrowse(w http.ResponseWriter, r *http.Request) {
	path := cleanPath(r.URL.Query().Get("path"), s.modelsDir)
	entries, err := os.ReadDir(path)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	type item struct {
		Name  string `json:"name"`
		Path  string `json:"path"`
		IsDir bool   `json:"is_dir"`
	}
	items := make([]item, 0, len(entries))
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if !info.IsDir() && !isModelFile(e.Name()) {
			continue
		}
		items = append(items, item{
			Name:  e.Name(),
			Path:  filepath.Join(path, e.Name()),
			IsDir: info.IsDir(),
		})
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].IsDir != items[j].IsDir {
			return items[i].IsDir
		}
		return strings.ToLower(items[i].Name) < strings.ToLower(items[j].Name)
	})
	parent := filepath.Dir(path)
	writeJSON(w, map[string]any{"path": path, "parent": parent, "items": items})
}

func (s *Server) handleFiles(w http.ResponseWriter, r *http.Request) {
	base := cleanPath(r.URL.Query().Get("path"), s.modelsDir)
	var files []string
	_ = filepath.WalkDir(base, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || !isModelFile(path) {
			return nil
		}
		rel, err := filepath.Rel(base, path)
		if err == nil {
			files = append(files, rel)
		}
		return nil
	})
	sort.Strings(files)
	writeJSON(w, map[string]any{"files": files})
}

func isModelFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".safetensors", ".gguf", ".ckpt", ".pt", ".bin":
		return true
	default:
		return false
	}
}

func cleanPath(value, fallback string) string {
	if value == "" {
		value = fallback
	}
	value = os.ExpandEnv(value)
	if strings.HasPrefix(value, "~") {
		if home, err := os.UserHomeDir(); err == nil {
			value = filepath.Join(home, strings.TrimPrefix(value, "~"))
		}
	}
	if abs, err := filepath.Abs(value); err == nil {
		value = abs
	}
	if real, err := filepath.EvalSymlinks(value); err == nil {
		value = real
	}
	return value
}

func (s *Server) runBridge(ctx context.Context, args ...string) ([]byte, error) {
	cmdArgs := append([]string{filepath.Join(s.rootDir, "scripts", "go_bridge.py")}, args...)
	cmd := exec.CommandContext(ctx, s.python, cmdArgs...)
	cmd.Dir = s.rootDir
	cmd.Env = s.commandEnv()
	out, err := cmd.CombinedOutput()
	if err != nil {
		return out, fmt.Errorf("%w: %s", err, strings.TrimSpace(string(out)))
	}
	return out, nil
}

func (s *Server) commandEnv() []string {
	env := os.Environ()
	venv := filepath.Join(s.rootDir, ".venv")
	venvBin := filepath.Join(venv, "bin")
	if info, err := os.Stat(venvBin); err == nil && info.IsDir() {
		env = setEnv(env, "VIRTUAL_ENV", venv)
		env = prependPath(env, venvBin)
	}
	env = setEnv(env, "PYTHONUNBUFFERED", "1")
	return env
}

func setEnv(env []string, key, value string) []string {
	prefix := key + "="
	for i, item := range env {
		if strings.HasPrefix(item, prefix) {
			env[i] = prefix + value
			return env
		}
	}
	return append(env, prefix+value)
}

func prependPath(env []string, value string) []string {
	const key = "PATH"
	prefix := key + "="
	for i, item := range env {
		if strings.HasPrefix(item, prefix) {
			current := strings.TrimPrefix(item, prefix)
			for _, part := range filepath.SplitList(current) {
				if part == value {
					return env
				}
			}
			env[i] = prefix + value + string(os.PathListSeparator) + current
			return env
		}
	}
	return append(env, prefix+value)
}

func (s *Server) handleInspect(w http.ResponseWriter, r *http.Request) {
	path := cleanPath(r.URL.Query().Get("path"), s.modelsDir)
	out, err := s.runBridge(r.Context(), "inspect", path)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(out)
}

func (s *Server) handleMetadataPreview(w http.ResponseWriter, r *http.Request) {
	args := []string{
		"metadata",
		"--name", r.URL.Query().Get("name"),
		"--architecture", r.URL.Query().Get("architecture"),
	}
	if r.URL.Query().Get("full") == "true" {
		args = append(args, "--full")
	}
	out, err := s.runBridge(r.Context(), args...)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(out)
}

func (s *Server) handleMetadataRead(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Path string `json:"path"`
	}
	_ = json.NewDecoder(r.Body).Decode(&in)
	if in.Path == "" {
		writeError(w, http.StatusBadRequest, "source path is required")
		return
	}
	out, err := s.runBridge(r.Context(), "read-metadata-path", cleanPath(in.Path, s.modelsDir))
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(out)
}

func (s *Server) handleMetadataInject(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Path     string `json:"path"`
		Metadata string `json:"metadata"`
	}
	_ = json.NewDecoder(r.Body).Decode(&in)
	if in.Path == "" || in.Metadata == "" {
		writeError(w, http.StatusBadRequest, "source path and metadata JSON are required")
		return
	}
	payload, _ := json.Marshal(map[string]string{
		"path":     cleanPath(in.Path, s.modelsDir),
		"metadata": in.Metadata,
	})
	out, err := s.runBridge(r.Context(), "inject-metadata-path", "--json", string(payload))
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(out)
}

type QuantizeRequest struct {
	ModelsDir      string   `json:"models_dir"`
	SourcePath     string   `json:"source_path"`
	ModelName      string   `json:"model_name"`
	Formats        []string `json:"formats"`
	Architecture   string   `json:"architecture"`
	Strategy       string   `json:"strategy"`
	Optimizer      string   `json:"optimizer"`
	LowVRAM        bool     `json:"low_vram"`
	FullCheckpoint bool     `json:"full_checkpoint"`
}

type LoraSpec struct {
	Path     string  `json:"path"`
	Strength float64 `json:"strength"`
	Strategy string  `json:"strategy"`
}

type LoraMergeRequest struct {
	BasePath       string     `json:"base_path"`
	ModelsDir      string     `json:"models_dir"`
	OutputPath     string     `json:"output_path"`
	OutputName     string     `json:"output_name"`
	Loras          []LoraSpec `json:"loras"`
	Strategy       string     `json:"strategy"`
	Architecture   string     `json:"architecture"`
	GlobalStrength float64    `json:"global_strength"`
	Adaptive       bool       `json:"adaptive"`
	DryRun         bool       `json:"dry_run"`
	StrictMatching bool       `json:"strict_matching"`
}

func (s *Server) handleQuantize(w http.ResponseWriter, r *http.Request) {
	var req QuantizeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.ModelsDir == "" {
		req.ModelsDir = s.modelsDir
	}
	if req.ModelName == "" || req.SourcePath == "" || len(req.Formats) == 0 {
		writeError(w, http.StatusBadRequest, "source, display name, and at least one format are required")
		return
	}
	id := newID()
	ctx, cancel := context.WithCancel(context.Background())
	job := &Job{
		ID:        id,
		CreatedAt: time.Now(),
		Events:    make(chan Event, 512),
		cancel:    cancel,
		Status:    "starting",
	}
	s.jobs.Add(job)
	go s.runQuantizeJob(ctx, job, req)
	writeJSON(w, map[string]string{"job_id": id})
}

func (s *Server) handleLoraMerge(w http.ResponseWriter, r *http.Request) {
	var req LoraMergeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.BasePath == "" || len(req.Loras) == 0 {
		writeError(w, http.StatusBadRequest, "base checkpoint and at least one LoRA are required")
		return
	}
	if req.ModelsDir == "" {
		req.ModelsDir = s.modelsDir
	}
	req.BasePath = cleanPath(req.BasePath, s.modelsDir)
	req.ModelsDir = cleanPath(req.ModelsDir, s.modelsDir)
	if req.OutputPath != "" {
		req.OutputPath = cleanPath(req.OutputPath, req.ModelsDir)
	}
	if req.Strategy == "" {
		req.Strategy = "Balanced"
	}
	if req.GlobalStrength == 0 {
		req.GlobalStrength = 1
	}
	for i := range req.Loras {
		req.Loras[i].Path = cleanPath(req.Loras[i].Path, s.modelsDir)
		if req.Loras[i].Strength == 0 {
			req.Loras[i].Strength = 1
		}
	}
	id := newID()
	ctx, cancel := context.WithCancel(context.Background())
	job := &Job{
		ID:        id,
		CreatedAt: time.Now(),
		Events:    make(chan Event, 512),
		cancel:    cancel,
		Status:    "starting lora merge",
	}
	s.jobs.Add(job)
	go s.runLoraMergeJob(ctx, job, req)
	writeJSON(w, map[string]string{"job_id": id})
}

func (s *Server) handleUpdate(w http.ResponseWriter, r *http.Request) {
	id := newID()
	ctx, cancel := context.WithCancel(context.Background())
	job := &Job{
		ID:        id,
		CreatedAt: time.Now(),
		Events:    make(chan Event, 512),
		cancel:    cancel,
		Status:    "starting update",
	}
	s.jobs.Add(job)
	go s.runUpdateJob(ctx, job)
	writeJSON(w, map[string]string{"job_id": id})
}

func (s *Server) handleMemoryClean(w http.ResponseWriter, r *http.Request) {
	before := readSystemStatus()

	runtime.GC()
	debug.FreeOSMemory()

	out, err := s.runBridge(r.Context(), "clean-memory")
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	runtime.GC()
	debug.FreeOSMemory()

	var bridge struct {
		Text string `json:"text"`
	}
	_ = json.Unmarshal(out, &bridge)

	after := readSystemStatus()
	text := formatMemoryCleanReport(before, after, bridge.Text)
	writeJSON(w, map[string]any{
		"text":   text,
		"before": before,
		"after":  after,
	})
}

func (s *Server) runQuantizeJob(ctx context.Context, job *Job, req QuantizeRequest) {
	defer close(job.Events)
	payload, _ := json.Marshal(req)
	cmd := exec.CommandContext(ctx, s.python, filepath.Join(s.rootDir, "scripts", "go_bridge.py"), "quantize", "--json", string(payload))
	cmd.Dir = s.rootDir
	cmd.Env = s.commandEnv()
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		job.Emit(Event{Type: "error", Text: err.Error()})
		return
	}
	cmd.Stderr = cmd.Stdout
	if err := cmd.Start(); err != nil {
		job.Emit(Event{Type: "error", Text: err.Error()})
		return
	}
	job.setStatus("running")
	job.Emit(Event{Type: "status", Status: "running"})
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		var ev Event
		if err := json.Unmarshal(line, &ev); err == nil && ev.Type != "" {
			job.Emit(ev)
			if ev.Status != "" {
				job.setStatus(ev.Status)
			}
			continue
		}
		job.Emit(Event{Type: "log", Text: string(line) + "\n"})
	}
	if err := scanner.Err(); err != nil && !errors.Is(ctx.Err(), context.Canceled) {
		job.Emit(Event{Type: "error", Text: err.Error()})
	}
	err = cmd.Wait()
	if errors.Is(ctx.Err(), context.Canceled) {
		job.setStatus("stopped")
		job.Emit(Event{Type: "done", Status: "stopped"})
		return
	}
	if err != nil {
		job.setStatus("failed")
		job.Emit(Event{Type: "error", Text: err.Error()})
		job.Emit(Event{Type: "done", Status: "failed"})
		return
	}
	job.setStatus("finished")
	job.Emit(Event{Type: "done", Status: "finished"})
}

func (s *Server) runLoraMergeJob(ctx context.Context, job *Job, req LoraMergeRequest) {
	defer close(job.Events)
	payload, _ := json.Marshal(req)
	cmd := exec.CommandContext(ctx, s.python, filepath.Join(s.rootDir, "scripts", "go_bridge.py"), "lora-merge", "--json", string(payload))
	cmd.Dir = s.rootDir
	cmd.Env = s.commandEnv()
	if err := streamCommand(ctx, cmd, job); err != nil {
		if errors.Is(ctx.Err(), context.Canceled) {
			job.setStatus("lora merge stopped")
			job.Emit(Event{Type: "done", Status: "lora merge stopped"})
			return
		}
		job.setStatus("lora merge failed")
		job.Emit(Event{Type: "error", Text: err.Error()})
		job.Emit(Event{Type: "done", Status: "lora merge failed"})
		return
	}
	job.setStatus("lora merge finished")
}

func (s *Server) runUpdateJob(ctx context.Context, job *Job) {
	defer close(job.Events)
	job.setStatus("updating")
	job.Emit(Event{Type: "status", Status: "updating"})

	steps := []struct {
		name string
		cmd  *exec.Cmd
	}{
		{
			name: "setup",
			cmd:  exec.CommandContext(ctx, "bash", filepath.Join(s.rootDir, "start-linux.sh"), "--setup-only"),
		},
		{
			name: "build",
			cmd:  exec.CommandContext(ctx, "go", "build", "-o", filepath.Join(s.rootDir, "quantstation.next"), "./cmd/dasiwa"),
		},
	}

	for _, step := range steps {
		job.Emit(Event{Type: "log", Text: fmt.Sprintf("\n-- %s --\n", step.name)})
		step.cmd.Dir = s.rootDir
		step.cmd.Env = s.commandEnv()
		if err := streamCommand(ctx, step.cmd, job); err != nil {
			if errors.Is(ctx.Err(), context.Canceled) {
				job.setStatus("update stopped")
				job.Emit(Event{Type: "done", Status: "update stopped"})
				return
			}
			job.setStatus("update failed")
			job.Emit(Event{Type: "error", Text: err.Error()})
			job.Emit(Event{Type: "done", Status: "update failed"})
			return
		}
		if step.name == "build" {
			nextPath := filepath.Join(s.rootDir, "quantstation.next")
			finalPath := filepath.Join(s.rootDir, "quantstation")
			if err := os.Rename(nextPath, finalPath); err != nil {
				job.setStatus("update failed")
				job.Emit(Event{Type: "error", Text: fmt.Sprintf("failed to replace binary: %v", err)})
				job.Emit(Event{Type: "done", Status: "update failed"})
				return
			}
		}
	}

	job.setStatus("restarting")
	job.Emit(Event{Type: "log", Text: "\nUpdate complete. Restarting Go app...\n"})
	job.Emit(Event{Type: "done", Status: "restarting"})

	go func() {
		time.Sleep(700 * time.Millisecond)
		if err := restartSelf(s.rootDir); err != nil {
			log.Printf("restart failed: %v", err)
			return
		}
		os.Exit(0)
	}()
}

func streamCommand(ctx context.Context, cmd *exec.Cmd, job *Job) error {
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	cmd.Stderr = cmd.Stdout
	if err := cmd.Start(); err != nil {
		return err
	}
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		var ev Event
		if err := json.Unmarshal(line, &ev); err == nil && ev.Type != "" {
			job.Emit(ev)
			if ev.Status != "" {
				job.setStatus(ev.Status)
			}
			continue
		}
		job.Emit(Event{Type: "log", Text: string(line) + "\n"})
	}
	if err := scanner.Err(); err != nil && !errors.Is(ctx.Err(), context.Canceled) {
		return err
	}
	return cmd.Wait()
}

func restartSelf(rootDir string) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	cmd := exec.Command(exe)
	cmd.Dir = rootDir
	cmd.Env = os.Environ()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Start()
}

func (s *Server) handleJobEvents(w http.ResponseWriter, r *http.Request) {
	job := s.jobs.Get(r.PathValue("id"))
	if job == nil {
		writeError(w, http.StatusNotFound, "job not found")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	for {
		select {
		case ev, ok := <-job.Events:
			if !ok {
				return
			}
			data, _ := json.Marshal(ev)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		case <-r.Context().Done():
			return
		}
	}
}

func (s *Server) handleJobStop(w http.ResponseWriter, r *http.Request) {
	job := s.jobs.Get(r.PathValue("id"))
	if job == nil {
		writeError(w, http.StatusNotFound, "job not found")
		return
	}
	job.cancel()
	writeJSON(w, map[string]string{"status": "stopping"})
}

func (s *Server) handleScan(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Path string `json:"path"`
	}
	_ = json.NewDecoder(r.Body).Decode(&in)
	out, err := s.runBridge(r.Context(), "scan", cleanPath(in.Path, s.modelsDir))
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(out)
}

func (s *Server) handleAudit(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Path         string `json:"path"`
		Architecture string `json:"architecture"`
	}
	_ = json.NewDecoder(r.Body).Decode(&in)
	out, err := s.runBridge(r.Context(), "audit", cleanPath(in.Path, s.modelsDir), "--architecture", in.Architecture)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(out)
}

func newID() string {
	var b [8]byte
	if _, err := io.ReadFull(rand.Reader, b[:]); err != nil {
		return strconv.FormatInt(time.Now().UnixNano(), 36)
	}
	return hex.EncodeToString(b[:])
}

type Event struct {
	Type   string `json:"type"`
	Text   string `json:"text,omitempty"`
	Status string `json:"status,omitempty"`
}

type Job struct {
	ID        string
	CreatedAt time.Time
	Events    chan Event
	cancel    context.CancelFunc
	mu        sync.Mutex
	Status    string
}

func (j *Job) Emit(ev Event) {
	select {
	case j.Events <- ev:
	default:
		log.Printf("job %s event buffer full, dropping event", j.ID)
	}
}

func (j *Job) setStatus(status string) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.Status = status
}

type JobStore struct {
	mu   sync.Mutex
	jobs map[string]*Job
}

func NewJobStore() *JobStore {
	return &JobStore{jobs: map[string]*Job{}}
}

func (s *JobStore) Add(job *Job) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.jobs[job.ID] = job
}

func (s *JobStore) Get(id string) *Job {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.jobs[id]
}

type systemStatus struct {
	CPUPercent  float64 `json:"cpu_percent"`
	RAMUsedGB   float64 `json:"ram_used_gb"`
	RAMTotalGB  float64 `json:"ram_total_gb"`
	GPU         string  `json:"gpu"`
	VRAM        string  `json:"vram"`
	GPUPercent  float64 `json:"gpu_percent"`
	VRAMPercent float64 `json:"vram_percent"`
}

func (s *Server) handleSystem(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, readSystemStatus())
}

func readSystemStatus() systemStatus {
	status := systemStatus{GPU: "Idle", VRAM: "n/a"}
	if data, err := os.ReadFile("/proc/meminfo"); err == nil {
		total, available := parseMeminfo(string(data))
		if total > 0 {
			status.RAMTotalGB = float64(total) / 1024 / 1024
			status.RAMUsedGB = float64(total-available) / 1024 / 1024
		}
	}
	status.CPUPercent = readLoadPercent()
	if gpu, vram, gpuPct, vramPct := readNvidia(); gpu != "" {
		status.GPU = gpu
		status.VRAM = vram
		status.GPUPercent = gpuPct
		status.VRAMPercent = vramPct
	}
	return status
}

func formatMemoryCleanReport(before, after systemStatus, bridgeText string) string {
	var b strings.Builder
	b.WriteString("Memory cleanup complete.\n")
	if before.RAMTotalGB > 0 && after.RAMTotalGB > 0 {
		freed := before.RAMUsedGB - after.RAMUsedGB
		b.WriteString(fmt.Sprintf("RAM: %.1f/%.1fGB -> %.1f/%.1fGB", before.RAMUsedGB, before.RAMTotalGB, after.RAMUsedGB, after.RAMTotalGB))
		if freed > 0 {
			b.WriteString(fmt.Sprintf(" (%.1fGB lower)", freed))
		}
		b.WriteString("\n")
	}
	if before.VRAM != "n/a" || after.VRAM != "n/a" {
		b.WriteString(fmt.Sprintf("VRAM: %s -> %s\n", before.VRAM, after.VRAM))
	}
	if strings.TrimSpace(bridgeText) != "" {
		b.WriteString("\n")
		b.WriteString(strings.TrimSpace(bridgeText))
		b.WriteString("\n")
	}
	b.WriteString("\nSafe mode: this does not kill processes, reset GPUs, or drop kernel page cache.")
	return b.String()
}

func parseMeminfo(text string) (totalKB, availableKB int64) {
	for _, line := range strings.Split(text, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		val, _ := strconv.ParseInt(fields[1], 10, 64)
		switch fields[0] {
		case "MemTotal:":
			totalKB = val
		case "MemAvailable:":
			availableKB = val
		}
	}
	return
}

func readLoadPercent() float64 {
	data, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(data))
	if len(fields) == 0 {
		return 0
	}
	load, _ := strconv.ParseFloat(fields[0], 64)
	cpus := float64(1)
	if n, err := os.ReadFile("/proc/cpuinfo"); err == nil {
		cpus = float64(strings.Count(string(n), "processor\t:"))
		if cpus < 1 {
			cpus = 1
		}
	}
	return min(load/cpus*100, 100)
}

func readNvidia() (string, string, float64, float64) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits")
	cmd.SysProcAttr = &syscall.SysProcAttr{}
	out, err := cmd.Output()
	if err != nil {
		return "", "", 0, 0
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	var utils []string
	var used, total float64
	var utilSum float64
	var utilCount float64
	for i, line := range lines {
		parts := strings.Split(line, ",")
		if len(parts) != 3 {
			continue
		}
		util := strings.TrimSpace(parts[0])
		utils = append(utils, fmt.Sprintf("G%d %s%%", i, util))
		if uPct, err := strconv.ParseFloat(util, 64); err == nil {
			utilSum += uPct
			utilCount++
		}
		u, _ := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		t, _ := strconv.ParseFloat(strings.TrimSpace(parts[2]), 64)
		used += u / 1024
		total += t / 1024
	}
	if len(utils) == 0 {
		return "", "", 0, 0
	}
	gpuPct := 0.0
	if utilCount > 0 {
		gpuPct = utilSum / utilCount
	}
	vramPct := 0.0
	if total > 0 {
		vramPct = used / total * 100
	}
	return strings.Join(utils, " "), fmt.Sprintf("%.1f/%.1fGB", used, total), gpuPct, vramPct
}
