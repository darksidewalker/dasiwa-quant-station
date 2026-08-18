package app

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/sha256"
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

// Server is the HTTP server.
type Server struct {
	rootDir   string
	modelsDir string
	python    string
	version   string // SHA-256 prefix of the running binary
	http      *http.Server
	jobs      *JobStore
}

func NewServer() (*Server, error) {
	root, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	models := ""
	if envModels := os.Getenv("DASIWA_MODELS_DIR"); envModels != "" {
		models = cleanPath(envModels, root)
	} else if home, err := os.UserHomeDir(); err == nil && home != "" {
		models = home
	} else {
		models = filepath.Join(root, "models")
	}
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

	s.version = computeBinaryHash()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/ping", s.handlePing)
	mux.HandleFunc("GET /api/config", s.handleConfig)
	mux.HandleFunc("GET /api/system", s.handleSystem)
	mux.HandleFunc("GET /api/browse", s.handleBrowse)
	mux.HandleFunc("GET /api/search", s.handleSearch)
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
	mux.HandleFunc("GET /api/jobs/{id}/summary", s.handleJobSummary)
	mux.HandleFunc("POST /api/tools/scan", s.handleScan)
	mux.HandleFunc("POST /api/tools/audit", s.handleAudit)
	mux.HandleFunc("POST /api/shutdown", s.handleShutdown)
	mux.Handle("/", noCache(http.FileServer(http.Dir(filepath.Join(root, "web")))))

	s.http = &http.Server{
		Addr:              "127.0.0.1:7878",
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 30 * time.Second, // SSE long-poll needs more than default 10s for large model processing
	}
	return s, nil
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

// handlePing is a lightweight health-check endpoint for the UI status dot.
func (s *Server) handlePing(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{"status": "ok"})
}

func (s *Server) handleConfig(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]any{
		"version":    s.version,
		"root_dir":   s.rootDir,
		"models_dir": s.modelsDir,
		"architectures": []string{
			"Not set", "WAN 2.2", "LTX-2.3", "Krea 2", "MiniMax H3", "Hunyuan Video", "Flux.2",
			"Qwen Image", "Z-Image", "Z-Image Refiner", "Anima", "Radiance",
			"Distillation Large", "Distillation Small", "NeRF Large", "NeRF Small",
			"T5-XXL", "Qwen 3.5", "Mistral", "Visual", "Generic Text",
		},
		"formats": []map[string]string{
			{"label": "FP8", "value": "FP8"},
			{"label": "NVFP4", "value": "NVFP4"},
			{"label": "MXFP8", "value": "MXFP8"},
			{"label": "Hybrid MXFP8", "value": "Hybrid MXFP8"},
			{"label": "INT8 Tensor-wise", "value": "INT8 Tensor-wise"},
			{"label": "INT8 Row-wise ConvRot (runtime)", "value": "INT8 Row-wise ConvRot Runtime"},
			{"label": "INT4 ConvRot (LTX/WAN/Krea experimental)", "value": "INT4 ConvRot Runtime"},
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

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	type itemSearch struct {
		Name string `json:"name"`
		Path string `json:"path"`
	}
	path := cleanPath(r.URL.Query().Get("path"), s.modelsDir)
	query := strings.TrimSpace(r.URL.Query().Get("q"))
	if query == "" {
		writeJSON(w, map[string]any{"path": path, "query": "", "items": []itemSearch{}})
		return
	}
	queryLower := strings.ToLower(query)
	var results []itemSearch
	err := filepath.WalkDir(path, func(fp string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // skip inaccessible dirs
		}
		name := d.Name()
		if strings.HasPrefix(name, ".") {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if !d.IsDir() && !isModelFile(name) {
			return nil
		}
		if d.IsDir() {
			return nil // files only in search results
		}
		if strings.Contains(strings.ToLower(name), queryLower) {
			results = append(results, itemSearch{Name: name, Path: fp})
		}
		return nil
	})
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	sort.Slice(results, func(i, j int) bool {
		return strings.ToLower(results[i].Name) < strings.ToLower(results[j].Name)
	})
	writeJSON(w, map[string]any{"path": path, "query": query, "items": results})
}

func isModelFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".safetensors", ".gguf", ".ckpt", ".pt", ".bin", ".tmp":
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
	OutputDir      string   `json:"output_dir"`
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
	OutputDir      string     `json:"output_dir"`
	OutputPath     string     `json:"output_path"`
	OutputName     string     `json:"output_name"`
	Loras          []LoraSpec `json:"loras"`
	Strategy       string     `json:"strategy"`
	Architecture   string     `json:"architecture"`
	GlobalStrength float64    `json:"global_strength"`
	Adaptive       bool       `json:"adaptive"`
	DryRun         bool       `json:"dry_run"`
	StrictMatching bool       `json:"strict_matching"`
	Krea2Unchain   bool       `json:"krea2_unchain"`
	MergeDevice    string     `json:"merge_device"`
	CUDADevice     string     `json:"cuda_device"`
	VRAMHeadroomMB int        `json:"vram_headroom_mb"`
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
	for _, format := range req.Formats {
		if format == "INT4 ConvRot Runtime" && req.Strategy != "Simple" {
			writeError(w, http.StatusBadRequest, "INT4 ConvRot requires the Simple strategy")
			return
		}
	}
	req.SourcePath = cleanPath(req.SourcePath, s.modelsDir)
	req.ModelsDir = cleanPath(req.ModelsDir, s.modelsDir)
	if req.OutputDir == "" {
		req.OutputDir = filepath.Dir(req.SourcePath)
	}
	req.OutputDir = cleanPath(req.OutputDir, filepath.Dir(req.SourcePath))
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
	if req.OutputDir == "" {
		req.OutputDir = filepath.Dir(req.BasePath)
	}
	req.OutputDir = cleanPath(req.OutputDir, filepath.Dir(req.BasePath))
	if req.OutputPath != "" {
		req.OutputPath = cleanPath(req.OutputPath, req.OutputDir)
	}
	if req.Strategy == "" {
		if req.Architecture == "" || req.Architecture == "LTX-2.3" {
			req.Strategy = "All"
		} else {
			req.Strategy = "Balanced"
		}
	}
	if req.GlobalStrength == 0 {
		req.GlobalStrength = 1
	}
	if req.MergeDevice == "" {
		req.MergeDevice = "auto"
	}
	if req.MergeDevice != "cpu" && req.MergeDevice != "auto" && req.MergeDevice != "cuda" {
		writeError(w, http.StatusBadRequest, "merge_device must be cpu, auto, or cuda")
		return
	}
	if req.CUDADevice == "" {
		req.CUDADevice = "cuda:0"
	}
	if req.VRAMHeadroomMB <= 0 {
		req.VRAMHeadroomMB = 1024
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

type updateStep struct {
	name string
	cmd  *exec.Cmd
}

func updateSteps(ctx context.Context, rootDir string) []updateStep {
	steps := []updateStep{
		{
			name: "source update",
			cmd:  exec.CommandContext(ctx, "git", "pull", "--ff-only"),
		},
		{
			name: "setup",
			cmd:  exec.CommandContext(ctx, "bash", filepath.Join(rootDir, "start-linux.sh"), "--setup-only"),
		},
		{
			name: "build",
			cmd:  exec.CommandContext(ctx, "go", "build", "-o", filepath.Join(rootDir, "quantstation.next"), "./cmd/quantstation"),
		},
	}
	for i := range steps {
		steps[i].cmd.Dir = rootDir
	}
	return steps
}

func (s *Server) runUpdateJob(ctx context.Context, job *Job) {
	defer close(job.Events)
	job.setStatus("updating")
	job.Emit(Event{Type: "status", Status: "updating"})

	steps := updateSteps(ctx, s.rootDir)

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

func (s *Server) handleJobSummary(w http.ResponseWriter, r *http.Request) {
	job := s.jobs.Get(r.PathValue("id"))
	if job == nil {
		writeError(w, http.StatusNotFound, "job not found")
		return
	}
	job.mu.Lock()
	status := job.Status
	job.mu.Unlock()

	if status == "" || status == "starting" {
		writeJSON(w, map[string]any{"id": job.ID, "status": "running"})
		return
	}
	// If the job is in a terminal state (finished/stopped/failed), mark it as no longer needed.
	switch status {
	case "finished", "stopped", "failed":
		writeJSON(w, map[string]any{"id": job.ID, "status": status})
	default:
		if strings.Contains(status, "merge") || strings.Contains(status, "update") {
			status = strings.ReplaceAll(strings.ToLower(status), " ", "_") + "-running"
		} else if !strings.Contains(status, "-running") && !strings.HasSuffix(status, "stopped") && status != "finished" && status != "failed" {
			if !strings.ContainsAny(status, "/\\ ") {
				status = status + "-running"
			}
		}
		writeJSON(w, map[string]any{"id": job.ID, "status": status})
	}
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

func (s *Server) handleShutdown(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{"status": "shutting down"})
	go func() {
		time.Sleep(500 * time.Millisecond)
		s.http.Shutdown(context.Background())
		os.Exit(0)
	}()
}

func computeBinaryHash() string {
	exe, err := os.Executable()
	if err != nil {
		return "unknown"
	}
	data, err := os.ReadFile(exe)
	if err != nil {
		return "unknown"
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])[:16]
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
