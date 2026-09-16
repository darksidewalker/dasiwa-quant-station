package app

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestUpdateStepsPullsLatestSourceBeforeSetupAndBuild(t *testing.T) {
	root := "/tmp/dasiwa-quant-station"
	steps := updateSteps(context.Background(), root)

	if len(steps) != 3 {
		t.Fatalf("expected 3 update steps, got %d", len(steps))
	}

	if steps[0].name != "source update" {
		t.Fatalf("first step = %q, want source update", steps[0].name)
	}
	if got, want := steps[0].cmd.Args, []string{"git", "pull", "--ff-only"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("source update command = %q, want %q", got, want)
	}
	if steps[0].cmd.Dir != root {
		t.Fatalf("source update directory = %q, want %q", steps[0].cmd.Dir, root)
	}

	if steps[1].name != "setup" || steps[1].cmd.Args[1] != filepath.Join(root, "start-linux.sh") || steps[1].cmd.Args[2] != "--setup-only" {
		t.Fatalf("second step should run setup from %q, got %#v", root, steps[1].cmd)
	}
	if steps[2].name != "build" || steps[2].cmd.Args[1] != "build" {
		t.Fatalf("third step should build the Go app, got %#v", steps[2].cmd)
	}
}

func TestHandleLoraExtractAcceptsGenericTwoCheckpointRecipe(t *testing.T) {
	s := &Server{modelsDir: t.TempDir(), rootDir: t.TempDir(), python: "false", jobs: NewJobStore()}
	req := httptest.NewRequest(http.MethodPost, "/api/lora/extract", bytes.NewBufferString(
		`{"base_path":"base.safetensors","merged_path":"modified.safetensors","architecture":"WAN 2.2","recipe":"generic","frobenius_energy":0.99}`,
	))
	res := httptest.NewRecorder()

	s.handleLoraExtract(res, req)

	if res.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", res.Code, res.Body.String())
	}
}

func TestHandleLoraComposeValidation(t *testing.T) {
	s := &Server{modelsDir: t.TempDir(), jobs: NewJobStore()}
	cases := []struct {
		name string
		body string
	}{
		{"needs two adapters", `{"loras":[{"path":"a.safetensors"}]}`},
		{"rejects output kind", `{"loras":[{"path":"a"},{"path":"b"}],"output_adapter":"bad"}`},
		{"rejects energy", `{"loras":[{"path":"a"},{"path":"b"}],"frobenius_energy":2}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/api/lora/compose", bytes.NewBufferString(tc.body))
			res := httptest.NewRecorder()
			s.handleLoraCompose(res, req)
			if res.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body=%s", res.Code, res.Body.String())
			}
		})
	}
}

func TestFormatSupportedFor(t *testing.T) {
	cases := []struct {
		format string
		arch   string
		want   bool
	}{
		{"W4A8", "MiniMax H3", true},
		{"W4A8", "LTX-2.3", false},
		{"W4A8", "WAN 2.2", false},
		{"NVFP4 HQ", "MiniMax H3", true},
		{"NVFP4 HQ", "LTX-2.3", false},
		{"NVFP4 HQ", "WAN 2.2", false},
		{"NVFP4", "LTX-2.3", true},
		{"INT4 ConvRot Runtime", "Krea 2", true},
		{"INT4 ConvRot Runtime", "Flux.2", false},
		{"INT4 ConvRot Runtime", "Not set", false},
		{"FP8", "Any Arch", true},
		{"GGUF_Q4_K", "Any Arch", true},
	}
	for _, c := range cases {
		if got := formatSupportedFor(c.format, c.arch); got != c.want {
			t.Errorf("formatSupportedFor(%q, %q) = %v, want %v", c.format, c.arch, got, c.want)
		}
	}
}

func TestQuantCapabilityAllows(t *testing.T) {
	cases := []struct {
		format, architecture, strategy string
		want                           bool
	}{
		{"FP8", "LTX-2.3", "Optimizer-driven", true},
		{"FP8", "LTX-2.3", "Simple", true},
		{"W4A8", "MiniMax H3", "Simple", true},
		{"W4A8", "MiniMax H3", "Optimizer-driven", false},
		{"W4A8", "WAN 2.2", "Simple", false},
		{"NVFP4 HQ", "MiniMax H3", "Optimizer-driven", true},
		{"NVFP4 HQ", "Krea 2", "Optimizer-driven", false},
		{"INT4 ConvRot Runtime", "Krea 2", "Simple", true},
		{"INT4 ConvRot Runtime", "Flux.2", "Simple", false},
		{"unknown", "MiniMax H3", "Simple", false},
	}
	for _, tc := range cases {
		t.Run(tc.format+"/"+tc.architecture+"/"+tc.strategy, func(t *testing.T) {
			if got := quantCapabilityAllows(tc.format, tc.architecture, tc.strategy); got != tc.want {
				t.Fatalf("quantCapabilityAllows() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestHandleConfigIncludesQuantCapabilities(t *testing.T) {
	s := &Server{rootDir: t.TempDir(), modelsDir: t.TempDir(), version: "test"}
	req := httptest.NewRequest(http.MethodGet, "/api/config", nil)
	res := httptest.NewRecorder()

	s.handleConfig(res, req)

	var body struct {
		QuantCapabilities map[string]quantCapability `json:"quant_capabilities"`
	}
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	w4a8 := body.QuantCapabilities["W4A8"]
	if !reflect.DeepEqual(w4a8.Architectures, []string{"MiniMax H3"}) {
		t.Fatalf("W4A8 architectures = %v", w4a8.Architectures)
	}
	if !reflect.DeepEqual(w4a8.Strategies, []string{"Simple"}) {
		t.Fatalf("W4A8 strategies = %v", w4a8.Strategies)
	}
}

type browserResponse struct {
	Items []struct {
		Name       string `json:"name"`
		Path       string `json:"path"`
		IsDir      bool   `json:"is_dir"`
		Size       int64  `json:"size"`
		ModifiedAt string `json:"modified_at"`
	} `json:"items"`
}

func TestHandleBrowseIncludesMetadata(t *testing.T) {
	dir := t.TempDir()
	filePath := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(filePath, bytes.Repeat([]byte{'x'}, 1536), 0o644); err != nil {
		t.Fatal(err)
	}
	modified := time.Date(2026, 9, 16, 5, 30, 0, 0, time.UTC)
	if err := os.Chtimes(filePath, modified, modified); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "models"), 0o755); err != nil {
		t.Fatal(err)
	}

	s := &Server{modelsDir: dir}
	res := httptest.NewRecorder()
	s.handleBrowse(res, httptest.NewRequest(http.MethodGet, "/api/browse?path="+dir, nil))

	var body browserResponse
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if len(body.Items) != 2 {
		t.Fatalf("items = %d, want 2", len(body.Items))
	}
	file := body.Items[1]
	if file.Size != 1536 || file.ModifiedAt != modified.Format(time.RFC3339) {
		t.Fatalf("file metadata = size %d, modified %q", file.Size, file.ModifiedAt)
	}
	if body.Items[0].Size != 0 || body.Items[0].ModifiedAt == "" {
		t.Fatalf("directory metadata = %#v", body.Items[0])
	}
}

func TestHandleSearchIncludesMetadata(t *testing.T) {
	dir := t.TempDir()
	filePath := filepath.Join(dir, "wanted.gguf")
	if err := os.WriteFile(filePath, bytes.Repeat([]byte{'x'}, 2048), 0o644); err != nil {
		t.Fatal(err)
	}

	s := &Server{modelsDir: dir}
	res := httptest.NewRecorder()
	s.handleSearch(res, httptest.NewRequest(http.MethodGet, "/api/search?path="+dir+"&q=wanted", nil))

	var body browserResponse
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if len(body.Items) != 1 || body.Items[0].Size != 2048 || body.Items[0].ModifiedAt == "" {
		t.Fatalf("search items = %#v", body.Items)
	}
}
