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

func TestModelMergeModalityOptionsRoundTrip(t *testing.T) {
	var req ModelMergeRequest
	if err := json.Unmarshal([]byte(`{"modality_mode":"split","audio_blocks":"all","text_blocks":"selected","audio_out_from_overlay":true}`), &req); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]interface{}
	if err := json.Unmarshal(encoded, &fields); err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]interface{}{"modality_mode": "split", "audio_blocks": "all", "text_blocks": "selected", "audio_out_from_overlay": true} {
		if fields[name] != want {
			t.Errorf("%s = %v, want %v", name, fields[name], want)
		}
	}
}

func TestHandleModelMergeRejectsUnknownModalityMode(t *testing.T) {
	s := &Server{modelsDir: t.TempDir(), jobs: NewJobStore()}
	req := httptest.NewRequest(http.MethodPost, "/api/model-merge", bytes.NewBufferString(`{"recipe":"h3_hybrid","modality_mode":"unknown"}`))
	res := httptest.NewRecorder()
	s.handleModelMerge(res, req)
	if res.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400", res.Code)
	}
}

func TestModelMergeOverlayBlocksJSONRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want string
	}{
		{"legacy range", `{}`, `false`},
		{"sparse", `{"overlay_blocks":[2,4,7]}`, `[2,4,7]`},
		{"explicit empty", `{"overlay_blocks":[]}`, `[]`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var req ModelMergeRequest
			if err := json.Unmarshal([]byte(tc.body), &req); err != nil {
				t.Fatal(err)
			}
			encoded, err := json.Marshal(req)
			if err != nil {
				t.Fatal(err)
			}
			var payload map[string]json.RawMessage
			if err := json.Unmarshal(encoded, &payload); err != nil {
				t.Fatal(err)
			}
			value, exists := payload["overlay_blocks"]
			if tc.want == `false` {
				if exists {
					t.Fatalf("unexpected overlay_blocks: %s", value)
				}
			} else if !exists || string(value) != tc.want {
				t.Fatalf("overlay_blocks = %s, exists=%t; want %s", value, exists, tc.want)
			}
		})
	}
}

func TestHandleModelMergeRejectsInvalidOverlayBlocks(t *testing.T) {
	s := &Server{modelsDir: t.TempDir(), jobs: NewJobStore()}
	for _, blocks := range []string{`[]`, `[0,0]`, `[-1]`, `[50]`} {
		req := httptest.NewRequest(http.MethodPost, "/api/model-merge", bytes.NewBufferString(
			`{"recipe":"h3_hybrid","overlay_blocks":`+blocks+`}`))
		res := httptest.NewRecorder()
		s.handleModelMerge(res, req)
		if res.Code != http.StatusBadRequest {
			t.Errorf("overlay_blocks=%s: status %d, want 400", blocks, res.Code)
		}
	}
}

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
	Path   string `json:"path"`
	Parent string `json:"parent"`
	Items  []struct {
		Name       string `json:"name"`
		Path       string `json:"path"`
		IsDir      bool   `json:"is_dir"`
		IsSymlink  bool   `json:"is_symlink"`
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

func TestHandleBrowseFollowsSymlinkDirectoriesAndFiles(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target")
	if err := os.Mkdir(target, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(target, "model.safetensors"), []byte("model"), 0o644); err != nil {
		t.Fatal(err)
	}
	linkDir := filepath.Join(dir, "linked")
	linkFile := filepath.Join(dir, "linked.safetensors")
	if err := os.Symlink(target, linkDir); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(target, "model.safetensors"), linkFile); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(dir, "missing"), filepath.Join(dir, "broken")); err != nil {
		t.Fatal(err)
	}

	s := &Server{modelsDir: dir}
	browse := func(path string) browserResponse {
		t.Helper()
		res := httptest.NewRecorder()
		s.handleBrowse(res, httptest.NewRequest(http.MethodGet, "/api/browse?path="+path, nil))
		if res.Code != http.StatusOK {
			t.Fatalf("browse %s: %d %s", path, res.Code, res.Body.String())
		}
		var body browserResponse
		if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		return body
	}

	root := browse(dir)
	if len(root.Items) != 3 {
		t.Fatalf("items = %#v, want target, linked, linked.safetensors", root.Items)
	}
	if root.Items[0].Name != "linked" || !root.Items[0].IsDir || !root.Items[0].IsSymlink || root.Items[0].Path != linkDir {
		t.Fatalf("linked directory = %#v", root.Items[0])
	}
	if root.Items[2].Name != "linked.safetensors" || root.Items[2].IsDir || !root.Items[2].IsSymlink || root.Items[2].Size != 5 {
		t.Fatalf("linked file = %#v", root.Items[2])
	}
	followed := browse(linkDir)
	if followed.Path != linkDir || followed.Parent != dir || len(followed.Items) != 1 || followed.Items[0].Path != filepath.Join(linkDir, "model.safetensors") {
		t.Fatalf("followed directory = %#v", followed)
	}
}

func TestHandleSearchFollowsSymlinksWithoutCycles(t *testing.T) {
	dir := t.TempDir()
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "wanted.gguf"), []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "linked")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(dir, filepath.Join(outside, "back")); err != nil {
		t.Fatal(err)
	}

	s := &Server{modelsDir: dir}
	res := httptest.NewRecorder()
	s.handleSearch(res, httptest.NewRequest(http.MethodGet, "/api/search?path="+dir+"&q=wanted", nil))
	var body browserResponse
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if res.Code != http.StatusOK || len(body.Items) != 1 || body.Items[0].Path != filepath.Join(dir, "linked", "wanted.gguf") {
		t.Fatalf("search status %d, items %#v", res.Code, body.Items)
	}
}

func TestHandleLoraExtractRejectsEmptyOrDuplicateBlocks(t *testing.T) {
	s := &Server{modelsDir: t.TempDir(), jobs: NewJobStore()}
	for _, blocks := range []string{`[]`, `["blocks.1","blocks.1"]`, `[""]`} {
		body := `{"base_path":"base.safetensors","merged_path":"modified.safetensors","recipe":"generic","selected_blocks":` + blocks + `}`
		res := httptest.NewRecorder()
		s.handleLoraExtract(res, httptest.NewRequest(http.MethodPost, "/api/lora/extract", bytes.NewBufferString(body)))
		if res.Code != http.StatusBadRequest {
			t.Errorf("selected_blocks=%s: status %d, want 400", blocks, res.Code)
		}
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
