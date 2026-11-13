package app

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestGravitonHTTPArtifacts(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	py := filepath.Join(root, ".venv", "bin", "python")
	if _, err := os.Stat(py); err != nil {
		t.Skip("project Python environment unavailable")
	}
	dir := t.TempDir()
	run := func(code string) {
		t.Helper()
		cmd := exec.Command(py, "-c", code, dir)
		cmd.Dir = root
		cmd.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES=")
		if output, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("Python: %v\n%s", err, output)
		}
	}
	run(`import sys, torch
from safetensors.torch import save_file
from pathlib import Path
p=Path(sys.argv[1]); key='diffusion_model.blocks.0.attn.qkv_proj'
save_file({'blocks.0.attn.qkv_proj.weight':torch.eye(2)},str(p/'base.safetensors'))
for name, value in [('a', .2), ('b', .1)]:
 save_file({key+'.lora_A.weight':torch.eye(2),key+'.lora_B.weight':torch.eye(2)*value},str(p/(name+'.safetensors')))
`)
	s := &Server{rootDir: root, modelsDir: dir, python: py, jobs: NewJobStore()}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/lora/merge", s.handleLoraMerge)
	mux.HandleFunc("POST /api/lora/compose", s.handleLoraCompose)
	mux.HandleFunc("GET /api/jobs/{id}/events", s.handleJobEvents)
	server := httptest.NewServer(mux)
	defer server.Close()
	client := &http.Client{Timeout: 60 * time.Second}
	submit := func(route, body string) {
		t.Helper()
		resp, err := client.Post(server.URL+route, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		data, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s: %s", route, data)
		}
		var job map[string]string
		if err := json.Unmarshal(data, &job); err != nil {
			t.Fatal(err)
		}
		resp, err = client.Get(server.URL + "/api/jobs/" + job["job_id"] + "/events")
		if err != nil {
			t.Fatal(err)
		}
		data, err = io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(data), `"status":"finished"`) || strings.Contains(string(data), `"type":"error"`) {
			t.Fatalf("SSE: %s", data)
		}
		t.Logf("%s: finished through real HTTP -> Go -> bridge -> CPU artifact", route)
	}
	body, _ := json.Marshal(map[string]any{"loras": []map[string]any{{"path": filepath.Join(dir, "a.safetensors"), "strength": 1}, {"path": filepath.Join(dir, "b.safetensors"), "strength": -1}}, "global_strength": 1, "architecture": "MiniMax H3", "merge_algorithm": "graviton", "merge_device": "cpu", "output_adapter": "lora", "output_rank": 2, "frobenius_energy": 1, "output_path": filepath.Join(dir, "composed.safetensors")})
	submit("/api/lora/compose", string(body))
	body, _ = json.Marshal(map[string]any{"base_path": filepath.Join(dir, "base.safetensors"), "loras": []map[string]any{{"path": filepath.Join(dir, "composed.safetensors"), "strength": 1}}, "global_strength": 1, "architecture": "MiniMax H3", "merge_algorithm": "graviton", "merge_device": "cpu", "output_path": filepath.Join(dir, "baked.safetensors")})
	submit("/api/lora/merge", string(body))
	run(`import sys, torch
from pathlib import Path
from safetensors.torch import load_file
from safetensors import safe_open
from core.model_merge_engine import _graviton_merge_matrix
p=Path(sys.argv[1]); k='diffusion_model.blocks.0.attn.qkv_proj'
adapter=load_file(str(p/'composed.safetensors'))
delta=adapter[k+'.lora_B.weight'].float()@adapter[k+'.lora_A.weight'].float()
torch.testing.assert_close(delta,torch.eye(2)*.1,atol=.002,rtol=.02)
expected=_graviton_merge_matrix(torch.eye(2),torch.eye(2)+delta,1.,-1)[0]
torch.testing.assert_close(load_file(str(p/'baked.safetensors'))['blocks.0.attn.qkv_proj.weight'],expected)
with safe_open(str(p/'composed.safetensors'),framework='pt') as f:
 assert f.metadata()['dasiwa.adapter_merge']=='graviton'
for name in ['composed','baked']:
 assert 'GRAVITON' in (p/(name+'.safetensors.txt')).read_text()
assert not list(p.glob('*.tmp'))
`)
}

func TestGravitonBakeRejectsIncompatibleControls(t *testing.T) {
	s := &Server{modelsDir: t.TempDir(), jobs: NewJobStore()}
	for _, field := range []string{`"adaptive":true`, `"krea2_unchain":true`} {
		r := httptest.NewRecorder()
		s.handleLoraMerge(r, httptest.NewRequest("POST", "/api/lora/merge", strings.NewReader(`{"base_path":"base","loras":[{"path":"a"}],"merge_algorithm":"graviton",`+field+`}`)))
		if r.Code != 400 || !strings.Contains(r.Body.String(), "GRAVITON cannot") {
			t.Fatalf("status=%d body=%s", r.Code, r.Body.String())
		}
	}
}
