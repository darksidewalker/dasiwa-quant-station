package app

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestH3RoutesRejectInvalidBeforeJob(t *testing.T) {
	s, err := NewServer()
	if err != nil {
		t.Fatal(err)
	}
	for _, route := range []string{"/api/h3/prune", "/api/h3/adapter-convert"} {
		r := httptest.NewRecorder()
		s.http.Handler.ServeHTTP(r, httptest.NewRequest("POST", route, strings.NewReader(`{}`)))
		if r.Code != 400 {
			t.Fatalf("%s status=%d body=%s", route, r.Code, r.Body.String())
		}
	}
}
func TestH3TurboValidation(t *testing.T) {
	s := &Server{modelsDir: t.TempDir(), jobs: NewJobStore()}
	for _, payload := range []string{
		`{"base_path":"base","loras":[{"path":"a"}],"architecture":"WAN 2.2","h3_turbo_complete":true}`,
		`{"base_path":"base","loras":[{"path":"a"},{"path":"b"}],"architecture":"MiniMax H3","merge_algorithm":"consensus","h3_turbo_complete":true}`,
	} {
		r := httptest.NewRecorder()
		s.handleLoraMerge(r, httptest.NewRequest("POST", "/api/lora/merge", strings.NewReader(payload)))
		if r.Code != 400 {
			t.Fatalf("accepted incompatible Turbo request: %s", r.Body.String())
		}
	}
}
func TestH3QuantPolicyValidation(t *testing.T) {
	s := &Server{modelsDir: t.TempDir(), jobs: NewJobStore()}
	for _, payload := range []string{
		`{"source_path":"base","model_name":"out","formats":["FP8"],"architecture":"MiniMax H3","strategy":"Simple","h3_quant_policy":"upstream_int8_convrot"}`,
		`{"source_path":"base","model_name":"out","formats":["INT8 Row-wise ConvRot Runtime"],"architecture":"MiniMax H3","strategy":"Optimizer-driven","h3_quant_policy":"upstream_int8_convrot"}`,
		`{"source_path":"base","model_name":"out","formats":["FP8"],"architecture":"MiniMax H3","strategy":"Simple","verbose_level":"INVALID"}`,
	} {
		r := httptest.NewRecorder()
		s.handleQuantize(r, httptest.NewRequest("POST", "/api/quantize", strings.NewReader(payload)))
		if r.Code != 400 {
			t.Fatalf("accepted incompatible quant request: %s", r.Body.String())
		}
	}
}
func TestH3ExactRecipeDestination(t *testing.T) {
	d := t.TempDir()
	src := filepath.Join(d, "base.safetensors")
	os.WriteFile(src, []byte("fixture"), 0600)
	out := filepath.Join(d, "out.SAFETENSORS")
	os.WriteFile(out+".txt", []byte("owned recipe"), 0600)
	s := &Server{modelsDir: d}
	req := H3Request{BasePath: src, FoldMode: "independent", OutputPath: out}
	if err := s.validateH3Request(&req, false); err == nil {
		t.Fatal("accepted existing exact writer sidecar")
	}
	os.Remove(out + ".txt")
	os.WriteFile(filepath.Join(d, "out.txt"), []byte("unrelated"), 0600)
	req = H3Request{BasePath: src, FoldMode: "independent", OutputPath: out}
	if err := s.validateH3Request(&req, false); err != nil {
		t.Fatalf("unrelated sidecar blocked output: %v", err)
	}
	if req.OutputPath != out {
		t.Fatal("uppercase output changed")
	}
}

func TestH3CancellationCleansOnlyOwnedStaging(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	real := &Server{rootDir: root, python: filepath.Join(root, ".venv", "bin", "python")}
	d := t.TempDir()
	os.Mkdir(filepath.Join(d, "scripts"), 0700)
	// Real Python TensorSpool child, intentionally blocked while writing its payload.
	script := "import sys,time,os\nsys.path.insert(0," + strconv.Quote(real.rootDir) + ")\nfrom core.safetensors_stream import TensorSpool\nimport json\np=json.loads(sys.argv[-1])\nwith TensorSpool(p['output_path']) as spool:\n spool.data.write(b'partial payload')\n spool.data.flush()\n print(json.dumps({'type':'log','text':'ready'}),flush=True)\n time.sleep(60)\n"
	os.WriteFile(filepath.Join(d, "scripts", "go_bridge.py"), []byte(script), 0600)
	src := filepath.Join(d, "base.safetensors")
	os.WriteFile(src, []byte("fixture"), 0600)
	foreign := filepath.Join(d, ".h3_stage_foreign")
	os.Mkdir(foreign, 0700)
	os.WriteFile(filepath.Join(foreign, "payload"), []byte("foreign"), 0600)
	s := &Server{rootDir: d, modelsDir: d, python: real.python, jobs: NewJobStore()}
	r := httptest.NewRecorder()
	s.handleH3Prune(r, httptest.NewRequest("POST", "/api/h3/prune", strings.NewReader(`{"base_path":"`+src+`","fold_mode":"independent","output_name":"out"}`)))
	if r.Code != 200 {
		t.Fatal(r.Body.String())
	}
	var response map[string]string
	json.Unmarshal(r.Body.Bytes(), &response)
	job := s.jobs.Get(response["job_id"])
	if job == nil {
		t.Fatal("missing job")
	}
	select {
	case event := <-job.Events:
		if event.Text != "ready" {
			t.Fatalf("child did not start: %+v", event)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("child startup timed out")
	}
	job.cancel()
	done := make(chan struct{})
	go func() {
		for range job.Events {
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("cancellation did not finish")
	}
	entries, _ := os.ReadDir(d)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".h3_") && e.Name() != ".h3_stage_foreign" {
			t.Errorf("leaked staging: %s", e.Name())
		}
	}
	if data, err := os.ReadFile(filepath.Join(foreign, "payload")); err != nil || string(data) != "foreign" {
		t.Fatal("foreign staging changed")
	}
	for _, path := range []string{"out.safetensors", "out.safetensors.txt"} {
		if _, err := os.Lstat(filepath.Join(d, path)); !os.IsNotExist(err) {
			t.Fatalf("published cancelled artifact: %s", path)
		}
	}
}

func TestConfigUsesBridgeH3Capability(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{rootDir: root, modelsDir: t.TempDir(), python: filepath.Join(root, ".venv", "bin", "python")}
	for _, available := range []bool{true, false} {
		if !available {
			s.python = filepath.Join(t.TempDir(), "missing-python")
		}
		r := httptest.NewRecorder()
		s.handleConfig(r, httptest.NewRequest("GET", "/api/config", nil))
		var data map[string]json.RawMessage
		if err := json.Unmarshal(r.Body.Bytes(), &data); err != nil {
			t.Fatal(err)
		}
		var capability struct {
			Supported bool   `json:"supported"`
			Detail    string `json:"detail"`
		}
		if len(data["h3_ctq"]) == 0 {
			t.Fatal("config missing runtime H3 ctq capability")
		}
		if err := json.Unmarshal(data["h3_ctq"], &capability); err != nil {
			t.Fatal(err)
		}
		if capability.Detail == "" {
			t.Fatal("missing capability evidence")
		}
		if !available && capability.Supported {
			t.Fatal("unavailable bridge advertised supported")
		}
	}
}

func TestCompositionRejectsInvalidEnumsBeforeJob(t *testing.T) {
	for _, field := range []string{"merge_device", "consensus_preset", "cuda_device"} {
		s := &Server{modelsDir: t.TempDir(), jobs: NewJobStore()}
		r := httptest.NewRecorder()
		s.handleLoraCompose(r, httptest.NewRequest("POST", "/api/lora/compose", strings.NewReader(`{"loras":[{"path":"a"},{"path":"b"}],"dry_run":true,"`+field+`":"INVALID"}`)))
		if r.Code != 400 {
			t.Errorf("%s invalid accepted: %s", field, r.Body.String())
		}
	}
}

func TestH3EnumsRejectBeforeJob(t *testing.T) {
	d := t.TempDir()
	src := filepath.Join(d, "base.safetensors")
	os.WriteFile(src, []byte("fixture"), 0600)
	s := &Server{modelsDir: d, jobs: NewJobStore()}
	for _, field := range []string{"fold_mode", "merge_device", "cuda_device"} {
		r := httptest.NewRecorder()
		s.handleH3Prune(r, httptest.NewRequest("POST", "/api/h3/prune", strings.NewReader(`{"base_path":"`+src+`","fold_mode":"independent","output_name":"out","`+field+`":"INVALID"}`)))
		if r.Code != 400 {
			t.Errorf("%s invalid accepted: %s", field, r.Body.String())
		}
	}
}

func TestH3StreamFailureReapsChildBeforeCleanup(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cmd := exec.CommandContext(ctx, filepath.Join(root, ".venv", "bin", "python"), "-c", "import sys,time;sys.stdout.write('x'*5000000+'\\n');sys.stdout.flush();time.sleep(60)")
	cmd.Env = os.Environ()
	job := &Job{Events: make(chan Event, 512)}
	s := &Server{}
	err = s.runH3Command(ctx, cmd, job, t.TempDir())
	if err == nil {
		t.Fatal("expected oversized stream failure")
	}
	if cmd.ProcessState == nil {
		// Baseline regression must not leave its own test child running.
		cancel()
		cmd.Wait()
		t.Fatal("returned without waiting for child exit")
	}
}

func TestExtractionCancellationCleansOnlyOwnedStaging(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	real := &Server{rootDir: root, python: filepath.Join(root, ".venv", "bin", "python")}
	d := t.TempDir()
	os.Mkdir(filepath.Join(d, "scripts"), 0700)
	// Real Python TensorSpool child, intentionally blocked while writing its payload.
	script := "import sys,time,os\nsys.path.insert(0," + strconv.Quote(real.rootDir) + ")\nfrom core.safetensors_stream import TensorSpool\nimport json\np=json.loads(sys.argv[-1])\nwith TensorSpool(p['output_path']) as spool:\n spool.data.write(b'partial payload')\n spool.data.flush()\n print(json.dumps({'type':'log','text':'ready'}),flush=True)\n time.sleep(60)\n"
	os.WriteFile(filepath.Join(d, "scripts", "go_bridge.py"), []byte(script), 0600)
	src := filepath.Join(d, "base.safetensors")
	os.WriteFile(src, []byte("fixture"), 0600)
	foreign := filepath.Join(d, ".h3_stage_foreign")
	os.Mkdir(foreign, 0700)
	os.WriteFile(filepath.Join(foreign, "payload"), []byte("foreign"), 0600)
	s := &Server{rootDir: d, modelsDir: d, python: real.python, jobs: NewJobStore()}
	ctx, cancel := context.WithCancel(context.Background())
	job := &Job{Events: make(chan Event, 512), cancel: cancel}
	go s.runLoraExtractJob(ctx, job, LoraExtractRequest{OutputPath: filepath.Join(d, "out.safetensors"), ModelsDir: d})
	select {
	case event := <-job.Events:
		if event.Text != "ready" {
			t.Fatalf("child did not start: %+v", event)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("child startup timed out")
	}
	job.cancel()
	done := make(chan struct{})
	go func() {
		for range job.Events {
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("cancellation did not finish")
	}
	entries, _ := os.ReadDir(d)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".h3_") && e.Name() != ".h3_stage_foreign" {
			t.Errorf("leaked staging: %s", e.Name())
		}
	}
	if data, err := os.ReadFile(filepath.Join(foreign, "payload")); err != nil || string(data) != "foreign" {
		t.Fatal("foreign staging changed")
	}
	for _, path := range []string{"out.safetensors", "out.safetensors.txt"} {
		if _, err := os.Lstat(filepath.Join(d, path)); !os.IsNotExist(err) {
			t.Fatalf("published cancelled artifact: %s", path)
		}
	}
}

func TestCompositionCancellationCleansOnlyOwnedStaging(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	real := &Server{rootDir: root, python: filepath.Join(root, ".venv", "bin", "python")}
	d := t.TempDir()
	os.Mkdir(filepath.Join(d, "scripts"), 0700)
	// Real Python TensorSpool child, intentionally blocked while writing its payload.
	script := "import sys,time,os\nsys.path.insert(0," + strconv.Quote(real.rootDir) + ")\nfrom core.safetensors_stream import TensorSpool\nimport json\np=json.loads(sys.argv[-1])\nwith TensorSpool(p['output_path']) as spool:\n spool.data.write(b'partial payload')\n spool.data.flush()\n print(json.dumps({'type':'log','text':'ready'}),flush=True)\n time.sleep(60)\n"
	os.WriteFile(filepath.Join(d, "scripts", "go_bridge.py"), []byte(script), 0600)
	src := filepath.Join(d, "base.safetensors")
	os.WriteFile(src, []byte("fixture"), 0600)
	foreign := filepath.Join(d, ".h3_stage_foreign")
	os.Mkdir(foreign, 0700)
	os.WriteFile(filepath.Join(foreign, "payload"), []byte("foreign"), 0600)
	s := &Server{rootDir: d, modelsDir: d, python: real.python, jobs: NewJobStore()}
	ctx, cancel := context.WithCancel(context.Background())
	job := &Job{Events: make(chan Event, 512), cancel: cancel}
	go s.runLoraComposeJob(ctx, job, LoraComposeRequest{OutputPath: filepath.Join(d, "out.safetensors"), ModelsDir: d})
	select {
	case event := <-job.Events:
		if event.Text != "ready" {
			t.Fatalf("child did not start: %+v", event)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("child startup timed out")
	}
	job.cancel()
	done := make(chan struct{})
	go func() {
		for range job.Events {
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("cancellation did not finish")
	}
	entries, _ := os.ReadDir(d)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".h3_") && e.Name() != ".h3_stage_foreign" {
			t.Errorf("leaked staging: %s", e.Name())
		}
	}
	if data, err := os.ReadFile(filepath.Join(foreign, "payload")); err != nil || string(data) != "foreign" {
		t.Fatal("foreign staging changed")
	}
	for _, path := range []string{"out.safetensors", "out.safetensors.txt"} {
		if _, err := os.Lstat(filepath.Join(d, path)); !os.IsNotExist(err) {
			t.Fatalf("published cancelled artifact: %s", path)
		}
	}
}

func TestH3CancellationDuringPublicationRemovesOwnedSidecar(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	real := &Server{rootDir: root, python: filepath.Join(root, ".venv", "bin", "python")}
	d := t.TempDir()
	os.Mkdir(filepath.Join(d, "scripts"), 0700)
	// Real Python TensorSpool child, intentionally blocked while writing its payload.
	script := "import sys,time,os,torch\nsys.path.insert(0," + strconv.Quote(real.rootDir) + ")\nfrom core.safetensors_stream import TensorSpool\nimport json\np=json.loads(sys.argv[-1])\nlink=os.link\ndef pause(src,dst):\n link(src,dst)\n print(json.dumps({'type':'log','text':'ready'}),flush=True)\n time.sleep(60)\nos.link=pause\nwith TensorSpool(p['output_path']) as spool:\n spool.tensor('weight',torch.ones(2,2))\n spool.publish({},'recipe')\n"

	os.WriteFile(filepath.Join(d, "scripts", "go_bridge.py"), []byte(script), 0600)
	src := filepath.Join(d, "base.safetensors")
	os.WriteFile(src, []byte("fixture"), 0600)
	foreign := filepath.Join(d, ".h3_stage_foreign")
	os.Mkdir(foreign, 0700)
	os.WriteFile(filepath.Join(foreign, "payload"), []byte("foreign"), 0600)
	s := &Server{rootDir: d, modelsDir: d, python: real.python, jobs: NewJobStore()}
	r := httptest.NewRecorder()
	s.handleH3Prune(r, httptest.NewRequest("POST", "/api/h3/prune", strings.NewReader(`{"base_path":"`+src+`","fold_mode":"independent","output_name":"out"}`)))
	if r.Code != 200 {
		t.Fatal(r.Body.String())
	}
	var response map[string]string
	json.Unmarshal(r.Body.Bytes(), &response)
	job := s.jobs.Get(response["job_id"])
	if job == nil {
		t.Fatal("missing job")
	}
	select {
	case event := <-job.Events:
		if event.Text != "ready" {
			t.Fatalf("child did not start: %+v", event)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("child startup timed out")
	}
	job.cancel()
	done := make(chan struct{})
	go func() {
		for range job.Events {
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("cancellation did not finish")
	}
	entries, _ := os.ReadDir(d)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".h3_") && e.Name() != ".h3_stage_foreign" {
			t.Errorf("leaked staging: %s", e.Name())
		}
	}
	if data, err := os.ReadFile(filepath.Join(foreign, "payload")); err != nil || string(data) != "foreign" {
		t.Fatal("foreign staging changed")
	}
	for _, path := range []string{"out.safetensors", "out.safetensors.txt"} {
		if _, err := os.Lstat(filepath.Join(d, path)); !os.IsNotExist(err) {
			t.Fatalf("published cancelled artifact: %s", path)
		}
	}
}

func TestInspectAPIExposesH3Coordinates(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{rootDir: root, modelsDir: t.TempDir(), python: filepath.Join(root, ".venv", "bin", "python")}
	path := filepath.Join(s.modelsDir, "pruned.safetensors")
	script := "import sys,torch;sys.path.insert(0," + strconv.Quote(root) + ");from tests.test_h3_variant_contract import full_tensors;from core.h3_curve import TIME_KEYS;from safetensors.torch import save_file;t={k:v for k,v in full_tensors().items() if k not in TIME_KEYS};t['adaln_t_table']=torch.zeros(65,3);save_file(t,sys.argv[1])"
	if out, err := exec.Command(s.python, "-c", script, path).CombinedOutput(); err != nil {
		t.Fatalf("fixture: %v %s", err, out)
	}
	r := httptest.NewRecorder()
	s.handleInspect(r, httptest.NewRequest("GET", "/api/inspect?path="+path, nil))
	if r.Code != 200 {
		t.Fatalf("inspect failed: %s", r.Body.String())
	}
	var data struct {
		H3 struct {
			Variant    string `json:"variant"`
			Coordinate string `json:"adaln_coordinate_table_sha256"`
		} `json:"h3"`
	}
	if err := json.Unmarshal(r.Body.Bytes(), &data); err != nil {
		t.Fatal(err)
	}
	if data.H3.Variant != "pruned" || len(data.H3.Coordinate) != 64 {
		t.Fatalf("missing coordinate metadata: %s", r.Body.String())
	}
}

func TestH3PruneRealBridgeExplicitUppercaseDestination(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	d := t.TempDir()
	src := filepath.Join(d, "base.safetensors")
	s := &Server{rootDir: root, modelsDir: d, python: filepath.Join(root, ".venv", "bin", "python"), jobs: NewJobStore()}
	script := "import sys;sys.path.insert(0," + strconv.Quote(root) + ");from tests.test_h3_variant_contract import full_tensors;from safetensors.torch import save_file;save_file(full_tensors(),sys.argv[1])"
	if out, err := exec.Command(s.python, "-c", script, src).CombinedOutput(); err != nil {
		t.Fatalf("fixture: %v %s", err, out)
	}
	chosen := filepath.Join(d, "chosen")
	os.Mkdir(chosen, 0700)
	output := filepath.Join(chosen, "out.SAFETENSORS")
	payload, _ := json.Marshal(H3Request{BasePath: src, FoldMode: "independent", OutputPath: output, MergeDevice: "cpu"})
	r := httptest.NewRecorder()
	s.handleH3Prune(r, httptest.NewRequest("POST", "/api/h3/prune", strings.NewReader(string(payload))))
	if r.Code != 200 {
		t.Fatal(r.Body.String())
	}
	var response map[string]string
	json.Unmarshal(r.Body.Bytes(), &response)
	job := s.jobs.Get(response["job_id"])
	if job == nil {
		t.Fatal("missing job")
	}
	t.Cleanup(job.cancel)
	timer := time.NewTimer(30 * time.Second)
	defer timer.Stop()
	for {
		select {
		case event, ok := <-job.Events:
			if !ok {
				goto finished
			}
			if event.Type == "error" {
				t.Fatal(event.Text)
			}
		case <-timer.C:
			t.Fatal("real pruning timed out")
		}
	}
finished:
	for _, path := range []string{output, output + ".txt"} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("missing exact artifact: %s: %v", path, err)
		}
	}
	if _, err := os.Stat(output + ".safetensors"); !os.IsNotExist(err) {
		t.Fatal("uppercase suffix duplicated")
	}
	entries, _ := os.ReadDir(chosen)
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".h3_") {
			t.Fatal("successful job leaked staging")
		}
	}
}

// Exercise the real bridge and engines, pausing only after a successful publication
// hard link. A killed Python context manager cannot perform its own cleanup.
func testAdapterPublicationCancellation(t *testing.T, bake bool, outputName string, linkNumber int, replaceArtifact bool) {
	t.Helper()
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	d := t.TempDir()
	python := filepath.Join(root, ".venv", "bin", "python")
	base := filepath.Join(d, "base.safetensors")
	modified := filepath.Join(d, "modified.safetensors")
	adapter := filepath.Join(d, "adapter.safetensors")
	fixture := "import sys,torch;sys.path.insert(0," + strconv.Quote(root) + ");from tests.test_h3_variant_contract import full_tensors;from safetensors.torch import save_file;t=full_tensors();save_file(t,sys.argv[1]);k='blocks.0.attn.qkv_proj.weight';delta=torch.ones_like(t[k])*.1;t[k]+=delta;save_file(t,sys.argv[2]);save_file({'diffusion_model.blocks.0.attn.qkv_proj.diff':delta},sys.argv[3])"
	if out, err := exec.Command(python, "-c", fixture, base, modified, adapter).CombinedOutput(); err != nil {
		t.Fatalf("fixture: %v %s", err, out)
	}
	if err := os.Mkdir(filepath.Join(d, "scripts"), 0700); err != nil {
		t.Fatal(err)
	}
	script := "import sys,os,json,time,runpy\nsys.path.insert(0," + strconv.Quote(root) + ")\nlink=os.link\ncount=0\ndef pause(src,dst):\n global count\n link(src,dst)\n count+=1\n if count==" + strconv.Itoa(linkNumber) + ":\n  print(json.dumps({'type':'log','text':'publication-ready'}),flush=True)\n  time.sleep(60)\nos.link=pause\nrunpy.run_path(" + strconv.Quote(filepath.Join(root, "scripts", "go_bridge.py")) + ",run_name='__main__')\n"
	if err := os.WriteFile(filepath.Join(d, "scripts", "go_bridge.py"), []byte(script), 0600); err != nil {
		t.Fatal(err)
	}
	chosen := filepath.Join(d, "chosen")
	if err := os.Mkdir(chosen, 0700); err != nil {
		t.Fatal(err)
	}
	foreign := filepath.Join(chosen, ".h3_stage_foreign")
	if err := os.Mkdir(foreign, 0700); err != nil {
		t.Fatal(err)
	}
	foreignPayload := filepath.Join(foreign, "payload")
	if err := os.WriteFile(foreignPayload, []byte("foreign"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(chosen, "unrelated.txt"), []byte("foreign"), 0600); err != nil {
		t.Fatal(err)
	}
	s := &Server{rootDir: d, modelsDir: d, python: python, jobs: NewJobStore()}
	r := httptest.NewRecorder()
	output := filepath.Join(chosen, outputName)
	if outputName == "" {
		output = filepath.Join(chosen, "minimax_h3_extracted_lora.safetensors")
	}
	if !strings.HasSuffix(strings.ToLower(output), ".safetensors") {
		output += ".safetensors"
	}
	if bake {
		// Explicit suffixless destination must be normalized before launching Python.
		payload, _ := json.Marshal(LoraMergeRequest{BasePath: base, Loras: []LoraSpec{{Path: adapter, Strength: 1}}, Architecture: "MiniMax H3", GlobalStrength: 1, OutputPath: filepath.Join(chosen, outputName), MergeDevice: "cpu"})
		s.handleLoraMerge(r, httptest.NewRequest("POST", "/api/lora/merge", strings.NewReader(string(payload))))
	} else {
		payload, _ := json.Marshal(LoraExtractRequest{BasePath: base, MergedPath: modified, Architecture: "MiniMax H3", Recipe: "h3_full", OutputDir: chosen, OutputName: outputName})
		s.handleLoraExtract(r, httptest.NewRequest("POST", "/api/lora/extract", strings.NewReader(string(payload))))
	}
	if r.Code != 200 {
		t.Fatal(r.Body.String())
	}
	var response map[string]string
	if err := json.Unmarshal(r.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	job := s.jobs.Get(response["job_id"])
	if job == nil {
		t.Fatal("missing job")
	}
	t.Cleanup(job.cancel)
	timer := time.NewTimer(30 * time.Second)
	defer timer.Stop()
waitReady:
	for {
		select {
		case event, ok := <-job.Events:
			if !ok {
				t.Fatal("child exited before publication")
			}
			if event.Type == "error" {
				t.Fatal(event.Text)
			}
			if event.Text == "publication-ready" {
				break waitReady
			}
		case <-timer.C:
			t.Fatal("publication timed out")
		}
	}
	// Verify exact publication targets are hard links to the private job's inode.
	published := map[string]string{"recipe.txt": output + ".txt"}
	if linkNumber == 2 {
		published["artifact.safetensors"] = output
	}
	for stagedName, target := range published {
		staged, err := filepath.Glob(filepath.Join(chosen, ".h3_job_*", ".h3_stage_*", stagedName))
		if err != nil || len(staged) != 1 {
			t.Fatalf("missing unique owned stage for %s: %v %v", target, staged, err)
		}
		owned, e1 := os.Lstat(staged[0])
		linked, e2 := os.Lstat(target)
		if e1 != nil || e2 != nil || !owned.Mode().IsRegular() || !linked.Mode().IsRegular() || !os.SameFile(owned, linked) {
			t.Fatalf("publication is not an owned regular-file hard link: %s (%v, %v)", target, e1, e2)
		}
	}
	if replaceArtifact {
		if err := os.Remove(output); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(output, []byte("foreign"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	job.cancel()
	done := make(chan struct{})
	go func() {
		for range job.Events {
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("cancellation did not finish")
	}
	for _, path := range []string{output, output + ".txt"} {
		if replaceArtifact && path == output {
			if data, err := os.ReadFile(path); err != nil || string(data) != "foreign" {
				t.Fatal("foreign replacement changed")
			}
		} else if _, err := os.Lstat(path); !os.IsNotExist(err) {
			t.Errorf("leaked owned publication: %s", path)
		}
	}
	entries, err := os.ReadDir(chosen)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".h3_") && entry.Name() != ".h3_stage_foreign" {
			t.Errorf("leaked owned staging: %s", entry.Name())
		}
	}
	for _, path := range []string{foreignPayload, filepath.Join(chosen, "unrelated.txt")} {
		if data, err := os.ReadFile(path); err != nil || string(data) != "foreign" {
			t.Fatalf("foreign file changed: %s", path)
		}
	}
}

func TestBakeCancellationAfterPublicationCleansOwnedArtifacts(t *testing.T) {
	for _, phase := range []int{1, 2} {
		t.Run(strconv.Itoa(phase), func(t *testing.T) { testAdapterPublicationCancellation(t, true, "baked", phase, false) })
	}
	t.Run("foreign-replacement", func(t *testing.T) { testAdapterPublicationCancellation(t, true, "baked", 2, true) })
}

func TestNameOnlyExtractionCancellationAfterPublicationCleansOwnedArtifacts(t *testing.T) {
	for _, phase := range []int{1, 2} {
		t.Run(strconv.Itoa(phase), func(t *testing.T) {
			testAdapterPublicationCancellation(t, false, "extracted.SAFETENSORS", phase, false)
		})
	}
	t.Run("default-name", func(t *testing.T) { testAdapterPublicationCancellation(t, false, "", 1, false) })
	t.Run("foreign-replacement", func(t *testing.T) { testAdapterPublicationCancellation(t, false, "extracted", 2, true) })
}

func TestAdapterDestinationsRejectUnsafeBeforeLaunch(t *testing.T) {
	d := t.TempDir()
	base := filepath.Join(d, "base.safetensors")
	modified := filepath.Join(d, "modified.safetensors")
	adapter := filepath.Join(d, "adapter.safetensors")
	for _, path := range []string{base, modified, adapter} {
		if err := os.WriteFile(path, []byte("foreign"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	out := filepath.Join(d, "out.SAFETENSORS")
	if err := os.WriteFile(out+".txt", []byte("foreign"), 0600); err != nil {
		t.Fatal(err)
	}
	dangling := filepath.Join(d, "dangling.safetensors")
	if err := os.Symlink(filepath.Join(d, "missing"), dangling); err != nil {
		t.Fatal(err)
	}
	for _, bake := range []bool{false, true} {
		for _, destination := range []struct{ name, path string }{
			{name: "../escape"}, {name: ".."}, {name: "base.safetensors"},
			{path: modified}, {path: adapter}, {path: out}, {path: dangling},
		} {
			s := &Server{modelsDir: d, jobs: NewJobStore()}
			r := httptest.NewRecorder()
			if bake {
				payload, _ := json.Marshal(LoraMergeRequest{BasePath: base, Loras: []LoraSpec{{Path: modified}, {Path: adapter}}, OutputName: destination.name, OutputPath: destination.path})
				s.handleLoraMerge(r, httptest.NewRequest("POST", "/api/lora/merge", strings.NewReader(string(payload))))
			} else {
				payload, _ := json.Marshal(LoraExtractRequest{BasePath: base, MergedPath: modified, PrunedTargetPath: adapter, Architecture: "MiniMax H3", Recipe: "h3_pruned", OutputName: destination.name, OutputPath: destination.path})
				s.handleLoraExtract(r, httptest.NewRequest("POST", "/api/lora/extract", strings.NewReader(string(payload))))
			}
			if r.Code != 400 {
				t.Fatalf("bake=%v accepted unsafe destination %+v: %s", bake, destination, r.Body.String())
			}
		}
	}
	for _, path := range []string{base, modified, adapter, out + ".txt"} {
		if data, err := os.ReadFile(path); err != nil || string(data) != "foreign" {
			t.Fatalf("foreign destination changed: %s", path)
		}
	}
	// Resolve symlink parents exactly as Python realpath, preserving uppercase suffix.
	alias := filepath.Join(d, "alias")
	if err := os.Symlink(d, alias); err != nil {
		t.Fatal(err)
	}
	resolved, err := resolveAdapterOutput("", "safe.SAFETENSORS", alias, "unused")
	if err != nil || resolved != filepath.Join(d, "safe.SAFETENSORS") {
		t.Fatalf("destination mismatch: %s %v", resolved, err)
	}
}

func TestH3DestinationSafety(t *testing.T) {
	d := t.TempDir()
	src := filepath.Join(d, "base.safetensors")
	os.WriteFile(src, []byte("fixture"), 0600)
	s := &Server{modelsDir: d}
	for _, name := range []string{"../escape", "base.safetensors"} {
		req := H3Request{BasePath: src, FoldMode: "independent", OutputName: name}
		if err := s.validateH3Request(&req, false); err == nil {
			t.Fatalf("unsafe destination accepted: %s", name)
		}
	}
	req := H3Request{BasePath: src, FoldMode: "independent", OutputName: "out", MergeDevice: "cpu"}
	if err := s.validateH3Request(&req, false); err != nil {
		t.Fatal(err)
	}
	if req.Architecture != "MiniMax H3" || req.OutputPath != filepath.Join(d, "out.safetensors") {
		t.Fatalf("bad normalization: %+v", req)
	}
}
