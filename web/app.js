const state = {
  rootDir: "",
  modelsDir: "",
  sourcePath: "",
  architecture: "WAN 2.2",
  strategy: "Optimizer-driven",
  workflowMode: "quantize",
  formats: new Set(),
  browserMode: "file",
  pendingLoraSlot: -1,
  loras: [],
  browserPath: "",
  jobId: "",
  events: null,
  logBuffer: "",
  logFlushPending: false,
};

const $ = (id) => document.getElementById(id);

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).error || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
}

function log(text) {
  state.logBuffer += text;
  if (state.logFlushPending) return;
  state.logFlushPending = true;
  requestAnimationFrame(flushLog);
}

function flushLog() {
  const el = $("console");
  if (!el) return;
  if (state.logBuffer) {
    el.append(document.createTextNode(state.logBuffer));
    state.logBuffer = "";
  }
  const maxChars = 250000;
  if (el.textContent.length > maxChars) {
    el.textContent = el.textContent.slice(-maxChars);
  }
  el.scrollTop = el.scrollHeight;
  state.logFlushPending = false;
}

function setStatus(text) {
  $("status").textContent = text;
}

function shortPath(path) {
  if (!path) return "";
  const parts = path.split("/");
  return parts.slice(-2).join("/");
}

async function init() {
  const cfg = await api("/api/config");
  state.rootDir = cfg.root_dir;
  state.modelsDir = cfg.models_dir;
  state.browserPath = cfg.models_dir;
  $("folder-label").textContent = shortPath(cfg.models_dir);

  const arch = $("architecture");
  cfg.architectures.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    arch.appendChild(opt);
  });
  arch.value = state.architecture;

  renderFormats(cfg.formats);

  wireEvents();
  refreshMetadata();
  refreshSystem();
  setInterval(refreshSystem, 5000);
}

function renderFormats(formats) {
  const root = $("formats");
  root.textContent = "";
  const groups = [
    {title: "Safetensors", items: formats.filter((fmt) => !fmt.value.startsWith("GGUF_"))},
    {title: "GGUF", items: formats.filter((fmt) => fmt.value.startsWith("GGUF_"))},
  ];

  groups.forEach((group) => {
    const section = document.createElement("section");
    section.className = "format-group";
    const title = document.createElement("h4");
    title.textContent = group.title;
    const list = document.createElement("div");
    list.className = "format-list";
    group.items.forEach((fmt) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
      btn.textContent = compactFormatLabel(fmt.label);
    btn.dataset.value = fmt.value;
    btn.addEventListener("click", () => {
      if (state.formats.has(fmt.value)) {
        state.formats.delete(fmt.value);
        btn.classList.remove("active");
      } else {
        state.formats.add(fmt.value);
        btn.classList.add("active");
      }
    });
      list.appendChild(btn);
    });
    section.appendChild(title);
    section.appendChild(list);
    root.appendChild(section);
  });
}

function compactFormatLabel(label) {
  return label
    .replace("Safetensors: ", "")
    .replace("GGUF ", "")
    .replace("INT8 Row-wise ConvRot (runtime)", "INT8 ConvRot")
    .replace("INT8 Tensor-wise", "INT8 Tensor")
    .replace("GGUF ", "");
}

function wireEvents() {
  $("pick-folder").addEventListener("click", () => openBrowser("folder"));
  $("pick-source").addEventListener("click", () => openBrowser("file"));
  document.querySelectorAll("#workflow-mode button").forEach((btn) => {
    btn.addEventListener("click", () => setWorkflowMode(btn.dataset.mode));
  });
  $("add-lora").addEventListener("click", addLoraRow);
  renderLoras();
  $("browser-close").addEventListener("click", () => $("browser").close());
  $("browser-up").addEventListener("click", () => browse($("browser").dataset.parent || state.browserPath));
  $("browser-select-folder").addEventListener("click", () => selectFolder(state.browserPath));
  $("refresh-files").addEventListener("click", () => browse(state.modelsDir));
  $("clear-console").addEventListener("click", () => {
    state.logBuffer = "";
    $("console").textContent = "";
  });
  $("refresh-meta").addEventListener("click", refreshMetadata);
  $("read-meta").addEventListener("click", readMetadata);
  $("inject-meta").addEventListener("click", injectMetadata);
  $("copy-meta").addEventListener("click", () => navigator.clipboard.writeText($("metadata").value));

  $("architecture").addEventListener("change", () => {
    state.architecture = $("architecture").value;
    refreshMetadata();
  });
  $("model-name").addEventListener("input", refreshMetadata);
  $("full-checkpoint").addEventListener("change", refreshMetadata);

  document.querySelectorAll("#strategy button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#strategy button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.strategy = btn.dataset.value;
    });
  });

  $("start").addEventListener("click", () => {
    if (state.workflowMode === "lora") {
      startLoraMerge();
    } else {
      startJob();
    }
  });
  $("stop").addEventListener("click", stopJob);
  $("clean-memory").addEventListener("click", cleanMemory);
  $("update-app").addEventListener("click", updateApp);
  $("scan").addEventListener("click", () => runTool("scan"));
  $("audit").addEventListener("click", () => runTool("audit"));
}

function setWorkflowMode(mode) {
  state.workflowMode = mode;
  document.querySelectorAll("#workflow-mode button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  document.querySelectorAll("[data-workflow-panel]").forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.workflowPanel !== mode);
  });
  // Show start button in both modes; label changes to match context.
  const startBtn = $("start");
  startBtn.classList.remove("hidden");
  startBtn.textContent = mode === "lora" ? "Start Merge" : "Start Batch";
  // Update checkpoint hint label per mode.
  const hint = $("source-label-hint");
  if (hint) {
    hint.textContent = mode === "lora" ? "Base checkpoint" : "Checkpoint";
  }
  setStatus(mode === "quantize" ? "Quantize mode" : "LoRA merge mode");
}

async function openBrowser(mode) {
  state.browserMode = mode;
  $("browser-title").textContent = mode === "folder" ? "Choose Model Folder" : (mode === "lora" ? "Choose LoRA" : "Choose Checkpoint");
  $("browser-select-folder").style.display = mode === "folder" ? "" : "none";
  await browse(mode === "folder" ? state.modelsDir : (state.modelsDir || state.rootDir));
  $("browser").showModal();
}

async function browse(path) {
  try {
    const data = await api(`/api/browse?path=${encodeURIComponent(path || state.modelsDir)}`);
    state.browserPath = data.path;
    $("browser").dataset.parent = data.parent;
    $("browser-path").textContent = data.path;
    const list = $("browser-list");
    list.textContent = "";
    data.items.forEach((item) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "browser-item";
      btn.innerHTML = `<span>${item.name}</span><span class="kind">${item.is_dir ? "folder" : "file"}</span>`;
      btn.addEventListener("click", () => {
        if (item.is_dir) {
          browse(item.path);
        } else if (state.browserMode === "file") {
          selectSource(item.path);
        } else if (state.browserMode === "lora") {
          selectLora(item.path);
        }
      });
      list.appendChild(btn);
    });
  } catch (err) {
    log(`Browse error: ${err.message}\n`);
  }
}

function selectFolder(path) {
  state.modelsDir = path;
  $("folder-label").textContent = shortPath(path);
  $("browser").close();
}

function addLoraRow() {
  state.loras.push({path: "", strength: 0.65, strategy: "Balanced", enabled: true});
  renderLoras();
}

function selectLora(path) {
  if (state.pendingLoraSlot >= 0 && state.loras[state.pendingLoraSlot]) {
    state.loras[state.pendingLoraSlot].path = path;
  } else {
    state.loras.push({path, strength: 0.65, strategy: "Balanced", enabled: true});
  }
  state.pendingLoraSlot = -1;
  renderLoras();
  $("browser").close();
}

function renderLoras() {
  const root = $("lora-list");
  root.textContent = "";
  if (state.loras.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted-note";
    empty.textContent = "No LoRAs selected. Add one or more LoRAs and choose a strategy for each.";
    root.appendChild(empty);
    return;
  }
  // WAN 2.2 has no audio components, so Audio strategy is not useful.
  const allStrategies = ["Balanced", "Motion", "Visuals", "Audio"];
  const strategies = state.architecture === "WAN 2.2"
    ? allStrategies.filter((s) => s !== "Audio")
    : allStrategies;
  state.loras.forEach((lora, idx) => {
    const row = document.createElement("div");
    row.className = "lora-row";
    row.innerHTML = `
      <label class="checkline" title="Enable this LoRA for the merge"><input type="checkbox" ${lora.enabled ? "checked" : ""} data-role="enabled"><span>Merge</span></label>
      <button class="ghost lora-pick" type="button" title="Choose a LoRA file">${lora.path ? shortPath(lora.path) : "Choose LoRA"}</button>
      <select data-role="strategy" title="Merge strategy: how LoRA weights are applied">${strategies.map((s) => `<option value="${s}" ${s === lora.strategy ? "selected" : ""}>${s}</option>`).join("")}</select>
      <input data-role="strength" type="number" min="0" max="2" step="0.05" value="${lora.strength}" title="Per-LoRA strength multiplier (0=none, 1=full)">
      <button class="ghost" data-role="remove" type="button" title="Remove this LoRA from the list">Remove</button>
    `;
    row.querySelector(".lora-pick").addEventListener("click", () => {
      state.pendingLoraSlot = idx;
      openBrowser("lora");
    });
    row.querySelector('[data-role="enabled"]').addEventListener("change", (ev) => lora.enabled = ev.target.checked);
    row.querySelector('[data-role="strategy"]').addEventListener("change", (ev) => lora.strategy = ev.target.value);
    row.querySelector('[data-role="strength"]').addEventListener("input", (ev) => lora.strength = Number(ev.target.value) || 0);
    row.querySelector('[data-role="remove"]').addEventListener("click", () => {
      state.loras.splice(idx, 1);
      renderLoras();
    });
    root.appendChild(row);
  });
}

async function selectSource(path) {
  state.sourcePath = path;
  $("source-label").textContent = shortPath(path);
  $("browser").close();
  if (!$("model-name").value) {
    $("model-name").value = path.split("/").pop().replace(/\.(safetensors|gguf|ckpt|pt|bin)$/i, "");
  }
  if (path.endsWith(".safetensors")) {
    try {
      const info = await api(`/api/inspect?path=${encodeURIComponent(path)}`);
      if (["WAN 2.2", "LTX-2.3"].includes(info.architecture)) {
        state.architecture = info.architecture;
        $("architecture").value = info.architecture;
      }
      $("full-checkpoint").checked = !!info.full_checkpoint;
      log(`Source inspected: ${path}\nArchitecture: ${info.architecture}\nFull checkpoint: ${info.full_checkpoint ? "yes" : "no"}\n\n`);
      refreshMetadata();
    } catch (err) {
      log(`Source inspection failed: ${err.message}\n`);
    }
  }
}

async function refreshMetadata() {
  const name = $("model-name").value || "TreasureChest";
  const arch = $("architecture").value || "WAN 2.2";
  const full = $("full-checkpoint").checked;
  try {
    const data = await api(`/api/metadata-preview?name=${encodeURIComponent(name)}&architecture=${encodeURIComponent(arch)}&full=${full}`);
    $("metadata").value = data.metadata;
  } catch {
    // Keep typing smooth if metadata generation is temporarily unavailable.
  }
}

async function startLoraMerge() {
  const basePath = state.sourcePath;
  const selected = state.loras.filter((lora) => lora.enabled && lora.path);
  if (!basePath) return log("Select a base checkpoint in the sidebar first.\n");
  if (selected.length === 0) return log("Add and enable at least one LoRA.\n");
  const dryRun = $("lora-dry-run").checked;
  if (!dryRun && !$("lora-output").value) return log("Enter an output name before writing a merged checkpoint.\n");

  $("start").disabled = true;
  $("stop").disabled = false;
  setStatus(dryRun ? "Starting LoRA dry run" : "Starting LoRA merge");
  log(`\nStarting ${dryRun ? "LoRA dry run" : "LoRA merge"} with ${selected.length} LoRA(s)...\n`);

  try {
    const data = await api("/api/lora/merge", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        base_path: basePath,
        models_dir: state.modelsDir,
        output_name: $("lora-output").value,
        loras: selected.map((lora) => ({
          path: lora.path,
          strength: Number(lora.strength) || 0,
          strategy: lora.strategy,
        })),
        strategy: "Balanced",
        architecture: state.architecture,
        global_strength: Number($("lora-global-strength").value) || 1,
        adaptive: $("lora-adaptive").checked,
        dry_run: dryRun,
        strict_matching: $("lora-strict").checked,
      }),
    });
    state.jobId = data.job_id;
    attachEvents(data.job_id);
  } catch (err) {
    log(`LoRA merge failed to start: ${err.message}\n`);
    $("start").disabled = false;
    $("stop").disabled = true;
    setStatus("Error");
  }
}

async function startJob() {
  if (!state.sourcePath) return log("Select a source checkpoint first.\n");
  if (!$("model-name").value) return log("Enter a display name.\n");
  if (state.formats.size === 0) return log("Choose at least one target format.\n");

  $("start").disabled = true;
  $("stop").disabled = false;
  setStatus("Starting");
  log("\nStarting quantization job...\n");

  try {
    const data = await api("/api/quantize", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        models_dir: state.modelsDir,
        source_path: state.sourcePath,
        model_name: $("model-name").value,
        formats: [...state.formats],
        architecture: $("architecture").value,
        strategy: state.strategy,
        optimizer: "prodigy",
        low_vram: $("low-vram").checked,
        full_checkpoint: $("full-checkpoint").checked,
      }),
    });
    state.jobId = data.job_id;
    attachEvents(data.job_id);
  } catch (err) {
    log(`Start failed: ${err.message}\n`);
    $("start").disabled = false;
    $("stop").disabled = true;
    setStatus("Error");
  }
}

function attachEvents(id) {
  if (state.events) state.events.close();
  state.events = new EventSource(`/api/jobs/${id}/events`);
  state.events.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.text) log(ev.text);
    if (ev.status) setStatus(ev.status);
    if (ev.type === "done" || ev.type === "error") {
      if (ev.type === "error" && ev.text) log(`Error: ${ev.text}\n`);
      $("start").disabled = false;
      $("stop").disabled = true;
      $("update-app").disabled = false;
      state.events.close();
      if (ev.status === "restarting") {
        setTimeout(() => window.location.reload(), 1800);
      }
    }
  };
  state.events.onerror = () => {
    log("Log stream disconnected.\n");
    $("start").disabled = false;
    $("stop").disabled = true;
    $("update-app").disabled = false;
    if (state.events) state.events.close();
  };
}

async function stopJob() {
  if (!state.jobId) return;
  await api(`/api/jobs/${state.jobId}/stop`, {method: "POST"});
  setStatus("Stopping");
}

async function updateApp() {
  if (!confirm("Run dependency update, rebuild the Go app, and restart DaSiWa?")) return;
  $("update-app").disabled = true;
  $("start").disabled = true;
  $("stop").disabled = false;
  setStatus("Updating");
  log("\nStarting app update...\n");

  try {
    const data = await api("/api/update", {method: "POST"});
    state.jobId = data.job_id;
    attachEvents(data.job_id);
  } catch (err) {
    log(`Update failed to start: ${err.message}\n`);
    $("update-app").disabled = false;
    $("start").disabled = false;
    $("stop").disabled = true;
    setStatus("Error");
  }
}

async function cleanMemory() {
  const btn = $("clean-memory");
  btn.disabled = true;
  setStatus("Cleaning memory");
  log("\nCleaning RAM / VRAM caches...\n");

  try {
    const data = await api("/api/memory/clean", {method: "POST"});
    log(`\n${data.text}\n`);
    await refreshSystem();
    setStatus("Idle");
  } catch (err) {
    log(`Memory cleanup failed: ${err.message}\n`);
    setStatus("Error");
  } finally {
    btn.disabled = false;
  }
}

async function runTool(kind) {
  if (!state.sourcePath) return log("Select a source checkpoint first.\n");
  const url = kind === "scan" ? "/api/tools/scan" : "/api/tools/audit";
  const payload = kind === "scan"
    ? {path: state.sourcePath}
    : {path: state.sourcePath, architecture: $("architecture").value};
  try {
    const data = await api(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    log(`\n${data.text}\n`);
  } catch (err) {
    log(`${kind} failed: ${err.message}\n`);
  }
}

async function readMetadata() {
  if (!state.sourcePath) return log("Select a source checkpoint first.\n");
  try {
    const data = await api("/api/metadata/read", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path: state.sourcePath}),
    });
    log(`\n${data.text}\n`);
  } catch (err) {
    log(`Read metadata failed: ${err.message}\n`);
  }
}

async function injectMetadata() {
  if (!state.sourcePath) return log("Select a source checkpoint first.\n");
  try {
    JSON.parse($("metadata").value);
  } catch (err) {
    return log(`Metadata JSON is invalid: ${err.message}\n`);
  }

  try {
    const data = await api("/api/metadata/inject", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        path: state.sourcePath,
        metadata: $("metadata").value,
      }),
    });
    log(`\n${data.ok ? "Metadata injected" : "Metadata injection failed"}: ${data.text}\n`);
  } catch (err) {
    log(`Inject metadata failed: ${err.message}\n`);
  }
}

async function refreshSystem() {
  try {
    const s = await api("/api/system");
    const cpuPct = clampPercent(s.cpu_percent || 0);
    const ramPct = s.ram_total_gb ? clampPercent((s.ram_used_gb / s.ram_total_gb) * 100) : 0;
    const gpuPct = clampPercent(s.gpu_percent || parseFirstPercent(s.gpu));
    const vramPct = clampPercent(s.vram_percent || parseVramPercent(s.vram));

    $("cpu").textContent = `${Math.round(cpuPct)}%`;
    $("ram").textContent = `${Math.round(ramPct)}% ${s.ram_used_gb.toFixed(1)}/${s.ram_total_gb.toFixed(1)}GB`;
    $("gpu").textContent = s.gpu && s.gpu !== "Idle" ? `${Math.round(gpuPct)}% ${s.gpu}` : "Idle";
    $("vram").textContent = vramPct > 0 ? `${Math.round(vramPct)}% ${s.vram}` : s.vram;

    setBar("cpu-bar", cpuPct);
    setBar("ram-bar", ramPct);
    setBar("gpu-bar", gpuPct);
    setBar("vram-bar", vramPct);
  } catch {
    $("cpu").textContent = "--";
  }
}

function clampPercent(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, n));
}

function setBar(id, percent) {
  $(id).style.width = `${clampPercent(percent)}%`;
}

function parseFirstPercent(text) {
  const match = String(text || "").match(/(\d+(?:\.\d+)?)%/);
  return match ? clampPercent(Number(match[1])) : 0;
}

function parseVramPercent(text) {
  const match = String(text || "").match(/(\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)GB/);
  if (!match) return parseFirstPercent(text);
  const used = Number(match[1]);
  const total = Number(match[2]);
  return total > 0 ? clampPercent((used / total) * 100) : 0;
}

init().catch((err) => log(`Initialization failed: ${err.message}\n`));
