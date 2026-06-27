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
  browserItems: [],
  browserSearchQuery: "",
  browserSearchRecursive: false,
  outputDir: "",
  jobId: "",
  events: null,
  logBuffer: "",
  logFlushPending: false,
  selectedLoraPaths: new Set(),
  lastLoraDir: "",
  lastFileDir: "",
  appVersion: "",
};

const SETTINGS_COOKIE = "dasiwa_settings";
const SETTINGS_MAX_AGE_DAYS = 90;

function saveSettings() {
  const settings = {
    v: state.appVersion,
    mode: state.workflowMode,
    arch: state.architecture,
    strategy: state.strategy,
    formats: [...state.formats],
    modelName: $("model-name").value,
    fullCheckpoint: $("full-checkpoint").checked,
    lowVram: $("low-vram").checked,
    loraOutput: $("lora-output").value,
    loraGlobalStrength: $("lora-global-strength").value,
    loraMergeDevice: $("lora-merge-device").value,
    loraCudaDevice: $("lora-cuda-device").value,
    loraVramHeadroom: $("lora-vram-headroom").value,
    loraAdaptive: $("lora-adaptive").checked,
    loraDryRun: $("lora-dry-run").checked,
    loraStrict: $("lora-strict").checked,
    krea2Unchain: $("krea2-unchain").checked,
    loras: state.loras.map((l) => ({
      path: l.path,
      strength: l.strength,
      strategy: l.strategy,
      enabled: l.enabled,
    })),
  };
  try {
    const json = JSON.stringify(settings);
    // Browser cookie limit is ~4096 bytes. URL encoding expands it further.
    // Encode first so we check the real size that hits the wire.
    const encoded = encodeURIComponent(json);
    if (encoded.length > 3800) {
      // Cookie too large — drop LoRA paths (keep other settings) and retry
      const slim = {...settings, loras: settings.loras.map((l) => ({
        path: shortPath(l.path),
        strength: l.strength,
        strategy: l.strategy,
        enabled: l.enabled,
      }))};
      const slimJson = JSON.stringify(slim);
      const slimEncoded = encodeURIComponent(slimJson);
      if (slimEncoded.length > 3800) return; // Still too large, skip entirely
      const expires = new Date(Date.now() + SETTINGS_MAX_AGE_DAYS * 864e5).toUTCString();
      document.cookie = `${SETTINGS_COOKIE}=${slimEncoded};expires=${expires};path=/;SameSite=Lax`;
      return;
    }
    const expires = new Date(Date.now() + SETTINGS_MAX_AGE_DAYS * 864e5).toUTCString();
    document.cookie = `${SETTINGS_COOKIE}=${encoded};expires=${expires};path=/;SameSite=Lax`;
  } catch {
    // Cookie full or blocked — silently skip
  }
}

function loadSettings() {
  const name = `${SETTINGS_COOKIE}=`;
  const cookies = document.cookie.split(";");
  for (const c of cookies) {
    const trimmed = c.trim();
    if (!trimmed.startsWith(name)) continue;
    try {
      const json = decodeURIComponent(trimmed.substring(name.length));
      const s = JSON.parse(json);
      // Only restore if version matches (prevents stale settings after updates)
      if (s.v !== state.appVersion) return false;
      state.workflowMode = s.mode || "quantize";
      state.architecture = s.arch || "WAN 2.2";
      state.strategy = s.strategy || "Optimizer-driven";
      state.formats = new Set(s.formats || []);
      if (s.modelName) $("model-name").value = s.modelName;
      $("full-checkpoint").checked = !!s.fullCheckpoint;
      $("low-vram").checked = !!s.lowVram;
      if (s.loraOutput != null) $("lora-output").value = s.loraOutput;
      if (s.loraGlobalStrength != null) $("lora-global-strength").value = s.loraGlobalStrength;
      if (s.loraMergeDevice) $("lora-merge-device").value = s.loraMergeDevice;
      if (s.loraCudaDevice) $("lora-cuda-device").value = s.loraCudaDevice;
      if (s.loraVramHeadroom != null) $("lora-vram-headroom").value = s.loraVramHeadroom;
      $("lora-adaptive").checked = s.loraAdaptive ?? true;
      $("lora-dry-run").checked = s.loraDryRun ?? true;
      $("lora-strict").checked = s.loraStrict ?? true;
      $("krea2-unchain").checked = !!s.krea2Unchain;
      if (Array.isArray(s.loras)) {
        state.loras = s.loras.map((l) => ({
          path: l.path || "",
          strength: l.strength ?? 0.65,
          strategy: l.strategy || "Balanced",
          enabled: l.enabled ?? true,
        }));
      }
      return true;
    } catch {
      return false;
    }
  }
  return false;
}

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
  state.appVersion = cfg.version || "unknown";
  $("folder-label").textContent = shortPath(cfg.models_dir);

  const arch = $("architecture");
  cfg.architectures.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    arch.appendChild(opt);
  });

  // Try to restore settings from cookie (version-gated)
  const restored = loadSettings();

  arch.value = state.architecture;

  renderFormats(cfg.formats);
  // Re-highlight format chips that were restored
  if (restored) {
    document.querySelectorAll(".chip").forEach((btn) => {
      btn.classList.toggle("active", state.formats.has(btn.dataset.value));
    });
    // Restore strategy button highlight
    document.querySelectorAll("#strategy button").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.value === state.strategy);
    });
    // Apply workflow mode to UI
    setWorkflowMode(state.workflowMode);
  }

  wireEvents();
  updateArchDependentUI();
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
      saveSettings();
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

function updateArchDependentUI() {
  const unchainLabel = $("krea2-unchain-label");
  if (unchainLabel) {
    unchainLabel.style.display = state.architecture === "Krea 2" ? "" : "none";
  }
}

function wireEvents() {
  $("pick-folder").addEventListener("click", () => openBrowser("folder"));
  $("pick-output").addEventListener("click", () => openBrowser("output"));
  $("pick-source").addEventListener("click", () => openBrowser("file"));
  wireDropTarget($("pick-source"), "source");
  wireDropTarget($("lora-list"), "lora");
  wireDropTarget($("add-lora"), "lora");

  // Global OS file-manager drop support.
  document.addEventListener("dragover", (ev) => {
    ev.preventDefault();
    ev.dataTransfer.dropEffect = "copy";
  });
  document.addEventListener("drop", (ev) => {
    ev.preventDefault();
    const paths = pathsFromDrop(ev.dataTransfer);
    if (paths.length === 0) return;
    const isSidebar = ev.target.closest(".side") !== null;
    if (isSidebar) {
      if (paths.length > 1) log(`Dropped ${paths.length} files on sidebar; using the first as the checkpoint.\n`);
      selectSource(paths[0]);
    } else {
      addLoraPaths(paths);
    }
  });
  document.querySelectorAll("#workflow-mode button").forEach((btn) => {
    btn.addEventListener("click", () => setWorkflowMode(btn.dataset.mode));
  });
  $("add-lora").addEventListener("click", () => openBrowser("lora"));
  renderLoras();
  $("browser-close").addEventListener("click", () => {
    $("browser-search-input").value = "";
    state.browserSearchQuery = "";
    state.browserSearchRecursive = false;
    $("browser").close();
  });
  $("browser-up").addEventListener("click", () => browse($("browser").dataset.parent || state.browserPath));
  $("browser-select-folder").addEventListener("click", () => {
    if (state.browserMode === "output") selectOutput(state.browserPath);
    else selectFolder(state.browserPath);
  });
  $("browser-add-selected").addEventListener("click", () => {
    if (state.selectedLoraPaths.size > 0) {
      addLoraPaths(Array.from(state.selectedLoraPaths));
      state.selectedLoraPaths.clear();
      $("browser").close();
    }
  });

  // Search/filter in browser dialog
  $("browser-search-input").addEventListener("input", () => {
    state.browserSearchQuery = $("browser-search-input").value.trim();
    state.browserSearchRecursive = false;
    renderBrowserList();
  });
  $("browser-search-input").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      state.browserSearchQuery = $("browser-search-input").value.trim();
      if (state.browserSearchQuery) {
        state.browserSearchRecursive = true;
        searchFiles(state.browserSearchQuery, state.browserPath);
      }
    }
    if (ev.key === "Escape") {
      clearSearch();
    }
  });
  $("browser-clear-search").addEventListener("click", clearSearch);

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
    updateArchDependentUI();
    saveSettings();
  });
  $("model-name").addEventListener("input", () => { refreshMetadata(); saveSettings(); });
  $("full-checkpoint").addEventListener("change", () => { refreshMetadata(); saveSettings(); });
  $("low-vram").addEventListener("change", saveSettings);

  document.querySelectorAll("#strategy button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#strategy button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.strategy = btn.dataset.value;
      saveSettings();
    });
  });

  // LoRA control change listeners
  $("lora-output").addEventListener("input", saveSettings);
  $("lora-global-strength").addEventListener("input", saveSettings);
  $("lora-merge-device").addEventListener("change", saveSettings);
  $("lora-cuda-device").addEventListener("input", saveSettings);
  $("lora-vram-headroom").addEventListener("input", saveSettings);
  $("lora-adaptive").addEventListener("change", saveSettings);
  $("lora-dry-run").addEventListener("change", saveSettings);
  $("lora-strict").addEventListener("change", saveSettings);
  $("krea2-unchain").addEventListener("change", saveSettings);

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
  saveSettings();
}

async function openBrowser(mode) {
  state.browserMode = mode;
  $("browser-title").textContent = mode === "folder" ? "Choose Model Folder"
    : (mode === "output" ? "Choose Output Folder"
    : (mode === "lora" ? "Choose LoRA" : "Choose Checkpoint"));
  $("browser-select-folder").style.display = (mode === "folder" || mode === "output") ? "" : "none";
  const addSel = $("browser-add-selected");
  if (addSel) addSel.style.display = mode === "lora" ? "" : "none";
  state.selectedLoraPaths.clear();
  state.browserSearchQuery = "";
  state.browserSearchRecursive = false;
  $("browser-search-input").value = "";

  let startPath;
  if (mode === "lora" && state.lastLoraDir) startPath = state.lastLoraDir;
  else if (mode === "file" && state.lastFileDir) startPath = state.lastFileDir;
  else startPath = mode === "folder" ? state.modelsDir : (state.modelsDir || state.rootDir);

  await browse(startPath);
  $("browser").showModal();
}

async function browse(path) {
  try {
    const data = await api(`/api/browse?path=${encodeURIComponent(path || state.modelsDir)}`);
    state.browserPath = data.path;
    $("browser").dataset.parent = data.parent;
    $("browser-path").textContent = data.path;
    state.browserItems = data.items;
    state.browserSearchQuery = "";
    state.browserSearchRecursive = false;
    $("browser-search-input").value = "";
    renderBrowserList();
  } catch (err) {
    log(`Browse error: ${err.message}\n`);
  }
}

// Recursive search across all subdirectories (always from models root)
async function searchFiles(query, path) {
  try {
    const data = await api(`/api/search?path=${encodeURIComponent(state.modelsDir)}&q=${encodeURIComponent(query)}`);
    state.browserPath = data.path;
    $("browser-path").textContent = `Search "${query}" in ${data.path}`;
    // Convert search results to browser-item format with relative paths
    const root = state.modelsDir;
    state.browserItems = data.items.map((item) => ({
      name: item.path.startsWith(root + "/")
        ? item.path.slice(root.length + 1)
        : item.name,
      path: item.path,
      is_dir: false,
    }));
    state.browserSearchRecursive = true;
    renderBrowserList();
  } catch (err) {
    log(`Search error: ${err.message}\n`);
  }
}

// Render the browser list from state.browserItems, optionally filtered by search query
function renderBrowserList() {
  const list = $("browser-list");
  list.textContent = "";
  const query = state.browserSearchQuery.toLowerCase();

  let items = state.browserItems;
  if (query && !state.browserSearchRecursive) {
    // Client-side filter of current directory
    items = items.filter((item) => item.name.toLowerCase().includes(query));
  }

  if (!items.length && query) {
    const empty = document.createElement("p");
    empty.className = "muted-note";
    empty.textContent = state.browserSearchRecursive
      ? `No files matching "${state.browserSearchQuery}" found.`
      : `No items matching "${state.browserSearchQuery}" in this folder.`;
    list.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "browser-item";
    btn.draggable = !item.is_dir;
    btn.addEventListener("dragstart", (ev) => {
      ev.dataTransfer.effectAllowed = "copy";
      ev.dataTransfer.setData("text/plain", item.path);
      ev.dataTransfer.setData("text/uri-list", `file://${item.path}`);
    });

    if (state.browserMode === "lora" && !item.is_dir) {
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = state.selectedLoraPaths.has(item.path);
      cb.addEventListener("click", (ev) => ev.stopPropagation());
      cb.addEventListener("change", () => {
        if (cb.checked) state.selectedLoraPaths.add(item.path);
        else state.selectedLoraPaths.delete(item.path);
      });
      btn.appendChild(cb);
    }

    const nameSpan = document.createElement("span");
    nameSpan.textContent = item.name;
    btn.appendChild(nameSpan);

    const kindSpan = document.createElement("span");
    kindSpan.className = "kind";
    kindSpan.textContent = item.is_dir ? "folder" : "file";
    btn.appendChild(kindSpan);

    btn.addEventListener("click", () => {
      if (item.is_dir && !state.browserSearchRecursive) {
        browse(item.path);
      } else if (state.browserMode === "file") {
        selectSource(item.path);
      } else if (state.browserMode === "lora") {
        selectLora(item.path);
      }
    });
    list.appendChild(btn);
  });
}

function clearSearch() {
  state.browserSearchQuery = "";
  state.browserSearchRecursive = false;
  $("browser-search-input").value = "";
  renderBrowserList();
}

function selectFolder(path) {
  state.modelsDir = path;
  $("folder-label").textContent = shortPath(path);
  $("browser").close();
}

function selectOutput(path) {
  state.outputDir = path;
  $("output-label").textContent = shortPath(path);
  $("browser").close();
}

function wireDropTarget(el, kind) {
  if (!el) return;
  ["dragenter", "dragover"].forEach((eventName) => {
    el.addEventListener(eventName, (ev) => {
      ev.preventDefault();
      ev.dataTransfer.dropEffect = "copy";
      el.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    el.addEventListener(eventName, () => el.classList.remove("drag-over"));
  });
  el.addEventListener("drop", (ev) => {
    ev.preventDefault();
    const paths = pathsFromDrop(ev.dataTransfer);
    if (paths.length === 0) {
      log("Drop did not expose file paths. Drag from the in-app browser, or use the picker if your browser hides local paths.\n");
      return;
    }
    if (kind === "source") {
      if (paths.length > 1) log(`Dropped ${paths.length} files; using the first as the checkpoint.\n`);
      selectSource(paths[0]);
    } else {
      addLoraPaths(paths);
    }
  });
}

function pathsFromDrop(dataTransfer) {
  const values = [];
  const addTextPaths = (text) => {
    String(text || "").split(/\r?\n/).forEach((line) => {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) return;
      values.push(trimmed);
    });
  };
  addTextPaths(dataTransfer.getData("text/uri-list"));
  addTextPaths(dataTransfer.getData("text/plain"));

  // Some Chromium/Electron contexts expose File.path for local drops. Normal
  // browsers usually hide full local paths, so this is best-effort only.
  Array.from(dataTransfer.files || []).forEach((file) => {
    if (file.path) values.push(file.path);
    else if (file.mozFullPath) values.push(file.mozFullPath);
    else if (file.webkitRelativePath) values.push(`${state.modelsDir}/${file.webkitRelativePath}`);
  });

  const seen = new Set();
  return values
    .map(normalizeDroppedPath)
    .filter(Boolean)
    .filter((path) => {
      if (seen.has(path)) return false;
      seen.add(path);
      return true;
    });
}

function normalizeDroppedPath(value) {
  let path = String(value || "").trim();
  if (!path) return "";
  if (path.startsWith("file://")) {
    try {
      path = decodeURIComponent(new URL(path).pathname);
    } catch {
      path = decodeURIComponent(path.replace(/^file:\/\//, ""));
    }
  }
  return path.startsWith("/") ? path : "";
}

function selectLora(path) {
  const dir = path.substring(0, path.lastIndexOf("/"));
  if (dir) state.lastLoraDir = dir;
  if (state.pendingLoraSlot >= 0 && state.loras[state.pendingLoraSlot]) {
    state.loras[state.pendingLoraSlot].path = path;
  } else {
    state.loras.push({path, strength: 0.65, strategy: "Balanced", enabled: true});
  }
  state.pendingLoraSlot = -1;
  renderLoras();
  saveSettings();
  $("browser").close();
}

function addLoraPaths(paths) {
  const existing = new Set(state.loras.map((lora) => lora.path).filter(Boolean));
  const added = [];
  paths.forEach((path) => {
    if (!path || existing.has(path)) return;
    state.loras.push({path, strength: 0.65, strategy: "Balanced", enabled: true});
    existing.add(path);
    added.push(path);
  });
  renderLoras();
  if (added.length > 0) log(`Added ${added.length} LoRA file(s) from drop.\n`);
}

function renderLoras() {
  const root = $("lora-list");
  root.textContent = "";
  if (state.loras.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted-note";
    empty.textContent = "No LoRAs selected. Drag files from the in-app browser, or click + Add LoRA.";
    root.appendChild(empty);
    return;
  }
  // WAN 2.2 has no audio components, so Audio strategy is not useful.
  // Krea 2 is an image model with its own strategy set.
  const allStrategies = ["Balanced", "Motion", "Visuals", "Audio"];
  const imageStrategies = ["Balanced", "Style", "Content", "Detail"];
  const strategies = state.architecture === "Krea 2"
    ? imageStrategies
    : state.architecture === "WAN 2.2"
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
    row.querySelector('[data-role="enabled"]').addEventListener("change", (ev) => { lora.enabled = ev.target.checked; saveSettings(); });
    row.querySelector('[data-role="strategy"]').addEventListener("change", (ev) => { lora.strategy = ev.target.value; saveSettings(); });
    row.querySelector('[data-role="strength"]').addEventListener("input", (ev) => { lora.strength = Number(ev.target.value) || 0; saveSettings(); });
    row.querySelector('[data-role="remove"]').addEventListener("click", () => {
      state.loras.splice(idx, 1);
      renderLoras();
      saveSettings();
    });
    root.appendChild(row);
  });
}

async function selectSource(path) {
  const dir = path.substring(0, path.lastIndexOf("/"));
  if (dir) state.lastFileDir = dir;
  state.sourcePath = path;
  $("source-label").textContent = shortPath(path);
  $("browser").close();
  if (!$("model-name").value) {
    $("model-name").value = path.split("/").pop().replace(/\.(safetensors|gguf|ckpt|pt|bin)$/i, "");
  }
  if (path.endsWith(".safetensors")) {
    try {
      const info = await api(`/api/inspect?path=${encodeURIComponent(path)}`);
      if (["WAN 2.2", "LTX-2.3", "Krea 2"].includes(info.architecture)) {
        state.architecture = info.architecture;
        $("architecture").value = info.architecture;
        updateArchDependentUI();
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
        merge_device: $("lora-merge-device").value,
        cuda_device: $("lora-cuda-device").value || "cuda:0",
        vram_headroom_mb: Number($("lora-vram-headroom").value) || 1024,
        adaptive: $("lora-adaptive").checked,
        dry_run: dryRun,
        strict_matching: $("lora-strict").checked,
        krea2_unchain: $("krea2-unchain").checked,
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
