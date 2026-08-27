const state = {
  rootDir: "",
  modelsDir: "",
  sourcePath: "",
  architecture: "WAN 2.2",
  allFormats: [],
  formatSupport: {},
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
  jobId: "",
  events: null,
  logBuffer: "",
  logFlushPending: false,
  selectedLoraPaths: new Set(),
  lastLoraDir: "",
  lastFileDir: "",
  lastOverlayDir: "",
  appVersion: "",
  mmOverlayPath: "",
  mmRecipe: "h3_hybrid",
  mmRank: 1024,
  mmStrength: 1,
};

const JOB_PERSIST_KEY = "dasiwa_active_job";
let consoleLogPersisted = "";
const LOG_PERSIST_KEY = "dasiwa_console_log";

const SETTINGS_COOKIE = "dasiwa_settings";
const SETTINGS_MAX_AGE_DAYS = 90;
const PING_INTERVAL_MS = 30000;
const MAX_EFFECTIVE_LORA_STRENGTH = 3;

let pingTimer = null;

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
    loraGlobalStrength: $("lora-global-strength").value,
    loraMergeDevice: $("lora-merge-device").value,
    loraCudaDevice: $("lora-cuda-device").value,
    loraVramHeadroom: $("lora-vram-headroom").value,
    loraAdaptive: $("lora-adaptive").checked,
    loraStrict: $("lora-strict").checked,
    krea2Unchain: $("krea2-unchain").checked,
    watermark: $("watermark").checked,
    mmOverlayPath: state.mmOverlayPath,
    mmRecipe: $("mm-recipe").value,
    mmRank: $("mm-rank").value,
    mmStrength: $("mm-strength").value,
    mmDryRun: $("mm-dry-run").checked,
    lastFileDir: state.lastFileDir,
    lastLoraDir: state.lastLoraDir,
    lastOverlayDir: state.lastOverlayDir,
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
      state.lastFileDir = s.lastFileDir || "";
      state.lastLoraDir = s.lastLoraDir || "";
      state.lastOverlayDir = s.lastOverlayDir || "";
      // Only restore if version matches (prevents stale settings after updates)
      if (s.v !== state.appVersion) return false;
      state.workflowMode = s.mode || "quantize";
      state.architecture = s.arch || "WAN 2.2";
      state.strategy = s.strategy || "Optimizer-driven";
      state.formats = new Set(s.formats || []);
      if (s.modelName) $("model-name").value = s.modelName;
      $("full-checkpoint").checked = !!s.fullCheckpoint;
      $("low-vram").checked = !!s.lowVram;
      if (s.loraGlobalStrength != null) $("lora-global-strength").value = s.loraGlobalStrength;
      if (s.loraMergeDevice) $("lora-merge-device").value = s.loraMergeDevice;
      if (s.loraCudaDevice) $("lora-cuda-device").value = s.loraCudaDevice;
      if (s.loraVramHeadroom != null) $("lora-vram-headroom").value = s.loraVramHeadroom;
      $("lora-adaptive").checked = s.loraAdaptive ?? false;
      $("lora-strict").checked = s.loraStrict ?? true;
      $("krea2-unchain").checked = !!s.krea2Unchain;
      $("watermark").checked = s.watermark ?? true;
      // Migrate old loraDryRun → mmDryRun (dry-run now lives in the Strategy panel,
      // shared between LoRA and Model Merge modes).
      state.mmOverlayPath = s.mmOverlayPath || "";
      $("mm-overlay-label").textContent = state.mmOverlayPath ? shortPath(state.mmOverlayPath) : "Pick…";
      if (s.mmRecipe) $("mm-recipe").value = s.mmRecipe;
      if (s.mmRank != null) $("mm-rank").value = s.mmRank;
      if (s.mmStrength != null) $("mm-strength").value = s.mmStrength;
      updateDeltaOptionsVisibility();
      $("mm-dry-run").checked = s.mmDryRun ?? (s.loraDryRun ?? true);
      if (Array.isArray(s.loras)) {
        state.loras = s.loras.map((l) => ({
          path: l.path || "",
          strength: l.strength ?? 0.65,
          strategy: l.strategy || defaultLoraStrategy(s.arch || "WAN 2.2"),
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

// --- Job persistence helpers ---------------------------------------------------
function saveActiveJob() {
  try { localStorage.setItem(JOB_PERSIST_KEY, state.jobId); } catch {}
}
function loadPersistedJob() {
  try { return localStorage.getItem(JOB_PERSIST_KEY); } catch { return null; }
}
function clearPersistedJob(id) {
  if (id && id === localStorage.getItem(JOB_PERSIST_KEY)) {
    try { localStorage.removeItem(JOB_PERSIST_KEY); } catch {}
  }
}

// --- Console log persistence ---------------------------------------------------
function persistConsoleLog() {
  const el = $("console");
  if (!el) return;
  const text = el.textContent || "";
  // Truncate to last 30k chars so localStorage stays under the ~5MB budget.
  const truncated = text.length > 30000 ? text.slice(-30000) : text;
  try { localStorage.setItem(LOG_PERSIST_KEY, truncated); } catch {}
}

async function restoreConsoleLog() {
  let saved = "";
  try { saved = localStorage.getItem(LOG_PERSIST_KEY) || ""; } catch {}
  if (saved) {
    const el = $("console");
    if (el) el.textContent = saved;
  }
}

// --- Job status polling --------------------------------------------------------
const POLL_INTERVAL_MS = 60_000; // every minute in background
let pollTimerId = null;         // monotonically increasing counter for clearTimeout
function startPolling() {
  stopPolling();
  if (state.jobId) doJobStatusCheck(state.jobId);
}

async function doJobStatusCheck(jobIdToCheck) {
  try {
    const data = await api(`/api/jobs/${jobIdToCheck}/summary`);
    // Update UI only when the job is still running — don't overwrite a "finished" state.
    if (data.status && !["finished", "stopped", "failed"].includes(data.status)) {
      setStatus(data.status || data.summary_status);
    } else if (data.id) {
      // Job completed while we were polling → clean up persisted job + events
      clearPersistedJob(state.jobId);
      if (state.events && state.events.readyState !== EventSource.CLOSED) {
        state.events.close();
      }
      state.jobId = "";
    }
  } catch {} // silently fail — user sees stale status but app stays responsive.
}

function stopPolling() {
  const idToClear = pollTimerId;
  if (idToClear !== null) clearTimeout(idToClear);
  pollTimerId = null;
}
// --- /Job status polling -------------------------------------------------------

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
  persistConsoleLog(); // save to localStorage after every flush
  state.logFlushPending = false;
}

function setStatus(text) {
  const el = $("status");
  if (el) el.textContent = text;
  startPolling(); // keep polling once a job is running.
}

// Progress ticks (type === "progress") never touch the scrollback <pre> or
// the topbar status line: they overwrite this single line in place, so the
// rest of the console window stays intact while a job streams progress.
function showProgressLine(text) {
  const el = $("console-progress");
  if (!el) return;
  el.textContent = text || "";
  el.hidden = false;
}

function clearProgressLine() {
  const el = $("console-progress");
  if (!el) return;
  el.hidden = true;
  el.textContent = "";
}

function shortPath(path) {
  if (!path) return "";
  const parts = path.split("/");
  return parts.slice(-2).join("/");
}

// Format strength with English decimal separator (always ".") and up to 2 decimals
function formatStrength(v) {
  return Number(v || 0).toFixed(2);
}

async function init() {
  const cfg = await api("/api/config");
  state.rootDir = cfg.root_dir;
  state.modelsDir = cfg.models_dir;
  state.browserPath = cfg.models_dir;
  state.appVersion = cfg.version || "unknown";

  const arch = $("architecture");
  state.formatSupport = cfg.format_support || {};
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
  refreshWatermarkStatus();
  refreshModelMergeHint();
  setInterval(refreshSystem, 5000);

  // Restore persisted console log so progress isn't lost across reloads.
  restoreConsoleLog().then(() => {
    // After logs are restored, check if there was a job running and try to reconnect.
    const savedJobId = loadPersistedJob();
    if (savedJobId) {
      state.jobId = savedJobId;
      attachEvents(savedJobId);
    }
  });

  // Background ping updates the green status dot next to the headline.
  // Uses self-scheduling setTimeout instead of setInterval so visibility
  // changes can force an immediate ping without creating duplicate loops.
  schedulePing(0);

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      schedulePing(0);
      tryReconnectEvents();
    }
  });
}

async function pingServer() {
  try {
    await fetch("/api/ping", {cache: "no-store"});
    $("status-dot").classList.add("online");
  } catch {
    $("status-dot").classList.remove("online");
  } finally {
    schedulePing();
  }
}

function schedulePing(delay = PING_INTERVAL_MS) {
  if (pingTimer !== null) clearTimeout(pingTimer);
  pingTimer = setTimeout(pingServer, delay);
}

function renderFormats(formats) {
  state.allFormats = formats;
  renderFormatsForArch(state.allFormats, state.architecture);
}

// Re-render the format chips, showing only formats the selected
// architecture supports (per the server's format_support map). Selected
// formats that are no longer supported are dropped from state with a log
// line. "Not set" / unknown architectures show every format.
function renderFormatsForArch(formats, architecture) {
  const arch = architecture && architecture !== "Not set" ? architecture : null;
  const supported = (fmt) => {
    if (!arch) return true;
    const list = state.formatSupport[fmt.value];
    return !list || list.includes(arch);
  };
  const root = $("formats");
  root.textContent = "";
  const groups = [
    {title: "Safetensors", items: formats.filter(supported).filter((fmt) => !fmt.value.startsWith("GGUF_"))},
    {title: "GGUF", items: formats.filter(supported).filter((fmt) => fmt.value.startsWith("GGUF_"))},
  ];
  const visible = new Set();
  groups.forEach((group) => {
    if (group.items.length === 0) return;
    const section = document.createElement("section");
    section.className = "format-group";
    const title = document.createElement("h4");
    title.textContent = group.title;
    const list = document.createElement("div");
    list.className = "format-list";
    group.items.forEach((fmt) => {
      visible.add(fmt.value);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip";
      btn.textContent = compactFormatLabel(fmt.label);
      btn.dataset.value = fmt.value;
      const tip = formatTitle(fmt.value);
      if (tip) btn.title = tip;
      btn.classList.toggle("active", state.formats.has(fmt.value));
      btn.addEventListener("click", () => {
        if (state.formats.has(fmt.value)) {
          state.formats.delete(fmt.value);
          btn.classList.remove("active");
        } else {
          state.formats.add(fmt.value);
          btn.classList.add("active");
        }
        enforceInt4ConvRotStrategy();
        saveSettings();
      });
      list.appendChild(btn);
    });
    section.appendChild(title);
    section.appendChild(list);
    root.appendChild(section);
  });
  // Drop selected formats that are no longer offered for this architecture.
  let dropped = 0;
  state.formats.forEach((fmt) => {
    if (!visible.has(fmt)) {
      state.formats.delete(fmt);
      dropped += 1;
    }
  });
  if (dropped) {
    log(`Architecture "${arch}" dropped ${dropped} unsupported format selection(s).\n`);
    saveSettings();
  }
  return visible;
}

function applyArchFormatFilter() {
  if (!state.allFormats || !state.allFormats.length) return;
  renderFormatsForArch(state.allFormats, state.architecture);
}

function compactFormatLabel(label) {
  return label
    .replace("Safetensors: ", "")
    .replace("GGUF ", "")
    .replace("NVFP4 HQ (H3 mixed profile)", "NVFP4 HQ")
    .replace("INT8 Row-wise ConvRot (runtime)", "INT8 ConvRot")
    .replace("INT4 ConvRot (LTX/WAN/Krea/H3 experimental)", "INT4 ConvRot")
    .replace("W4A8 asymmetric ConvRot (MiniMax H3)", "W4A8 ConvRot")
    .replace("INT8 Tensor-wise", "INT8 Tensor")
    .replace("GGUF ", "");
}

const FORMAT_TITLES = {
  FP8:              "FP8 — Balanced mixed-precision quantization. Best quality/size trade-off for most use cases.",
  NVFP4:            "NVFP4 (E2M1) — Low-bit tensor format for Blackwell GPUs (RTX 5090). Smallest size, good quality on supported hardware.",
  "NVFP4 HQ":       "NVFP4 HQ — MiniMax H3 mixed profile. Same dedicated NVFP4 path, but 30 per-block heavy linears (27 out_proj + 3 fc2, community-verified HQ plan) are kept at source precision; all other heavy linears are NVFP4-packed.",
  MXFP8:            "MXFP8 (Block FP8) — Block-norm quantized FP8. Good balance of compression and fidelity, works across GPU generations.",
  "Hybrid MXFP8":   "Hybrid MXFP8 — Combines block-norm and per-channel scaling for improved accuracy over pure MXFP8.",
  "INT8 Tensor-wise": "INT8 (Tensor-wise) — Per-tensor scale quantization to INT8. Fastest inference, slightly lower quality than FP8.",
  "INT8 Row-wise ConvRot Runtime": "INT8 ConvRot (Row-wise) — Per-row scaling with Hadamard rotation at runtime. Best INT8 fidelity; requires ConvRot support in the loader (e.g., ComfyUI).",
  "INT4 ConvRot Runtime": "INT4 ConvRot — Experimental LTX-2.3, WAN 2.2 MoE High/Low, Krea 2, and MiniMax H3 ComfyUI format. Requires a current ComfyUI ConvRot runtime and an Ampere-or-newer NVIDIA GPU. Uses ConvRot group 256 and INT4 group 64.",
  "W4A8": "W4A8 asymmetric ConvRot — MiniMax H3 low-bit format (asym_w4a8_int8): packed 4-bit INT8 weights with a 16-value codebook, FP8 group scales, and ConvRot group 256. Only the heavy linears (qkv/out/fc1/fc2) are packed; structural layers stay at source precision. Output matches the reference MiniMax w4a8 quants and loads in ComfyUI with a ConvRot-capable runtime on Ampere-or-newer GPUs.",
  "GGUF_F32":       "GGUF F32 — Full FP32 precision stored in GGUF format. Largest file size, maximum compatibility.",
  "GGUF_BF16":      "GGUF BF16 — Bfloat16 quantization (one bit less than FP16). Negligible quality loss vs full FP16; good for modern GPUs.",
  "GGUF_F16":       "GGUF F16 — Float16. Good compatibility across GGUF-based runners like llama.cpp, Ollama, etc.",
  "GGUF_Q8_0":      "GGUF Q8_0 — High-quality quantization (~7 bits). Very close to original quality; recommended for most GGUF use cases.",
  "GGUF_Q6_K":      "GGUF Q6_K — Good compression with minimal quality loss. Decent balance of size and fidelity.",
  "GGUF_Q5_K":      "GGUF Q5_K — Medium quantization (~4 bits). Noticeable quality drop but much smaller files; acceptable for many models.",
  "GGUF_Q4_K":      "GGUF Q4_K — Popular compression point. Good balance of size and quality for GGUF runners on consumer GPUs.",
  "GGUF_Q3_K":      "GGUF Q3_K — Aggressive quantization (~3 bits). Significant quality loss; use only when file size is critical.",
  "GGUF_Q2_K":      "GGUF Q2_K — Extreme compression. Only for very constrained environments (e.g., mobile, edge devices).",
};

function formatTitle(value) {
  return FORMAT_TITLES[value] || null;
}

function enforceInt4ConvRotStrategy() {
  const simpleOnly =
    (state.formats.has("INT4 ConvRot Runtime") || state.formats.has("W4A8"))
      && state.strategy !== "Simple";
  if (!simpleOnly) return;
  state.strategy = "Simple";
  document.querySelectorAll("#strategy button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.value === "Simple");
  });
  log("INT4 ConvRot / W4A8 requires Simple strategy; switched from Optimizer.\n");
}

function updateArchDependentUI() {
  applyArchFormatFilter();
  const unchainLabel = $("krea2-unchain-label");
  if (unchainLabel) {
    unchainLabel.style.display = state.architecture === "Krea 2" ? "" : "none";
  }
  if (state.workflowMode === "model") refreshModelMergeHint();
}

// --- Model Merge (model-level, not LoRA) ------------------------------------

function selectModelMergeOverlay(path) {
  state.mmOverlayPath = path;
  const dir = path.substring(0, path.lastIndexOf("/"));
  if (dir) state.lastOverlayDir = dir;
  $("mm-overlay-label").textContent = shortPath(path);
  $("browser").close();
  saveSettings();
  refreshModelMergeHint();
}

function refreshModelMergeHint() {
  const hint = $("mm-hint");
  if (!hint) return;
  const base = state.sourcePath;
  const overlay = state.mmOverlayPath;
  const recipe = $("mm-recipe").value;
  let text, warn = false;
  if (state.architecture !== "MiniMax H3") {
    warn = true;
    text = "MiniMax H3 model merge requires two MiniMax H3 checkpoints. " +
      "Select MiniMax H3 as the architecture (or pick an H3 checkpoint in the Source panel).";
  } else if (!base || !overlay) {
    warn = true;
    text = !base
      ? "Pick a base (fl2va) checkpoint in the Source panel to start."
      : "Pick the second (ref2va) checkpoint to start.";
  } else if (base === overlay) {
    warn = true;
    text = "Base (Source panel) and second checkpoint must be two different files.";
  } else if (recipe === "h3_delta") {
    const rank = parseInt($("mm-rank").value, 10) || 0;
    text = "Delta recipe: fuses the full ref2va − fl2va weight delta into the base " +
      (rank === 0
        ? "(exact mode — output ≈ ref2va-equivalent, largest file)."
        : `(SVD rank ${rank} — trunk deltas compressed, norms/biases/timestep exact).`) +
      " Works on pruned and full key sets (auto-detected).";
  } else {
    text = "Base = Source-panel checkpoint, overlay = ref2va. Role auto-detection by filename; " +
      "only blocks.{25..49} adaln_proj tensors come from the overlay.";
  }
  hint.textContent = text;
  hint.classList.toggle("warn", warn);
}

function updateDeltaOptionsVisibility() {
  const wrap = $("mm-delta-options");
  if (!wrap) return;
  wrap.style.display = $("mm-recipe").value === "h3_delta" ? "" : "none";
}

function updateModelMergeVisibility() {
  refreshModelMergeHint();
}

async function startModelMerge() {
  const base = state.sourcePath;
  const overlay = state.mmOverlayPath;
  if (!base) return log("Pick a base (fl2va) checkpoint in the Source panel first.\n");
  if (!overlay) return log("Pick an overlay (ref2va) checkpoint in the Model Merge sidebar first.\n");
  const dryRun = $("mm-dry-run").checked;
  if (state.architecture !== "MiniMax H3") {
    return log("MiniMax H3 model merge needs MiniMax H3 as the architecture. Pick an H3 checkpoint or set the architecture to MiniMax H3.\n");
  }
  const outputName = $("model-name").value.trim();
  if (!dryRun && !outputName) return log("Enter a Display Name before writing a merged checkpoint.\n");
  const recipe = $("mm-recipe").value;
  let rank = 0;
  let strength = 1.0;
  if (recipe === "h3_delta") {
    const r = parseInt($("mm-rank").value, 10);
    rank = Number.isFinite(r) && r >= 0 ? r : 0;
    const s = parseFloat($("mm-strength").value);
    strength = Number.isFinite(s) && s > 0 ? s : 1.0;
  }

  $("start").disabled = true;
  $("stop").disabled = false;
  setStatus(dryRun ? "Starting model merge dry run" : "Starting model merge");
  log(`\nStarting ${dryRun ? "model merge dry run" : "model merge"} (recipe ${recipe}${recipe === "h3_delta" ? `, rank ${rank}, strength ${strength}` : ""}, output "${outputName || "(dry run)"}")...\n`);

  try {
    const data = await api("/api/model-merge", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        base_path: base,
        overlay_path: overlay,
        models_dir: state.modelsDir,
        output_name: outputName,
        architecture: state.architecture,
        recipe,
        rank,
        strength,
        dry_run: dryRun,
        watermark: $("watermark").checked,
      }),
    });
    state.jobId = data.job_id;
    attachEvents(data.job_id);
  } catch (err) {
    log(`Model merge error: ${err.message}\n`);
    $("start").disabled = false;
    $("stop").disabled = true;
    setStatus("Error");
  }
}

function wireEvents() {
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
  $("load-recipe-btn").addEventListener("click", loadRecipe);

  renderLoras();
  $("browser-close").addEventListener("click", () => {
    $("browser-search-input").value = "";
    state.browserSearchQuery = "";
    state.browserSearchRecursive = false;
    $("browser").close();
  });
  $("browser-up").addEventListener("click", () => browse($("browser").dataset.parent || state.browserPath));

  $("browser-add-all").addEventListener("click", () => {
    const paths = state.browserItems.filter((item) => !item.is_dir).map((item) => item.path);
    for (const p of paths) {
      state.selectedLoraPaths.add(p);
    }
    renderBrowserList();
  });

  $("browser-add-selected").addEventListener("click", () => {
    if (state.selectedLoraPaths.size > 0) {
      addLoraPaths(Array.from(state.selectedLoraPaths));
      state.selectedLoraPaths.clear();
      $("browser").close();
    } else {
      log("No LoRA files selected.\n");
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
    const currentJobId = state.jobId;
    state.logBuffer = "";
    $("console").textContent = "";
    clearProgressLine();
    try { localStorage.removeItem(LOG_PERSIST_KEY); } catch {}
    if (currentJobId) clearPersistedJob(currentJobId);
  });

  // Copy the console window (scrollback + live progress line, if shown)
  // to the clipboard; flashes the "Copied" state briefly.
  let copyResetTimer = null;
  async function copyConsole() {
    const pre = $("console");
    let text = pre ? (pre.textContent || "") : "";
    const prog = $("console-progress");
    if (prog && !prog.hidden && prog.textContent) {
      text = (text ? text + "\n" : "") + prog.textContent;
    }
    let ok = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        ok = true;
      }
    } catch {}
    if (!ok && text) {
      // Fallback for contexts without the async Clipboard API.
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try { ok = document.execCommand("copy"); } catch {}
      ta.remove();
    }
    const btn = $("copy-console");
    if (!btn) return;
    btn.classList.add("copied");
    btn.querySelector(".copy-label").textContent = ok ? "Copied" : "Copy failed";
    if (copyResetTimer) clearTimeout(copyResetTimer);
    copyResetTimer = setTimeout(() => {
      btn.classList.remove("copied");
      btn.querySelector(".copy-label").textContent = "";
    }, 1600);
  }
  $("copy-console").addEventListener("click", copyConsole);
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
  $("watermark").addEventListener("change", () => { refreshWatermarkStatus(); saveSettings(); });

  document.querySelectorAll("#strategy button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#strategy button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.strategy = btn.dataset.value;
      saveSettings();
    });
  });

  // LoRA control change listeners
  $("lora-global-strength").addEventListener("input", saveSettings);
  $("lora-merge-device").addEventListener("change", saveSettings);
  $("lora-cuda-device").addEventListener("input", saveSettings);
  $("lora-vram-headroom").addEventListener("input", saveSettings);
  $("lora-adaptive").addEventListener("change", saveSettings);
  $("lora-strict").addEventListener("change", saveSettings);
  $("krea2-unchain").addEventListener("change", saveSettings);

  // Model Merge (model-level) — base reuses the Source panel; only overlay is picked here
  $("mm-pick-overlay").addEventListener("click", () => openBrowser("mm-overlay"));
  wireDropTarget($("mm-pick-overlay"), "mm-overlay");
  $("mm-recipe").addEventListener("change", () => {
    updateDeltaOptionsVisibility();
    refreshModelMergeHint();
    saveSettings();
  });
  $("mm-rank").addEventListener("change", saveSettings);
  $("mm-strength").addEventListener("change", saveSettings);
  $("mm-dry-run").addEventListener("change", saveSettings);

  $("start").addEventListener("click", () => {
    if (state.workflowMode === "lora") {
      startLoraMerge();
    } else if (state.workflowMode === "model") {
      startModelMerge();
    } else {
      startJob();
    }
  });
  $("stop").addEventListener("click", stopJob);
  $("clean-memory").addEventListener("click", cleanMemory);
  $("update-app").addEventListener("click", updateApp);
  $("quit-btn").addEventListener("click", shutdownServer);
  $("scan").addEventListener("click", () => runTool("scan"));
  $("audit").addEventListener("click", () => runTool("audit"));
}

async function shutdownServer() {
  try {
    await fetch("/api/shutdown", {method: "POST"});
  } catch {} finally {
    // Try to close the tab — works for tabs opened by scripts, and leaves a hint otherwise.
    window.close();
    document.body.innerHTML = "<div style='display:flex;align-items:center;justify-content:center;height:100vh;background:#101214;color:#edf2f7;font-family:sans-serif'>Server shut down.</div>";
  }
}

function setWorkflowMode(mode) {
  state.workflowMode = mode;
  document.querySelectorAll("#workflow-mode button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  document.querySelectorAll("[data-workflow-panel]").forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.workflowPanel !== mode);
  });
  // Show the sidebar Model Merge section only in model mode.
  $("mm-side-panel").classList.toggle("hidden", mode !== "model");
  updateDeltaOptionsVisibility();
  // Show the dry-run checkbox (under Strategy) for LoRA merge + Model Merge modes only
  // (the two merge workflows that support a plan-only, no-write run).
  $("mm-dry-run-wrap").style.display = mode === "model" || mode === "lora" ? "" : "none";
  // Show start button in all modes; label changes to match context.
  const startBtn = $("start");
  startBtn.classList.remove("hidden");
  startBtn.textContent = mode === "lora" ? "Start Merge" : mode === "model" ? "Start Model Merge" : "Start Batch";
  // Update checkpoint hint label per mode.
  const hint = $("source-label-hint");
  if (hint) {
    hint.textContent = mode === "lora" ? "Base checkpoint" : mode === "model" ? "Checkpoint (base)" : "Checkpoint";
  }
  // Show Model Merge hint only when arch is MiniMax H3.
  updateModelMergeVisibility();
  setStatus(mode === "quantize" ? "Quantize mode" : mode === "model" ? "Model merge mode" : "LoRA merge mode");
  saveSettings();
}

async function openBrowser(mode) {
  state.browserMode = mode;
  const titleMap = { lora: "Choose LoRA", "mm-overlay": "Pick overlay checkpoint (ref2va)" };
  $("browser-title").textContent = titleMap[mode] || "Choose Checkpoint";
  const addSel = $("browser-add-selected");
  const addAllBtn = $("browser-add-all");
  if (addAllBtn) addAllBtn.style.display = mode === "lora" ? "" : "none";
  if (addSel) addSel.style.display = mode === "lora" ? "" : "none";
  state.selectedLoraPaths.clear();
  state.browserSearchQuery = "";
  state.browserSearchRecursive = false;
  $("browser-search-input").value = "";

  let startPath;
  if (mode === "lora" && state.lastLoraDir) startPath = state.lastLoraDir;
  else if (mode === "file" && state.lastFileDir) startPath = state.lastFileDir;
  else if (mode === "mm-overlay" && state.lastOverlayDir) startPath = state.lastOverlayDir;
  else startPath = state.modelsDir || state.rootDir;

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
    if (state.browserMode === "lora" && !item.is_dir && state.selectedLoraPaths.has(item.path)) {
      btn.classList.add("selected");
    }
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
      } else if (state.browserMode === "mm-overlay" && !item.is_dir) {
        selectModelMergeOverlay(item.path);
      } else if (state.browserMode === "lora" && !item.is_dir) {
        // Toggle selection instead of immediately adding
        const cb = btn.querySelector('input[type="checkbox"]');
        if (cb) {
          cb.checked = !cb.checked;
          if (cb.checked) {
            state.selectedLoraPaths.add(item.path);
            btn.classList.add("selected");
          } else {
            state.selectedLoraPaths.delete(item.path);
            btn.classList.remove("selected");
          }
        }
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
    } else if (kind === "mm-overlay") {
      if (paths.length > 1) log(`Dropped ${paths.length} files; using the first as the overlay checkpoint.\n`);
      selectModelMergeOverlay(paths[0]);
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

function defaultLoraStrategy(architecture = state.architecture) {
  return architecture === "LTX-2.3" ? "All" : "Balanced";
}

function selectLora(path) {
  const dir = path.substring(0, path.lastIndexOf("/"));
  if (dir) state.lastLoraDir = dir;
  if (state.pendingLoraSlot >= 0 && state.loras[state.pendingLoraSlot]) {
    state.loras[state.pendingLoraSlot].path = path;
  } else {
    state.loras.push({path, strength: 0.65, strategy: defaultLoraStrategy(), enabled: true});
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
    state.loras.push({path, strength: 0.65, strategy: defaultLoraStrategy(), enabled: true});
    const dir = path.substring(0, path.lastIndexOf("/"));
    if (dir) state.lastLoraDir = dir;
    existing.add(path);
    added.push(path);
  });
  renderLoras();
  if (added.length > 0) log(`Added ${added.length} LoRA file(s) from drop.\n`);
  saveSettings();
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
  // LTX mirrors its ComfyUI loader: all weights, video branch, or audio branch.
  const ltxStrategies = ["All", "Video", "Audio"];
  const wan22Strategies = ["Balanced", "Motion", "Visuals"];
  const krea2Strategies = ["Balanced", "Style", "Content", "Detail"];
  // H3 is an omni-modal joint video+audio DiT; offer video+audio emphasis presets.
  const h3Strategies = ["Balanced", "Motion", "Visuals", "Audio"];
  const strategies = state.architecture === "Krea 2"
    ? krea2Strategies
    : state.architecture === "WAN 2.2"
      ? wan22Strategies
      : state.architecture === "MiniMax H3"
        ? h3Strategies
        : ltxStrategies;
  state.loras.forEach((lora, idx) => {
    if (!strategies.includes(lora.strategy)) lora.strategy = defaultLoraStrategy();
    const row = document.createElement("div");
    row.className = "lora-row";
    row.innerHTML = `
      <label class="checkline" title="Enable this LoRA for the merge"><input type="checkbox" ${lora.enabled ? "checked" : ""} data-role="enabled"><span>Merge</span></label>
      <button class="ghost lora-pick" type="button" title="Choose a LoRA file">${lora.path ? shortPath(lora.path) : "Choose LoRA"}</button>
      <select data-role="strategy" title="Merge strategy: how LoRA weights are applied">${strategies.map((s) => `<option value="${s}" ${s === lora.strategy ? "selected" : ""}>${s}</option>`).join("")}</select>
      <div class="strength-control" title="Per-LoRA strength multiplier (-3..3; negative subtracts the LoRA, 0=none, 1=full)">
        <input data-role="strength-slider" type="range" min="-3" max="3" step="0.05" value="${lora.strength}">
        <input class="strength-num" type="number" min="-3" max="3" step="0.01" value="${formatStrength(lora.strength)}" title="Type a value directly (-3..3)">
      </div>
      <button class="ghost" data-role="remove" type="button" title="Remove this LoRA from the list">Remove</button>
    `;
    row.querySelector(".lora-pick").addEventListener("click", () => {
      state.pendingLoraSlot = idx;
      openBrowser("lora");
    });
    row.querySelector('[data-role="enabled"]').addEventListener("change", (ev) => { lora.enabled = ev.target.checked; saveSettings(); });
    row.querySelector('[data-role="strategy"]').addEventListener("change", (ev) => { lora.strategy = ev.target.value; saveSettings(); });
    const sliderEl = row.querySelector('[data-role="strength-slider"]');
    const numInput = row.querySelector('.strength-num');
    function syncFromValue(cause) {
      // cause: 'slider' or 'num' — update the other control without recursion
      numInput.value = formatStrength(lora.strength);
      sliderEl.value = lora.strength;
    }
    sliderEl.addEventListener("input", (ev) => {
      lora.strength = Number(ev.target.value);
      syncFromValue('slider');
      saveSettings();
    });
    numInput.addEventListener("change", (ev) => {
      const v = parseFloat(String(Number(ev.target.value).toFixed(2)));
      if (!isNaN(v)) {
        lora.strength = Math.max(-3, Math.min(3, v));
        syncFromValue('num');
        saveSettings();
      } else {
        // Invalid input — revert to current value
        numInput.value = formatStrength(lora.strength);
      }
    });
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
  saveSettings();
  if (!$("model-name").value) {
    $("model-name").value = path.split("/").pop().replace(/\.(safetensors|gguf|ckpt|pt|bin)$/i, "");
  }
  if (path.endsWith(".safetensors")) {
    try {
      const info = await api(`/api/inspect?path=${encodeURIComponent(path)}`);
      if (["WAN 2.2", "LTX-2.3", "Krea 2", "MiniMax H3"].includes(info.architecture)) {
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
  saveSettings();
  if (state.workflowMode === "model") refreshModelMergeHint();
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
  const dryRun = $("mm-dry-run").checked;
  const globalStrength = Number($("lora-global-strength").value) || 1;
  const unsafe = selected.find((lora) => Math.abs((Number(lora.strength) || 0) * globalStrength) > MAX_EFFECTIVE_LORA_STRENGTH);
  if (unsafe) {
    return log(`${shortPath(unsafe.path)} effective strength is too high. Keep per-LoRA × global strength within ±${MAX_EFFECTIVE_LORA_STRENGTH}.\n`);
  }
  if (!dryRun && !$("model-name").value) return log("Enter a Display Name before writing a merged checkpoint.\n");

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
        output_name: $("model-name").value,
        loras: selected.map((lora) => ({
          path: lora.path,
          strength: Number(lora.strength) || 0,
          strategy: lora.strategy,
        })),
        strategy: defaultLoraStrategy(),
        architecture: state.architecture,
        global_strength: globalStrength,
        merge_device: $("lora-merge-device").value,
        cuda_device: $("lora-cuda-device").value || "cuda:0",
        vram_headroom_mb: Number($("lora-vram-headroom").value) || 1024,
        adaptive: $("lora-adaptive").checked,
        dry_run: dryRun,
        strict_matching: $("lora-strict").checked,
        krea2_unchain: $("krea2-unchain").checked,
        watermark: $("watermark").checked,
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
  enforceInt4ConvRotStrategy();

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
        watermark: $("watermark").checked,
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
  saveActiveJob(); // persist jobId so reloads can reconnect to running jobs.
  if (state.events) state.events.close();
  clearProgressLine(); // stale line from a finished job must not linger.
  state.events = new EventSource(`/api/jobs/${id}/events`);
  let doneHandled = false;

  function markDone() {
    if (doneHandled) return;
    doneHandled = true;
    clearPersistedJob(id);
    stopPolling();
  }

  state.events.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.type === "progress") {
      // Self-updating single line inside the console window. Deliberately NOT
      // appended to the scrollback and NOT written to the topbar status line.
      showProgressLine(ev.text);
    } else {
      if (ev.text) log(ev.text);
      if (ev.status) setStatus(ev.status);
    }
    if (ev.type === "done" || ev.type === "error") {
      markDone();
      clearProgressLine(); // progress is over; keep the final log lines only.
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
  // Do NOT close on error — let the browser's built-in auto-reconnect handle it.
  // Visibilitychange + tryReconnectEvents() provides an extra push when tab comes back.
  state.events.onerror = () => {
    log("Log stream disconnected (will retry automatically).\n");
  };

  // Periodic poll to catch completion if SSE drops while page is open but idle.
  let pollIntervalId = setInterval(() => doJobStatusCheck(id), POLL_INTERVAL_MS);
  const origClose = state.events.close.bind(state.events);
  state.events.close = () => {
    clearInterval(pollIntervalId);
    markDone();
    origClose();
  };
}

function tryReconnectEvents() {
  // If there's no active event stream but we still have a job ID, reconnect.
  if (state.jobId && (!state.events || state.events.readyState === EventSource.CLOSED)) {
    log("Tab visible again — reconnecting to log stream...\n");
    attachEvents(state.jobId);
  }
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

async function refreshWatermarkStatus() {
  const hint = $("watermark-hint");
  const box = $("watermark");
  if (!hint || !box) return;
  let status;
  try {
    status = await api("/api/watermark");
  } catch {
    return;
  }
  const enabled = box.checked;
  // enabled-but-inert: box checked but no key -> pulse the label red.
  const danger = enabled && !status.available;
  const line = box.closest(".checkline");
  if (line) line.classList.toggle("danger", danger);
  let text, warn = false;
  if (!enabled) {
    text = "Watermarking off for this run — outputs will NOT carry a modelspec.watermark token.";
  } else if (status.available) {
    text = `Watermark key available (${status.source}); outputs will carry modelspec.watermark.`;
  } else {
    warn = true;
    text = "No watermark key configured — outputs will NOT carry modelspec.watermark. " +
      "Set DASIWA_WATERMARK_PASSPHRASE (or run `go_bridge.py watermark-key`) to enable provenance.";
  }
  hint.textContent = text;
  hint.classList.toggle("warn", warn);
  hint.classList.remove("hidden");
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

function loadRecipe() {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".txt";
  var picker = input; // keep ref for cleanup in error path.
  input.addEventListener("change", async (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      await parseRecipeAndApply(text, file.name);
    } catch (err) {
      log("Load recipe failed: " + err.message + "\n");
    } finally {
      picker.remove();
    }
  });
  document.body.appendChild(input);
  input.click();
}

async function resolveRecipeModelPath(name, kind) {
  var trimmed = String(name || "").trim();
  if (!trimmed) return "";
  if (trimmed.startsWith("/")) return trimmed;

  try {
    var data = await api(`/api/search?path=${encodeURIComponent(state.modelsDir)}&q=${encodeURIComponent(trimmed)}`);
    var exact = (data.items || []).filter((item) => item.name === trimmed);
    if (exact.length === 1) return exact[0].path;
    if (exact.length > 1) {
      log(`Recipe ${kind} "${trimmed}" matched multiple files; using ${exact[0].path}\n`);
      return exact[0].path;
    }
    log(`Recipe ${kind} "${trimmed}" was not found under ${state.modelsDir}; keeping filename only.\n`);
  } catch (err) {
    log(`Recipe ${kind} lookup failed for "${trimmed}": ${err.message}\n`);
  }
  return trimmed;
}

async function parseRecipeAndApply(recipeText, fileName) {
  var lines = recipeText.split("\n");

  // Helper: read a "Key: value" line by searching for the label.
  var i = 0;
  function field(label) {
    while (i < lines.length) {
      var escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      var m = lines[i].match(new RegExp("^\\s*" + escaped + "[: ]+(.*)"));
      if (m) { i++; return m[1].trim(); }
      i++;
    }
    return "";
  }

  // Scan for the recipe header block.
  var headerStart = -1;
  for (var j = 0; j < lines.length; j++) {
    if (~lines[j].indexOf("DaSiWa LoRA Merge Recipe")) { headerStart = j; break; }
  }

  if (headerStart >= 0) {
    i = headerStart + 1;
    var outputName        = field("Output");
    var baseCheckpoint    = field("Base checkpoint");
    var architecture      = field("Architecture");
    var defaultStrategy   = field("Default strategy") || "Balanced";
    var globalStrengthStr = field("Global strength");
    var adaptive          = field("Adaptive scaling") === "yes";
    var dryRun            = field("Dry run first") === "yes";
    var strictMatch       = field("Strict matching") === "yes";
    var krea2Unchain      = field("Krea2 unchain") === "yes";

    // Move to the LoRA section before parsing entries. Header field scanning
    // intentionally stops before the separator, so skip section labels/blanks.
    while (i < lines.length && lines[i].indexOf("LoRAs") < 0) i++;
    if (i < lines.length) i++;
    while (i < lines.length && (!lines[i].trim() || lines[i].indexOf("-".repeat(8)) >= 0)) i++;

    // Parse LoRAs (each starts with a number and filename).
    state.loras = [];
    while (i < lines.length) {
      var mLine = lines[i];
      if (!mLine || !mLine.trim()) { i++; continue; }
      if (mLine.indexOf("-".repeat(8)) >= 0) break;
      var loraMatch = mLine.match(/^\s*(\d+)\.\s+(.+)/);
      if (loraMatch) {
        i++; // skip the "1. filename" line
        var name = loraMatch[2].trim();
        var strength = 0.65;
        var strategy = defaultStrategy;

        while (i < lines.length) {
          var sM = lines[i].match(/^\s*Strength:\s+(.*)/);
          if (sM) {
            i++;
            var parsedStrength = parseFloat(sM[1]);
            strength = isNaN(parsedStrength) ? 0.65 : parsedStrength;
            continue;
          }
          var stM = lines[i].match(/^\s*Strategy:\s+(.*)/);
          if (stM) { i++; strategy = stM[1] || defaultStrategy; continue; }
          break;
        }

        state.loras.push({ path: name, strength: strength, strategy: strategy, enabled: true });
      } else {
        i++;
      }
    }

    // Apply to UI.
    if (architecture) {
      var archSel = $("architecture");
      for (var k = 0; k < archSel.options.length; k++) {
        var opt = archSel.options[k];
        if (opt.value === architecture) { state.architecture = opt.value; archSel.value = opt.value; break; }
      }
    }

    if (baseCheckpoint) {
      var resolvedBase = await resolveRecipeModelPath(baseCheckpoint, "base checkpoint");
      await selectSource(resolvedBase);
    }

    if (outputName) {
      $("model-name").value = outputName.replace(/\.(safetensors|gguf|ckpt|pt|bin)$/i, "");
    }

    for (var r = 0; r < state.loras.length; r++) {
      state.loras[r].path = await resolveRecipeModelPath(state.loras[r].path, "LoRA");
    }

    var globalStrVal = Number(globalStrengthStr);
    if (!isNaN(globalStrVal)) $("lora-global-strength").value = globalStrVal;
    $("lora-adaptive").checked = adaptive;
    $("mm-dry-run").checked = dryRun;
    $("lora-strict").checked = strictMatch;

    // Krea2 unchain checkbox visibility + state.
    updateArchDependentUI();
    var ucbLabel = $("krea2-unchain-label");
    if (ucbLabel) {
      ucbLabel.style.display = "none"; // only show when arch is actually Krea 2 in the dropdown.
      $("krea2-unchain").checked = krea2Unchain;
      if (architecture === "Krea 2") updateArchDependentUI();
    }

    renderLoras();
    saveSettings();
    log("Loaded recipe \"" + fileName + "\": " + (state.sourcePath ? "base=" + shortPath(state.sourcePath) + ", " : "") + state.loras.length + " LoRA(s), arch=" + state.architecture + ", strength=" + globalStrVal + "\n");
  } else {
    log("Not a DaSiWa LoRA Merge Recipe file.\n");
    return;
  }
}


init().catch(function(err) { log("Initialization failed: " + err.message + "\n"); });
