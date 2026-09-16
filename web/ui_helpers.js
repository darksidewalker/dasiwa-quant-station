(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.QuantStationUI = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const STRATEGIES = ["Optimizer-driven", "Simple"];
  const EMPTY_PICKER_DIRECTORIES = Object.freeze({
    source: "",
    lora: "",
    "model-overlay": "",
    "extract-modified": "",
    "extract-pruned": "",
  });
  const WORKFLOW_CONTROLS = Object.freeze({
    quantize: ["source", "architecture", "output-name", "full-checkpoint", "quant-strategy", "low-vram", "watermark", "preserve-metadata"],
    lora: ["source", "architecture", "output-name", "full-checkpoint", "dry-run", "watermark", "preserve-metadata"],
    compose: ["architecture", "output-name", "dry-run"],
    extract: ["source", "architecture", "output-name", "full-checkpoint", "dry-run"],
    model: ["source", "architecture", "output-name", "full-checkpoint", "dry-run", "watermark", "preserve-metadata"],
  });

  function allowedStrategies(formats, capabilities) {
    if (!formats.length) return STRATEGIES.slice();
    return STRATEGIES.filter((strategy) => formats.every((format) =>
      (capabilities[format]?.strategies || []).includes(strategy)));
  }

  function formatAllowed(format, architecture, capabilities) {
    const rule = capabilities[format];
    if (!rule) return false;
    return !rule.architectures?.length || rule.architectures.includes(architecture);
  }

  function normalizeQuantSelection(selection, capabilities) {
    const formats = selection.formats.filter((format) =>
      formatAllowed(format, selection.architecture, capabilities));
    const strategies = allowedStrategies(formats, capabilities);
    return {
      formats,
      strategy: strategies.includes(selection.strategy) ? selection.strategy : strategies[0],
    };
  }

  function sortBrowserItems(items, key, direction) {
    const sign = direction === "desc" ? -1 : 1;
    return items.slice().sort((a, b) => {
      if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1;
      if (key === "size") return sign * ((a.size || 0) - (b.size || 0));
      if (key === "date") return sign * (dateValue(a.modified_at) - dateValue(b.modified_at));
      return sign * a.name.localeCompare(b.name, undefined, {sensitivity: "base", numeric: true});
    });
  }

  function dateValue(value) {
    const parsed = Date.parse(value || "");
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function directoryForMode(directories, mode, fallback) {
    return directories[mode] || fallback;
  }

  function rememberDirectory(directories, mode, path) {
    return {...directories, [mode]: path};
  }

  function decodeStoredSettings(raw) {
    if (!raw) return null;
    try {
      const value = typeof raw === "string" ? JSON.parse(raw) : raw;
      return value && typeof value === "object" ? value : null;
    } catch (_) {
      return null;
    }
  }

  function migrateSettings(settings) {
    const source = settings || {};
    if (source.schema === 2) {
      return {
        ...source,
        pickerDirectories: {...EMPTY_PICKER_DIRECTORIES, ...(source.pickerDirectories || {})},
      };
    }
    return {
      ...source,
      schema: 2,
      pickerDirectories: {
        ...EMPTY_PICKER_DIRECTORIES,
        source: source.lastFileDir || "",
        lora: source.lastLoraDir || "",
        "model-overlay": source.lastOverlayDir || "",
      },
    };
  }

  function applicableControls(mode) {
    return new Set(WORKFLOW_CONTROLS[mode] || []);
  }

  function formatBytes(size, isDirectory) {
    if (isDirectory) return "—";
    const bytes = Number(size) || 0;
    if (bytes < 1024) return `${bytes} B`;
    const units = ["KiB", "MiB", "GiB", "TiB"];
    let value = bytes / 1024;
    let unit = units[0];
    for (let index = 1; value >= 1024 && index < units.length; index += 1) {
      value /= 1024;
      unit = units[index];
    }
    const digits = value >= 10 ? 0 : 1;
    return `${value.toFixed(digits)} ${unit}`;
  }

  function formatDate(value) {
    const parsed = Date.parse(value || "");
    if (!Number.isFinite(parsed)) return "—";
    return new Intl.DateTimeFormat(undefined, {dateStyle: "medium", timeStyle: "short"}).format(parsed);
  }

  return {
    EMPTY_PICKER_DIRECTORIES,
    STRATEGIES,
    WORKFLOW_CONTROLS,
    allowedStrategies,
    applicableControls,
    decodeStoredSettings,
    directoryForMode,
    formatAllowed,
    formatBytes,
    formatDate,
    migrateSettings,
    normalizeQuantSelection,
    rememberDirectory,
    sortBrowserItems,
  };
});
