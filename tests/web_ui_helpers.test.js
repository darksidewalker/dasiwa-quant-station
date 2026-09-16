'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
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
} = require('../web/ui_helpers.js');

const capabilities = {
  FP8: {strategies: ['Optimizer-driven', 'Simple']},
  W4A8: {architectures: ['MiniMax H3'], strategies: ['Simple']},
  'INT4 ConvRot Runtime': {architectures: ['WAN 2.2', 'LTX-2.3', 'Krea 2', 'MiniMax H3'], strategies: ['Simple']},
};

test('intersects strategies across all selected formats', () => {
  assert.deepEqual(allowedStrategies(['FP8', 'W4A8'], capabilities), ['Simple']);
  assert.deepEqual(allowedStrategies([], capabilities), ['Optimizer-driven', 'Simple']);
});

test('normalizes stale quant selections after architecture changes', () => {
  assert.deepEqual(normalizeQuantSelection({
    formats: ['FP8', 'W4A8'], architecture: 'WAN 2.2', strategy: 'Optimizer-driven',
  }, capabilities), {formats: ['FP8'], strategy: 'Optimizer-driven'});
  assert.deepEqual(normalizeQuantSelection({
    formats: ['FP8', 'W4A8'], architecture: 'MiniMax H3', strategy: 'Optimizer-driven',
  }, capabilities), {formats: ['FP8', 'W4A8'], strategy: 'Simple'});
});

test('fails closed for unknown formats', () => {
  assert.equal(formatAllowed('unknown', 'MiniMax H3', capabilities), false);
});

test('sorts files without mutating input and keeps directories first', () => {
  const input = [
    {name: 'z.bin', is_dir: false, size: 2, modified_at: '2026-01-02T00:00:00Z'},
    {name: 'Folder', is_dir: true, size: 0, modified_at: '2026-01-03T00:00:00Z'},
    {name: 'a.bin', is_dir: false, size: 10, modified_at: '2026-01-01T00:00:00Z'},
  ];
  assert.deepEqual(sortBrowserItems(input, 'name', 'asc').map((item) => item.name), ['Folder', 'a.bin', 'z.bin']);
  assert.deepEqual(sortBrowserItems(input, 'name', 'desc').map((item) => item.name), ['Folder', 'z.bin', 'a.bin']);
  assert.deepEqual(sortBrowserItems(input, 'size', 'desc').map((item) => item.name), ['Folder', 'a.bin', 'z.bin']);
  assert.deepEqual(sortBrowserItems(input, 'date', 'asc').map((item) => item.name), ['Folder', 'a.bin', 'z.bin']);
  assert.equal(input[0].name, 'z.bin');
});

test('remembers picker directories independently', () => {
  const initial = {source: '/a', lora: '/b', 'model-overlay': '/c', 'extract-modified': '/d', 'extract-pruned': '/e'};
  const changed = rememberDirectory(initial, 'extract-modified', '/new');
  assert.equal(directoryForMode(changed, 'extract-modified', '/fallback'), '/new');
  assert.equal(changed.source, '/a');
  assert.equal(changed.lora, '/b');
  assert.equal(changed['model-overlay'], '/c');
  assert.equal(changed['extract-pruned'], '/e');
});

test('migrates legacy settings without executable version gating', () => {
  const old = {v: 'old-build', mode: 'lora', lastFileDir: '/source', lastLoraDir: '/loras', lastOverlayDir: '/overlay'};
  const migrated = migrateSettings(old);
  assert.equal(migrated.mode, 'lora');
  assert.deepEqual(migrated.pickerDirectories, {
    source: '/source', lora: '/loras', 'model-overlay': '/overlay', 'extract-modified': '', 'extract-pruned': '',
  });
  assert.equal(decodeStoredSettings(JSON.stringify(old)).v, 'old-build');
  assert.equal(decodeStoredSettings('{broken'), null);
});

test('maps controls to workflows', () => {
  assert.equal(applicableControls('quantize').has('quant-strategy'), true);
  assert.equal(applicableControls('quantize').has('dry-run'), false);
  assert.equal(applicableControls('compose').has('source'), false);
  assert.equal(applicableControls('model').has('preserve-metadata'), true);
});

test('formats browser metadata', () => {
  assert.equal(formatBytes(0, true), '—');
  assert.equal(formatBytes(1536, false), '1.5 KiB');
  assert.equal(formatDate('not-a-date'), '—');
  assert.notEqual(formatDate('2026-09-16T05:30:00Z'), '—');
});
