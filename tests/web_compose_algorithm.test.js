'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

// Regression: the compose algorithm select must persist across reloads.
// Without a change listener, the dropdown silently falls back to "consensus"
// on every settings reload, so a user-selected "additive" run executes as consensus.
test('compose-merge-algorithm is persisted by the settings change listeners', () => {
  const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
  const listenerBlock = source.slice(
    source.indexOf('["compose-merge-algorithm"'),
    source.indexOf('.forEach((id) => $(id).addEventListener("change", saveSettings));',
      source.indexOf('["compose-merge-algorithm"')),
  );
  assert.ok(listenerBlock.includes('"compose-merge-algorithm"'),
    'compose-merge-algorithm must be in the saveSettings change-listener list');
});

test('compose settings round-trip restores the saved merge algorithm', () => {
  const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
  const loader = source.slice(
    source.indexOf('const compose = s.compose || {};'),
    source.indexOf('const modelMerge = s.modelMerge || {};'),
  );
  const elements = {
    'compose-merge-algorithm': {value: 'consensus'},
    'compose-preset': {value: 'balanced'},
    'compose-output-adapter': {value: 'auto'},
    'compose-output-rank': {value: '0'},
    'compose-energy': {value: '0.99'},
    'compose-mismatch': {value: 'error'},
  };
  const settings = {compose: {mergeAlgorithm: 'additive', preset: 'neutral', outputAdapter: 'lora', outputRank: 128, energy: 1, mismatch: 'skip'}};
  const context = {s: settings, $: (id) => elements[id]};
  eval(`(function(){ const s = settings; const $ = context.$; ${loader} })()`);
  assert.equal(elements['compose-merge-algorithm'].value, 'additive',
    'saved additive choice must survive the settings reload');
});
