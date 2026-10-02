'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
function fn(name) {
  const start = source.search(new RegExp(`^(?:async )?function ${name}\\(`, 'm'));
  const rest = source.slice(start);
  const end = rest.slice(1).search(/\n(?:(?:async )?function |const \$|init\(\))/);
  return end < 0 ? rest : rest.slice(0, end + 1);
}
function context() {
  const elements = {};
  const $ = (id) => elements[id] ||= {value: '', checked: false, style: {}, options: [{value: 'MiniMax H3'}]};
  return {elements, $, state: {architecture: 'MiniMax H3', sourcePath: '/base.safetensors', modelsDir: '/models',
    loras: [{path: '/a.safetensors', strength: 1, strategy: 'Balanced', enabled: true},
            {path: '/b.safetensors', strength: 1, strategy: 'Balanced', enabled: true}],
    formats: new Set(), mmSelectedBlocks: new Set(), extractSelectedBlocks: new Set(), browserSort: {}}, MAX_EFFECTIVE_LORA_STRENGTH: 3,
    defaultLoraStrategy: () => 'Balanced', shortPath: (s) => s, log: () => {}, setStatus: () => {},
    attachEvents: () => {}, updateArchDependentUI: () => {}, renderLoras: () => {}, saveSettings: () => {},
    resolveRecipeModelPath: async (p) => p, selectSource: async () => {}};
}
test('baking request sends checkbox for additive and consensus', async () => {
  for (const algorithm of ['additive', 'consensus']) {
    for (const enabled of [true, false]) {
      const ctx = context();
      ctx.$('lora-merge-algorithm').value = algorithm;
      ctx.$('lora-global-strength').value = '1';
      ctx.$('model-name').value = 'out';
      ctx.$('protect-token-refiner').checked = enabled;
      let body;
      ctx.api = async (url, options) => {assert.equal(url, '/api/lora/merge'); body = JSON.parse(options.body); return {job_id: 'test'};};
      await vm.runInNewContext(`${fn('startLoraMerge')}\nstartLoraMerge()`, ctx);
      assert.equal(body.protect_token_refiner, enabled);
      assert.equal(body.merge_algorithm, algorithm);
    }
  }
});
test('recipe loads protection and preserves old recipes with missing optional field', async () => {
  for (const option of ['', 'Protect Token Refiner: yes\n', 'Protect Token Refiner: no\n']) {
    const ctx = context();
    ctx.$('protect-token-refiner').checked = true;
    const recipe = `DaSiWa LoRA Merge Recipe\nOutput: out.safetensors\nBase checkpoint: /base.safetensors\nArchitecture: MiniMax H3\nDefault strategy: Balanced\nMerge algorithm: consensus\nConsensus preset: neutral\nGlobal strength: 1\nAdaptive scaling: no\nDry run first: yes\nStrict matching: yes\nKrea2 unchain: no\n${option}\n--------\nLoRAs\n--------\n1. /a.safetensors\n Strength: 0\n Strategy: Balanced\n2. /b.safetensors\n Strength: -0.5\n Strategy: Balanced\n--------\nMerge Summary\n`;
    ctx.recipe = recipe;
    await vm.runInNewContext(`${fn('parseRecipeAndApply')}\nparseRecipeAndApply(recipe, 'test.txt')`, ctx);
    assert.equal(ctx.$('protect-token-refiner').checked, option.includes('yes'));
    assert.equal(ctx.$('lora-merge-algorithm').value, 'consensus');
    assert.equal(ctx.$('lora-consensus-preset').value, 'neutral');
    assert.equal(ctx.state.loras.length, 2);
    assert.equal(ctx.state.loras[0].strength, 0);
    assert.equal(ctx.state.loras[1].strength, -0.5);
  }
});
test('settings restore true, false, and default off for older settings', () => {
  for (const enabled of [true, false, undefined]) {
    const ctx = context();
    ctx.$('protect-token-refiner').checked = true;
    ctx.SETTINGS_STORAGE_KEY = 'test';
    ctx.QuantStationUI = {decodeStoredSettings: JSON.parse, migrateSettings: (s) => s};
    ctx.localStorage = {getItem: () => JSON.stringify({lora: {protectTokenRefiner: enabled}}), setItem: () => {}};
    for (const name of ['renderModelMergeBlockGrid', 'updateModelMergeBlockSelection',
      'updateModelMergeModalityVisibility', 'updateExtractBlockVisibility', 'setPathLabel',
      'updateDeltaOptionsVisibility', 'updateExtractRecipeVisibility']) ctx[name] = () => {};
    vm.runInNewContext(`${fn('loadSettings')}\nloadSettings()`, ctx);
    assert.equal(ctx.$('protect-token-refiner').checked, enabled ?? false);
  }
});

test('current settings persist protection and HTML defaults off in baking panel', () => {
  const ctx = context();
  ctx.$('protect-token-refiner').checked = true;
  const settings = vm.runInNewContext(`${fn('currentSettings')}\ncurrentSettings()`, ctx);
  assert.equal(settings.lora.protectTokenRefiner, true);
  const html = fs.readFileSync(path.join(__dirname, '../web/index.html'), 'utf8');
  const input = html.match(/<input id="protect-token-refiner"[^>]*>/)[0];
  assert.ok(!input.includes('checked'));
  assert.ok(html.includes('<span>Protect Token Refiner</span>'));
  assert.ok(source.includes('$("protect-token-refiner").checked = lora.protectTokenRefiner ?? false;'));
});
