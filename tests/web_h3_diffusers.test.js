'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
function fn(name) {
  const start = source.search(new RegExp(`^(?:async )?function ${name}\\(`, 'm'));
  assert.ok(start >= 0, `missing ${name}`);
  const rest = source.slice(start); const end = rest.slice(1).search(/\n(?:(?:async )?function |const \$|init\(\))/);
  return end < 0 ? rest : rest.slice(0, end + 1);
}
function context() {
  const elements = {};
  const $ = id => elements[id] ||= {value: '', checked: false, textContent: '', style: {}, classList: {toggle() {}, remove() {}}};
  const c = {$, elements, setPathLabel() {}, saveSettings() {}, attachEvents() {},
    state: {architecture: 'MiniMax H3', workflowMode: 'h3-diffusers-lora', sourcePath: '', h3AdapterPath: '/lora.safetensors', h3TargetPath: '', h3CurvePath: '',
      h3ReferencePath: '', loras: [], formats: new Set(), mmSelectedBlocks: new Set(), extractSelectedBlocks: new Set()}};
  c.log = text => { c.logged = text; }; c.setWorkflowMode = mode => { c.mode = mode; };
  return c;
}
async function start(c) {
  let call;
  c.api = async (url, opts) => { call = {url, body: JSON.parse(opts.body)}; return {job_id: 'job'}; };
  await vm.runInNewContext(`${fn('startH3Workflow')}\nstartH3Workflow()`, c);
  return call;
}

test('Diffusers mode posts its own route and omits unused optional fields', async () => {
  const c = context(); c.$('model-name').value = 'native'; c.$('mm-dry-run').checked = true; c.$('h3-representation').value = 'exact';
  const {url, body} = await start(c);
  assert.equal(url, '/api/h3/diffusers-lora');
  assert.equal(body.adapter_path, '/lora.safetensors'); assert.equal(body.representation, 'exact'); assert.equal(body.dry_run, true);
  for (const key of ['checkpoint_path', 'curve_path', 'energy', 'max_rank', 'base_path', 'merge_device']) assert.ok(!(key in body), key);
});

test('Diffusers SVD payload carries checkpoint, curve, energy and rank cap', async () => {
  const c = context(); c.state.h3TargetPath = '/ckpt.safetensors'; c.state.h3CurvePath = '/curve.safetensors';
  c.$('model-name').value = 'native'; c.$('h3-representation').value = 'svd'; c.$('h3-energy').value = '0.995'; c.$('h3-max-rank').value = '512';
  const {body} = await start(c);
  assert.equal(body.checkpoint_path, '/ckpt.safetensors'); assert.equal(body.curve_path, '/curve.safetensors');
  assert.equal(body.energy, 0.995); assert.equal(body.max_rank, 512); assert.equal(body.output_name, 'native');
});

test('curve without target checkpoint never starts a job', async () => {
  const c = context(); c.state.h3CurvePath = '/curve.safetensors'; c.$('mm-dry-run').checked = true;
  assert.equal(await start(c), undefined);
  assert.match(c.logged, /target checkpoint/);
});

test('saved Diffusers controls have change listeners and round-trip through settings', () => {
  const listeners = source.slice(source.indexOf('["h3-turbo-complete"'), source.indexOf('.forEach((id) => $(id).addEventListener("change", saveSettings));', source.indexOf('["h3-turbo-complete"')));
  for (const id of ['h3-energy', 'h3-max-rank']) assert.ok(listeners.includes(`"${id}"`), id);
  assert.match(source, /\$\("h3-representation"\)\.addEventListener\("change", \(\) => \{updateH3Visibility\(\); saveSettings\(\);\}\)/);
  const c = context(); c.state.h3CurvePath = '/curve.safetensors'; c.$('h3-representation').value = 'svd'; c.$('h3-energy').value = '0.99'; c.$('h3-max-rank').value = '256';
  for (const id of ['lora-global-strength', 'compose-merge-algorithm', 'mm-recipe', 'extract-mode']) c.$(id);
  const saved = vm.runInNewContext(`${fn('currentSettings')}\ncurrentSettings()`, c);
  assert.deepEqual([saved.h3.curvePath, saved.h3.representation, saved.h3.energy, saved.h3.maxRank], ['/curve.safetensors', 'svd', '0.99', '256']);
  const restored = context(); restored.QuantStationUI = require('../web/ui_helpers.js');
  restored.localStorage = {getItem: () => JSON.stringify(saved), setItem() {}}; restored.SETTINGS_STORAGE_KEY = 'k';
  for (const name of ['renderModelMergeBlockGrid', 'updateModelMergeBlockSelection', 'updateModelMergeModalityVisibility', 'updateExtractBlockVisibility', 'updateDeltaOptionsVisibility', 'updateExtractRecipeVisibility', 'defaultLoraStrategy']) restored[name] = () => {};
  vm.runInNewContext(`${fn('loadSettings')}\nloadSettings()`, restored);
  assert.equal(restored.state.h3CurvePath, '/curve.safetensors'); assert.equal(restored.$('h3-representation').value, 'svd');
  assert.equal(restored.$('h3-energy').value, '0.99'); assert.equal(restored.$('h3-max-rank').value, '256');
});

test('Diffusers recipe reload restores inputs, representation and destination', async () => {
  const c = context();
  const settings = {adapter_path: '/lora.safetensors', checkpoint_path: '/ckpt.safetensors', curve_path: '/curve.safetensors', representation: 'svd', energy: 0.999, max_rank: 512, output_path: '/out/native.safetensors'};
  await vm.runInNewContext(`${fn('parseRecipeAndApply')}\nparseRecipeAndApply(recipe, 'native.txt')`, {...c, recipe: 'DaSiWa H3 Diffusers LoRA Conversion Recipe\n' + JSON.stringify({settings})});
  assert.equal(c.state.h3TargetPath, '/ckpt.safetensors'); assert.equal(c.state.h3CurvePath, '/curve.safetensors');
  assert.equal(c.$('h3-representation').value, 'svd'); assert.equal(c.$('h3-max-rank').value, 512); assert.equal(c.$('model-name').value, 'native');
  assert.equal(c.state.h3OutputPath, '/out/native.safetensors'); assert.equal(c.$('mm-dry-run').checked, true); assert.equal(c.mode, 'h3-diffusers-lora');
  const {body} = await start(c);
  assert.equal(body.output_path, settings.output_path);
  assert.equal(body.dry_run, true);
});

test('exact conversion copy distinguishes rank preservation from BF16 rounding', () => {
  const html = fs.readFileSync(path.join(__dirname, '../web/index.html'), 'utf8');
  assert.match(html, /no rank truncation; BF16 storage rounding still applies/);
  assert.doesNotMatch(html, /bit-faithful deltas/);
});

test('Diffusers mode shows its pickers and options, without source or compute', () => {
  const c = context(); let hidden; let moved = false;
  c.$('h3-side-panel').classList.toggle = (_, value) => { hidden = value; };
  c.$('lora-merge-device').closest = () => ({});
  c.$('shared-compute').appendChild = () => { moved = true; }; c.$('lora-bake-controls').insertBefore = () => {};
  c.$('compose-preset').closest = () => ({style: {}}); c.$('h3-turbo-complete').closest = () => null;
  c.$('h3-representation').value = 'exact';
  vm.runInNewContext(`${fn('updateH3Visibility')}\nupdateH3Visibility()`, c);
  assert.equal(hidden, false); assert.equal(moved, false);
  for (const id of ['h3-pick-target', 'h3-pick-adapter', 'h3-pick-curve', 'h3-diffusers-options']) assert.equal(c.$(id).style.display, '', id);
  assert.equal(c.$('h3-pick-reference').style.display, 'none'); assert.equal(c.$('h3-energy-label').style.display, 'none');
  assert.equal(c.$('h3-adapter-hint').textContent, 'Diffusers LoRA');
  assert.deepEqual([...require('../web/ui_helpers.js').applicableControls('h3-diffusers-lora')], ['architecture', 'output-name', 'dry-run']);
  const html = fs.readFileSync(path.join(__dirname, '../web/index.html'), 'utf8');
  assert.ok(html.includes('data-mode="h3-diffusers-lora"'));
  assert.match(fn('setWorkflowMode'), /"h3-diffusers-lora": \{title: "Diffusers → H3"/);
});
