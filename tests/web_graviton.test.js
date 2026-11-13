'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('web/index.html', 'utf8');
const js = fs.readFileSync('web/app.js', 'utf8');
test('both existing algorithm selectors offer opt-in GRAVITON', () => {
 for (const id of ['lora-merge-algorithm', 'compose-merge-algorithm']) {
  const select = html.match(new RegExp(`<select id="${id}"[^>]*>([\\s\\S]*?)</select>`))[1];
  assert.match(select, /value="graviton"/);
 }
 assert.match(html, /base-free spectral/i);
});
test('GRAVITON compose recipe restores signed/zero strengths, strategies and compute settings', async () => {
 const start = js.indexOf('async function parseRecipeAndApply(');
 const rest = js.slice(start);
 const end = rest.slice(1).search(/\n(?:(?:async )?function |const \$|init\(\))/);
 const source = end < 0 ? rest : rest.slice(0, end + 1);
 const elements = {};
 const $ = id => elements[id] ||= {value:'', checked:false};
 const state = {loras:[]};
 const settings = {global_strength:0, merge_device:'cpu', cuda_device:'cuda:1', vram_headroom_mb:2048,
   mismatch_mode:'error', loras:[{strategy:'Motion'}, {strategy:'Visuals'}]};
 const recipe = 'DaSiWa LoRA Compose Recipe\nArchitecture: MiniMax H3\nMerge algorithm: graviton\nOutput adapter: lora\nOutput rank: 8\nFrobenius energy: 1\nGRAVITON settings: '+JSON.stringify(settings)+'\n1. /a.safetensors\n   Strength: -1\n2. /b.safetensors\n   Strength: 0\n';
 await vm.runInNewContext(`${source}\nparseRecipeAndApply(recipe, 'recipe.txt')`, {recipe, $, state,
   resolveRecipeModelPath: async path => path, defaultLoraStrategy: () => 'Balanced',
   setWorkflowMode() {}, renderLoras() {}, saveSettings() {}, log() {}});
 assert.equal($('compose-merge-algorithm').value, 'graviton');
 assert.equal($('lora-global-strength').value, 0);
 assert.deepEqual(Array.from(state.loras, item => [item.strength, item.strategy]), [[-0,'Motion'],[0,'Visuals']]);
 assert.equal($('lora-vram-headroom').value, 2048);
});

test('saved GRAVITON algorithm round-trips through existing compose settings', () => {
 const loader = js.slice(js.indexOf('const compose = s.compose || {};'), js.indexOf('const modelMerge = s.modelMerge || {};'));
 const elements = Object.fromEntries(['compose-merge-algorithm','compose-preset','compose-output-adapter','compose-output-rank','compose-energy','compose-mismatch'].map(id => [id, {value:''}]));
 const s = {compose:{mergeAlgorithm:'graviton'}};
 const $ = id => elements[id];
 // Evaluate only this repository's settings-loader code, never user input.
 eval(loader);
 assert.equal(elements['compose-merge-algorithm'].value, 'graviton');
});
