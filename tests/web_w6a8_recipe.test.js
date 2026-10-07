'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require.resolve('../web/app.js'), 'utf8');
const parser = source.slice(source.indexOf('async function parseRecipeAndApply('), source.indexOf('init().catch'));

test('W6 recipe restores quantization settings and output folder', async () => {
 const nodes = new Map();
 const context = {
  state: {}, Set,
  $: id => { if(!nodes.has(id)) nodes.set(id, {}); return nodes.get(id); },
  selectSource: async path => { context.state.sourcePath = path; },
  resolveRecipeModelPath: async path => path,
  updateArchDependentUI: () => {},
  setWorkflowMode: mode => { context.state.workflowMode = mode; },
  saveSettings: () => {}, log: () => {},
 };
 vm.createContext(context); vm.runInContext(parser,context);
 await context.parseRecipeAndApply('DaSiWa Quantization Recipe\nSource path: /models/h3.safetensors\nOutput path: /outputs/video_w6a8.safetensors\nModel name: video\nArchitecture: MiniMax H3\nFormat: W6A8 (w6a8_int8)\nStrategy: Simple\nPreserve loader metadata: no\n', 'recipe.txt');
 assert.equal(context.state.sourcePath, '/models/h3.safetensors');
 assert.equal(context.state.architecture, 'MiniMax H3');
 assert.deepEqual([...context.state.formats], ['W6A8']);
 assert.equal(context.state.strategy, 'Simple');
 assert.equal(context.state.outputDir, '/outputs');
 assert.equal(context.state.workflowMode, 'quantize');
 assert.equal(nodes.get('model-name').value, 'video');
 assert.equal(nodes.get('preserve-loader-metadata').checked, false);
});
