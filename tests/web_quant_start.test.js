'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

// Exercise the actual Start Batch handler without loading a multi-GB checkpoint.
test('Krea 2 Simple starts a quantization request', async () => {
  const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
  const handler = source.slice(source.indexOf('async function startJob() {'), source.indexOf('\nfunction attachEvents(', source.indexOf('async function startJob() {')));
  const elements = {
    'model-name': {value: 'test-krea'},
    architecture: {value: 'Krea 2'},
    start: {disabled: false},
    stop: {disabled: true},
    'low-vram': {checked: false},
    'full-checkpoint': {checked: false},
    'preserve-loader-metadata': {checked: true},
    watermark: {checked: false},
  };
  let request;
  const context = {
    state: {sourcePath: '/models/krea.safetensors', modelsDir: '/models', strategy: 'Simple', formats: new Set(['FP8']), jobId: ''},
    $: (id) => elements[id],
    api: async (url, options) => { request = {url, body: JSON.parse(options.body)}; return {job_id: 'test-job'}; },
    attachEvents: (id) => { assert.equal(id, 'test-job'); },
    setStatus: () => {},
    log: () => {},
  };
  vm.runInNewContext(`${handler}\nstartJob()`, context);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(request?.url, '/api/quantize');
  assert.equal(request.body.architecture, 'Krea 2');
  assert.equal(request.body.strategy, 'Simple');
  assert.deepEqual(request.body.formats, ['FP8']);
});
