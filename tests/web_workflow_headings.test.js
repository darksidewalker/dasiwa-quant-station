'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../web/index.html'), 'utf8');
const start = source.indexOf('function setWorkflowMode(');
const end = source.indexOf('\nasync function openBrowser(', start);
const modes = [...html.matchAll(/data-mode="([^"]+)"[^>]*>([^<]+)<\/button>/g)];

test('all mode transitions update one title, purpose and matching start action', () => {
  const elements = {};
  const $ = id => elements[id] ||= {textContent: '', style: {}, classList: {toggle() {}, remove() {}}};
  const context = {
    $, state: {sourcePath: '', jobId: ''}, document: {querySelectorAll: () => []},
    updateComposeOutputIndicator() {}, updateExtractRecipeVisibility() {},
    updateDeltaOptionsVisibility() {}, updateH3Visibility() {}, applyControlVisibility() {},
    updateModelMergeVisibility() {}, saveSettings() {}, setStatus(text) { $('status').textContent = text; },
  };
  vm.runInNewContext(source.slice(start, end), context);
  for (const [_, mode, title] of [...modes, ...modes.slice().reverse()]) {
    context.setWorkflowMode(mode);
    assert.equal($('workflow-title').textContent, title, mode);
    assert.ok($('workflow-description').textContent.length > 30, `${mode} needs a purpose hint`);
    assert.ok($('start').title.length > 20, `${mode} needs a matching action hint`);
    assert.equal($('status').textContent, 'Ready');
  }
  context.state.jobId = 'running-job';
  $('status').textContent = 'Running';
  context.setWorkflowMode('h3-prune');
  assert.equal($('start').textContent, 'Start Prune');
  assert.equal($('status').textContent, 'Running', 'switching modes must preserve job status');
  context.setWorkflowMode('h3-adapter-convert');
  assert.equal($('start').textContent, 'Convert Adapter');
});

test('mode titles and purpose use one shared header without duplicate mode cards', () => {
  assert.equal((html.match(/id="workflow-title"/g) || []).length, 1);
  assert.equal((html.match(/id="workflow-description"/g) || []).length, 1);
  assert.doesNotMatch(html, /LoRA Merge \/ Compose|<h[23]>(LoRA Extract|Model Merge|H3 Workflow)<\/h[23]>/);
  assert.doesNotMatch(html, /class="lora-card hidden" data-workflow-panel="(?:extract|model)"/);
});
