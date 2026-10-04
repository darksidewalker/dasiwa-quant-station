'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../web/app.js'), 'utf8');
function fn(name) {
 const start = source.search(new RegExp(`^(?:async )?function ${name}\\(`, 'm'));
 assert.ok(start >= 0, `missing ${name}`);
 const rest = source.slice(start); const end = rest.slice(1).search(/\n(?:(?:async )?function |const \$|init\(\))/);
 return end < 0 ? rest : rest.slice(0, end + 1);
}
function context() {
 const elements = {}; const $ = id => elements[id] ||= {value:'',checked:false,style:{},classList:{toggle(){},remove(){}}, options:[{value:'MiniMax H3'}]};
 return {$, elements, state:{architecture:'MiniMax H3',workflowMode:'h3-prune',sourcePath:'/full.safetensors',modelsDir:'/models',h3ReferencePath:'/ref.safetensors',h3TargetPath:'/target.safetensors',h3AdapterPath:'/adapter.safetensors',loras:[{path:'/adapter.safetensors',strength:0,enabled:true}],formats:new Set(),mmSelectedBlocks:new Set(),extractSelectedBlocks:new Set()},log(){},setStatus(){},attachEvents(){},saveSettings(){},setPathLabel(){},updateArchDependentUI(){},updateH3Visibility(){},setWorkflowMode(){},selectSource:async function(p){this.state.sourcePath=p},resolveRecipeModelPath:async p=>p,defaultLoraStrategy:()=> 'Balanced',MAX_EFFECTIVE_LORA_STRENGTH:3};
}
test('H3 start traverses shared source/name/compute and distinct pickers',async()=>{
 for (const mode of ['h3-prune','h3-adapter-convert']) {
  const c=context();c.state.workflowMode=mode;c.$('model-name').value='out';c.$('h3-fold-mode').value='reference';c.$('lora-merge-device').value='cpu';c.$('lora-cuda-device').value='cuda:1';c.$('lora-vram-headroom').value='2048';c.$('mm-dry-run').checked=true;
  let payload;c.api=async(url,opts)=>{assert.equal(url,'/api/h3/'+(mode==='h3-prune'?'prune':'adapter-convert'));payload=JSON.parse(opts.body);return {job_id:'test'}};
  await vm.runInNewContext(`${fn('startH3Workflow')}\nstartH3Workflow()`,c);
  assert.equal(payload.base_path,'/full.safetensors');assert.equal(payload.output_name,'out');assert.equal(payload.merge_device,'cpu');assert.equal(payload.vram_headroom_mb,2048);assert.equal(payload.dry_run,true);
  assert.equal(payload.reference_path,mode==='h3-prune'?'/ref.safetensors':'');assert.equal(payload.adapter_path,mode==='h3-prune'?'':'/adapter.safetensors');
 }
});
test('zero global baking strength is not replaced with one and complete Turbo is sent',async()=>{
 const c=context();c.$('lora-global-strength').value='0';c.$('model-name').value='out';c.$('lora-merge-algorithm').value='additive';c.$('h3-turbo-complete').checked=true;
 let payload;c.api=async(u,o)=>{payload=JSON.parse(o.body);return {job_id:'test'}};
 await vm.runInNewContext(`${fn('startLoraMerge')}\nstartLoraMerge()`,c);
 assert.equal(payload.global_strength,0);assert.equal(payload.h3_turbo_complete,true);
});
test('settings persist H3 paths and independent mode plus flags',()=>{
 const c=context(); c.$('h3-fold-mode').value='independent';c.$('h3-turbo-complete').checked=true;c.$('h3-quant-policy').value='upstream_int8_convrot';c.$('quant-verbose').value='VERBOSE';c.$('compose-merge-algorithm').value='additive';
 const settings=vm.runInNewContext(`${fn('currentSettings')}\ncurrentSettings()`,c);
 assert.equal(settings.h3.referencePath,'/ref.safetensors');assert.equal(settings.h3.foldMode,'independent');assert.equal(settings.lora.h3TurboComplete,true);assert.equal(settings.compose.mergeAlgorithm,'additive');assert.equal(settings.quant.h3QuantPolicy,'upstream_int8_convrot');
});
test('H3 visibility exposes required pickers and compute without duplicates',()=>{
 const c=context();c.$('h3-fold-mode').value='reference'; c.$('h3-side-panel').classList.toggle=(key,hidden)=>{c.hidden=hidden};
 c.$('lora-merge-device').closest=()=>({parentElement:c.$('shared-compute'),dataset:{}});
 c.$('shared-compute').appendChild=()=>{};c.$('lora-bake-controls').insertBefore=()=>{};
 c.$('compose-preset').closest=()=>({style:{}});
 vm.runInNewContext(`${fn('updateH3Visibility')}\nupdateH3Visibility()`,c);
 assert.equal(c.hidden,false);assert.equal(c.$('h3-pick-reference').style.display,'');assert.equal(c.$('h3-pick-target').style.display,'none');
 c.state.workflowMode='h3-adapter-convert';vm.runInNewContext(`${fn('updateH3Visibility')}\nupdateH3Visibility()`,c);
 assert.equal(c.$('h3-pick-reference').style.display,'none');assert.equal(c.$('h3-pick-target').style.display,'');assert.equal(c.$('h3-pick-adapter').style.display,'');
});
test('H3 workflows are wired to actual Start, visibility and picker listeners',()=>{
 assert.match(source, /state\.workflowMode\.startsWith\("h3-"\)[\s\S]*?startH3Workflow\(\)/);
 assert.match(source, /h3-pick-\$\{role\}/);
 assert.match(source, /h3-fold-mode.*addEventListener/);
 assert.match(fn('setWorkflowMode'), /updateH3Visibility\(\)/);
});

test('H3 workflows retain shared source, name, architecture and dry run',()=>{
 const ui = require('../web/ui_helpers.js');
 for (const mode of ['h3-prune','h3-adapter-convert']) {
  const controls = ui.applicableControls(mode);
  for (const field of ['source','architecture','output-name','dry-run']) assert.ok(controls.has(field), `${mode}:${field}`);
 }
});

test('H3 JSON sidecar recipes restore paths and effective compute settings',async()=>{
 const c=context(); c.rememberPickerDirectory=()=>{}; c.selectSource=async p=>{c.state.sourcePath=p};
 const settings={base_path:'/full.safetensors', reference_path:'/ref.safetensors', output_path:'/outputs/fold.safetensors', fold_mode:'independent', merge_device:'cpu',cuda_device:'cuda:1',vram_headroom_mb:2048,dry_run:false};
 await vm.runInNewContext(`${fn('parseRecipeAndApply')}\nparseRecipeAndApply(recipe,'fold.txt')`, {...c, recipe:'DaSiWa H3 Prune Recipe\\n'+JSON.stringify({settings})});
 assert.equal(c.$('h3-fold-mode').value,'independent'); assert.equal(c.$('lora-merge-device').value,'cpu'); assert.equal(c.$('model-name').value,'fold');
});

test('H3 destination uses explicit output directory and restored path',async()=>{
 for (const outputDir of ['/chosen','']) {
  const c=context();c.state.outputDir=outputDir;c.state.config={output_dir:'/configured'};c.$('model-name').value='out';c.$('h3-fold-mode').value='independent';
  let payload;c.api=async(u,o)=>{payload=JSON.parse(o.body);return {job_id:'test'}};
  await vm.runInNewContext(`${fn('startH3Workflow')}\nstartH3Workflow()`,c);
  assert.equal(payload.output_dir,outputDir||'/configured');
 }
 const c=context();c.selectSource=async p=>{c.state.sourcePath=p};
 const recipe='DaSiWa H3 Prune Recipe\n'+JSON.stringify({settings:{base_path:'/full.safetensors',output_path:'/saved/out.safetensors',fold_mode:'independent'}});
 await vm.runInNewContext(`${fn('parseRecipeAndApply')}\nparseRecipeAndApply(recipe,'out.txt')`,{...c,recipe});
 assert.equal(c.state.h3OutputPath,'/saved/out.safetensors');
 let payload;c.api=async(u,o)=>{payload=JSON.parse(o.body);return {job_id:'test'}};
 await vm.runInNewContext(`${fn('startH3Workflow')}\nstartH3Workflow()`,c);
 assert.equal(payload.output_path,'/saved/out.safetensors');
});
test('saved H3 destination survives settings reload',()=>{
 const c=context();c.state.h3OutputPath='/saved/out.safetensors';c.state.h3OutputName='out';c.state.outputDir='/saved';c.$('model-name').value='out';
 const saved=vm.runInNewContext(`${fn('currentSettings')}\ncurrentSettings()`,c);
 assert.equal(saved.h3.outputPath,'/saved/out.safetensors');assert.equal(saved.outputDir,'/saved');
 const restored=context();restored.QuantStationUI=require('../web/ui_helpers.js');restored.localStorage={getItem:()=>JSON.stringify(saved),setItem(){}};restored.SETTINGS_STORAGE_KEY='test';
 for(const name of ['renderModelMergeBlockGrid','updateModelMergeBlockSelection','updateModelMergeModalityVisibility','updateExtractBlockVisibility','updateDeltaOptionsVisibility','updateExtractRecipeVisibility'])restored[name]=()=>{};
 vm.runInNewContext(`${fn('loadSettings')}\nloadSettings()`,restored);
 assert.equal(restored.state.h3OutputPath,'/saved/out.safetensors');assert.equal(restored.state.h3OutputName,'out');assert.equal(restored.state.outputDir,'/saved');
});
test('extraction recipe reload preserves effective payload and exact destination',async()=>{
 const c=context();c.selectSource=async p=>{c.state.sourcePath=p};
 for(const n of ['renderExtractBlocks','updateExtractBlockVisibility','updateExtractRecipeVisibility']) c[n]=()=>{};
 c.scanExtractBlocks=async()=>{c.state.extractScannedPath=c.state.sourcePath};
 const recipe=['DaSiWa Quant Station LoRA Extract Recipe','Recipe: h3_pruned','Architecture: MiniMax H3','Base checkpoint: /base.safetensors','Modified checkpoint: /modified.safetensors','Pruned target: /pruned.safetensors','Output mode: pruned','Output: /saved/extracted.safetensors','Frobenius energy: 0.95','Minimum rank: 2','Maximum rank: 0','Dry run first: no','Selected blocks: blocks.0, blocks.7'].join('\n');
 await vm.runInNewContext(`${fn('parseRecipeAndApply')}\nparseRecipeAndApply(recipe,'extracted.txt')`,{...c,recipe});
 assert.equal(c.state.sourcePath,'/base.safetensors');assert.equal(c.state.extractMergedPath,'/modified.safetensors');assert.equal(c.state.extractPrunedPath,'/pruned.safetensors');
 assert.equal(c.$('extract-mode').value,'h3_pruned');assert.equal(c.$('extract-energy').value,'0.95');assert.equal(c.$('extract-min-rank').value,'2');assert.equal(c.$('extract-max-rank').value,'0');assert.equal(c.$('extract-block-filter').checked,true);assert.deepEqual([...c.state.extractSelectedBlocks],['blocks.0','blocks.7']);
 let payload;c.api=async(u,o)=>{assert.equal(u,'/api/lora/extract');payload=JSON.parse(o.body);return {job_id:'test'}};
 await vm.runInNewContext(`${fn('startLoraExtract')}\nstartLoraExtract()`,c);
 assert.equal(payload.output_path,'/saved/extracted.safetensors');assert.equal(payload.max_rank,0);assert.equal(payload.dry_run,false);assert.deepEqual(payload.selected_blocks,['blocks.0','blocks.7']);
});
test('generic extraction recipe restores no target and unfiltered blocks',async()=>{
 const c=context();c.selectSource=async p=>{c.state.sourcePath=p};c.scanExtractBlocks=async()=>{};
 for(const n of ['renderExtractBlocks','updateExtractBlockVisibility','updateExtractRecipeVisibility'])c[n]=()=>{};
 const recipe='DaSiWa Quant Station LoRA Extract Recipe\nRecipe: generic\nArchitecture: WAN 2.2\nBase checkpoint: /base.safetensors\nModified checkpoint: /modified.safetensors\nPruned target: none\nSelected blocks: all tensors\n';
 await vm.runInNewContext(`${fn('parseRecipeAndApply')}\nparseRecipeAndApply(recipe,'extracted.txt')`,{...c,recipe});
 assert.equal(c.$('extract-mode').value,'generic');assert.equal(c.state.extractPrunedPath,'');assert.equal(c.$('extract-block-filter').checked,false);
});
test('extraction destination persists across application reload',()=>{
 const c=context();c.state.extractOutputPath='/saved/extract.safetensors';c.state.extractOutputName='extract';
 const saved=vm.runInNewContext(`${fn('currentSettings')}\ncurrentSettings()`,c);
 assert.equal(saved.extract.outputPath,'/saved/extract.safetensors');
 const restored=context();restored.QuantStationUI=require('../web/ui_helpers.js');restored.localStorage={getItem:()=>JSON.stringify(saved),setItem(){}};restored.SETTINGS_STORAGE_KEY='test';
 for(const name of ['renderModelMergeBlockGrid','updateModelMergeBlockSelection','updateModelMergeModalityVisibility','updateExtractBlockVisibility','updateDeltaOptionsVisibility','updateExtractRecipeVisibility'])restored[name]=()=>{};
 vm.runInNewContext(`${fn('loadSettings')}\nloadSettings()`,restored);
 assert.equal(restored.state.extractOutputPath,'/saved/extract.safetensors');assert.equal(restored.state.extractOutputName,'extract');
});
test('composition recipe restores selected algorithm',async()=>{
 const c=context();c.renderLoras=()=>{};
 await vm.runInNewContext(`${fn('parseRecipeAndApply')}\nparseRecipeAndApply(recipe,'out.txt')`,{...c,recipe:'DaSiWa LoRA Compose Recipe\nArchitecture: MiniMax H3\nMerge algorithm: additive\n'});
 assert.equal(c.$('compose-merge-algorithm').value,'additive');
});
test('bake recipe restores complete Turbo intent',async()=>{
 const c=context();c.renderLoras=()=>{};c.shortPath=p=>p;
 await vm.runInNewContext(`${fn('parseRecipeAndApply')}\nparseRecipeAndApply(recipe,'out.txt')`,{...c,recipe:'DaSiWa LoRA Merge Recipe\nArchitecture: MiniMax H3\nH3 Turbo complete: yes\nLoRAs\n'});
 assert.equal(c.$('h3-turbo-complete').checked,true);
});

test('H3 controls have distinct workflow modes and tooltips',()=>{
 const html=fs.readFileSync(require('node:path').join(__dirname,'../web/index.html'),'utf8');
 for(const id of ['h3-fold-mode','h3-pick-reference','h3-pick-target','h3-pick-adapter','h3-turbo-complete','h3-quant-policy','quant-verbose','compose-merge-algorithm']) {
  const control=html.match(new RegExp(`<[^>]+id="${id}"[^>]*>`));assert.ok(control, id);assert.ok(control[0].includes('title='),id+' tooltip');
 }
 for (const mode of ['h3-prune','h3-adapter-convert']) assert.ok(html.includes(`data-mode="${mode}"`));
});
