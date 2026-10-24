import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../assets/outlook.js',import.meta.url),'utf8');
const payload=JSON.parse(readFileSync(new URL('../data/dashboard.json',import.meta.url),'utf8'));
function render(stale=false,expired=false){
 const a=structuredClone(payload.assets.gold);
 a.market_brief={summary:'Downward pressure',bias:'BEARISH',asof:a.asof,horizon:'1–5 sessions',headwinds:[],tailwinds:[],factors:[],reports:[{topic:'Report',status:'DATED REPORT',review_after:expired?'2000-01-01':'2099-01-01',finding:'<img src=x onerror=alert(1)>',meaning:'Interpretation',url:'https://www.gold.org/',publisher:'WGC'}],limitations:[]};
 const c=vm.createContext({a,stale,Date,Number,Object,data:payload,n:x=>Number.isFinite(x)?String(x):'—',time:x=>String(x),pct:x=>String(x),escape:x=>String(x??'').replaceAll('<','&lt;').replaceAll('>','&gt;'),badge:x=>String(x)});
 vm.runInContext(source,c);
 return vm.runInContext('outlookPanel(a,stale)',c);
}
test('renders all four numerical projections with unvalidated labels and futures basis',()=>{
 const s=render();assert.equal((s.match(/data-outlook-horizon=/g)||[]).length,4);
 assert.match(s,/EXPERIMENTAL|ELAPSED/);assert.match(s,/not spot targets/);assert.match(s,/Measured recent range coverage/);
 assert.ok(!s.includes('<img'));assert.ok(s.includes('&lt;img'));
});
test('stale outlook and expired reports cannot masquerade as current',()=>{
 const s=render(true,true);assert.match(s,/STALE \/ INCOMPLETE/);assert.match(s,/REVIEW DUE/);assert.match(s,/STALE · PRIOR ESTIMATE/);
});
