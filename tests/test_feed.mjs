import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {webcrypto,createHash} from 'node:crypto';

const source=readFileSync(new URL('../assets/feed.js',import.meta.url),'utf8');
function bundle(id='one',asof=new Date().toISOString()){
 const metal={run_id:id,asof,quote:{price:100,time:asof},timeframes:Object.fromEntries(['15m','1h','4h','1d','1w'].map(tf=>[tf,{last_completed:asof,chart:[{time:asof,close:100}]}]))};
 const payload={schema_version:1,run_id:id,asof,assets:{gold:metal,silver:metal}};
 const raw=JSON.stringify(payload),manifest={run_id:id,asof,files:{'dashboard.json':createHash('sha256').update(raw).digest('hex')}};
 return {raw,manifest};
}
function harness(){
 const elements=new Map(),intervals=[],requests=[];
 let current=bundle(),broken=false,badHash=false;
 const context=vm.createContext({console,Date,Math,Number,String,JSON,Array,Uint8Array,TextEncoder,AbortController,crypto:webcrypto,setTimeout,clearTimeout,setInterval:(fn,ms)=>intervals.push({fn,ms}),document:{hidden:false,addEventListener(){}},data:undefined,asset:'gold',window:{},render(){},time:x=>x,$:key=>{if(!elements.has(key))elements.set(key,{textContent:'',disabled:false,classList:{add(){}}});return elements.get(key);},fetch:async url=>{requests.push(url);if(broken)throw Error('offline');return {ok:true,json:async()=>badHash?{...current.manifest,files:{'dashboard.json':'bad'}}:current.manifest,text:async()=>current.raw};}});
 vm.runInContext(source,context);
 return {context,elements,intervals,requests,set:(v)=>current=v,fail:()=>broken=true,corrupt:()=>badHash=true,run:code=>vm.runInContext(code,context)};
}
test('verifies source hash, refreshes new publications and rejects older data',async()=>{
 const h=harness();await h.run('loadMarketSnapshot()');assert.equal(h.context.data.run_id,'one');
 const newer=new Date(Date.now()+1000).toISOString();h.set(bundle('two',newer));await h.run('loadMarketSnapshot()');assert.equal(h.context.data.run_id,'two');
 h.set(bundle('old','2020-01-01T00:00:00Z'));await h.run('loadMarketSnapshot()');assert.equal(h.context.data.run_id,'two');assert.match(h.run('feedError'),/Older publication/);
});
test('mismatched publication and network outage preserve the last verified snapshot',async()=>{
 const h=harness();await h.run('loadMarketSnapshot()');h.set(bundle('changed'));h.corrupt();await h.run('loadMarketSnapshot()');assert.equal(h.context.data.run_id,'one');assert.match(h.run('feedError'),/checksum mismatch/);
 h.fail();await h.run('loadMarketSnapshot()');assert.equal(h.context.data.run_id,'one');assert.match(h.run('feedError'),/offline/);
});
test('polling runs each minute and unchanged runs avoid another large download',async()=>{
 const h=harness();await h.run('loadMarketSnapshot()');const before=h.requests.length;await h.run('loadMarketSnapshot()');assert.equal(h.requests.length-before,1);
 h.run('startMarketFeed()');assert.ok(h.intervals.some(i=>i.ms===60000));assert.ok(h.intervals.some(i=>i.ms===1000));
});
test('old snapshots pause current eligibility and invalid quote timestamps are rejected',async()=>{
 const h=harness();h.set(bundle('stale','2020-01-01T00:00:00Z'));await h.run('loadMarketSnapshot()');assert.equal(h.run('snapshotIsStale()'),true);
 const b=bundle('future');const p=JSON.parse(b.raw);p.assets.gold.quote.time='2099-01-01T00:00:00Z';assert.throws(()=>h.context.validateSnapshot(p,b.manifest),/Invalid gold/);
});
