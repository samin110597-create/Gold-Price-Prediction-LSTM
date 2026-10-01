import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
const code=readFileSync(new URL('../assets/live.js',import.meta.url),'utf8');
const now=Date.now(),iso=n=>new Date(n).toISOString();
const raw=(symbol='XAU',stamp=now)=>({symbol,currency:'USD',price:symbol==='XAU'?4100:60,updatedAt:iso(stamp)});
function context(extra={}){return vm.createContext({Date,Number,Object,Error,AbortController,setTimeout,clearTimeout,...extra});}
const c=context();vm.runInContext(code,c);
test('direct spot rejects currency, symbol, invalid value, missing/future and regressed timestamps',()=>{
 const q=c.validateDirectSpot(raw(),'gold',null,now);
 for(const patch of [{symbol:'XAG'},{currency:'CAD'},{price:'4100'},{price:0},{price:NaN},{updatedAt:null},{updatedAt:iso(now+1)}])assert.throws(()=>c.validateDirectSpot({...raw(),...patch},'gold',null,now));
 assert.throws(()=>c.validateDirectSpot(raw('XAU',now-1),'gold',q,now),/Older/);
 assert.equal(c.validateDirectSpot(raw('XAG'),'silver',null,now).symbol,'XAG');
});
test('freshest spot wins; futures cannot displace it; it ages without new requests',()=>{
 const gold=c.validateDirectSpot(raw(),'gold',null,now);
 const old={...gold,symbol:'GOLD',source_time:iso(now-3600000)};
 const p={gold_api:{observations:[gold]},alpha_vantage:{observations:[old]},fmp:{observations:[{...gold,price:4200}]}};
 assert.equal(c.selectSpotQuote(p,'gold',now).source,'Gold API');
 assert.equal(c.selectSpotQuote(p,'gold',now+1200000).stale,false);
 assert.equal(c.selectSpotQuote(p,'gold',now+1200001).stale,true);
 p.alpha_vantage.observations=[{...old,source_time:iso(now+1)}];
 assert.equal(c.selectSpotQuote(p,'gold',now+1).source,'Alpha Vantage');
});
test('independent metal failures retain source time and request budget throttles clicks',async()=>{
 let clock=now,calls=0,fail=false;
 class Clock extends Date{static now(){return clock;}}
 const ctx=context({Date:Clock,data:null,fetch:async url=>{calls++;if(fail)throw Error('offline');return {ok:true,json:async()=>raw(url.endsWith('XAU')?'XAU':'XAG')};}});
 vm.runInContext(code,ctx);
 await ctx.loadSpotQuotes();assert.equal(calls,2);
 await ctx.loadSpotQuotes();assert.equal(calls,2);
 const first=vm.runInContext('JSON.stringify(directSpot.observations)',ctx);
 clock+=900000;fail=true;await ctx.loadSpotQuotes();assert.equal(calls,4);
 assert.equal(vm.runInContext('JSON.stringify(directSpot.observations)',ctx),first);
 assert.equal(vm.runInContext('Object.keys(spotErrors).length',ctx),2);
 clock+=900000;fail=false;await ctx.loadSpotQuotes();assert.equal(calls,6);
 assert.equal(vm.runInContext('Object.keys(spotErrors).length',ctx),0);
});
test('one invalid metal response does not discard the other metal',async()=>{
 const ctx=context({data:null,fetch:async url=>({ok:true,json:async()=>raw(url.endsWith('XAU')?'XAU':'BAD')})});vm.runInContext(code,ctx);
 await ctx.loadSpotQuotes();
 assert.equal(vm.runInContext('directSpot.observations.length',ctx),1);
 assert.equal(vm.runInContext('directSpot.observations[0].asset',ctx),'gold');
 assert.equal(vm.runInContext('Object.keys(spotErrors)[0]',ctx),'silver');
});
