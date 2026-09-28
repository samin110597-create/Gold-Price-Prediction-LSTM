import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
const c=vm.createContext({Date,Number,Object,Error});
vm.runInContext(readFileSync(new URL('../assets/live.js',import.meta.url),'utf8'),c);
const now=Date.now(),iso=x=>new Date(x).toISOString();
const f={research:{origin:iso(now-3600000),reference_price:100,price:102,interval80:[95,105]},outlook:{target_time:iso(now+3600000),recent_mean_absolute_error_dollars:2}};
test('current-price assessment rejects expired and stale forecasts without rewriting targets',()=>{
 const before=JSON.stringify(f);
 assert.match(c.forecastApplicability(f,{price:110,time:iso(now)},now).status,/OUTSIDE/);
 assert.equal(c.forecastApplicability(f,{price:110,time:iso(now)},now).errorRatio,5);
 assert.match(c.forecastApplicability(f,{price:100,time:iso(now-3600000)},now).status,/TOO OLD/);
 assert.match(c.forecastApplicability(f,{price:100,time:iso(now)},now+7200000).status,/EXPIRED/);
 assert.equal(JSON.stringify(f),before);
});
test('current-market publication rejects wrong instrument and future bars',()=>{
 const p={schema_version:1,asof:iso(now),assets:{gold:{symbol:'GC=F',asof:iso(now),quote:{symbol:'GC=F',price:100,time:iso(now)},technical_brief:{rows:[{known_at:iso(now-3600000)}]}}}};
 c.validateCurrentMarket(p);
 p.assets.gold.quote.symbol='GLD';assert.throws(()=>c.validateCurrentMarket(p),/Invalid/);
 p.assets.gold.quote.symbol='GC=F';p.assets.gold.technical_brief.rows[0].known_at=iso(now+1000);assert.throws(()=>c.validateCurrentMarket(p),/Invalid/);
});
