import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../assets/current-outlook.js',import.meta.url),'utf8');
const now=Date.parse('2026-10-26T14:30:00Z');
function setup(){
 const f={missingLatest:false,lastCompleted:'2026-10-26T14:00:00Z',confirmed:{trend:'Bearish',support:98,resistance:105,ema20:102,atr:2},closed:Array.from({length:20},(_,i)=>({high:104+i%3,low:96+i%3}))};
 const bundle={asof:new Date(now).toISOString(),quote:{price:100,time:new Date(now).toISOString(),marketState:'open',providerStale:false},frames:{'1h':structuredClone(f),'4h':structuredClone(f),'1d':structuredClone(f)}};
 const c=vm.createContext({Date,Math,Number,asset:'gold',spotTechnicalState:{gold:bundle},spotTechnicalErrors:{},spotTechnicalBusy:false});vm.runInContext(source,c);return {c,bundle};
}
test('current outlook needs no scheduled snapshot and derives same-feed levels on the correct side',()=>{
 const {c,bundle}=setup(),before=JSON.stringify(bundle),r=c.spotDecision('gold',now);
 assert.equal(r.usable,true);assert.equal(r.bias,'DOWN');assert.ok(r.next<100);
 assert.ok(r.support.every(x=>x<100));assert.ok(r.resistance.every(x=>x>100));assert.equal(JSON.stringify(bundle),before);
 bundle.quote.price=97;assert.notEqual(c.spotDecision('gold',now).next,r.next);
});
test('stale quotes, closed markets, failed refresh and missing closed bars cannot imply a current prediction',()=>{
 const {c,bundle}=setup();assert.equal(c.spotDecision('gold',now+900001).usable,false);
 bundle.quote.marketState='closed';assert.match(c.spotDecision('gold',now).status,/CLOSED/);bundle.quote.marketState='open';
 bundle.frames['4h'].missingLatest=true;assert.equal(c.spotDecision('gold',now).usable,false);bundle.frames['4h'].missingLatest=false;
 c.spotTechnicalErrors.gold='offline';assert.equal(c.spotDecision('gold',now).usable,false);
});
test('conflicting timeframes are two-sided, and an out-of-range price does not create invented objectives',()=>{
 const {c,bundle}=setup();bundle.frames['4h'].confirmed.trend='Bullish';assert.equal(c.spotDecision('gold',now).bias,'TWO-SIDED');
 bundle.frames['4h'].confirmed.trend='Bearish';bundle.quote.price=90;
 const r=c.spotDecision('gold',now);assert.equal(r.next,undefined);assert.equal(r.support.length,0);assert.equal(r.beyond,'BELOW');
});
test('manual refresh bypasses automatic fifteen-minute delay but duplicate clicks are throttled',async()=>{
 let clock=now,calls=0,forced=[];
 class Clock extends Date {static now(){return clock;}}
 const c=vm.createContext({Date:Clock,Number,Object,Error,AbortController,setTimeout,clearTimeout,data:null,loadSpotTechnicals:async force=>forced.push(force),fetch:async url=>{calls++;return {ok:true,json:async()=>({symbol:url.endsWith('XAU')?'XAU':'XAG',currency:'USD',price:100,updatedAt:new Date(clock).toISOString()})};}});
 vm.runInContext(readFileSync(new URL('../assets/live.js',import.meta.url),'utf8'),c);
 await c.loadSpotQuotes();await c.loadSpotQuotes(true);assert.equal(calls,2);
 clock+=5001;await c.loadSpotQuotes(true);assert.equal(calls,4);assert.deepEqual(forced,[false,true]);
 await c.loadSpotQuotes();assert.equal(calls,4);
});
