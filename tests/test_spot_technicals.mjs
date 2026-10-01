import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../assets/spot-technicals.js',import.meta.url),'utf8');
const now=Date.parse('2026-09-30T23:05:00Z'),iso=v=>new Date(v).toISOString();
function context(extra={}){const c=vm.createContext({Date,Number,Object,Error,Math,Set,AbortController,setTimeout,clearTimeout,...extra});vm.runInContext(source,c);return c;}
const c=context();
function candles(tf='1h',symbol='XAUUSD',count=100){
 const duration={'15m':900000,'1h':3600000,'4h':14400000,'1d':86400000}[tf],end=Math.floor(now/duration)*duration;
 return {symbol,interval:tf,bars:Array.from({length:count},(_,i)=>({openTime:iso(end-(count-i)*duration),open:100+i,close:100+i,high:101+i,low:99+i,isOpen:false}))};
}
function quote(symbol='XAUUSD',at=now){return {symbol,timestamp:iso(at),source:'MetaTrader 5 (Broker 1)',mid:200,bid:199.9,ask:200.1,marketState:'open',stale:false};}
test('indicator arithmetic matches analytical trends and handles flat prices',()=>{
 const bars=Array.from({length:100},(_,i)=>({close:i+1,open:i+1,high:i+2,low:i}));
 const v=c.spotIndicators(bars);
 assert.ok(Math.abs(v.ema20-90.5)<1e-9);assert.ok(Math.abs(v.ema50-75.5)<1e-9);
 assert.equal(v.rsi,100);assert.equal(v.atr,2);assert.equal(v.adx,100);
 assert.ok(Math.abs(v.macd-7)<1e-9);assert.ok(Math.abs(v.hist)<1e-9);assert.equal(v.ema200,null);
 const f=c.spotIndicators(bars.map(()=>({open:50,high:50,low:50,close:50})));
 assert.equal(f.rsi,50);assert.equal(f.atr,0);assert.equal(f.adx,0);assert.equal(f.bbLower,50);assert.equal(f.bbUpper,50);
});
test('quote validation refuses foreign instruments, bad spreads, future times and regressions',()=>{
 const valid=c.validateTechnicalQuote(quote(),'gold',null,now);
 for(const patch of [{symbol:'GC=F'},{source:'Yahoo Finance'},{bid:201},{mid:205},{timestamp:iso(now+1)}])assert.throws(()=>c.validateTechnicalQuote({...quote(),...patch},'gold',null,now));
 assert.throws(()=>c.validateTechnicalQuote(quote('XAUUSD',now-1),'gold',valid,now));
});
test('candle validation rejects wrong symbol, timeframe, duplicates, bad OHLC and future closes',()=>{
 const good=candles();c.validateTechnicalBars(good,'gold','1h',null,now);
 for(const patch of [{symbol:'GC=F'},{interval:'1d'},{bars:[...good.bars,good.bars[0]]},{bars:good.bars.map((b,i)=>i?b:{...b,high:b.low-1})},{bars:good.bars.map((b,i)=>i?b:{...b,openTime:iso(now)})}])assert.throws(()=>c.validateTechnicalBars({...good,...patch},'gold','1h',null,now));
});
test('a changing current quote updates developing RSI and EMA, not completed readings or bars',()=>{
 const raw=candles();raw.bars.push({openTime:'2026-09-30T23:00:00Z',open:200,high:202,low:198,close:200,isOpen:true});
 const series=c.validateTechnicalBars(raw,'gold','1h',null,now),before=JSON.stringify(series),q=c.validateTechnicalQuote(quote(),'gold',null,now);
 const first=c.buildTechnicalFrame(series,q,now),second=c.buildTechnicalFrame(series,{...q,price:190},now);
 assert.equal(JSON.stringify(first.confirmed),JSON.stringify(second.confirmed));
 assert.notEqual(first.developing.rsi,second.developing.rsi);assert.notEqual(first.developing.ema20,second.developing.ema20);
 assert.equal(JSON.stringify(series),before);assert.equal(first.closed.length,100);
});
test('missing completed candles stay flagged despite a fresh quote',()=>{
 const raw=candles();raw.bars.pop();const series=c.validateTechnicalBars(raw,'gold','1h',null,now);
 const f=c.buildTechnicalFrame(series,c.validateTechnicalQuote(quote(),'gold',null,now),now);assert.equal(f.missingLatest,true);
});
test('a new completed candle changes confirmed readings; older series cannot replace it',()=>{
 const old=c.validateTechnicalBars(candles(),'gold','1h',null,now),raw=candles();
 raw.bars.push({openTime:'2026-09-30T23:00:00Z',open:200,high:211,low:199,close:210,isOpen:false});
 const later=now+3600000,newer=c.validateTechnicalBars(raw,'gold','1h',old,later);
 assert.notEqual(c.spotIndicators(newer.closed).ema20,c.spotIndicators(old.closed).ema20);
 assert.throws(()=>c.validateTechnicalBars(candles(),'gold','1h',newer,later),/Regressed/);
});
test('refresh commits each metal atomically and retains original times after partial failure',async()=>{
 let clock=now,fail=false,calls=0;
 class Clock extends Date{static now(){return clock;}}
 const ctx=context({Date:Clock,fetch:async url=>{calls++;if(url.includes('/latest'))return {ok:true,json:async()=>({XAUUSD:quote(),XAGUSD:quote('XAGUSD')})};const symbol=url.includes('XAUUSD')?'XAUUSD':'XAGUSD',tf=new URL(url).searchParams.get('interval');if(fail&&symbol==='XAUUSD'&&tf==='4h')throw Error('offline');return {ok:true,json:async()=>candles(tf,symbol)};}});
 await ctx.loadSpotTechnicals();assert.equal(calls,9);
 assert.equal(vm.runInContext('Object.keys(spotTechnicalState.gold.frames).length',ctx),4);
 const before=vm.runInContext('JSON.stringify(spotTechnicalState.gold)',ctx);
 await ctx.loadSpotTechnicals();assert.equal(calls,9);
 clock+=900000;fail=true;await ctx.loadSpotTechnicals();assert.equal(calls,18);
 assert.equal(vm.runInContext('JSON.stringify(spotTechnicalState.gold)',ctx),before);
 assert.ok(vm.runInContext('spotTechnicalErrors.gold',ctx));
 assert.equal(ctx.technicalQuote('gold',clock),null);
});
test('automatic price refresh uses fifteen minutes, not thirty seconds',()=>{
 const intervals=[],doc={hidden:false,addEventListener(){}};
 const ctx=vm.createContext({Date,Number,Object,Error,document:doc,setInterval:(fn,ms)=>intervals.push(ms)});
 vm.runInContext(readFileSync(new URL('../assets/live.js',import.meta.url),'utf8'),ctx);
 vm.runInContext('loadCurrentMarket=()=>{};loadSpotQuotes=()=>{};startCurrentMarket()',ctx);
 assert.ok(intervals.includes(900000));assert.ok(!intervals.includes(30000));assert.ok(!intervals.includes(60000));
});
