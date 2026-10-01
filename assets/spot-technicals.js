'use strict';
// One provider/instrument per technical bundle; no futures-to-spot conversions.
const technicalFrames={'15m':900000,'1h':3600000,'4h':14400000,'1d':86400000};
let spotTechnicalState={gold:null,silver:null},spotTechnicalErrors={},spotTechnicalBusy=false,spotTechnicalAttempt=null,spotTechnicalFrame='1h';
function validateTechnicalQuote(raw,key,previous,now=Date.now()){
 const symbol={gold:'XAUUSD',silver:'XAGUSD'}[key],stamp=Date.parse(raw?.timestamp);
 if(!symbol||raw?.symbol!==symbol||!String(raw.source).startsWith('MetaTrader 5')||!['open','closed'].includes(raw.marketState)||typeof raw.stale!=='boolean'||!Number.isFinite(stamp)||stamp>now||!Number.isFinite(raw.bid)||raw.bid<=0||!Number.isFinite(raw.ask)||raw.ask<raw.bid||!Number.isFinite(raw.mid)||Math.abs(raw.mid-(raw.ask+raw.bid)/2)>Math.max(.000001,raw.mid*1e-8))throw Error('Invalid broker quote');
 if(previous&&stamp<Date.parse(previous.time))throw Error('Regressed broker quote');
 return {price:raw.mid,time:raw.timestamp,symbol,source:'biquote / '+raw.source,basis:'biquote broker midpoint · USD per troy ounce',marketState:raw.marketState,providerStale:raw.stale};
}
function validateTechnicalBars(raw,key,tf,previous,now=Date.now()){
 const symbol={gold:'XAUUSD',silver:'XAGUSD'}[key],duration=technicalFrames[tf];
 if(!duration||raw?.symbol!==symbol||raw.interval!==tf||!Array.isArray(raw.bars))throw Error('Wrong candle series');
 const seen=new Set(),bars=raw.bars.map(b=>{
  const at=Date.parse(b.openTime);
  if(!Number.isFinite(at)||at>now||seen.has(at)||typeof b.isOpen!=='boolean'||['open','high','low','close'].some(k=>!Number.isFinite(b[k])||b[k]<=0)||b.high<Math.max(b.open,b.close,b.low)||b.low>Math.min(b.open,b.close,b.high)||(!b.isOpen&&at+duration>now))throw Error('Invalid candle');
  seen.add(at);return {time:b.openTime,at,end:at+duration,open:b.open,high:b.high,low:b.low,close:b.close,forming:b.isOpen};
 }).sort((a,b)=>a.at-b.at);
 const closed=bars.filter(b=>!b.forming),open=bars.filter(b=>b.forming);
 if(closed.length<60||open.length>1||(open.length&&open[0].at<=closed.at(-1).at))throw Error('Insufficient or overlapping candles');
 if(previous&&closed.at(-1).at<previous.closed.at(-1).at)throw Error('Regressed candle series');
 return {closed,open:open[0]||null,duration};
}
function spotEma(values,period){
 const out=Array(values.length).fill(null);if(values.length<period)return out;
 let v=values.slice(0,period).reduce((a,b)=>a+b,0)/period;out[period-1]=v;
 for(let i=period;i<values.length;i++){v+=(values[i]-v)*2/(period+1);out[i]=v;}return out;
}
function spotWilder(values,period){
 if(values.length<period)return null;
 let v=values.slice(0,period).reduce((a,b)=>a+b,0)/period;
 for(const x of values.slice(period))v=(v*(period-1)+x)/period;return v;
}
function spotIndicators(bars){
 if(bars.length<60)throw Error('Insufficient indicator history');
 const c=bars.map(b=>b.close),last=c.at(-1),delta=c.slice(1).map((v,i)=>v-c[i]);
 const gain=spotWilder(delta.map(v=>Math.max(0,v)),14),loss=spotWilder(delta.map(v=>Math.max(0,-v)),14);
 const rsi=loss===0?(gain===0?50:100):100-100/(1+gain/loss);
 const tr=[],plus=[],minus=[];
 for(let i=1;i<bars.length;i++){
  const a=bars[i],b=bars[i-1],up=a.high-b.high,down=b.low-a.low;
  tr.push(Math.max(a.high-a.low,Math.abs(a.high-b.close),Math.abs(a.low-b.close)));
  plus.push(up>down&&up>0?up:0);minus.push(down>up&&down>0?down:0);
 }
 const atr=spotWilder(tr,14),plusDI=atr?100*spotWilder(plus,14)/atr:0,minusDI=atr?100*spotWilder(minus,14)/atr:0;
 let at=tr.slice(0,14).reduce((a,b)=>a+b,0)/14,p=plus.slice(0,14).reduce((a,b)=>a+b,0)/14,m=minus.slice(0,14).reduce((a,b)=>a+b,0)/14;const dx=[];
 for(let i=13;i<tr.length;i++){
  if(i>13){at=(at*13+tr[i])/14;p=(p*13+plus[i])/14;m=(m*13+minus[i])/14;}
  dx.push(p+m?100*Math.abs(p-m)/(p+m):0);
 }
 const adx=spotWilder(dx,14),ema20=spotEma(c,20).at(-1),ema50=spotEma(c,50).at(-1),ema200=spotEma(c,200).at(-1);
 const fast=spotEma(c,12),slow=spotEma(c,26),macds=c.map((_,i)=>slow[i]===null?null:fast[i]-slow[i]).filter(v=>v!==null),macd=macds.at(-1),signal=spotEma(macds,9).at(-1);
 const tail=c.slice(-20),sma20=tail.reduce((a,b)=>a+b,0)/20,std=Math.sqrt(tail.reduce((a,b)=>a+(b-sma20)**2,0)/20);
 const range=bars.slice(-20),support=Math.min(...range.map(b=>b.low)),resistance=Math.max(...range.map(b=>b.high));
 const prior=bars.slice(-21,-1),priorLow=Math.min(...prior.map(b=>b.low)),priorHigh=Math.max(...prior.map(b=>b.high));
 return {close:last,rsi,ema20,ema50,ema200,macd,signal,hist:macd-signal,atr,adx,plusDI,minusDI,bbLower:sma20-2*std,bbUpper:sma20+2*std,support,resistance,
  trend:last>ema20&&ema20>ema50?'Bullish':last<ema20&&ema20<ema50?'Bearish':'Mixed',
  breakout:last>priorHigh?'Closed above prior 20-bar high':last<priorLow?'Closed below prior 20-bar low':'Inside prior 20-bar range'};
}
function buildTechnicalFrame(series,q,now=Date.now()){
 const last=series.closed.at(-1),confirmed=spotIndicators(series.closed);
 const canDevelop=series.open&&series.open.at<=Date.parse(q.time)&&Date.parse(q.time)<series.open.end&&now<series.open.end&&!q.providerStale;
 const forming=canDevelop?{...series.open,close:q.price,high:Math.max(series.open.high,q.price),low:Math.min(series.open.low,q.price)}:null;
 const developing=forming?spotIndicators([...series.closed,forming]):null;
 // A fresh tick cannot certify a missing completed candle.
 const reference=Math.min(now,Date.parse(q.time));
 const missingLatest=reference-last.end>series.duration+120000;
 return {...series,forming,confirmed,developing,lastCompleted:new Date(last.end).toISOString(),missingLatest};
}
async function technicalJson(url){
 const controller=new AbortController(),deadline=setTimeout(()=>controller.abort(),12000);
 try{const response=await fetch(url,{signal:controller.signal,credentials:'omit',referrerPolicy:'no-referrer'});if(!response.ok)throw Error('Candle feed unavailable');return await response.json();}
 finally{clearTimeout(deadline);}
}
async function loadSpotTechnicals(){
 if(spotTechnicalBusy||(spotTechnicalAttempt!==null&&Date.now()-spotTechnicalAttempt<900000))return;
 spotTechnicalBusy=true;spotTechnicalAttempt=Date.now();
 try{
  // Fetch bars first, then the matching provider's quote; commit each metal atomically.
  const rows=await Promise.all(['gold','silver'].map(async key=>{
   const symbol=key==='gold'?'XAUUSD':'XAGUSD';
   const entries=await Promise.all(Object.keys(technicalFrames).map(async tf=>{
    try{return [tf,validateTechnicalBars(await technicalJson('https://biquote.io/api/'+symbol+'/ohlc?interval='+tf+'&limit=300'),key,tf,spotTechnicalState[key]?.frames[tf])];}
    catch{return [tf,null];}
   }));return [key,Object.fromEntries(entries)];
  }));
  const quotes=await technicalJson('https://biquote.io/api/latest?symbols=XAUUSD&symbols=XAGUSD');
  for(const [key,frames] of rows){
   try{
    const symbol=key==='gold'?'XAUUSD':'XAGUSD',q=validateTechnicalQuote(quotes[symbol],key,spotTechnicalState[key]?.quote);
    if(Object.values(frames).some(v=>!v))throw Error('Incomplete bundle');
    const asof=new Date().toISOString(),next={quote:q,asof,frames:Object.fromEntries(Object.entries(frames).map(([tf,s])=>[tf,buildTechnicalFrame(s,q)]))};
    spotTechnicalState[key]=next;delete spotTechnicalErrors[key];
   }catch{spotTechnicalErrors[key]='Price/candle refresh failed validation. Previous technicals retain their original time.';}
  }
 }catch{for(const key of ['gold','silver'])spotTechnicalErrors[key]='Price/candle feed unavailable. Previous technicals retain their original time.';}
 finally{spotTechnicalBusy=false;}
}
function technicalQuote(key,now=Date.now()){
 const state=spotTechnicalState[key];if(!state||spotTechnicalErrors[key])return null;
 const q=state.quote;if(q.providerStale||now-Date.parse(q.time)>1200000)return null;
 return {...q,stale:false,retained:false};
}
function technicalPanel(){
 const state=spotTechnicalState[asset],q=state?.quote,frame=state?.frames[spotTechnicalFrame];
 const bad=!state||spotTechnicalErrors[asset]||q.providerStale||Date.now()-Date.parse(q.time)>1200000||Date.now()-Date.parse(state.asof)>1200000;
 const status=!state?'Loading price and candles…':bad?'STALE / UNAVAILABLE — WAIT':frame.missingLatest?'LATEST COMPLETED BAR MISSING':'PRICE + TECHNICALS · SAME REFRESH';
 const header=`<section class="panel" id="spot-technicals" style="margin-bottom:16px"><div class="panel-head"><h2>Live metal technicals · refreshed every 15 minutes</h2><span class="badge wait" id="technical-status">${escape(status)}</span></div>`;
 if(!state)return header+`<p>${escape(spotTechnicalErrors[asset]||'Fetching matching XAU/USD or XAG/USD prices and candles…')}</p><p class="footnote">No futures history is substituted into these technicals.</p></section>`;
 const c=frame.confirmed,d=frame.developing,keys=[['RSI (14)','rsi'],['EMA20','ema20'],['EMA50','ema50'],['EMA200','ema200'],['MACD (12,26)','macd'],['MACD signal (9)','signal'],['MACD histogram','hist'],['ATR (14)','atr'],['ADX (14)','adx'],['Bollinger lower (20,2)','bbLower'],['Bollinger upper (20,2)','bbUpper']];
 return header+`<p>${escape(q.symbol)} · ${escape(q.source)} · broker midpoint $${n(q.price)} · Quote ${time(q.time)} · Calculated ${time(state.asof)}. <strong>${escape(spotTechnicalErrors[asset]||'')}</strong></p><div class="controls" aria-label="Live technical timeframe">${Object.keys(technicalFrames).map(tf=>`<button data-spot-tf="${tf}" aria-pressed="${tf===spotTechnicalFrame}">${tf.toUpperCase()}</button>`).join('')}</div><p style="margin-top:12px"><strong>${escape(c.trend)} on completed candles</strong> · RSI ${n(c.rsi,1)} · ADX ${n(c.adx,1)} · ${escape(c.breakout)}. ${bad||frame.missingLatest?'Wait for current, complete data.':d?`Developing candle: ${escape(d.trend)}, RSI ${n(d.rsi,1)} — not confirmed.`:'No eligible forming candle.'}</p><div class="chart-wrap" style="height:280px"><canvas id="spot-chart" role="img" aria-label="${escape(q.symbol)} ${spotTechnicalFrame} broker price candles"></canvas></div><p class="footnote">Completed candles only · EMA20 gold / EMA50 blue · Hover for OHLC and Eastern time. ${frame.closed.length} completed bars; ${time(frame.lastCompleted)} is the last close.</p><div class="table-wrap"><table><thead><tr><th>Technical</th><th>Last completed ${spotTechnicalFrame.toUpperCase()}</th><th>Developing candle · can change</th></tr></thead><tbody>${keys.map(([name,k])=>`<tr><td>${name}</td><td>${n(c[k])}</td><td>${d?n(d[k]):'—'}</td></tr>`).join('')}</tbody></table></div><p>20-bar support $${n(c.support)} · resistance $${n(c.resistance)}. These are observed price extremes, not promised targets. ATR measures typical range; ADX measures strength, not direction.</p><div class="table-wrap"><table><thead><tr><th>Frame</th><th>Closed-bar trend</th><th>RSI</th><th>MACD histogram</th><th>ATR</th><th>Last completed (ET)</th><th>Bar check</th></tr></thead><tbody>${Object.entries(state.frames).map(([tf,f])=>`<tr data-live-technical-frame="${tf}"><td>${tf.toUpperCase()}</td><td>${escape(f.confirmed.trend)}</td><td>${n(f.confirmed.rsi,1)}</td><td>${n(f.confirmed.hist)}</td><td>${n(f.confirmed.atr)}</td><td>${time(f.lastCompleted)}</td><td>${f.missingLatest?'GAP — WAIT':'Available at refresh'}</td></tr>`).join('')}</tbody></table></div><details><summary>How to use these live technicals</summary><p>Confirmed values use completed candles. A 1H candle closes hourly even though data refreshes every 15 minutes. Developing values use the forming candle and the matching broker quote; they can change before the close and do not confirm entries.</p><p>The candle provider labels these series XAUUSD/XAGUSD and documents MT5 history, tick aggregation and historical bootstrap sources. These are indicative broker prices, not a verified exchange tape. Daily boundaries follow the provider (UTC); times are displayed in Eastern time. Missing EMA200 means fewer than 200 bars. Trade volume is unavailable, so volume indicators and VWAP are not invented. These technicals do not validate the separate futures forecasts.</p><a href="https://biquote.io/docs/" target="_blank" rel="noopener noreferrer">Candle and quote source documentation</a></details></section>`;
}
function renderSpotTechnicals(){
 const panel=document.querySelector('#spot-technicals');if(panel)panel.outerHTML=technicalPanel();bindSpotTechnicals();
}
function bindSpotTechnicals(){
 document.querySelectorAll('[data-spot-tf]').forEach(b=>b.onclick=()=>{spotTechnicalFrame=b.dataset.spotTf;renderSpotTechnicals();});drawSpotTechnicals();
}
function drawSpotTechnicals(){
 const canvas=document.querySelector('#spot-chart'),f=spotTechnicalState[asset]?.frames[spotTechnicalFrame];if(!canvas||!f)return;
 const rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;canvas.width=rect.width*ratio;canvas.height=rect.height*ratio;
 const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);const bars=f.closed.slice(-90),offset=f.closed.length-bars.length,ema20=spotEma(f.closed.map(b=>b.close),20).slice(offset),ema50=spotEma(f.closed.map(b=>b.close),50).slice(offset);
 const w=rect.width,h=rect.height,lo=Math.min(...bars.map(b=>b.low)),hi=Math.max(...bars.map(b=>b.high)),pad=(hi-lo)*.08||1,dx=(w-75)/bars.length,y=v=>12+(hi+pad-v)/(hi-lo+2*pad)*(h-40),x=i=>6+(i+.5)*dx;
 ctx.strokeStyle='#293647';ctx.fillStyle='#9eafc3';ctx.font='10px system-ui';for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;ctx.beginPath();ctx.moveTo(0,y(v));ctx.lineTo(w-65,y(v));ctx.stroke();ctx.fillText(n(v),w-63,y(v)+3);}
 bars.forEach((b,i)=>{ctx.fillStyle=ctx.strokeStyle=b.close>=b.open?'#6ed4b8':'#f18c94';ctx.beginPath();ctx.moveTo(x(i),y(b.high));ctx.lineTo(x(i),y(b.low));ctx.stroke();ctx.fillRect(x(i)-dx*.3,Math.min(y(b.open),y(b.close)),Math.max(1,dx*.6),Math.max(1,Math.abs(y(b.close)-y(b.open))));});
 for(const [values,color] of [[ema20,'#ecc777'],[ema50,'#97acde']]){ctx.strokeStyle=color;ctx.beginPath();let started=false;values.forEach((v,i)=>{if(v===null)return;if(!started){ctx.moveTo(x(i),y(v));started=true;}else ctx.lineTo(x(i),y(v));});ctx.stroke();}
 canvas.onpointermove=ev=>{const i=Math.max(0,Math.min(bars.length-1,Math.floor((ev.clientX-canvas.getBoundingClientRect().left-6)/dx))),b=bars[i];canvas.title=time(b.time)+' · O '+n(b.open)+' H '+n(b.high)+' L '+n(b.low)+' C '+n(b.close);};
}
