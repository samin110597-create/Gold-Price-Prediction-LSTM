'use strict';
let lastOutlookState='';
function spotDecision(key,now=Date.now()){
 const s=spotTechnicalState[key],q=s?.quote;
 if(!s)return {status:spotTechnicalBusy?'FETCHING LATEST DATA':'FEED UNAVAILABLE',usable:false};
 if(spotTechnicalErrors[key])return {status:'REFRESH FAILED',usable:false,quote:q};
 if(q.marketState==='closed')return {status:'MARKET CLOSED · LAST SESSION',usable:false,quote:q};
 if(q.providerStale||now-Date.parse(q.time)>900000||now-Date.parse(s.asof)>900000)return {status:'SOURCE HAS NOT PROVIDED A CURRENT QUOTE',usable:false,quote:q};
 const frames=['1h','4h'].map(t=>s.frames[t]);
 if(frames.some(f=>!f||f.missingLatest))return {status:'LATEST COMPLETED CANDLE MISSING',usable:false,quote:q};
 const trends=frames.map(f=>f.confirmed.trend),bias=trends.every(t=>t==='Bearish')?'DOWN':trends.every(t=>t==='Bullish')?'UP':'TWO-SIDED';
 // Confirmed pivots require two later CLOSED bars; forming candles never set levels.
 const levels=[];
 for(const tf of ['1h','4h','1d']){
  const f=s.frames[tf];if(!f||f.missingLatest)continue;
  const bars=f.closed.slice(-150);
  for(let i=2;i<bars.length-2;i++){
   const nearby=bars.slice(i-2,i+3).filter((_,j)=>j!==2),b=bars[i];
   if(nearby.every(v=>b.high>v.high))levels.push(b.high);
   if(nearby.every(v=>b.low<v.low))levels.push(b.low);
  }
  // Observed range boundaries also remain valid levels when a trend lacks pivots.
  levels.push(f.confirmed.support,f.confirmed.resistance);
 }
 const tolerance=Math.max(q.price*.00005,frames[0].confirmed.atr*.15);
 const distinct=values=>values.reduce((out,v)=>{if(!out.some(x=>Math.abs(x-v)<tolerance))out.push(v);return out;},[]).slice(0,2);
 const support=distinct(levels.filter(v=>Number.isFinite(v)&&v<q.price-tolerance).sort((a,b)=>b-a));
 const resistance=distinct(levels.filter(v=>Number.isFinite(v)&&v>q.price+tolerance).sort((a,b)=>a-b));
 const beyond=q.price>frames[0].confirmed.resistance?'ABOVE':q.price<frames[0].confirmed.support?'BELOW':null;
 return {status:'LATEST SPOT PRICE + CANDLES',usable:true,quote:q,checked:s.asof,bias,support,resistance,beyond,
  next:bias==='DOWN'?support[0]:bias==='UP'?resistance[0]:null,
  hourlyClose:frames[0].lastCompleted,fourHourClose:frames[1].lastCompleted,
  extended:Math.abs(q.price-frames[0].confirmed.ema20)>2*frames[0].confirmed.atr};
}
function currentOutlookPanel(){
 const d=spotDecision(asset),money=v=>Number.isFinite(v)?'$'+n(v,2):'No further observed level',q=d.quote;
 const header=`<section class="panel" id="current-outlook" style="margin:16px 0"><div class="panel-head"><h2>${asset==='gold'?'Gold':'Silver'} · current spot outlook</h2>${badge(d.status,d.usable?'up':'wait')}</div><button id="refresh-outlook" type="button">Refresh latest data now</button>`;
 if(!d.usable)return header+`<p class="outlook-summary" style="margin-top:16px">${d.status==='MARKET CLOSED · LAST SESSION'?'The market is closed. The last session price is available; there is no new trading price yet.':spotBusy?'Requesting the latest quote and matching candles now…':'The source has not supplied a usable current price-and-candle bundle. Press refresh to request it again.'}</p>${q?`<p>Last source quote ${money(q.price)} · ${time(q.time)}.</p>`:''}<p>Current direction and targets are withheld while this check fails. Older futures estimates are in the separate research section below.</p></section>`;
 const direction=d.bias==='DOWN'?`Downward bias. ${d.next?'The next observed downside level is '+money(d.next)+'.':'Price is below the available support map; a lower target cannot be justified from this history.'}`:d.bias==='UP'?`Upward bias. ${d.next?'The next observed upside level is '+money(d.next)+'.':'Price is above the available resistance map; a higher target cannot be justified from this history.'}`:'No clear direction: the 1H and 4H trends disagree or are mixed.';
 const watch=d.bias==='DOWN'?'Wait for a completed 1H close back above resistance and a retest that holds before treating a bounce as a recovery.':d.bias==='UP'?'Watch for a pullback that holds support; a completed 1H close below it would weaken the upside view.':'Wait for a completed 1H break of support or resistance and a retest; price inside the range gives no confirmed breakout.';
 return header+`<p class="outlook-summary" data-current-direction style="margin-top:16px">${escape(direction)}</p><p><strong>${money(q.price)} · ${escape(q.symbol)} spot broker midpoint</strong><br>Source quote ${time(q.time)} · checked ${time(d.checked)}<br><small>Latest completed 1H ${time(d.hourlyClose)} · 4H ${time(d.fourHourClose)}. A fresh request does not change when a candle closes.</small></p><div class="path-grid"><article><h3>Where next?</h3><p>Downside levels: <strong>${d.support.map(money).join(' → ')||'No confirmed lower level'}</strong></p><p>Upside levels: <strong>${d.resistance.map(money).join(' → ')||'No confirmed higher level'}</strong></p><small>Nearest observed levels from the same spot feed, ordered from the current price. These are possible tests, not promised destinations.</small></article><article><h3>What to wait for</h3><p>${escape(watch)}</p><p>${d.extended?'Price is stretched more than two hourly ATRs from its EMA20. Chasing carries pullback risk.':'Use a completed candle and a retest; a moving quote alone is not confirmation.'}</p></article><article><h3>What would change this view?</h3><p>${d.bias==='DOWN'?(d.resistance.length?'A completed 1H close above '+money(d.resistance[0])+' weakens the downside case.':'A new confirmed recovery structure is needed.'):d.bias==='UP'?(d.support.length?'A completed 1H close below '+money(d.support[0])+' weakens the upside case.':'A new confirmed breakdown structure is needed.'):'A confirmed range break would establish a new candidate direction.'}</p><p>${d.beyond?'The current quote is '+d.beyond.toLowerCase()+' the recent hourly range. This live break is still developing until a candle closes.':'Current price remains inside the recent hourly range.'}</p></article></div><p class="footnote" style="margin-top:14px">Time window: next several hours to 1–2 sessions, conditional on confirmation; arrival time and target-hit probability are unvalidated. This is a fresh technical assessment, not a guaranteed prediction. It updates on opening the page, manual refresh and every 15 minutes while visible.</p></section>`;
}
function renderCurrentOutlook(){
 const panel=document.querySelector('#current-outlook');if(panel)panel.outerHTML=currentOutlookPanel();
 lastOutlookState=asset+':'+spotDecision(asset).status;
}
function checkCurrentOutlookAge(){
 if(lastOutlookState!==asset+':'+spotDecision(asset).status)renderCurrentOutlook();
}
