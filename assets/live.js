'use strict';
let currentMarket=null,currentBusy=false,currentError='';
// Public, keyless spot feed. Do not route secret-bearing APIs through the browser.
let directSpot={observations:[],checked_at:null},spotBusy=false,spotLastAttempt=null,spotErrors={};
function validateDirectSpot(raw,key,previous,now=Date.now()){
 const symbol={gold:'XAU',silver:'XAG'}[key],at=Date.parse(raw?.updatedAt);
 if(!symbol||raw?.symbol!==symbol||raw.currency!=='USD'||!Number.isFinite(raw.price)||raw.price<=0||!Number.isFinite(at)||at>now)throw Error('Invalid spot response');
 if(previous&&at<Date.parse(previous.source_time))throw Error('Older spot observation rejected');
 return {asset:key,symbol,basis:'Spot USD/oz',price:raw.price,source_time:raw.updatedAt,retrieved_at:new Date(now).toISOString(),currency:'USD'};
}
async function loadSpotQuotes(){
 // Respect the provider's documented 30-second cache, including manual clicks.
 if(spotBusy||(spotLastAttempt!==null&&Date.now()-spotLastAttempt<30000))return;
 spotBusy=true;spotLastAttempt=Date.now();
 await Promise.all(['gold','silver'].map(async key=>{
  const controller=new AbortController(),deadline=setTimeout(()=>controller.abort(),10000);
  try{
   const response=await fetch('https://api.gold-api.com/price/'+(key==='gold'?'XAU':'XAG'),{signal:controller.signal,credentials:'omit',referrerPolicy:'no-referrer'});
   if(!response.ok)throw Error('Spot HTTP '+response.status);
   const previous=directSpot.observations.find(q=>q.asset===key);
   const q=validateDirectSpot(await response.json(),key,previous);
   directSpot.observations=[...directSpot.observations.filter(q=>q.asset!==key),q];
   directSpot.checked_at=q.retrieved_at;delete spotErrors[key];
  }catch(error){spotErrors[key]=error.name==='AbortError'?'Direct spot request timed out':'Direct spot request failed or returned invalid data';}
  finally{clearTimeout(deadline);}
 }));
 spotBusy=false;
 if(data)refreshSpotDisplay();
}
function refreshSpotDisplay(){
 const q=primarySpotQuote();
 const price=document.querySelector('[data-testid="price"]'),basis=document.querySelector('#spot-price-basis');
 if(price)price.textContent=q?'$'+n(q.price):'Unavailable';
 if(basis)basis.textContent=q?q.basis+' · Last reported '+time(q.time):'No timestamped spot quote available';
 const panel=document.querySelector('#current-market');if(panel)panel.outerHTML=currentPanel();
 const health=document.querySelector('#provider-health');if(health)health.outerHTML=providerPanel();
 refreshSpotLabels();
}
function refreshSpotLabels(){
 if(!data)return;
 const q=primarySpotQuote();
 for(const id of ['spot-price-status','spot-panel-status','spot-detail-status']){const el=document.querySelector('#'+id);if(el)el.textContent=spotStatus(q);}
 const age=document.querySelector('#spot-quote-age');if(age&&q)age.textContent=ageLabel(q.time);
 const button=document.querySelector('#refresh-spot');if(button){const remaining=spotLastAttempt===null?0:Math.max(0,Math.ceil((30000-(Date.now()-spotLastAttempt))/1000));button.disabled=spotBusy||remaining>0;button.textContent=spotBusy?'Refreshing spot…':remaining?'Refresh spot in '+remaining+'s':'Refresh spot prices';button.onclick=()=>loadSpotQuotes();}
}

function validateCurrentMarket(next){
 const at=Date.parse(next.asof);
 if(next.schema_version!==1||!Number.isFinite(at)||at>Date.now()+120000)throw Error('Invalid current-market publication');
 if(currentMarket&&at<Date.parse(currentMarket.asof))throw Error('Older current-market publication');
 for(const [key,item] of Object.entries(next.assets||{})){
  const symbol={gold:'GC=F',silver:'SI=F'}[key],q=item.quote;
  if(!symbol||item.symbol!==symbol||q?.symbol!==symbol||!Number.isFinite(q.price)||q.price<=0||!Number.isFinite(Date.parse(q.time))||Date.parse(q.time)>at||!Number.isFinite(Date.parse(item.asof))||Date.parse(item.asof)>at)throw Error('Invalid current-market quote');
  if(currentMarket?.assets[key]&&Date.parse(q.time)<Date.parse(currentMarket.assets[key].quote.time))throw Error('Regressed source timestamp');
  if(!item.technical_brief?.rows?.length||item.technical_brief.rows.some(r=>!Number.isFinite(Date.parse(r.known_at))||Date.parse(r.known_at)>Date.parse(item.asof)))throw Error('Invalid current technicals');
 }
 if(!Object.keys(next.assets||{}).length)throw Error('Empty current-market publication');
 return next;
}
function currentAsset(){
 const item=currentMarket?.assets?.[asset];
 return item&&data&&Date.parse(item.asof)>=Date.parse(data.asof)?item:null;
}
function newestFuturesQuote(){
 const old=data.assets[asset].quote,item=currentMarket?.assets?.[asset];
 return item&&Date.parse(item.quote.time)>Date.parse(old.time)?item.quote:old;
}
function forecastApplicability(f,q,now=Date.now()){
 const r=f.research,o=f.outlook;
 if(!r||!o?.target_time)return {status:'NO FORECAST'};
 if(Date.parse(o.target_time)<=now)return {status:'EXPIRED — AWAIT NEXT FORECAST'};
 if(now-Date.parse(q.time)>1800000)return {status:'CURRENT QUOTE TOO OLD'};
 if(Date.parse(q.time)<Date.parse(r.origin))return {status:'QUOTE PREDATES FORECAST'};
 const outside=q.price<r.interval80[0]||q.price>r.interval80[1];
 const direction=q.price>r.reference_price?'UP':q.price<r.reference_price?'DOWN':'UNCHANGED';
 return {status:outside?'OUTSIDE ORIGINAL RANGE — REASSESS':'WITHIN ORIGINAL RANGE',
  direction,move:q.price/r.reference_price-1,remaining:r.price/q.price-1,
  errorRatio:o.recent_mean_absolute_error_dollars>0?Math.abs(q.price-r.reference_price)/o.recent_mean_absolute_error_dollars:null};
}
function currentProviders(){
 const providers={...data.external_data?.providers};
 for(const [key,p] of Object.entries(currentMarket?.external_data?.providers||{})){
  if(!providers[key]||Date.parse(p.checked_at)>Date.parse(providers[key].checked_at))providers[key]=p;
 }
 providers.gold_api={...directSpot,status:Object.keys(spotErrors).length?(directSpot.observations.length?'PARTIAL':'UNAVAILABLE'):(directSpot.observations.length?'CONNECTED':'CONNECTING'),errors:Object.values(spotErrors),refresh_seconds:30,documentation:'https://gold-api.com/docs'};
 return providers;
}
function selectSpotQuote(providers,key,now=Date.now()){
 // Only a documented spot endpoint can populate the spot headline.
 // ETF shares, prior-day bars and unidentified contracts are never fallbacks.
 const valid=[];
 for(const [provider,symbol] of [['gold_api',{gold:'XAU',silver:'XAG'}[key]],['alpha_vantage',{gold:'GOLD',silver:'SILVER'}[key]]]){
  const source=providers[provider];
  for(const q of source?.observations||[]){
   if(symbol&&q.asset===key&&q.symbol===symbol&&q.basis==='Spot USD/oz'&&(!q.currency||q.currency==='USD')&&Number.isFinite(q.price)&&q.price>0&&Number.isFinite(Date.parse(q.source_time))&&Date.parse(q.source_time)<=now)valid.push({...q,provider,retained:!!source.retained_previous});
  }
 }
 if(!valid.length)return null;
 const q=valid.sort((a,b)=>Date.parse(b.source_time)-Date.parse(a.source_time))[0];
 const source=q.provider==='gold_api'?'Gold API':'Alpha Vantage';
 return {price:q.price,time:q.source_time,symbol:key==='gold'?'XAU/USD':'XAG/USD',source,basis:source+' indicative spot · USD per troy ounce',stale:now-Date.parse(q.source_time)>120000,retained:q.retained};
}
function primarySpotQuote(){return selectSpotQuote(currentProviders(),asset);}
function spotStatus(q){return !q?'SPOT QUOTE UNAVAILABLE':q.stale?'STALE SPOT — NOT A CURRENT PRICE':'CURRENT SPOT · SOURCE UNDER 2 MIN OLD';}
function currentPanel(){
 const item=currentAsset(),q=newestFuturesQuote();
 const providers=currentProviders(),spot=selectSpotQuote(providers,asset);
 const quotes=Object.entries(providers).flatMap(([name,p])=>(p.observations||[]).filter(v=>v.asset===asset).map(v=>({...v,provider:name,checked:v.retrieved_at||p.checked_at}))).sort((a,b)=>(Date.parse(b.source_time)||0)-(Date.parse(a.source_time)||0));
 return `<section class="panel" id="current-market"><div class="panel-head"><h2>Spot price and separate futures research</h2><span class="badge wait" id="spot-panel-status">${escape(spotStatus(spot))}</span></div><p><strong id="spot-detail-status">${escape(spotStatus(spot))}</strong>${spot?` · ${escape(spot.source)} ${time(spot.time)} · age <span id="spot-quote-age">${ageLabel(spot.time)}</span>`:" · No substitute price is displayed."}.</p><p>Spot prices are requested directly every 30 seconds while this page is visible, independently of GitHub builds. <button id="refresh-spot" type="button">Refresh spot prices</button></p><p class="footnote">Indicative USD per troy ounce, not an executable broker bid/ask. Source timestamps determine freshness; quotes over two minutes old are flagged. Alpha Vantage remains a timestamped backup on its existing quota budget. Last direct attempt: ${spotLastAttempt?time(new Date(spotLastAttempt).toISOString()):'Connecting…'}. ${escape(spotErrors[asset]||'')}</p><p><strong>Separate Yahoo futures quote: $${n(q.price)}</strong>. This is not the spot headline and is not independently verified against a matching futures contract. Futures source ${time(q.time)} · age <strong id="current-quote-age">${ageLabel(q.time)}</strong>. Technical refresh ${time(item?.asof||data.asof)}. Full model evaluation ${time(data.asof)}.</p><p class="footnote">Fast refresh requested every 5 minutes; this page checks every 30 seconds. GitHub schedules can run late. Refresh checks published data; it does not bypass provider delays.</p><div class="table-wrap"><table><thead><tr><th>API / instrument</th><th>Price</th><th>Source observation</th><th>API checked</th><th>Basis</th></tr></thead><tbody>${quotes.map(v=>`<tr><td>${escape(v.provider)} / ${escape(v.symbol)}</td><td>$${n(v.price)}</td><td>${v.source_time?time(v.source_time):'No quote timestamp; context only'}</td><td>${time(v.checked)}</td><td>${escape(v.basis)}</td></tr>`).join('')}</tbody></table></div><p class="footnote">Sorted by source timestamp. Spot, ETF and unverified commodity contracts are comparisons; they never replace futures prices in model calculations.</p><h3>Forecast versus the current futures price</h3><div class="table-wrap"><table><thead><tr><th>Horizon</th><th>Applicability now</th><th>Since model origin</th><th>Move / recent model error</th></tr></thead><tbody>${Object.entries(data.assets[asset].forecasts).map(([h,f])=>{const a=forecastApplicability(f,q);return `<tr><td>${escape(h)}</td><td>${escape(a.status)}</td><td>${a.move===undefined?'—':pct(a.move)}</td><td>${a.errorRatio===undefined?'—':n(a.errorRatio,2)+'×'}</td></tr>`;}).join('')}</tbody></table></div><p class="footnote">Checks the original issued forecast against new observations. Being inside its range does not validate accuracy. Targets and probabilities are never shifted to follow the market. ${escape(item?.history_note||'')}</p><p id="current-refresh-status" class="footnote">${escape(currentError||currentMarket?.errors?.[asset]||'')}</p></section>`;
}
async function loadCurrentMarket(){
 if(currentBusy)return;
 currentBusy=true;
 const controller=new AbortController(),deadline=setTimeout(()=>controller.abort(),15000);
 try{
  const response=await fetch('data/live.json?t='+Date.now(),{cache:'no-store',signal:controller.signal});
  if(!response.ok)throw Error('Current refresh unavailable (HTTP '+response.status+')');
  const next=validateCurrentMarket(await response.json()),changed=next.asof!==currentMarket?.asof;
  currentMarket=next;currentError='';
  if(data&&changed)render();
  else if(data){const panel=document.querySelector('#current-market');if(panel)panel.outerHTML=currentPanel();refreshSpotLabels();}
 }catch(error){currentError=error.name==='AbortError'?'Current refresh timed out':error.message;
  const el=document.querySelector('#current-refresh-status');if(el)el.textContent=currentError+'; original timestamps preserved.';
 }finally{clearTimeout(deadline);currentBusy=false;}
}
function startCurrentMarket(){
 loadCurrentMarket();loadSpotQuotes();
 setInterval(()=>{if(!document.hidden){loadCurrentMarket();loadSpotQuotes();}},30000);
 setInterval(()=>{const el=document.querySelector('#current-quote-age');if(el&&data)el.textContent=ageLabel(newestFuturesQuote().time);refreshSpotLabels();},1000);
 document.addEventListener('visibilitychange',()=>{if(!document.hidden){loadCurrentMarket();loadSpotQuotes();}});
}
