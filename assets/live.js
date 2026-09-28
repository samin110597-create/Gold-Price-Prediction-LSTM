'use strict';
let currentMarket=null,currentBusy=false,currentError='';
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
 return providers;
}
function currentPanel(){
 const item=currentAsset(),q=newestFuturesQuote(),closed=(item||data.assets[asset]).session?.status==='MARKET CLOSED';
 const stale=Date.now()-Date.parse(q.time)>1800000;
 const providers=currentProviders();
 const quotes=Object.entries(providers).flatMap(([name,p])=>(p.observations||[]).filter(v=>v.asset===asset).map(v=>({...v,provider:name,checked:p.checked_at}))).sort((a,b)=>(Date.parse(b.source_time)||0)-(Date.parse(a.source_time)||0));
 return `<section class="panel" id="current-market"><div class="panel-head"><h2>Latest available market data</h2>${badge(closed?'MARKET CLOSED':stale?'SOURCE OVER 30 MIN OLD':'TIMESTAMPED · DELAYED')}</div><p>Futures source ${time(q.time)} · age <strong id="current-quote-age">${ageLabel(q.time)}</strong>. Technical refresh ${time(item?.asof||data.asof)}. Full model evaluation ${time(data.asof)}.</p><p class="footnote">Fast refresh requested every 5 minutes; this page checks every 30 seconds. GitHub schedules can run late. Refresh checks published data; it does not bypass provider delays.</p><div class="table-wrap"><table><thead><tr><th>API / instrument</th><th>Price</th><th>Source observation</th><th>API checked</th><th>Basis</th></tr></thead><tbody>${quotes.map(v=>`<tr><td>${escape(v.provider)} / ${escape(v.symbol)}</td><td>$${n(v.price)}</td><td>${v.source_time?time(v.source_time):'No quote timestamp; context only'}</td><td>${time(v.checked)}</td><td>${escape(v.basis)}</td></tr>`).join('')}</tbody></table></div><p class="footnote">Sorted by source timestamp. Spot, ETF and unverified commodity contracts are comparisons; they never replace futures prices in model calculations.</p><h3>Forecast versus the current futures price</h3><div class="table-wrap"><table><thead><tr><th>Horizon</th><th>Applicability now</th><th>Since model origin</th><th>Move / recent model error</th></tr></thead><tbody>${Object.entries(data.assets[asset].forecasts).map(([h,f])=>{const a=forecastApplicability(f,q);return `<tr><td>${escape(h)}</td><td>${escape(a.status)}</td><td>${a.move===undefined?'—':pct(a.move)}</td><td>${a.errorRatio===undefined?'—':n(a.errorRatio,2)+'×'}</td></tr>`;}).join('')}</tbody></table></div><p class="footnote">Checks the original issued forecast against new observations. Being inside its range does not validate accuracy. Targets and probabilities are never shifted to follow the market. ${escape(item?.history_note||'')}</p><p id="current-refresh-status" class="footnote">${escape(currentError||currentMarket?.errors?.[asset]||'')}</p></section>`;
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
  else if(data){const panel=document.querySelector('#current-market');if(panel)panel.outerHTML=currentPanel();}
 }catch(error){currentError=error.name==='AbortError'?'Current refresh timed out':error.message;
  const el=document.querySelector('#current-refresh-status');if(el)el.textContent=currentError+'; original timestamps preserved.';
 }finally{clearTimeout(deadline);currentBusy=false;}
}
function startCurrentMarket(){
 loadCurrentMarket();
 setInterval(()=>{if(!document.hidden)loadCurrentMarket();},30000);
 setInterval(()=>{const el=document.querySelector('#current-quote-age');if(el&&data)el.textContent=ageLabel(newestFuturesQuote().time);},1000);
 document.addEventListener('visibilitychange',()=>{if(!document.hidden)loadCurrentMarket();});
}
