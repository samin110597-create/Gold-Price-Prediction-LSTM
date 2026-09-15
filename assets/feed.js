'use strict';
let feedBusy=false,lastFeedCheck=null,feedError='',lastFreshnessState='';
function snapshotIsStale(){return !data||Date.now()-Date.parse(data.asof)>(data.feed?.stale_after_seconds||1800)*1000;}
function ageLabel(timestamp){
 const seconds=Math.max(0,Math.floor((Date.now()-Date.parse(timestamp))/1000));
 if(!Number.isFinite(seconds))return 'unknown';
 return seconds<60?seconds+'s':seconds<3600?Math.floor(seconds/60)+'m '+seconds%60+'s':Math.floor(seconds/3600)+'h '+Math.floor(seconds%3600/60)+'m';
}
function refreshFeedLabels(){
 if(!data)return;
 const a=data.assets[asset];
 if($('#quote-age'))$('#quote-age').textContent=ageLabel(a.quote.time);
 if($('#snapshot-age'))$('#snapshot-age').textContent=ageLabel(data.asof);
 if($('#page-check'))$('#page-check').textContent=lastFeedCheck?time(lastFeedCheck):'Checking…';
 if($('#feed-status'))$('#feed-status').textContent=feedBusy?'Checking published data…':feedError?'Refresh failed: '+feedError+'. Showing the last verified snapshot.':snapshotIsStale()?'Snapshot is over 30 minutes old; current-watch eligibility is paused.':'Latest published snapshot loaded. Source quote '+time(a.quote.time)+'.';
 if($('#refresh-market')){$('#refresh-market').disabled=feedBusy;$('#refresh-market').onclick=()=>loadMarketSnapshot();}
}
function validateSnapshot(d,manifest){
 if(d.schema_version!==1||!d.run_id||d.run_id!==manifest.run_id||d.asof!==manifest.asof||!Number.isFinite(Date.parse(d.asof)))throw Error('Inconsistent snapshot metadata');
 for(const key of ['gold','silver']){
  const a=d.assets?.[key];
  if(!a||a.run_id!==d.run_id||a.asof!==d.asof||!Number.isFinite(a.quote?.price)||a.quote.price<=0||!Number.isFinite(Date.parse(a.quote.time))||Date.parse(a.quote.time)>Date.parse(d.asof))throw Error('Invalid '+key+' market observations');
  for(const tf of ['15m','1h','4h','1d','1w'])if(!a.timeframes?.[tf]?.chart?.length||Date.parse(a.timeframes[tf].last_completed)>Date.parse(d.asof))throw Error('Invalid completed bars');
 }
 if(data&&Date.parse(d.asof)<Date.parse(data.asof))throw Error('Older publication received');
 return d;
}
async function loadMarketSnapshot(){
 if(feedBusy)return;
 feedBusy=true;refreshFeedLabels();
 const controller=new AbortController(),deadline=setTimeout(()=>controller.abort(),20000);
 try{
  const options={cache:'no-store',signal:controller.signal};
  const mr=await fetch('data/run_manifest.json?t='+Date.now(),options);
  if(!mr.ok)throw Error('Manifest HTTP '+mr.status);
  const manifest=await mr.json();
  if(!manifest.run_id||!manifest.files?.['dashboard.json'])throw Error('Invalid manifest');
  if(!data||manifest.run_id!==data.run_id){
   const response=await fetch('data/dashboard.json?t='+Date.now(),options);
   if(!response.ok)throw Error('Snapshot HTTP '+response.status);
   const raw=await response.text();
   const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(raw))),b=>b.toString(16).padStart(2,'0')).join('');
   if(hash!==manifest.files['dashboard.json'])throw Error('Publication still changing; checksum mismatch');
   const next=validateSnapshot(JSON.parse(raw),manifest);
   data=next;$('#loading').classList.add('hidden');render();window.__metalsReady=true;
  }
  lastFeedCheck=new Date().toISOString();feedError='';
 }catch(error){
  feedError=error.name==='AbortError'?'request timed out':error.message;
  if(!data){$('#loading').classList.add('error');$('#loading').textContent='Market snapshot could not be verified. '+feedError+'. Retrying automatically.';}
 }finally{clearTimeout(deadline);feedBusy=false;refreshFeedLabels();}
}
function startMarketFeed(){
 loadMarketSnapshot();
 setInterval(()=>{if(!document.hidden)loadMarketSnapshot();},60000);
 setInterval(()=>{
  if(!data||document.hidden)return;
  const state=String(snapshotIsStale());
  if(lastFreshnessState&&state!==lastFreshnessState)render();
  lastFreshnessState=state;refreshFeedLabels();
 },1000);
 document.addEventListener('visibilitychange',()=>{if(!document.hidden)loadMarketSnapshot();});
}
