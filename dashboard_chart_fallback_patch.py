from pathlib import Path

P=Path('index.html')
s=P.read_text(encoding='utf-8')

js=r'''
/* CLIENT_TECH_OVERLAY_FALLBACK */
function clientChartOverlay(t){
  const rows=(t?.chart||[]).map(x=>({date:x.date,open:+x.open,high:+x.high,low:+x.low,close:+x.close,ema20:+x.ema20,ema50:+x.ema50})).filter(x=>Number.isFinite(x.close));
  if(rows.length<25)return {levels:[],patterns:[],divergences:[],events:[]};
  const last=rows.at(-1).close, f=technicalFrameworks(t)||{}, conf=t?.confirmation||{};
  const levels=[];
  (f.supports||[]).slice(0,2).forEach(x=>levels.push({label:'Support',price:+x.center,kind:'support',strength:x.strength}));
  (f.resistances||[]).slice(0,2).forEach(x=>levels.push({label:'Resistance',price:+x.center,kind:'resistance',strength:x.strength}));
  if(Number.isFinite(+conf.bullish_above))levels.push({label:'Bull confirm',price:+conf.bullish_above,kind:'bull_trigger'});
  if(Number.isFinite(+conf.bearish_below))levels.push({label:'Bear confirm',price:+conf.bearish_below,kind:'bear_trigger'});
  const patterns=(f.patterns||[]).filter(p=>!/Mixed swing structure/i.test(p.name||'')).slice(0,5).map((p,i)=>({
    name:p.name,bias:p.bias||'Neutral',status:/breakout|breakdown/i.test(p.name||'')?'Confirmed':'Detected',confidence:p.confidence||'Medium',
    structural_quality:/High/i.test(p.confidence||'')?72:/Medium/i.test(p.confidence||'')?60:50,
    start:rows[Math.max(0,rows.length-40-i*4)].date,end:rows.at(-1).date,
    confirmation:/Bull/i.test(p.bias||'')?+conf.bullish_above:/Bear/i.test(p.bias||'')?+conf.bearish_below:null,
    invalidation:/Bull/i.test(p.bias||'')?+conf.bearish_below:/Bear/i.test(p.bias||'')?+conf.bullish_above:null,target:null,detail:p.meaning||'Rule-based structural detection.',segments:[]
  }));
  const events=[]; const add=(i,name,bias,kind,detail)=>events.push({date:rows[i].date,price:rows[i].close,name,bias,kind,detail});
  for(let i=Math.max(21,rows.length-70);i<rows.length;i++){
    const a=rows[i],p=rows[i-1];
    if(Number.isFinite(a.ema20)&&Number.isFinite(a.ema50)&&Number.isFinite(p.ema20)&&Number.isFinite(p.ema50)){
      if(a.ema20>a.ema50&&p.ema20<=p.ema50)add(i,'EMA20/50 bullish cross','Bullish','trend','EMA20 crossed above EMA50.');
      if(a.ema20<a.ema50&&p.ema20>=p.ema50)add(i,'EMA20/50 bearish cross','Bearish','trend','EMA20 crossed below EMA50.');
    }
    const prev=rows.slice(i-20,i); const ph=Math.max(...prev.map(x=>x.high)),pl=Math.min(...prev.map(x=>x.low));
    if(Number.isFinite(ph)&&a.close>ph)add(i,'20D breakout','Bullish','breakout','Close exceeded the previous 20-day high.');
    if(Number.isFinite(pl)&&a.close<pl)add(i,'20D breakdown','Bearish','breakout','Close fell below the previous 20-day low.');
  }
  const seen=new Set(),keep=[]; for(const e of [...events].reverse()){if(!seen.has(e.name)){keep.push(e);seen.add(e.name)}if(keep.length>=8)break}
  return {levels,patterns,divergences:[],events:keep.reverse(),note:'Client fallback uses the OHLC already loaded on the page; server-side advanced scan overrides it when available.'};
}
'''
if '/* CLIENT_TECH_OVERLAY_FALLBACK */' not in s:
    s=s.replace('function tradingViewSection(asset)',js+'\nfunction tradingViewSection(asset)')

old="const ps=t?.advanced_chart_patterns||[],ds=t?.divergence_signals||[],ev=t?.chart_signal_events||[];"
new="const fb=clientChartOverlay(t),ps=(t?.advanced_chart_patterns?.length?t.advanced_chart_patterns:fb.patterns)||[],ds=(t?.divergence_signals?.length?t.divergence_signals:fb.divergences)||[],ev=(t?.chart_signal_events?.length?t.chart_signal_events:fb.events)||[];"
s=s.replace(old,new)
s=s.replace("overlay=t.chart_overlay||{};chart=new Chart", "overlay=(t.chart_overlay&&(((t.chart_overlay.events||[]).length)+((t.chart_overlay.patterns||[]).length)))?t.chart_overlay:clientChartOverlay(t);chart=new Chart")

P.write_text(s,encoding='utf-8')
print('Client technical overlay fallback patched')
