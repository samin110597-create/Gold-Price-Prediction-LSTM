from pathlib import Path
import re

P = Path('index.html')
s = P.read_text(encoding='utf-8')

css = r'''
/* PRECISION_GATE_V7 */
.pg-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:10px 0}.pg-card{background:#0c1621;border:1px solid #27394b;border-radius:12px;padding:11px}.pg-card.validated{border-color:rgba(91,211,154,.45)}.pg-card.mixed{border-color:rgba(239,190,104,.45)}.pg-card.low{border-color:rgba(255,125,134,.32)}.pg-call{font-size:20px;font-weight:950;margin:4px 0}.pg-call.up{color:var(--green)}.pg-call.down{color:var(--red)}.pg-call.range{color:var(--amber)}.pg-price{font-size:19px;font-weight:900;margin:3px 0}.pg-move{font-size:11px;font-weight:850}.pg-row{display:flex;justify-content:space-between;gap:8px;border-top:1px solid rgba(38,56,75,.55);padding-top:5px;margin-top:5px;font-size:10px}.pg-row span{color:var(--muted)}.pg-scen{font-size:9px;color:var(--muted);line-height:1.45;margin-top:6px}.pg-trust{display:inline-block;padding:4px 7px;border-radius:999px;font-size:9px;font-weight:900;border:1px solid #33495f}.pg-trust.validated{color:var(--green);border-color:rgba(91,211,154,.45)}.pg-trust.mixed{color:var(--amber);border-color:rgba(239,190,104,.45)}.pg-trust.low{color:var(--red);border-color:rgba(255,125,134,.4)}.pg-note{font-size:11px;color:#b8c6d4;line-height:1.55}.pg-summary{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}.pg-summary span{font-size:9px;border:1px solid #33495f;border-radius:999px;padding:4px 7px}@media(max-width:1050px){.pg-grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:700px){.pg-grid{grid-template-columns:1fr}}
'''

if '/* PRECISION_GATE_V7 */' in s:
    s, n = re.subn(r'/\* PRECISION_GATE_V7 \*/.*?(?=</style>)', css.strip()+'\n', s, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError('Could not replace Precision Gate CSS')
else:
    s = s.replace('</style>', css + '</style>')

js = r'''
/* PRECISION_GATE_V7_JS */
function pgTone(c){return c==='UP'?'up':c==='DOWN'?'down':'range'}
function pgTrustClass(t){return t==='VALIDATED'?'validated':t==='MIXED'?'mixed':'low'}
function pgMove(x){const n=Number(x);return Number.isFinite(n)?`${n>=0?'+':''}${n.toFixed(2)}%`:'—'}
function precisionGateHTML(g){
  if(!g||g.status!=='ok')return '';
  const cards=(g.horizons||[]).map(x=>{
    const tc=pgTrustClass(x.trust),sc=x.scenario_20_50_80||[],risk=x.risk_10_90||[];
    return `<div class="pg-card ${tc}">
      <div class="projection-status"><div class="label">${x.horizon}</div><span class="pg-trust ${tc}">${x.trust||'LOW TRUST'}</span></div>
      <div class="pg-call ${pgTone(x.call)}">${x.call||'RANGE'}</div>
      <div class="pg-price">${money(x.predicted_price)}</div>
      <div class="pg-move ${Number(x.projected_return_pct)>0?'good':Number(x.projected_return_pct)<0?'bad':'neutral'}">${pgMove(x.projected_return_pct)}</div>
      <div class="pg-row"><span>Probability</span><b>UP ${pct(x.probability_up)} / DOWN ${pct(x.probability_down)}</b></div>
      <div class="pg-row"><span>Probability lean</span><b>${x.probability_lean||'NEUTRAL'}</b></div>
      <div class="pg-row"><span>Technical</span><b class="${x.technical_lean==='UP'?'good':x.technical_lean==='DOWN'?'bad':'neutral'}">${x.technical_label||x.technical_lean||'—'}</b></div>
      <div class="pg-row"><span>Evidence alignment</span><b>${x.alignment||'MIXED'}</b></div>
      <div class="pg-row"><span>Historical audit</span><b class="${x.audit_ok?'good':'bad'}">${x.audit_status||'FAIL'}</b></div>
      ${Number.isFinite(Number(x.directional_accuracy))?`<div class="pg-row"><span>OOS accuracy</span><b>${pct(x.directional_accuracy)}</b></div>`:''}
      ${Number.isFinite(Number(x.directional_edge))?`<div class="pg-row"><span>OOS edge</span><b class="${Number(x.directional_edge)>=0?'good':'bad'}">${pct(x.directional_edge)}</b></div>`:''}
      ${Number.isFinite(Number(x.mae_skill_vs_baseline))?`<div class="pg-row"><span>MAE skill</span><b class="${Number(x.mae_skill_vs_baseline)>=0?'good':'bad'}">${pct(x.mae_skill_vs_baseline)}</b></div>`:''}
      ${Number.isFinite(Number(x.forward_direction_accuracy))?`<div class="pg-row"><span>Live forward</span><b>${x.forward_samples||0} • ${pct(x.forward_direction_accuracy)}</b></div>`:''}
      ${sc.length?`<div class="pg-scen">20/50/80 scenario: ${sc.map(money).join(' / ')}</div>`:''}
      ${risk.length?`<div class="pg-scen">10–90 risk range: ${money(risk[0])} – ${money(risk[1])}</div>`:''}
    </div>`
  }).join('');
  return `<div class="section-title">Directional Readout & Trust <span class="section-kicker">always show the model • validation changes trust, not visibility</span></div>
    <div class="card"><div class="pg-note">The forecast is never blanked out. <b>UP / DOWN / RANGE</b> is the model's current price-path lean. The trust badge tells you whether that lean is historically validated, mixed, or low-trust. Probability and technical structure stay separate so disagreement is visible instead of being hidden behind “ABSTAIN.”</div>
    <div class="pg-summary"><span>${g.validated_horizons||0} validated</span><span>${g.mixed_horizons||0} mixed</span><span>${g.low_trust_horizons||0} low-trust</span><span>Current ref ${money(g.current_price)}</span></div><div class="pg-grid">${cards}</div></div>`
}
async function hydratePrecisionGate(asset){const el=document.getElementById('precisionGateV7');if(!el)return;try{const r=await fetch(`data/${asset}_precision_gate_v7.json?ts=${Date.now()}`,{cache:'no-store'});if(r.ok)el.innerHTML=precisionGateHTML(await r.json())}catch(e){}}
'''

if '/* PRECISION_GATE_V7_JS */' in s:
    s, n = re.subn(r'/\* PRECISION_GATE_V7_JS \*/.*?(?=function tradingViewSection\(asset\))', js.strip()+'\n\n', s, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError('Could not replace Precision Gate JS')
else:
    s = s.replace('function tradingViewSection(asset)', js+'\nfunction tradingViewSection(asset)')

s = s.replace("function futurePriceActionShell(){return '<div id=\"futurePriceActionV6\"></div>'}", "function futurePriceActionShell(){return '<div id=\"futurePriceActionV6\"></div><div id=\"precisionGateV7\"></div>'}")
if 'hydratePrecisionGate(asset)' not in s.split('async function loadAsset',1)[1]:
    s = s.replace('drawChart(t||{},asset);hydrateFuturePriceAction(asset)}','drawChart(t||{},asset);hydrateFuturePriceAction(asset);hydratePrecisionGate(asset)}')

P.write_text(s, encoding='utf-8')
print('Directional Readout V7.1 dashboard patched')
