from pathlib import Path

P=Path('index.html');s=P.read_text(encoding='utf-8')
css='''
/* PRECISION_GATE_V7 */
.pg-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:10px 0}.pg-card{background:#0c1621;border:1px solid #27394b;border-radius:12px;padding:10px}.pg-call{font-size:18px;font-weight:950;margin:4px 0}.pg-call.up{color:var(--green)}.pg-call.down{color:var(--red)}.pg-call.noedge{color:var(--amber)}.pg-reason{font-size:9px;color:var(--muted);line-height:1.4;margin-top:6px}@media(max-width:900px){.pg-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:520px){.pg-grid{grid-template-columns:1fr}}
'''
if '/* PRECISION_GATE_V7 */' not in s:s=s.replace('</style>',css+'</style>')
js=r'''
/* PRECISION_GATE_V7_JS */
function pgTone(c){return c==='UP'?'up':c==='DOWN'?'down':'noedge'}
function precisionGateHTML(g){if(!g||g.status!=='ok')return '';const cards=(g.horizons||[]).map(x=>`<div class="pg-card"><div class="projection-status"><div class="label">${x.horizon}</div><span class="pass-pill ${x.release?'pass':'fail'}">${x.release?'RELEASED':'ABSTAIN'}</span></div><div class="pg-call ${pgTone(x.call)}">${x.call}</div><div class="small">Watch: ${x.watch_direction||'NO EDGE'} • UP ${pct(x.probability_up)}</div>${Number.isFinite(Number(x.forward_direction_accuracy))?`<div class="small">Live: ${x.forward_samples||0} • ${pct(x.forward_direction_accuracy)} direction</div>`:''}<div class="pg-reason">${x.reason||''}</div></div>`).join('');return `<div class="section-title">Precision Gate V7 <span class="section-kicker">publish fewer calls • demand stronger evidence</span></div><div class="card"><div class="note">This is the decision layer: UP/DOWN is released only when the underlying model passes its audit and the probability is outside the 42–58% no-edge band. Otherwise the dashboard deliberately abstains.</div><div class="pg-grid">${cards}</div></div>`}
async function hydratePrecisionGate(asset){const el=document.getElementById('precisionGateV7');if(!el)return;try{const r=await fetch(`data/${asset}_precision_gate_v7.json?ts=${Date.now()}`,{cache:'no-store'});if(r.ok)el.innerHTML=precisionGateHTML(await r.json())}catch(e){}}
'''
if '/* PRECISION_GATE_V7_JS */' not in s:s=s.replace('function tradingViewSection(asset)',js+'\nfunction tradingViewSection(asset)')
s=s.replace("function futurePriceActionShell(){return '<div id=\"futurePriceActionV6\"></div>'}","function futurePriceActionShell(){return '<div id=\"futurePriceActionV6\"></div><div id=\"precisionGateV7\"></div>'}")
s=s.replace('drawChart(t||{},asset);hydrateFuturePriceAction(asset)}','drawChart(t||{},asset);hydrateFuturePriceAction(asset);hydratePrecisionGate(asset)}')
P.write_text(s,encoding='utf-8');print('Precision Gate V7 dashboard patched')
