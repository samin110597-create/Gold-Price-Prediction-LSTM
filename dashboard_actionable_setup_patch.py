from pathlib import Path
import re

P=Path('index.html')
s=P.read_text(encoding='utf-8')

css=r'''
/* MARKET_SETUP_V8 */
.ms8{margin-bottom:14px}.ms8-main{display:grid;grid-template-columns:1.25fr .75fr;gap:10px}.ms8-hero{background:linear-gradient(145deg,rgba(16,26,37,.98),rgba(10,20,30,.98));border:1px solid var(--line);border-radius:16px;padding:16px}.ms8-state{font-size:25px;font-weight:950;margin:5px 0}.ms8-bias{font-size:13px;font-weight:900;letter-spacing:.06em}.ms8-bias.bull{color:var(--green)}.ms8-bias.bear{color:var(--red)}.ms8-bias.neutral{color:var(--amber)}.ms8-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:10px}.ms8-box{background:#0c1621;border:1px solid #27394b;border-radius:10px;padding:9px}.ms8-box span{display:block;font-size:8px;color:var(--muted);text-transform:uppercase}.ms8-box b{display:block;font-size:14px;margin-top:3px}.ms8-targets{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px}.ms8-target{background:#0b1720;border:1px solid #2b3c50;border-radius:9px;padding:8px}.ms8-target b{font-size:16px}.ms8-reasons{display:flex;gap:5px;flex-wrap:wrap;margin-top:9px}.ms8-chip{font-size:9px;border:1px solid #33495f;border-radius:999px;padding:4px 7px;background:#0c1621}.ms8-warn{font-size:10px;color:var(--amber);line-height:1.45;margin-top:8px}.ms8-side{background:rgba(16,26,37,.96);border:1px solid var(--line);border-radius:16px;padding:14px}.ms8-score{font-size:36px;font-weight:950}.ms8-table{width:100%;border-collapse:collapse;font-size:10px;margin-top:8px}.ms8-table td{padding:6px 2px;border-bottom:1px solid rgba(38,56,75,.55)}.ms8-table td:last-child{text-align:right;font-weight:850}.ms8-rule{font-size:10px;color:#b8c6d4;line-height:1.5;margin-top:8px}@media(max-width:850px){.ms8-main{grid-template-columns:1fr}.ms8-grid,.ms8-targets{grid-template-columns:1fr}}
'''
if '/* MARKET_SETUP_V8 */' not in s:
    s=s.replace('</style>',css+'</style>')

js=r'''
/* MARKET_SETUP_V8_JS */
function ms8Tone(b){return b==='BULLISH'?'bull':b==='BEARISH'?'bear':'neutral'}
function ms8Money(x){return Number.isFinite(Number(x))?money(x):'—'}
function marketSetupHTML(x){if(!x||x.status!=='ok')return '';const ts=x.targets||[],rr=x.risk_reward||[],retest=x.retest_zone||[],ap=x.active_pattern||{},se=x.recent_signal_edge_5d||{},ext=x.extension_flags||[];const tgt=ts.map((t,i)=>`<div class="ms8-target"><span class="label">Target ${i+1}</span><b>${money(t)}</b><div class="small">R/R ${Number.isFinite(Number(rr[i]))?Number(rr[i]).toFixed(2):'—'}</div></div>`).join('');const chips=(x.evidence||[]).slice(0,10).map(r=>`<span class="ms8-chip">${r}</span>`).join('');return `<div class="section-title">Market Setup V8 <span class="section-kicker">direction • location • trigger • invalidation • targets</span></div><section class="ms8"><div class="ms8-main"><div class="ms8-hero"><div class="ms8-bias ${ms8Tone(x.market_bias)}">${x.market_bias} MARKET BIAS</div><div class="ms8-state">${x.setup_state}</div><div class="small">Current ${money(x.current_price)} • ${x.market_structure||'—'}</div><div class="ms8-grid"><div class="ms8-box"><span>Confirmation / trigger</span><b>${ms8Money(x.trigger)}</b></div><div class="ms8-box"><span>Retest zone</span><b>${retest.length?`${money(retest[0])} – ${money(retest[1])}`:'—'}</b></div><div class="ms8-box"><span>Invalidation</span><b>${ms8Money(x.invalidation)}</b></div></div><div class="ms8-targets">${tgt||'<div class="small">No clean structural target yet.</div>'}</div>${ext.length?`<div class="ms8-warn"><b>Extension risk:</b> ${ext.join(' • ')}</div>`:''}<div class="ms8-reasons">${chips}</div></div><div class="ms8-side"><div class="label">Directional evidence score</div><div class="ms8-score ${ms8Tone(x.market_bias)}">${Number(x.evidence_score).toFixed(1)}/100</div><div class="small">Explainable confluence score — not a probability.</div><table class="ms8-table"><tr><td>Technical score</td><td>${Number(x.technical_score).toFixed(1)}/100</td></tr><tr><td>RSI / ADX</td><td>${x.rsi14??'—'} / ${x.adx14??'—'}</td></tr><tr><td>+DI / -DI</td><td>${x.plus_di??'—'} / ${x.minus_di??'—'}</td></tr><tr><td>MACD hist.</td><td>${x.macd_hist??'—'}</td></tr><tr><td>Volume vs 20D</td><td>${x.volume_vs_20d??'—'}x</td></tr><tr><td>OBV</td><td>${x.obv_trend||'—'}</td></tr><tr><td>20D VWAP</td><td>${ms8Money(x.vwap20)}</td></tr><tr><td>Recent audited 5D signal edge</td><td>${Number(se.edge||0)>=0?'+':''}${(Number(se.edge||0)*100).toFixed(1)}pp</td></tr><tr><td>4H / 1W / 1Y statistical trust</td><td>${x.statistical_trust?.['4H']||'—'} / ${x.statistical_trust?.['1W']||'—'} / ${x.statistical_trust?.['1Y']||'—'}</td></tr></table>${ap.name?`<div class="ms8-rule"><b>Active structure:</b> ${ap.name} • ${ap.status||'—'} • quality ${ap.structural_quality??'—'}/100${ap.target?` • measured objective ${money(ap.target)}`:''}</div>`:''}<div class="ms8-rule">Use this as market research: the setup tells you direction and location; the audit tells you how much historical trust it deserves. A bullish bias can still say WAIT if price is extended or the risk/reward is poor.</div></div></div></section>`}
function marketSetupShell(){return '<div id="marketSetupV8"></div>'}
async function hydrateMarketSetup(asset){const el=document.getElementById('marketSetupV8');if(!el)return;try{const r=await fetch(`data/${asset}_actionable_setup_v8.json?ts=${Date.now()}`,{cache:'no-store'});if(r.ok)el.innerHTML=marketSetupHTML(await r.json())}catch(e){}}
'''
if '/* MARKET_SETUP_V8_JS */' not in s:
    s=s.replace('function tradingViewSection(asset)',js+'\nfunction tradingViewSection(asset)')

# Put the action-oriented panel before the model diagnostics.
if '+marketSetupShell()+predictionSnapshot' not in s:
    s=s.replace('+predictionSnapshot(p||{projections:[]},aa)', '+marketSetupShell()+predictionSnapshot(p||{projections:[]},aa)')

# Ensure hydration survives every patch ordering.
if 'hydrateMarketSetup(asset)' not in s.split('async function loadAsset',1)[1]:
    s=s.replace('drawChart(t||{},asset);hydrateFuturePriceAction(asset);hydratePrecisionGate(asset)}','drawChart(t||{},asset);hydrateMarketSetup(asset);hydrateFuturePriceAction(asset);hydratePrecisionGate(asset)}')
    s=s.replace('drawChart(t||{},asset);hydrateFuturePriceAction(asset)}','drawChart(t||{},asset);hydrateMarketSetup(asset);hydrateFuturePriceAction(asset)}')

P.write_text(s,encoding='utf-8')
print('Market Setup V8 dashboard patched')
