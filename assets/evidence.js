'use strict';
function forecastCards(a,stale){
 return Object.entries(a.forecasts).map(([h,f])=>{
  const r=f.research,m=f.oos,ho=f.holdout,d=f.data_evidence||{},o=f.outlook||{},review=f.challenger_review;
  const expired=o.target_time&&Date.parse(o.target_time)<=Date.now();
  const verified=!stale&&!expired&&f.status==='VALIDATED';
  const ref=a.timeframes[h==='4H'?'1h':'1d'].chart.at(-1);
  const selected=r?.selection;
  return `<article class="forecast" data-forecast="${h}">
   <div class="panel-head"><h3>${h}</h3>${badge(stale?'STALE ESTIMATE':expired?'TARGET ELAPSED':verified?'VALIDATED':'RESEARCH · UNVALIDATED',verified?'up':'wait')}</div>
   <div class="forecast-direction ${o.direction==='UPWARD'?'uptext':o.direction==='DOWNWARD'?'downtext':''}">${escape(o.direction||'PENDING')} BIAS</div>
   <div class="forecast-label">${verified?'Validated expected price':'Model price estimate · not a market quote'}</div>
   <div class="headline" data-estimate="${h}">${r?'$'+n(r.price,asset==='silver'?3:2):'Insufficient complete data'}</div>
   <p class="forecast-date" data-target-date="${h}">For <strong>${time(o.target_time)}</strong></p>
   <small>${r?'Model 80% range $'+n(r.interval80[0])+' – $'+n(r.interval80[1]):'Waiting for sufficient complete calibration cases.'}</small>
   ${r?`<p class="forecast-strength">${escape(o.strength||'DIRECTION UNVALIDATED')}<br><small>${o.range_contains_no_change?'The range includes both gains and losses.':''}</small></p>`:''}
   <p class="observed-price">Observed reference <strong>$${n(ref.close,asset==='silver'?3:2)}</strong><br>${time(ref.time)}</p>
   ${r?`<p class="footnote">Expected change ${pct(r.return)} (${o.expected_change>=0?'+':''}$${n(o.expected_change)}). Recent average error ≈ $${n(o.recent_mean_absolute_error_dollars)}. This is an error scale, not an additional confidence interval.</p>`:''}
   <div class="forecast-evidence"><span>Earlier / recent tests</span><strong>${n(m.n,0)} / ${n(ho.n,0)}</strong><span>Recent direction hit rate</span><strong>${pct(ho.directional_accuracy)}</strong><span>Recent price error</span><strong>${pct(Number.isFinite(ho.mae_percent)?ho.mae_percent/100:null)}</strong><span>Recent 80% range coverage</span><strong>${pct(ho.coverage80)}</strong></div>
   <details><summary>Accuracy, model & decision rules</summary>
    <p>${escape(o.interpretation)} ${expired?'This forecast target has elapsed; wait for an updated run.':''}</p>
    <p>Recipe ${escape(f.recipe_version)}: ${escape(f.model)}. ${selected?'Past-only selection: '+escape(selected.candidates[selected.candidate_mae_percent.indexOf(Math.min(...selected.candidate_mae_percent))])+'.':''}</p>
    <p>Earlier baseline-relative directional edge ${pp(m.edge)}; price error ${pct((m.mae_percent??NaN)/100)} versus baseline ${pct((m.baseline_mae_percent??NaN)/100)} and no-change ${pct((m.zero_change_mae_percent??NaN)/100)}. Recent no-change error ${pct((ho.zero_change_mae_percent??NaN)/100)}.</p>
    ${r?`<p>Direction score ${pct(r.probability_up)} P(up), ${verified?'validated under this recipe':'not a validated trade probability'}. ${n(d.completed_bars,0)} completed bars; ${n(r.calibration_n,0)} pre-origin calibration cases.</p>`:''}
    ${review?`<p>New damped-model challenger: ${review.promoted?'passed research promotion criteria':'not promoted; current recipe retained'}. Matched recent price-error change ${pct(review.comparison.holdout.mae_improvement)}; positive means improvement. ${escape(review.note)}</p>`:''}
    <p>Failed gates: ${escape(f.failed_gates.join(', ')||'none')}. Estimates update with new data; issued trade stops and targets remain fixed.</p>
    <a href="${escape(f.evidence_file)}">Full timestamped evaluation</a>
   </details></article>`;
 }).join('');
}
function pricePaths(a,stale){
 const p=a.price_paths;if(!p)return '';
 const level=x=>Number.isFinite(x)?'$'+n(x):'No confirmed level';
 return `<section class="panel span12" id="price-paths"><div class="panel-head"><h2>What would confirm the next move?</h2>${badge(stale?'STALE · WAIT':'CONDITIONAL 4H SCENARIOS')}</div><div class="path-grid">
 <article><h3 class="uptext">Bull case</h3><strong>Reclaim ${level(p.bull.trigger)}</strong><p>${escape(p.bull.rule)}</p><small>Next known objective: ${level(p.bull.objective)}</small></article>
 <article><h3>Base case</h3><strong>${level(p.base.lower)} – ${level(p.base.upper)}</strong><p>${escape(p.base.rule)}</p><small>Zone map from ${time(p.known_at)}</small></article>
 <article><h3 class="downtext">Bear case</h3><strong>Lose ${level(p.bear.trigger)}</strong><p>${escape(p.bear.rule)}</p><small>Next known objective: ${level(p.bear.objective)}</small></article></div><p class="footnote">${escape(p.note)}</p></section>`;
}
function providerPanel(){
 const external=data.external_data;if(!external)return '';
 const names={finnhub:'Finnhub',fmp:'FMP',fred:'FRED',polygon:'Polygon / Massive',alpha_vantage:'Alpha Vantage'};
 return `<section class="panel span12" id="provider-health"><div class="panel-head"><h2>Live data sources & access</h2><span class="muted">Last access audit ${time(external.asof)}</span></div><div class="provider-grid">${Object.entries(external.providers).map(([key,p])=>`<article data-provider="${key}"><h3>${names[key]}</h3>${badge(p.status,p.status==='CONNECTED'?'up':'wait')}<p>Refresh budget: every ${n(p.refresh_seconds/3600,2)}h</p>${p.retained_previous?'<p>Retained older observations after a failed request.</p>':''}${p.observations.filter(v=>!v.asset||v.asset===asset).map(v=>`<div class="provider-value"><strong>${escape(v.symbol)} ${v.price?'$'+n(v.price):n(v.value)}</strong><small>${escape(v.basis)}${v.units?' · '+escape(v.units):''}<br>${v.source_time?'Source '+time(v.source_time):v.period_start?'Period starts '+time(v.period_start):'Source timestamp unavailable'}<br>Retrieved ${time(v.retrieved_at)}</small></div>`).join('')}${!p.observations.length?'<p>No verified observations from this provider.</p>':''}<a href="${escape(p.documentation)}" target="_blank" rel="noopener noreferrer">Provider documentation</a></article>`).join('')}</div><p class="footnote">Primary chart and forecasts: Yahoo GC=F / SI=F continuous futures, with delayed timestamped quotes. Spot USD/oz and GLD/SLV USD/share are separate context; their prices are never substituted into futures history. ${escape(external.note)}</p></section>`;
}
function feedPanel(){
 return '<section class="feed-panel" aria-label="Market data freshness"><div><strong>Real market data · delayed snapshots</strong><p>Yahoo GC=F / SI=F futures. Provider timestamps show the actual delay. Additional API access is reported below.</p></div><div class="feed-times"><span>Quote age <b id="quote-age">—</b></span><span>Snapshot age <b id="snapshot-age">—</b></span><span>Last page check <b id="page-check">—</b></span></div><div><button id="refresh-market">Check for new data</button><p id="feed-status" role="status">Checking…</p></div><small>Page checks every 60 seconds. Data builds are scheduled every 15 minutes; publication can be delayed. This is a periodically refreshed page, not a streaming exchange feed.</small></section>';
}
