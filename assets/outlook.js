'use strict';
function outlookPanel(a, stale){
 const b=a.market_brief;
 if(!b)return '<section class="panel span12" id="market-outlook"><h2>Price outlook & market drivers</h2><p>Waiting for a new evidence bundle. No market summary is available in this older snapshot.</p></section>';
 const price=v=>Number.isFinite(v)?'$'+n(v,2):'Unavailable';
 const fresh=!stale&&a.timeframes['4h'].quality.latest_expected_bar_present&&a.timeframes['1d'].quality.latest_expected_bar_present;
 const p=a.price_paths;
 const summary=fresh?b.summary:'This scheduled research snapshot is older or incomplete. Use the current spot outlook above for the latest price-and-candle assessment.';
 const factors=b.factors||[], reports=b.reports||[];
 const color=impact=>impact==='HEADWIND'?'down':impact==='TAILWIND'?'up':'wait';
 return `<section class="panel span12" id="market-outlook">
 <div class="panel-head"><h2>${escape(a.name)}: scheduled futures research</h2>${badge(fresh?b.bias+' TECHNICAL BIAS':'STALE / INCOMPLETE · WAIT')}</div>
 <p class="outlook-summary" data-outlook-summary>${escape(summary)}</p>
 <p>${escape(b.horizon)}<br><small>Data used from ${time(b.asof)}. Price levels below are ${escape(a.symbol)} continuous futures, USD/oz — not spot targets.</small></p>
 <div class="path-grid">
 <article><h3>What the chart says</h3><p>Daily support: <strong>${price(b.daily_support)}</strong><br>Daily resistance: <strong>${price(b.daily_resistance)}</strong></p><p>${b.bias==='BEARISH'?'Watch support for a pause or bounce; a confirmed break would keep downside risk open.':b.bias==='BULLISH'?'Watch resistance for a pause; a confirmed break would keep upside potential open.':'Wait for a confirmed break or reversal; the chart does not support a single strong direction.'}</p><small>These are observed zones, not guaranteed turning points. Missing support means no defensible lower structural target.</small></article>
 <article><h3>What would change the view?</h3><p>4H upside confirmation: close above <strong>${price(p?.bull.trigger)}</strong> and hold the retest. Next known zone: ${price(p?.bull.objective)}.</p><p>4H downside confirmation: close below <strong>${price(p?.bear.trigger)}</strong> and fail to reclaim it. Next known zone: ${price(p?.bear.objective)}.</p><small>A failed upside break falls back below ${price(p?.bull.invalidation)}; a failed downside break reclaims ${price(p?.bear.invalidation)}. No predicted arrival time is established for these levels.</small></article>
 <article><h3>What is helping or hurting?</h3><p><strong>Measured headwinds:</strong> ${escape(b.headwinds.join(', ')||'None established from available changes')}.</p><p><strong>Measured tailwinds:</strong> ${escape(b.tailwinds.join(', ')||'None established from available changes')}.</p><small>Directional interpretations use lagged changes. Missing data are not neutral. Dollar and yield effects can overlap; they are not added into a confidence score.</small></article>
 </div>
 <h3 class="outlook-subtitle">Dated price projections — shown with their uncertainty</h3>
 <p class="footnote">These are existing model outputs, not a newly validated market-wide forecast. Targets remain tied to their original futures price and date; they are never shifted to match the live spot quote.</p>
 <div class="forecast-grid outlook-forecasts">${Object.entries(a.forecasts).map(([h,f])=>{
  const r=f.research,o=f.outlook||{};
  const expired=!o.target_time||Date.parse(o.target_time)<=Date.now();
  const status=stale?'STALE · PRIOR ESTIMATE':expired?'ELAPSED · PRIOR ESTIMATE':f.status==='VALIDATED'?'VALIDATED MODEL':'EXPERIMENTAL · UNVALIDATED';
  return `<article class="forecast" data-outlook-horizon="${h}"><h3>${h} · ${escape(o.direction||'UNAVAILABLE')}</h3>${badge(status)}<div class="forecast-label">Model central estimate</div><div class="headline">${price(r?.price)}</div><p class="forecast-date">For ${time(o.target_time)}</p><p>Model 80% range<br><strong>${r?.interval80?price(r.interval80[0])+' – '+price(r.interval80[1]):'Unavailable'}</strong></p><p class="footnote">Origin ${time(r?.origin)} · reference ${price(r?.reference_price)}<br>Recent average error ≈ ${price(o.recent_mean_absolute_error_dollars)}<br>Measured recent range coverage: ${pct(f.holdout?.coverage80)}.</p><p class="footnote">${o.range_contains_no_change?'Range includes both gains and losses; direction is uncertain.':'A one-sided model range is not a guarantee.'} ${f.status!=='VALIDATED'?'Accuracy checks have not passed.':''}</p></article>`;
 }).join('')}</div>
 <p class="footnote">${escape(b.note)}</p>
 <h3 class="outlook-subtitle">Market factors: observation → interpretation</h3>
 <div class="market-factor-grid">${factors.map(f=>`<article data-market-factor="${escape(f.key)}"><div class="panel-head"><h3>${escape(f.name)}</h3>${badge(f.impact,color(f.impact))}</div><p><strong>${n(f.value,3)} ${escape(f.units)}</strong> · ${escape(f.movement)}</p><small>${escape(f.status)} · observation ${time(f.source_time)}${f.comparison_time?'<br>Compared with '+time(f.comparison_time):'<br>No verified comparison series available.'}</small><p>${escape(f.meaning)}</p><a href="${escape(f.url)}" target="_blank" rel="noopener noreferrer">${escape(f.source)}</a></article>`).join('')}</div>
 <p class="footnote">Daily market comparisons use five available observations, with a conservative 24-hour availability delay; they are not live intraday changes. Observations older than five calendar days lose directional labels. FRED observations retain their own release timing. These relationships suggest possible pressure, not proof of what caused a move.</p>
 <h3 class="outlook-subtitle">Demand, supply & news: what is known</h3>
 <div class="market-factor-grid">${reports.map(r=>{
  const expired=r.expired||Date.now()>=Date.parse(r.review_after);
  return `<article data-market-report><h3>${escape(r.topic)}</h3>${badge(expired?'REVIEW DUE · BACKGROUND ONLY':r.status)}<p><strong>${escape(r.impact)}</strong></p><p>${escape(r.finding)}</p><p>${escape(r.meaning)}</p><small>Period: ${escape(r.period)}<br>Published ${escape(r.published_date)} · verified ${time(r.verified_at)}<br>Review due ${time(r.review_after)}. Manually researched; not an automatic news feed.</small><p><a href="${escape(r.url)}" target="_blank" rel="noopener noreferrer">${escape(r.publisher)} · source</a></p></article>`;
 }).join('')||'<p>No verified report available for this snapshot date.</p>'}</div>
 <p class="footnote">${escape(b.difference)}</p>
 <details><summary>Official Fed releases & coverage gaps</summary>
 <p>Official feed checked ${time(data.policy_releases?.checked_at)} · ${escape(data.policy_releases?.status||'UNAVAILABLE')}. Headlines are references only; they are not classified as rate surprises or used to move targets.</p>
 ${(data.policy_releases?.items||[]).map(r=>`<p><a href="${escape(r.url)}" target="_blank" rel="noopener noreferrer">${escape(r.title)}</a><br>${time(r.published_at)}</p>`).join('')}
 <ul>${b.limitations.map(s=>'<li>'+escape(s)+'</li>').join('')}</ul>
 <p>Watch the next verified inflation, employment and central-bank releases. Hotter inflation is not automatically bullish for metals: the dollar and real-yield response matters. No event dates or policy probabilities are inferred from headlines.</p>
 </details></section>`;
}
