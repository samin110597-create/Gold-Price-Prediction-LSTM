import json
import subprocess
from pathlib import Path

DATA = Path("data")
ASSETS = {
    "gold": {
        "forecast": "live_forecast.json",
        "technical": "gold_technicals.json",
        "projection": "gold_projections.json",
        "label": "Gold",
    },
    "silver": {
        "forecast": "silver_forecast.json",
        "technical": "silver_technicals.json",
        "projection": "silver_projections.json",
        "label": "Silver",
    },
}


def load(name):
    p = DATA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "—"


def pct(x):
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "—"


def compose(asset_key, cfg, audit_all):
    f = load(cfg["forecast"])
    t = load(cfg["technical"])
    p = load(cfg["projection"])
    audit = (audit_all.get("assets") or {}).get(asset_key, {})
    if not f:
        return

    projections = {x.get("horizon"): x for x in p.get("projections", [])}
    passed_prices = [x for x in audit.get("price_projections", []) if x.get("pass")]
    price_bits = []
    for a in passed_prices:
        pr = projections.get(a.get("horizon"))
        if not pr:
            continue
        price_bits.append({
            "horizon": pr.get("horizon"),
            "price": pr.get("model_price"),
            "zone": pr.get("tight_model_zone"),
            "probability_up": pr.get("probability_up"),
            "selected_model": a.get("selected_model", pr.get("selected_model", "Incumbent")),
            "edge": a.get("directional_edge"),
            "skill": a.get("mae_skill_vs_baseline"),
        })

    tech_score = t.get("technical_score")
    tech_bias = t.get("technical_bias", "Unavailable")
    structure = t.get("market_structure", "not clear")
    ind = t.get("indicators") or {}
    confirm = t.get("confirmation") or {}

    passed_probs = [x for x in audit.get("probability_forecasts", []) if x.get("pass")]
    if passed_probs:
        prob_text = "PASS probability signals: " + "; ".join(
            f"{x.get('horizon_days')}D {pct(x.get('probability_up'))} up"
            for x in passed_probs
        ) + "."
    else:
        prob_text = "Probability backtest: FAIL on all current horizons, so those percentages are shown only as model probabilities."

    tech_champs = audit.get("technical_champions") or []
    passed_tech = [x for x in tech_champs if x.get("pass")]
    if passed_tech:
        tech_pred_text = "PASS technical prediction: " + "; ".join(
            f"{x.get('horizon_days')}D {x.get('source')} with {pct(x.get('accuracy'))} accuracy and {pct(x.get('edge'))} edge"
            for x in passed_tech
        ) + "."
    else:
        tech_pred_text = "Technical prediction backtest: FAIL on all current horizons; technicals remain context only."

    if price_bits:
        nearest = price_bits[0]
        price_text = (
            f"PASS price horizons: {', '.join(x['horizon'] for x in price_bits)}. "
            f"Nearest PASS price is {nearest['horizon']} at {money(nearest['price'])} "
            f"with {pct(nearest.get('probability_up'))} model probability up."
        )
    else:
        price_text = "Price backtest: FAIL on all current horizons. Prices can still be displayed with probabilities, but none has earned PASS status."

    one_year = projections.get("1 Year") or {}
    one_year_model = one_year.get("selected_model", "Incumbent projection model")
    one_year_status = next((x for x in audit.get("price_projections", []) if x.get("horizon") == "1 Year"), {})
    one_year_text = (
        f"1-year model: {one_year_model}. Current model price is {money(one_year.get('model_price'))}. "
        f"Backtest: {'PASS' if one_year_status.get('pass') else 'FAIL'}. "
        f"Model probability up: {pct(one_year.get('probability_up'))}."
    )

    supportive = [x.get("name") for x in f.get("factors", []) if x.get("impact") == "Supportive"]
    headwinds = [x.get("name") for x in f.get("factors", []) if x.get("impact") == "Headwind"]
    factor_parts = []
    if supportive:
        factor_parts.append("helping: " + ", ".join(supportive[:3]))
    if headwinds:
        factor_parts.append("hurting: " + ", ".join(headwinds[:3]))
    factor_text = "; ".join(factor_parts) if factor_parts else "cross-market factors are mixed"

    audit_grade = audit.get("audit_grade", "PENDING")
    pass_rate = audit.get("component_pass_rate")
    readiness = audit.get("predictive_readiness", "NOT ASSESSED")

    chart_text = (
        f"Technical score is {tech_score:.0f}/100 ({tech_bias}) and chart structure is {structure}. "
        f"RSI is {ind.get('rsi14', '—')}, ADX is {ind.get('adx14', '—')}."
        if tech_score is not None else "Technical data is not ready."
    )

    model_view = (
        f"{cfg['label']} trend: {f.get('regime', {}).get('trend', 'mixed')}. "
        f"{chart_text} Cross-market picture: {factor_text}. "
        f"{prob_text} {tech_pred_text} {price_text} {one_year_text}"
    )

    bull = confirm.get("bullish_above") or f.get("levels", {}).get("resistance_20d")
    bear = confirm.get("bearish_below") or f.get("levels", {}).get("support_20d")
    if bull is not None and bear is not None:
        trigger = (
            f"Above {money(bull)} strengthens the higher-price case. "
            f"Below {money(bear)} strengthens the lower-price case. "
            "Between those levels, conviction should stay lower."
        )
    else:
        trigger = "Confirmation levels are not ready."

    risk = (
        f"Audit grade: {audit_grade}. Component pass rate: {pct(pass_rate)}. "
        f"Predictive readiness: {readiness}. Only PASS components receive predictive trust. "
        "FAIL components may still show a probability, but they have not proved enough historical edge. "
        "PASS is evidence, not a guarantee."
    )

    f["guru_summary"] = {
        "bias": f.get("regime", {}).get("overall", "Neutral / mixed"),
        "conviction": "Use PASS components only" if price_bits or passed_tech or passed_probs else "Low / no proven edge",
        "model_view": model_view,
        "trigger": trigger,
        "risk": risk,
        "audit_grade": audit_grade,
        "component_pass_rate": pass_rate,
        "passed_projection_horizons": [x["horizon"] for x in price_bits],
        "passed_technical_horizons": [x.get("horizon_days") for x in passed_tech],
        "passed_probability_horizons": [x.get("horizon_days") for x in passed_probs],
        "one_year_selected_model": one_year_model,
    }

    (DATA / cfg["forecast"]).write_text(json.dumps(f, indent=2), encoding="utf-8")


def patch_dashboard():
    """Keep the UI simple: every tested forecast is PASS or FAIL, plus probability."""
    path = Path("index.html")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    original = text

    replacements = [
        (
            "tech=a.technical_signals||[]",
            "tech=a.technical_champions||a.technical_signals||[]",
        ),
        (
            "function forecastState(x,a){if(a?.pass||x.forecast_status==='Validated')return {label:'VALIDATED',cls:'pass'};return {label:'ESTIMATE',cls:'fail'}}",
            "function forecastState(x,a){return a?.pass?{label:'PASS',cls:'pass'}:{label:'FAIL',cls:'fail'}}",
        ),
        (
            "No price horizon currently passes the full audit; estimates are still shown but clearly marked.",
            "No price horizon currently passes the full audit. Failed forecasts stay visible with probability only.",
        ),
        (
            "sub=`${x.horizon} • ${forecastState(x,best.a).label} • ML trust ${pct(x.ml_trust??0)}`",
            "sub=`${x.horizon} • ${forecastState(x,best.a).label} • UP ${pct(x.probability_up)} / DOWN ${pct(x.probability_down)}`",
        ),
        (
            "point price • focus zone • risk zone • validation",
            "point price • probability • backtest PASS/FAIL",
        ),
        (
            "Raw ML ${money(x.raw_ml_price??x.model_price)} • ML trust ${pct(x.ml_trust??0)}",
            "Probability: UP ${pct(x.probability_up)} • DOWN ${pct(x.probability_down)}",
        ),
        (
            "Up ${pct(x.probability_up)} • ${x.confidence||'—'} confidence",
            "Backtest: ${st.label} • ${x.confidence||'—'} model confidence",
        ),
        (
            "ML trust ${pct(x.ml_trust??0)} • model agreement ${pct(x.model_agreement)}",
            "UP ${pct(x.probability_up)} • DOWN ${pct(x.probability_down)} • model agreement ${pct(x.model_agreement)}",
        ),
        (
            "A PASS means that specific historical test beat its baseline. It is not a guarantee of future prices.",
            "Every forecast is backtested. PASS means it cleared the strict baseline, error, and sample-size rules. FAIL means it did not. Probability is still shown, but FAIL does not receive predictive trust.",
        ),
    ]

    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)

    if text != original:
        path.write_text(text, encoding="utf-8")
        # The workflow's later commit step commits all already-staged files.
        subprocess.run(["git", "add", "index.html"], check=True)
        print("Dashboard patched to PASS/FAIL + probability display")
    else:
        print("Dashboard PASS/FAIL patch already applied")


def main():
    audit = load("model_backtest.json")
    for key, cfg in ASSETS.items():
        compose(key, cfg, audit)
    patch_dashboard()
    print("Simple PASS/FAIL market-guru summaries written")


if __name__ == "__main__":
    main()
