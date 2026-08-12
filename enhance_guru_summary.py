import json
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
            "selected_model": a.get("selected_model", pr.get("selected_model", "Incumbent")),
            "edge": a.get("directional_edge"),
            "skill": a.get("mae_skill_vs_baseline"),
        })

    tech_score = t.get("technical_score")
    tech_bias = t.get("technical_bias", "Unavailable")
    structure = t.get("market_structure", "not clear")
    ind = t.get("indicators") or {}
    confirm = t.get("confirmation") or {}

    valid_probs = [x for x in f.get("forecasts", []) if x.get("use_in_summary")]
    if valid_probs:
        prob_text = "Validated probability signals: " + "; ".join(
            f"{x.get('horizon_days')}D {pct(x.get('actionable_probability_up', x.get('probability_up')))} up"
            for x in valid_probs
        ) + "."
    else:
        prob_text = "The probability engine currently has no validated edge, so it is excluded from this top-level view."

    tech_champs = audit.get("technical_champions") or []
    passed_tech = [x for x in tech_champs if x.get("pass")]
    if passed_tech:
        tech_pred_text = "Validated technical prediction: " + "; ".join(
            f"{x.get('horizon_days')}D {x.get('source')} with {pct(x.get('accuracy'))} accuracy and {pct(x.get('edge'))} edge"
            for x in passed_tech
        ) + "."
    else:
        tech_pred_text = "No technical prediction horizon currently proves positive edge; technicals are used as context, not prediction."

    if price_bits:
        nearest = price_bits[0]
        price_text = (
            f"Passed price horizons: {', '.join(x['horizon'] for x in price_bits)}. "
            f"Nearest passed target is {nearest['horizon']} at {money(nearest['price'])} using {nearest['selected_model']}."
        )
    else:
        price_text = "No price horizon currently passes the audit, so no model target should be treated as validated."

    one_year = projections.get("1 Year") or {}
    one_year_model = one_year.get("selected_model", "Incumbent projection model")
    one_year_status = next((x for x in audit.get("price_projections", []) if x.get("horizon") == "1 Year"), {})
    one_year_text = (
        f"1-year model: {one_year_model}. Current 1-year price estimate is {money(one_year.get('model_price'))}. "
        f"Audit status: {'PASS' if one_year_status.get('pass') else 'FAIL / estimate only'}."
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
        f"Predictive readiness: {readiness}. Failed components are excluded from the top-level view. "
        "A passing backtest is evidence, not a guarantee of future accuracy."
    )

    f["guru_summary"] = {
        "bias": f.get("regime", {}).get("overall", "Neutral / mixed"),
        "conviction": "Use passed components only" if price_bits or passed_tech or valid_probs else "Low / no proven edge",
        "model_view": model_view,
        "trigger": trigger,
        "risk": risk,
        "audit_grade": audit_grade,
        "component_pass_rate": pass_rate,
        "passed_projection_horizons": [x["horizon"] for x in price_bits],
        "passed_technical_horizons": [x.get("horizon_days") for x in passed_tech],
        "validated_probability_horizons": [x.get("horizon_days") for x in valid_probs],
        "one_year_selected_model": one_year_model,
    }

    (DATA / cfg["forecast"]).write_text(json.dumps(f, indent=2), encoding="utf-8")


def main():
    audit = load("model_backtest.json")
    for key, cfg in ASSETS.items():
        compose(key, cfg, audit)
    print("Simple evidence-gated market-guru summaries written")


if __name__ == "__main__":
    main()
