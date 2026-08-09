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
    return f"${float(x):,.2f}"


def pct(x):
    return f"{float(x) * 100:.1f}%"


def compose(asset_key, cfg, audit_all):
    f = load(cfg["forecast"])
    t = load(cfg["technical"])
    p = load(cfg["projection"])
    audit = (audit_all.get("assets") or {}).get(asset_key, {})
    if not f:
        return

    projections = {x.get("horizon"): x for x in p.get("projections", [])}
    audited_projections = audit.get("price_projections", [])
    passed = [x for x in audited_projections if x.get("pass")]

    passed_details = []
    for a in passed:
        pr = projections.get(a.get("horizon"))
        if not pr:
            continue
        zone = pr.get("tight_model_zone") or [None, None]
        passed_details.append({
            "horizon": pr.get("horizon"),
            "price": pr.get("model_price"),
            "zone": zone,
            "directional_accuracy": a.get("directional_accuracy"),
            "edge": a.get("directional_edge"),
            "mae_skill": a.get("mae_skill_vs_baseline"),
        })

    tech_score = t.get("technical_score")
    tech_bias = t.get("technical_bias", "Unavailable")
    structure = t.get("market_structure", "not clear")
    candle = t.get("candlestick", "no clear candle signal")
    indicators = t.get("indicators") or {}
    confirm = t.get("confirmation") or {}

    supportive = [x.get("name") for x in f.get("factors", []) if x.get("impact") == "Supportive"]
    headwinds = [x.get("name") for x in f.get("factors", []) if x.get("impact") == "Headwind"]
    risk_on = [x.get("name") for x in f.get("factors", []) if x.get("impact") == "Risk-on"]
    risk_off = [x.get("name") for x in f.get("factors", []) if x.get("impact") == "Risk-off"]

    f1 = next((x for x in f.get("forecasts", []) if x.get("horizon_days") == 1), {})
    f5 = next((x for x in f.get("forecasts", []) if x.get("horizon_days") == 5), {})
    f20 = next((x for x in f.get("forecasts", []) if x.get("horizon_days") == 20), {})

    audit_grade = audit.get("audit_grade", "PENDING")
    pass_rate = audit.get("component_pass_rate")
    pass_rate_text = pct(pass_rate) if pass_rate is not None else "not ready yet"

    existing = f.get("guru_summary") or {}
    bias = existing.get("bias") or f.get("regime", {}).get("overall", "Neutral")

    if audit_grade == "PASS":
        conviction = "Good"
    elif passed_details:
        conviction = "Use only passed forecasts"
    else:
        conviction = "Low"

    factor_bits = []
    if supportive:
        factor_bits.append("helping: " + ", ".join(supportive[:3]))
    if headwinds:
        factor_bits.append("hurting: " + ", ".join(headwinds[:3]))
    if risk_on:
        factor_bits.append("risk mood is positive: " + ", ".join(risk_on[:2]))
    if risk_off:
        factor_bits.append("risk mood is weak: " + ", ".join(risk_off[:2]))
    factor_text = "; ".join(factor_bits) if factor_bits else "outside market signals are mixed"

    if tech_score is not None:
        tech_text = (
            f"Technical score is {tech_score:.0f}/100, which is {tech_bias}. "
            f"The chart structure is {structure}. RSI is {indicators.get('rsi14', '—')}, "
            f"ADX is {indicators.get('adx14', '—')}, and the latest candle signal is {candle}."
        )
    else:
        tech_text = "Technical data is not ready."

    prob_text = (
        f"Model chance of price going up: 1 day {pct(f1.get('probability_up', .5))}, "
        f"1 week {pct(f5.get('probability_up', .5))}, 20 days {pct(f20.get('probability_up', .5))}."
    )

    if passed_details:
        first = passed_details[0]
        validated = (
            f"The backtest says {len(passed_details)} price forecast(s) currently pass. "
            f"The nearest passed forecast is {first['horizon']} with a model price of {money(first['price'])}. "
            f"Its model zone is {money(first['zone'][0])} to {money(first['zone'][1])}. "
            f"Its past direction accuracy was {pct(first['directional_accuracy'])}."
        )
    else:
        validated = "No price forecast currently passes the backtest, so do not rely on a model target right now."

    model_view = (
        f"{cfg['label']} is currently {f.get('regime', {}).get('overall', 'mixed')}. "
        f"The main trend is {f.get('regime', {}).get('trend', 'mixed')}. "
        f"The outside market picture is {f.get('regime', {}).get('macro', 'mixed').lower()}. "
        f"{tech_text} {prob_text} Other market signals are {factor_text}. {validated}"
    )

    bull = confirm.get("bullish_above") or f.get("levels", {}).get("resistance_20d")
    bear = confirm.get("bearish_below") or f.get("levels", {}).get("support_20d")
    if bull is not None and bear is not None:
        trigger = (
            f"Bullish confirmation: price needs to stay above {money(bull)}. "
            f"Bearish confirmation: price needs to stay below {money(bear)}. "
            f"A move is stronger when the chart indicators and a passed model agree with it."
        )
    else:
        trigger = existing.get("trigger", "Confirmation levels are not ready.")

    if passed_details:
        passed_names = ", ".join(x["horizon"] for x in passed_details)
        risk = (
            f"Overall audit result is {audit_grade}. About {pass_rate_text} of the tested parts passed. "
            f"Only these price forecasts currently pass: {passed_names}. "
            f"Ignore failed forecasts until they improve. Big news, interest rates, the US dollar, or sudden volatility can quickly change the setup."
        )
    else:
        risk = (
            f"Overall audit result is {audit_grade}. About {pass_rate_text} of the tested parts passed. "
            f"No price forecast passes right now, so the safest reading is: no proven model edge at this moment."
        )

    f["guru_summary"] = {
        "bias": bias,
        "conviction": conviction,
        "model_view": model_view,
        "trigger": trigger,
        "risk": risk,
        "regime": f.get("regime", {}).get("overall", "Neutral / mixed"),
        "audit_grade": audit_grade,
        "component_pass_rate": pass_rate,
        "passed_projection_horizons": [x["horizon"] for x in passed_details],
    }

    (DATA / cfg["forecast"]).write_text(json.dumps(f, indent=2), encoding="utf-8")


def main():
    audit = load("model_backtest.json")
    for key, cfg in ASSETS.items():
        compose(key, cfg, audit)
    print("Simple market-guru summaries written")


if __name__ == "__main__":
    main()
