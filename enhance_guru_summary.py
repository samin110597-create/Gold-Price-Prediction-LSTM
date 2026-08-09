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
    structure = t.get("market_structure", "structure unavailable")
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
    pass_rate_text = pct(pass_rate) if pass_rate is not None else "pending"

    existing = f.get("guru_summary") or {}
    bias = existing.get("bias") or f.get("regime", {}).get("overall", "Neutral")

    if audit_grade == "PASS":
        conviction = "Validated / selective"
    elif passed_details:
        conviction = "Selective only"
    else:
        conviction = "Low / no validated edge"

    factor_bits = []
    if supportive:
        factor_bits.append("supportive: " + ", ".join(supportive[:3]))
    if headwinds:
        factor_bits.append("headwinds: " + ", ".join(headwinds[:3]))
    if risk_on:
        factor_bits.append("risk-on: " + ", ".join(risk_on[:2]))
    if risk_off:
        factor_bits.append("risk-off: " + ", ".join(risk_off[:2]))
    factor_text = "; ".join(factor_bits) if factor_bits else "cross-market inputs are mixed"

    tech_text = (
        f"Technical confluence is {tech_score:.0f}/100 ({tech_bias}) with {structure}. "
        f"RSI {indicators.get('rsi14', '—')}, ADX {indicators.get('adx14', '—')}, "
        f"MACD histogram {indicators.get('macd_hist', '—')}; latest candle read: {candle}."
        if tech_score is not None
        else "Technical layer is not available."
    )

    prob_text = (
        f"Probability layer: 1D up {pct(f1.get('probability_up', .5))}, "
        f"1W up {pct(f5.get('probability_up', .5))}, 20D up {pct(f20.get('probability_up', .5))}."
    )

    if passed_details:
        pd = passed_details[0]
        validated = (
            f"Walk-forward validation currently passes {len(passed_details)} price horizon(s). "
            f"Nearest validated view: {pd['horizon']} model price {money(pd['price'])}, "
            f"model zone {money(pd['zone'][0])}–{money(pd['zone'][1])}, "
            f"directional accuracy {pct(pd['directional_accuracy'])} with edge {pct(pd['edge'])}."
        )
    else:
        validated = "No price-projection horizon currently passes the audit; treat model prices as research outputs only."

    model_view = (
        f"{cfg['label']} is {f.get('regime', {}).get('overall', 'mixed')} with "
        f"{f.get('regime', {}).get('trend', 'mixed trend')} and a "
        f"{f.get('regime', {}).get('macro', 'mixed').lower()} macro backdrop. "
        f"{tech_text} {prob_text} Cross-market read: {factor_text}. {validated}"
    )

    bull = confirm.get("bullish_above") or f.get("levels", {}).get("resistance_20d")
    bear = confirm.get("bearish_below") or f.get("levels", {}).get("support_20d")
    trigger = (
        f"Bullish confirmation requires sustained trade above {money(bull)}; bearish confirmation sits below {money(bear)}. "
        f"A breakout should ideally be confirmed by improving MACD/ADX, supportive cross-market factors, and a passing short/medium-horizon model rather than price alone."
        if bull is not None and bear is not None
        else existing.get("trigger", "Confirmation levels unavailable.")
    )

    if passed_details:
        passed_names = ", ".join(x["horizon"] for x in passed_details)
        risk = (
            f"Overall audit grade is {audit_grade} with {pass_rate_text} of tested components passing. "
            f"Only these projection horizons currently pass: {passed_names}. Do not promote failed horizons because their historical edge is not proven. "
            f"The main model risks are regime change, macro-event gaps, stale weekly positioning data, and sudden volatility expansion."
        )
    else:
        risk = (
            f"Overall audit grade is {audit_grade} with {pass_rate_text} of tested components passing. "
            f"No price horizon currently clears the validation rule, so the correct institutional read is 'no validated edge' rather than forcing a target."
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
    print("Enhanced market-guru summaries written")


if __name__ == "__main__":
    main()
