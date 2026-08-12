import json
from pathlib import Path

DATA = Path("data")


def main():
    path = DATA / "model_backtest.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))

    all_ready = True
    for asset, a in (data.get("assets") or {}).items():
        probs = a.get("probability_forecasts") or []
        prices = a.get("price_projections") or []
        techs = a.get("technical_champions") or []

        prob_pass = sum(bool(x.get("pass")) for x in probs)
        price_pass = sum(bool(x.get("pass")) for x in prices)
        tech_pass = sum(bool(x.get("pass")) for x in techs)
        one_year = next((x for x in prices if x.get("horizon") == "1 Year"), {})
        one_year_pass = bool(one_year.get("pass"))
        pass_rate = float(a.get("component_pass_rate") or 0)

        reasons = []
        if pass_rate < 0.60:
            reasons.append(f"overall champion pass rate is {pass_rate:.0%}, below 60%")
        if price_pass < 3:
            reasons.append(f"only {price_pass}/4 price horizons pass")
        if tech_pass < 2:
            reasons.append(f"only {tech_pass}/3 technical champion horizons pass")
        if prob_pass < 1:
            reasons.append("no probability horizon passes")
        if not one_year_pass:
            reasons.append("1-year champion does not pass")

        predictive_ready = not reasons
        all_ready = all_ready and predictive_ready
        a["process_readiness"] = "INSTITUTIONAL-STYLE CONTROLS ACTIVE"
        a["predictive_readiness"] = "READY" if predictive_ready else "NOT READY"
        a["readiness_reasons"] = reasons if reasons else ["pre-defined predictive-readiness criteria are met"]
        a["governance_controls"] = [
            "purged chronological walk-forward validation",
            "champion-challenger promotion rules",
            "selective technical abstention / no-edge state",
            "dedicated macro 1Y challenger",
            "one-month lag on FRED macro inputs",
            "weak probability gating",
            "challenger fail-safe: incumbent remains active on failure",
        ]

    data["system_process_readiness"] = "INSTITUTIONAL-STYLE CONTROLS ACTIVE"
    data["system_predictive_readiness"] = "READY" if all_ready else "NOT READY"
    data["institutional_grade_claim_allowed"] = bool(all_ready)
    data["institutional_grade_note"] = (
        "Process controls can be institutional-style even when predictive evidence is not yet strong enough. "
        "Do not describe the system as institutional-grade predictive quality unless institutional_grade_claim_allowed is true."
    )
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
