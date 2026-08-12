import json
from pathlib import Path

DATA = Path("data")
FILES = ("live_forecast.json", "silver_forecast.json")


def process(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    for f in data.get("forecasts", []):
        p = float(f.get("probability_up", 0.5) or 0.5)
        edge = float(f.get("backtest_edge", 0) or 0)
        brier = float(f.get("brier_score", 1) or 1)
        obs = int(f.get("test_observations", 0) or 0)
        passed = bool(edge > 0 and brier < 0.25 and obs >= 250)
        f["raw_probability_up"] = round(p, 4)
        f["forecast_status"] = "VALIDATED" if passed else "NO EDGE"
        f["use_in_summary"] = passed
        if not passed:
            f["signal"] = "NO EDGE / ABSTAIN"
            f["actionable_probability_up"] = 0.5
        else:
            f["actionable_probability_up"] = round(p, 4)
    data["probability_gate_note"] = (
        "Probability forecasts that do not beat baseline with acceptable Brier score are excluded from the top-level view. "
        "Raw probabilities remain visible for audit."
    )
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    for name in FILES:
        path = DATA / name
        if path.exists():
            process(path)


if __name__ == "__main__":
    main()
