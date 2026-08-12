import json
from datetime import datetime, timezone

import technical_selective_model as core


def main():
    for asset, cfg in core.ASSETS.items():
        rows = []
        errors = []
        try:
            df = core.download(cfg["ticker"])
            for h in core.HORIZONS:
                try:
                    rows.append(core.horizon_model(df, h))
                except Exception as exc:
                    errors.append(f"{h}D: {exc}")
                    rows.append({
                        "horizon_days": h,
                        "signal": "NO EDGE / UNAVAILABLE",
                        "active_signal": False,
                        "audit_pass": False,
                        "selective_edge": None,
                        "coverage": 0,
                        "oos_signals": 0,
                        "error": str(exc),
                    })
        except Exception as exc:
            errors.append(str(exc))

        payload = {
            "status": "ok" if any(x.get("selective_edge") is not None for x in rows) else "unavailable",
            "asset": asset,
            "symbol": cfg["ticker"],
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "method": "Selective technical model with nested chronological calibration and abstention.",
            "horizons": rows,
            "errors": errors,
            "note": "A challenger failure never replaces or disables the incumbent technical model.",
        }
        (core.DATA / cfg["output"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
