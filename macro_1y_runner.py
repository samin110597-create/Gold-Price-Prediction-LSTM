import json
from datetime import datetime, timezone

import macro_1y_model as core


def main():
    for asset, cfg in core.ASSETS.items():
        try:
            panel, fred_ok = core.build_panel(cfg)
            macro = core.walkforward(panel)
            promoted, comparison = core.maybe_promote(asset, cfg, macro)
            payload = {
                "status": "ok",
                "asset": asset,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "fred_inputs_available": fred_ok,
                "macro_forecast": macro,
                "selected_as_champion": promoted,
                "comparison": comparison,
                "method_note": "Dedicated 1Y challenger cannot replace the incumbent unless pre-defined OOS safeguards are met.",
            }
        except Exception as exc:
            payload = {
                "status": "unavailable",
                "asset": asset,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "selected_as_champion": False,
                "error": str(exc),
                "method_note": "Macro challenger failed safely; incumbent 1Y model remains active.",
            }
        (core.DATA / cfg["output"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
