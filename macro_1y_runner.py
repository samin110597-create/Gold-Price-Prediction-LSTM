import json
from datetime import datetime, timezone

import macro_1y_model_v3 as core


def main():
    for asset, cfg in core.ASSETS.items():
        try:
            panel, input_status = core.build_panel(cfg)
            macro = core.walkforward(panel)
            promoted, comparison = core.maybe_promote(asset, cfg, macro)
            payload = {
                "status": "ok",
                "asset": asset,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "inputs_available": input_status,
                "macro_forecast": macro,
                "selected_as_champion": promoted,
                "comparison": comparison,
                "method_note": (
                    "Dedicated 1Y challenger uses the exact live formula in its OOS audit; "
                    "it cannot replace the incumbent unless fixed promotion safeguards are met."
                ),
            }
        except Exception as exc:
            payload = {
                "status": "unavailable",
                "asset": asset,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "selected_as_champion": False,
                "error": str(exc),
                "method_note": (
                    "Macro V3 challenger failed safely; incumbent 1Y model remains active."
                ),
            }
        (core.DATA / cfg["output"]).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
