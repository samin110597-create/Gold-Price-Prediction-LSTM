import json
from datetime import datetime, timezone

import technical_context_meta_model as core


def main():
    for asset,cfg in core.ASSETS.items():
        try:
            core.run_asset(asset,cfg)
        except Exception as exc:
            payload={"status":"unavailable","asset":asset,"updated_utc":datetime.now(timezone.utc).isoformat(),"horizons":[],"error":str(exc),"note":"Technical context challenger failed safely; incumbent technical champion remains active."}
            (core.DATA/cfg["output"]).write_text(json.dumps(payload,indent=2),encoding="utf-8")

if __name__=="__main__":main()
