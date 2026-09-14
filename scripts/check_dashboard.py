"""Exercise both asset views, both modes and every timeframe, including the deployed site."""
import argparse
import json
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser()
parser.add_argument("--url",default="http://127.0.0.1:8000")
parser.add_argument("--expected-run")
parser.add_argument("--wait",type=int,default=0)
args=parser.parse_args()
Path("screenshots").mkdir(exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(viewport={"width":1440,"height":1100},device_scale_factor=1)
    errors=[]
    page.on("pageerror",lambda e:errors.append(str(e)))
    deadline=time.monotonic()+args.wait
    while True:
        page.goto(args.url,wait_until="networkidle",timeout=60000)
        try:
            page.wait_for_function("window.__metalsReady === true",timeout=20000)
            run_id=page.locator("#app").get_attribute("data-run-id")
            if not args.expected_run or run_id==args.expected_run:
                break
        except Exception:
            pass
        if time.monotonic()>=deadline:
            raise AssertionError("Expected coherent deployed snapshot did not become available")
        page.wait_for_timeout(10000)
    report={"url":args.url,"run_id":run_id,"assets":{},"errors":[]}
    for asset in ("gold","silver"):
        page.locator("#"+asset+"-tab").click()
        assert page.locator("#app").get_attribute("data-asset")==asset
        assert page.locator('[data-testid="price"]').inner_text().startswith("$")
        for mode in ("Strict","Adaptive"):
            page.locator('[data-mode="'+mode+'"]').click()
            assert page.locator('[data-mode="'+mode+'"]').get_attribute("aria-pressed")=="true"
        for tf in ("15m","1h","4h","1d","1w"):
            page.locator('[data-tf="'+tf+'"]').click()
            assert page.locator('[data-tf="'+tf+'"]').get_attribute("aria-pressed")=="true"
            assert page.locator("#chart").evaluate("(c)=>c.width>0 && c.height>0")
        page.locator('[data-tf="1d"]').click()
        page.locator('[data-mode="Strict"]').click()
        page.screenshot(path="screenshots/"+asset+"-desktop.png",full_page=True)
        report["assets"][asset]={"price":page.locator('[data-testid="price"]').inner_text(),"state":page.locator('[data-testid="setup-state"]').inner_text(),"modes_checked":2,"timeframes_checked":5}
        page.set_viewport_size({"width":390,"height":844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth+1"),"Mobile horizontal overflow"
        page.screenshot(path="screenshots/"+asset+"-mobile.png",full_page=True)
        page.set_viewport_size({"width":1440,"height":1100})
    assert not errors,errors
    report["errors"]=errors
    Path("screenshots/verification.json").write_text(json.dumps(report,indent=2))
    print("BROWSER VERIFIED",json.dumps(report),flush=True)
    browser.close()
