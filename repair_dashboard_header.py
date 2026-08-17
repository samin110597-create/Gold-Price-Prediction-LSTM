from pathlib import Path
import re

INDEX = Path('index.html')
PATCH = Path('dashboard_v4_patch.py')

# Keep the top of the dashboard intentionally minimal.
s = INDEX.read_text(encoding='utf-8')
s, n = re.subn(
    r'<div class="eyebrow">.*?</div><h1>Metals Intelligence</h1><div class="sub">.*?</div>',
    '<div class="eyebrow">Gold • Silver</div><h1>Metals Intelligence</h1><div class="sub">Forecasts, technicals and model audit.</div>',
    s,
    count=1,
    flags=re.S,
)
if n != 1:
    raise RuntimeError('Could not locate dashboard header')
INDEX.write_text(s, encoding='utf-8')

# Fix the V4 patch itself so repeated workflow runs never append header text again.
t = PATCH.read_text(encoding='utf-8')
old = "s=s.replace('calibrated price forecasts, spot Gold/Silver charts','visible V4 specialists, restored 4H comparison, calibrated price forecasts, spot Gold/Silver charts')"
new = "s=re.sub(r'<div class=\"eyebrow\">.*?</div><h1>Metals Intelligence</h1><div class=\"sub\">.*?</div>', '<div class=\"eyebrow\">Gold • Silver</div><h1>Metals Intelligence</h1><div class=\"sub\">Forecasts, technicals and model audit.</div>', s, count=1, flags=re.S)"
if old in t:
    t = t.replace(old, new)
PATCH.write_text(t, encoding='utf-8')

print('Dashboard header cleaned and V4 patch made idempotent')
