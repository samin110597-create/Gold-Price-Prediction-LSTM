from pathlib import Path

P=Path('index.html')
s=P.read_text(encoding='utf-8')

# Make the target cell support true measured moves, structural objectives, and dual targets.
old="<span>Measured target<br><b>${money(p.target)}</b></span>"
new="<span>${p.target_type||'Measured move'}<br><b>${Array.isArray(p.targets)&&p.targets.length?`${money(p.targets[0])} / ${money(p.targets[1])}`:money(p.target)}</b></span>"
s=s.replace(old,new)

# Add the basis beneath the pattern so the number is explainable.
old2="<div class=\"small\">${p.detail||''}</div></div>"
new2="<div class=\"small\">${p.detail||''}${p.target_basis?`<br><b>Target basis:</b> ${p.target_basis}`:''}</div></div>"
s=s.replace(old2,new2)

P.write_text(s,encoding='utf-8')
print('Measured move labels and target basis patched')
