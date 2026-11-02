"""Static sanity checks for paper/main.tex (no LaTeX install needed): cite keys, labels/refs, environments, input files, figures."""
import re, sys
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "paper"
tex = (P / "main.tex").read_text(encoding="utf-8")
bib = (P / "references.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cites = {k.strip() for c in re.findall(r"\\cite\{([^}]*)\}", tex) for k in c.split(",")}
ok = True
def chk(cond, msg):
    global ok
    if not cond: ok = False; print("PROBLEM:", msg)
chk(cites <= keys, f"cite keys missing in bib: {cites - keys}")
print("uncited bib entries:", sorted(keys - cites))
labels = set(re.findall(r"\\label\{([^}]*)\}", tex)); refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
chk(refs <= labels, f"refs without labels: {refs - labels}")
for env in set(re.findall(r"\\begin\{([^}]*)\}", tex)):
    chk(len(re.findall(r"\\begin\{%s\}" % re.escape(env), tex)) == len(re.findall(r"\\end\{%s\}" % re.escape(env), tex)), f"unbalanced env {env}")
chk(tex.count("{") == tex.count("}") or True, "")
depth = 0
for ch in re.sub(r"\\[{}]", "", tex):
    depth += (ch == "{") - (ch == "}")
    if depth < 0: break
chk(depth == 0, f"brace depth {depth}")
for f in re.findall(r"\\input\{([^}]*)\}", tex): chk((P / f).exists(), f"missing input {f}")
for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex): chk((P / f).exists(), f"missing figure {f}")
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
