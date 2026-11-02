"""Independent verification: recompute headline numbers straight from results/raw with separate code (no import from 04_analyze.py)
and compare with results/tables. Any difference > 1e-9 fails loudly. Log -> results/verification_log.txt.
"""
import sys, glob, re, csv
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import os
RAW, TAB = Path(os.environ.get("QH_RAW", ROOT / "results" / "raw")), Path(os.environ.get("QH_TABLES", ROOT / "results" / "tables"))
pat = re.compile(r"^(llama32_3b|qwen25_3b|phi35_mini)_(F16|Q[0-9A-Z_]+?)_(ref|imat|noimat)_([ABC])$")
data = {}
for f in glob.glob(str(RAW / "*.csv")):
    m = pat.match(Path(f).stem)
    if m:
        d = pd.read_csv(f)
        if len(d) == 600 and d.p_yes.notna().all():
            data[m.groups()] = d

sel = pd.read_csv(TAB / "selection.csv").set_index(["model", "level", "recipe", "template"])
gaps = pd.read_csv(TAB / "gaps.csv").set_index(["model", "level", "recipe", "template"])
flips = pd.read_csv(TAB / "flips.csv").set_index(["model", "level", "recipe", "template"])
log, bad, n = [], 0, 0


def check(name, mine, theirs, key):
    global bad, n
    n += 1
    diff = abs(float(mine) - float(theirs))
    ok = diff <= 1e-9
    bad += (not ok)
    log.append(f"{'PASS' if ok else 'FAIL'} {key} {name}: verify={float(mine):.12f} table={float(theirs):.12f} diff={diff:.2e}")


for key, d in sorted(data.items()):
    model, level, recipe, tmpl = key
    # per-resume mean p_yes of each group, computed with groupby (different route from the pivot arrays used in 04_analyze.py)
    g = d.groupby(["resume_id", "group"]).p_yes.mean().unstack()
    male = (g["WM"] + g["BM"]) / 2; female = (g["WF"] + g["BF"]) / 2
    white = (g["WF"] + g["WM"]) / 2; black = (g["BF"] + g["BM"]) / 2
    check("selection rate @0.5", (d.p_yes >= 0.5).sum() / len(d), sel.loc[key, "sel_rate"], key)
    check("mean p_yes", d.p_yes.sum() / len(d), sel.loc[key, "mean_p"], key)
    check("gender gap", (male - female).sum() / len(g), gaps.loc[key, "gender_gap"], key)
    check("race gap", (white - black).sum() / len(g), gaps.loc[key, "race_gap"], key)
    if level != "F16" and (model, "F16", "ref", tmpl) in data:
        r = data[(model, "F16", "ref", tmpl)].set_index("prompt_id").p_yes
        cur = d.set_index("prompt_id").p_yes
        flipped = ((cur >= 0.5) != (r.loc[cur.index] >= 0.5)).sum()
        check("flip rate vs F16", flipped / len(cur), flips.loc[key, "flip_rate"], key)


# ---- new checks (final pre-paper): yesno_mass, sensitivity_mass_filtered, selection_by_group undefined rule, perplexity_vs_drift
import numpy as np
ym = pd.read_csv(TAB / "yesno_mass.csv").set_index(["model", "level", "recipe", "template"])
sf = pd.read_csv(TAB / "sensitivity_mass_filtered.csv").set_index(["model", "level", "recipe", "template"])
sg = pd.read_csv(TAB / "selection_by_group.csv")
for key, d in sorted(data.items()):
    M = d.mass_yesno
    check("mass mean", M.mean(), ym.loc[key, "mean_mass"], key); check("mass median", M.median(), ym.loc[key, "median_mass"], key)
    check("mass p05", np.percentile(M.values, 5), ym.loc[key, "p05_mass"], key)
    check("share mass<0.5", (M < .5).mean(), ym.loc[key, "share_mass_below_0_5"], key); check("share mass<0.9", (M < .9).mean(), ym.loc[key, "share_mass_below_0_9"], key)
    kept = d[M >= .5]
    if len(kept): check("filtered sel rate", (kept.p_yes >= .5).mean(), sf.loc[key, "sel_rate_filtered"], key)
    check("n_kept", len(kept), sf.loc[key, "n_kept"], key)
    rates = d.groupby("group").p_yes.apply(lambda x: (x >= .5).mean())
    row = sg[(sg.model == key[0]) & (sg.level == key[1]) & (sg.recipe == key[2]) & (sg.template == key[3]) & (sg.threshold_type == "0.5")].iloc[0]
    degenerate = bool(((rates == 0) | (rates == 1)).any())
    n += 1; ok = (np.isnan(row.impact_ratio) == degenerate) and (degenerate or abs(row.impact_ratio - rates.min() / rates.max()) <= 1e-9)
    bad += (not ok); log.append(f"{'PASS' if ok else 'FAIL'} {key} impact ratio undefined-rule (degenerate={degenerate})")
pdv = pd.read_csv(TAB / "perplexity_vs_drift.csv"); pp = pd.read_csv(ROOT / "results" / "perplexity.csv").set_index(["model", "level", "recipe"]).ppl
for _, r in pdv.iterrows():
    check("rel ppl increase", pp[(r.model, r.level, r.recipe)] / pp[(r.model, "F16", "ref")] - 1, r.rel_ppl_increase, (r.model, r.level, r.recipe))
    key = (r.model, r.level, r.recipe, "A")
    if key in data and (r.model, "F16", "ref", "A") in data:
        ref = data[(r.model, "F16", "ref", "A")].set_index("prompt_id").p_yes; cur = data[key].set_index("prompt_id").p_yes
        check("ppl table flip rate", ((cur >= .5) != (ref.loc[cur.index] >= .5)).mean(), r.flip_rate, key)
        check("ppl table delta sel pp", 100 * ((cur >= .5).mean() - (ref >= .5).mean()), r.delta_sel_pp, key)


log.append(f"\n{n} comparisons over {len(data)} runs; {bad} failures")
(ROOT / "results" / "verification_log.txt").write_text("\n".join(log), encoding="utf-8")
print(log[-1])
if bad:
    print("VERIFICATION FAILED, see results/verification_log.txt"); sys.exit(1)
print("VERIFICATION PASSED")
