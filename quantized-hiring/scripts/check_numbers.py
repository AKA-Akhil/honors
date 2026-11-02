"""Cross-check the paper's headline numbers against results/tables (prints PASS/FAIL per claim)."""
import sys
from pathlib import Path
import pandas as pd
T = Path(__file__).resolve().parents[1] / "results" / "tables"
g, f, c, s, r = (pd.read_csv(T / n) for n in ("gaps.csv", "flips.csv", "capability.csv", "selection.csv", "template_robustness.csv"))
ga = g[g.template == "A"].set_index(["model", "level"]); fa = f[(f.template == "A") & (f.threshold_type == "0.5")].set_index(["model", "level"])
ca = c[c.template == "A"].set_index(["model", "level"]); sa = s[(s.template == "A") & (s.threshold_type == "0.5")].set_index(["model", "level"])
L, Q = "llama32_3b", "qwen25_3b"; bad = 0
def chk(name, val, exp, tol=0.0006):
    global bad
    ok = abs(val - exp) <= tol; bad += (not ok); print("PASS" if ok else "FAIL", name, round(float(val), 4), exp)
chk("max |gender gap| A", ga.gender_gap.abs().max(), 0.020, 0.0006)
chk("max race gap A", ga.race_gap.abs().max(), 0.036, 0.0006)
chk("llama Q8 gender gap", ga.loc[(L, "Q8_0"), "gender_gap"], -0.011)
chk("qwen Q3 flip A", fa.loc[(Q, "Q3_K_M"), "flip_rate"], 0.265)
chk("qwen Q2 flip A", fa.loc[(Q, "Q2_K"), "flip_rate"], 0.140)
chk("llama Q2 flip A", fa.loc[(L, "Q2_K"), "flip_rate"], 0.065)
chk("mean flip A", fa.flip_rate.mean(), 0.073, 0.0006)
chk("mean flip B", f[(f.template == "B") & (f.threshold_type == "0.5")].flip_rate.mean(), 0.0925, 0.0001)
chk("qwen Q8 sel", sa.loc[(Q, "Q8_0"), "overall"], 0.325); chk("qwen Q3 sel", sa.loc[(Q, "Q3_K_M"), "overall"], 0.060)
chk("qwen Q3 chi2", fa.loc[(Q, "Q3_K_M"), "chi2_yn_groups_p_raw"], 0.949, 0.001)
chk("AUC min", ca.auc_strong_vs_weak.min(), 1.0, 1e-9); chk("AUC min B", c[c.template == "B"].auc_strong_vs_weak.min(), 1.0, 1e-9)
chk("gender corr", r.loc[r.gap == "gender_gap", "pearson_r"].iloc[0], 0.516, 0.001)
chk("race corr", r.loc[r.gap == "race_gap", "pearson_r"].iloc[0], -0.128, 0.001)
chk("holm-sig gender tests A", (ga.gender_p_holm < .05).sum(), 5, 0); chk("holm-sig race tests (A,B)", (g.race_p_holm < .05).sum(), 0, 0)
chk("n prompts", pd.concat([pd.read_csv(p) for p in (T.parent / "raw").glob("*.csv")]).shape[0], 4000, 0)
print("ALL PASS" if not bad else f"{bad} FAILED"); sys.exit(bad > 0)
