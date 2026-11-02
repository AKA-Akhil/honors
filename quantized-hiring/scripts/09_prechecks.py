"""Final pre-paper checks (tasks 1b, 2): mass-filtered sensitivity and perplexity-vs-drift. Reads results/raw + results/tables; seed 42; rerunnable."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
import config as cfg

A = cfg.RAW
T = cfg.TABLES
import re, glob
RE = re.compile(r"^(llama32_3b|qwen25_3b|phi35_mini)_(F16|Q[0-9A-Z_]+?)_(ref|imat|noimat)_([ABC])$")
RUNS = {}
for f in sorted(glob.glob(str(A / "*.csv"))):
    m = RE.match(Path(f).stem)
    if m:
        d = pd.read_csv(f)
        if len(d) == 600 and d.p_yes.notna().all():
            RUNS[m.groups()] = d.set_index("prompt_id")


def gaps(d):
    g = d.groupby(["resume_id", "group"]).p_yes.mean().unstack()
    g = g.dropna()          # resumes with all 4 variants retained
    gen = ((g["WM"] + g["BM"]) / 2 - (g["WF"] + g["BF"]) / 2).mean()
    rac = ((g["WF"] + g["WM"]) / 2 - (g["BF"] + g["BM"]) / 2).mean()
    return gen, rac, len(g)


def sensitivity():
    rows = []
    for (m, l, rcp, t), d in RUNS.items():
        keep = d[d.mass_yesno >= 0.5]
        ref = RUNS.get((m, "F16", "ref", t))
        row = dict(model=m, level=l, recipe=rcp, template=t, n_prompts=len(d), n_kept=len(keep), share_excluded=1 - len(keep) / len(d))
        row["sel_rate_all"] = (d.p_yes >= .5).mean()
        row["sel_rate_filtered"] = (keep.p_yes >= .5).mean() if len(keep) else np.nan
        if len(keep) >= 8:
            gg, rr, nres = gaps(keep.reset_index())
        else:
            gg = rr = np.nan; nres = 0
        ga, ra, _ = gaps(d.reset_index())
        row.update(gender_gap_all=ga, gender_gap_filtered=gg, race_gap_all=ra, race_gap_filtered=rr, n_complete_resumes_filtered=nres)
        if ref is not None and l != "F16":
            both = d.index[(d.mass_yesno >= .5) & (ref.loc[d.index].mass_yesno >= .5)]
            fl_all = ((d.p_yes >= .5) != (ref.loc[d.index].p_yes >= .5)).mean()
            row["flip_rate_all"] = fl_all
            row["flip_rate_filtered"] = ((d.loc[both].p_yes >= .5) != (ref.loc[both].p_yes >= .5)).mean() if len(both) else np.nan
            dd = d.loc[both].reset_index(); rf = ref.loc[both].reset_index()
            if len(both) >= 32:
                g1 = gaps(dd); g0 = gaps(rf)
                row["gender_gap_change_filtered"] = g1[0] - g0[0]; row["race_gap_change_filtered"] = g1[1] - g0[1]
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in ("sel_rate", "gender_gap", "race_gap", "flip_rate"):
        df[f"{c}_abs_change"] = (df[f"{c}_filtered"] - df[f"{c}_all"]).abs() if f"{c}_filtered" in df else np.nan
    df["conclusion_changed"] = np.where(df.share_excluded == 0, False, True)   # runs with nothing excluded are identical by construction
    df["conclusion_changed"] = df.conclusion_changed & (df.sel_rate_abs_change.fillna(1) > 0.01)
    df["lo"] = df.level.map({l: i for i, l in enumerate(cfg.LEVEL_ORDER)}); df["mo"] = df.model.map({m: i for i, m in enumerate(cfg.MODELS)})
    df = df.sort_values(["mo", "template", "recipe", "lo"]).drop(columns=["lo", "mo"])
    df.to_csv(T / "sensitivity_mass_filtered.csv", index=False)
    (T / "sensitivity_mass_filtered.tex").write_text(df[df.share_excluded > 0].to_latex(index=False, float_format="%.3f", na_rep="--"), encoding="utf-8")
    aff = df[df.share_excluded > 0]
    print("runs with excluded prompts:\n", aff[["model", "level", "recipe", "template", "share_excluded", "sel_rate_all", "sel_rate_filtered", "gender_gap_all", "gender_gap_filtered", "race_gap_all", "race_gap_filtered"]].round(4).to_string())
    print("main (imat/ref) runs with exclusions:", int(((df.share_excluded > 0) & df.recipe.isin(["ref", "imat"])).sum()))


def ppl_vs_drift():
    ppl = pd.read_csv(cfg.ROOT / "results" / "perplexity.csv")
    sel = pd.read_csv(T / "selection.csv"); fl = pd.read_csv(T / "flips.csv")
    rows = []
    for _, p in ppl.iterrows():
        m, l, rcp = p.model, p.level, p.recipe
        r0 = ppl[(ppl.model == m) & (ppl.level == "F16")].iloc[0]
        s = sel[(sel.model == m) & (sel.level == l) & (sel.recipe == rcp) & (sel.template == "A")]
        f = fl[(fl.model == m) & (fl.level == l) & (fl.recipe == rcp) & (fl.template == "A")]
        if s.empty: continue
        s = s.iloc[0]
        rows.append(dict(model=m, level=l, recipe=rcp, ppl=p.ppl, ppl_f16=r0.ppl, rel_ppl_increase=p.ppl / r0.ppl - 1,
                         delta_sel_pp=s.get("delta_sel_pp", 0.0) if l != "F16" else 0.0, abs_delta_sel_pp=abs(s.get("delta_sel_pp", 0.0)) if l != "F16" else 0.0,
                         flip_rate=(f.iloc[0].flip_rate if len(f) else 0.0), size_gb=p.size_gb))
    df = pd.DataFrame(rows); df = df[df.level != "F16"]
    df.to_csv(T / "perplexity_vs_drift.csv", index=False)
    out = []
    for nm, sub in (("imatrix levels, all models", df[df.recipe == "imat"]), ("all quantized runs incl. no-imatrix", df),
                    ("imatrix, excl. Qwen", df[(df.recipe == "imat") & (df.model != "qwen25_3b")]),
                    ("imatrix, excl. Qwen Q2_K/Q3_K_M", df[(df.recipe == "imat") & ~((df.model == "qwen25_3b") & df.level.isin(["Q2_K", "Q3_K_M"]))]) ):
        # exclude the degenerate Qwen no-imatrix Q2_K (ppl ~ 2e4) from the 'all' line only if present: report both
        for target in ("abs_delta_sel_pp", "flip_rate"):
            rho = stats.spearmanr(sub.rel_ppl_increase, sub[target])
            out.append(dict(subset=nm, target=target, n=len(sub), spearman_rho=rho.statistic, p=rho.pvalue))
    for mname in cfg.MODELS:
        sub = df[(df.model == mname) & (df.recipe == "imat")]
        for target in ("abs_delta_sel_pp", "flip_rate"):
            rho = stats.spearmanr(sub.rel_ppl_increase, sub[target]); out.append(dict(subset=f"within {mname} (imat)", target=target, n=len(sub), spearman_rho=rho.statistic, p=rho.pvalue))
    pd.DataFrame(out).to_csv(T / "perplexity_vs_drift_spearman.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string())
    print(df[(df.level == "Q3_K_M") & (df.recipe == "imat")].round(4).to_string())
    print(df[(df.recipe == "imat")][["model", "level", "rel_ppl_increase", "delta_sel_pp", "flip_rate"]].round(3).to_string())


if __name__ == "__main__":
    sensitivity(); ppl_vs_drift()
