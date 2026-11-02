"""Task 4: refit the 9 mixed models with a fallback ladder until one converges; final fallback = OLS on logit(p_yes) with resume-clustered SEs.
Outputs: results/tables/mixed_methods.csv (one row per fit), results/tables/mixed_effects_final.csv (coefficients from the method used). Seed-free (deterministic)."""
import sys, re, glob, warnings, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import config as cfg
warnings.filterwarnings("ignore")
RE = re.compile(r"^(llama32_3b|qwen25_3b|phi35_mini)_(F16|Q[0-9A-Z_]+?)_(ref|imat)_([ABC])$")
RUN = {}
for f in glob.glob(str(cfg.RAW / "*.csv")):
    m = RE.match(Path(f).stem)
    if m:
        d = pd.read_csv(f)
        if len(d) == 600 and d.p_yes.notna().all(): RUN[m.groups()] = d
orig = pd.read_csv(cfg.TABLES / "mixed_effects.csv")
KINDS = (("gender x level", lambda k: "gender" in k and "level" in k and "race" not in k), ("race x level", lambda k: "race" in k and "level" in k and "gender" not in k),
         ("three-way", lambda k: "gender" in k and "race" in k and "level" in k))


def frame(m, t):
    parts = [RUN[(m, l, "ref" if l == "F16" else "imat", t)].assign(level=l) for l in cfg.LEVEL_ORDER if (m, l, "ref" if l == "F16" else "imat", t) in RUN]
    d = pd.concat(parts, ignore_index=True); p = d.p_yes.clip(1e-6, 1 - 1e-6); d["logit_p"] = np.log(p / (1 - p)); d["one"] = 1
    d["g_c"] = np.where(d.gender == "M", .5, -.5); d["r_c"] = np.where(d.race == "W", .5, -.5)        # centred (+-0.5) demographic predictors
    d["logit_s"] = (d.logit_p - d.logit_p.mean()) / d.logit_p.std()                                    # scaled outcome (mixed variants only)
    return d


LV = "C(level, Treatment('F16'))"
F_T = f"logit_p ~ C(gender, Treatment('F')) * C(race, Treatment('B')) * {LV} + C(tier, Treatment('weak')) + C(job)"
F_C = f"logit_s ~ g_c * r_c * {LV} + C(tier, Treatment('weak')) + C(job)"
VC = {"resume": "0 + C(resume_id)", "name": "0 + C(first_name)"}


def wald(res, nfe, names):
    out = {}
    for kind, cond in KINDS:
        ks = [k for k in names[:nfe] if cond(k)]
        R = np.zeros((len(ks), len(names)))
        for i, k in enumerate(ks): R[i, names.index(k)] = 1
        w = res.wald_test(R, scalar=True); out[kind] = (float(w.pvalue), float(w.statistic), len(ks))
    return out


def fit_ladder(d):
    tried = []
    attempts = []
    for form, tag in ((F_T, "treatment coding"), (F_C, "centred predictors + standardised outcome")):
        for struct in ("crossed resume+name", "resume only"):
            for meth in ("lbfgs", "bfgs", "powell", "nm", "cg"):
                attempts.append((form, tag, struct, meth))
    # order: original structure/coding first with every optimiser, then centred, then simpler RE
    for form, tag, struct, meth in attempts:
        label = f"MixedLM {struct}, {tag}, {meth}"
        try:
            if struct.startswith("crossed"):
                mod = smf.mixedlm(form, d, groups="one", vc_formula=VC)
            else:
                mod = smf.mixedlm(form, d, groups="resume_id")
            res = mod.fit(reml=True, method=[meth], maxiter=3000)
            tried.append(f"{label}: {'converged' if res.converged else 'no'}")
            if res.converged: return res, label, tried, "mixed"
        except Exception as e:
            tried.append(f"{label}: error {str(e)[:40]}")
    res = smf.ols(F_T, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.resume_id)[0]})
    tried.append("OLS on logit(p_yes), cluster-robust SE by resume_id")
    return res, "OLS logit(p_yes) + cluster-robust SE (resume_id)", tried, "ols"


rows, coefs = [], []
for m in cfg.MODELS:
    for t in cfg.TEMPLATES:
        d = frame(m, t); t0 = time.time()
        res, label, tried, kind_m = fit_ladder(d)
        names = list(res.params.index); nfe = len(res.fe_params) if kind_m == "mixed" else len(names)
        w = wald(res, nfe, names)
        o = orig[(orig.model == m) & (orig.template == t) & (orig.term.str.startswith("JOINT"))].set_index("kind")
        row = dict(model=m, template=t, method_used=label, family=kind_m, n_attempts=len(tried), seconds=round(time.time() - t0), original_converged=bool(orig[(orig.model == m) & (orig.template == t)].converged.iloc[0]))
        for kind, (p, st, df) in w.items():
            po = o.loc[kind, "p_raw"] if kind in o.index else np.nan
            row.update({f"{kind}_joint_p": p, f"{kind}_joint_p_original": po, f"{kind}_sig_changed": (p < .05) != (po < .05)})
        rows.append(row)
        ci = res.conf_int()
        for k in names[:nfe]:
            for kind, cond in KINDS:
                if cond(k): coefs.append(dict(model=m, template=t, kind=kind, term=k, coef=res.params[k], se=res.bse[k], ci_lo=ci.loc[k, 0], ci_hi=ci.loc[k, 1], p_raw=res.pvalues[k], method=label))
        print(m, t, "->", label, "| attempts", len(tried), flush=True)
mm = pd.DataFrame(rows)
cf = pd.DataFrame(coefs)
cf["p_holm"] = np.nan
for kind, g in cf.groupby("kind"): cf.loc[g.index, "p_holm"] = multipletests(g.p_raw, method="holm")[1]
for kind in ("gender x level", "race x level", "three-way"):
    mm[f"{kind}_joint_p_holm"] = multipletests(mm[f"{kind}_joint_p"], method="holm")[1]
    mm[f"{kind}_coef_min_p_holm"] = [cf[(cf.model == a) & (cf.template == b) & (cf.kind == kind)].p_holm.min() for a, b in zip(mm.model, mm.template)]
mm.to_csv(cfg.TABLES / "mixed_methods.csv", index=False); cf.to_csv(cfg.TABLES / "mixed_effects_final.csv", index=False)
(cfg.TABLES / "mixed_methods.tex").write_text(mm[["model", "template", "method_used"] + [c for c in mm.columns if c.endswith("joint_p")]].to_latex(index=False, float_format="%.3f", escape=True), encoding="utf-8")
print(mm[["model", "template", "method_used", "original_converged", "gender x level_joint_p", "race x level_joint_p", "three-way_joint_p"] + [c for c in mm.columns if c.endswith("sig_changed")]].round(3).to_string())
print("any coefficient Holm-significant:", (cf.p_holm < .05).sum())
