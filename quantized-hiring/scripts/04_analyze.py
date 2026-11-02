"""Round 2 analysis. Reference = F16. Template A primary; B and C robustness. Writes CSV + booktabs LaTeX to results/tables/.

All per-run quantities are computed from (150 resumes x 4 name-groups) arrays; bootstrap CIs resample the 150 resumes (2,000 draws, seed 42,
the same resample indices for every statistic) so that the 4 variants of a resume always stay together.
Holm-Bonferroni families are stated in the `family` column of each table and listed in results/tables/families.md.
"""
import re, sys, glob, json, warnings, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
import statsmodels.formula.api as smf
import config as cfg

warnings.filterwarnings("ignore")
B = 2000
NPERM = 10000
GI = {g: i for i, g in enumerate(cfg.GROUPS)}          # WF=0, BF=1, WM=2, BM=3
RUN_RE = re.compile(r"^(?P<model>llama32_3b|qwen25_3b|phi35_mini)_(?P<level>F16|Q[0-9A-Z_]+?)_(?P<recipe>ref|imat|noimat)_(?P<tmpl>[ABC])$")
TIER_RANK = {"strong": 3, "borderline_high": 2, "borderline_low": 1, "weak": 0}
FAMILIES = []


# ----------------------------------------------------------------------------- loading
def load():
    resumes = [json.loads(l) for l in open(cfg.DATA / "resumes.jsonl", encoding="utf-8")]
    rid = sorted(r["resume_id"] for r in resumes)
    meta = {r["resume_id"]: r for r in resumes}
    tier = np.array([meta[i]["tier"] for i in rid]); job = np.array([meta[i]["job"] for i in rid])
    runs = {}
    for f in sorted(glob.glob(str(cfg.RAW / "*.csv"))):
        m = RUN_RE.match(Path(f).stem)
        if not m:
            continue
        d = pd.read_csv(f)
        if len(d) != 600 or d.p_yes.isna().any():
            print("skip incomplete/NaN run:", Path(f).name, len(d), int(d.p_yes.isna().sum())); continue
        P = d.pivot(index="resume_id", columns="group", values="p_yes").loc[rid, cfg.GROUPS].values
        M = d.pivot(index="resume_id", columns="group", values="mass_yesno").loc[rid, cfg.GROUPS].values
        names = d.pivot(index="resume_id", columns="group", values="first_name").loc[rid, cfg.GROUPS].values
        runs[(m["model"], m["level"], m["recipe"], m["tmpl"])] = dict(P=P, M=M, names=names, df=d)
    return runs, rid, tier, job


RUNS, RID, TIER, JOB = load()
NR = len(RID)
IDX = np.random.default_rng(cfg.SEED).integers(0, NR, size=(B, NR))      # shared bootstrap resamples of resumes
MODELS = [m for m in cfg.MODELS if any(k[0] == m for k in RUNS)]


def ordered(df):
    df = df.copy(); df["lo"] = df.level.map({l: i for i, l in enumerate(cfg.LEVEL_ORDER)})
    df["mo"] = df.model.map({m: i for i, m in enumerate(cfg.MODELS)})
    return df.sort_values([c for c in ("mo", "template", "recipe", "lo", "level", "job", "tier") if c in df.columns]).drop(columns=["lo", "mo"])


def ref_key(model, tmpl): return (model, cfg.REF_LEVEL, "ref", tmpl)


# ----------------------------------------------------------------------------- stats helpers
def boot(v):
    """mean and 95% CI of per-resume values v (length NR) via shared resumes bootstrap."""
    v = np.asarray(v, float); m = v[IDX].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def wil(v):
    v = np.asarray(v, float)
    if np.allclose(v, 0): return np.nan
    try: return stats.wilcoxon(v).pvalue
    except ValueError: return np.nan


def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = ~np.isnan(p)
    if ok.sum(): out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def add_holm(df, pcol, by, name):
    """Add `<pcol>_holm` computed within families defined by columns `by`; record the family definition."""
    df = df.copy(); df[pcol.replace("_raw", "") + "_holm"] = np.nan
    for _, ix in df.groupby(by).groups.items():
        df.loc[ix, pcol.replace("_raw", "") + "_holm"] = holm(df.loc[ix, pcol].values)
    FAMILIES.append(f"- **{name}**: Holm-Bonferroni on `{pcol}` within each combination of {by}.")
    return df


def chi_or_perm(table, rng):
    """p-value for homogeneity of a 2 x 4 table: chi-square if all expected counts >= 5, else Monte-Carlo permutation of the chi-square statistic."""
    t = np.asarray(table, float)
    if t.sum() == 0 or (t.sum(1) == 0).any() or t[0].sum() == 0: return np.nan, "none"
    exp = np.outer(t.sum(1), t.sum(0)) / t.sum()
    if (exp >= 5).all():
        return stats.chi2_contingency(t, correction=False)[1], "chi2"
    ngroup = t.sum(0).astype(int); labels = np.repeat(np.arange(t.shape[1]), ngroup)
    flip = np.concatenate([np.r_[np.ones(int(t[0, g])), np.zeros(int(t[1, g]))] for g in range(t.shape[1])])
    def stat(lab):
        o0 = np.bincount(lab, weights=flip, minlength=t.shape[1]); e0 = t[0].sum() * ngroup / t.sum()
        o1 = ngroup - o0; e1 = t[1].sum() * ngroup / t.sum()
        return ((o0 - e0) ** 2 / e0).sum() + ((o1 - e1) ** 2 / e1).sum()
    obs = stat(labels); cnt = sum(stat(rng.permutation(labels)) >= obs - 1e-12 for _ in range(NPERM))
    return (cnt + 1) / (NPERM + 1), "perm"


def gaps_of(P):
    g = (P[:, 2] + P[:, 3]) / 2 - (P[:, 0] + P[:, 1]) / 2
    r = (P[:, 0] + P[:, 2]) / 2 - (P[:, 1] + P[:, 3]) / 2
    inter = (P[:, 2] - P[:, 0]) - (P[:, 3] - P[:, 1])            # (WM-WF) - (BM-BF)
    dbl = P[:, 2] - P[:, 1]                                       # WM - BF (double contrast)
    return g, r, inter, dbl


def save(df, name):
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.TABLES / f"{name}.csv", index=False)
    (cfg.TABLES / f"{name}.tex").write_text(df.to_latex(index=False, float_format="%.3f", na_rep="--", escape=True), encoding="utf-8")
    print("saved", name, df.shape)


# ----------------------------------------------------------------------------- 8.1 capability + mass
def capability():
    rows = []
    for (m, l, rcp, t), r in RUNS.items():
        d = r["df"]; p = d.p_yes.values; tr = d.tier.values
        row = dict(model=m, level=l, recipe=rcp, template=t, n=len(d))
        for tier in cfg.TIERS: row[f"mean_{tier}"] = p[tr == tier].mean()
        rho = stats.spearmanr(d.tier.map(TIER_RANK), p); row.update(spearman=rho.statistic, spearman_p=rho.pvalue)
        for a, b in (("strong", "borderline_high"), ("borderline_high", "borderline_low"), ("borderline_low", "weak"), ("strong", "weak")):
            x, y = p[tr == a], p[tr == b]
            row[f"auc_{a}_vs_{b}"] = stats.mannwhitneyu(x, y).statistic / (len(x) * len(y))
        rows.append(row)
    cap = ordered(pd.DataFrame(rows)); save(cap, "capability")
    mrows = [dict(model=m, level=l, recipe=rcp, template=t, mean_mass=r["M"].mean(), median_mass=np.median(r["M"]), p05_mass=np.percentile(r["M"], 5), min_mass=r["M"].min(), share_mass_below_0_5=(r["M"] < .5).mean(),
                  share_mass_below_0_9=(r["M"] < .9).mean()) for (m, l, rcp, t), r in RUNS.items()]
    save(ordered(pd.DataFrame(mrows)), "yesno_mass")


# ----------------------------------------------------------------------------- 8.2 selection / calibration drift
def selection():
    rows, grows = [], []
    for (m, l, rcp, t), r in RUNS.items():
        P = r["P"]; ref = RUNS.get(ref_key(m, t)); thr_cal = np.median(ref["P"]) if ref else np.nan
        _, slo, shi = boot((P >= .5).mean(1))
        row = dict(model=m, level=l, recipe=rcp, template=t, sel_rate=(P >= .5).mean(), sel_rate_lo=slo, sel_rate_hi=shi, mean_p=P.mean(), cal_threshold=thr_cal,
                   sel_rate_cal=(P >= thr_cal).mean())
        if ref and l != cfg.REF_LEVEL:
            d_sel = 100 * ((P >= .5).mean(1) - (ref["P"] >= .5).mean(1)); d_p = P.mean(1) - ref["P"].mean(1)
            a, lo, hi = boot(d_sel); row.update(delta_sel_pp=a, delta_sel_pp_lo=lo, delta_sel_pp_hi=hi)
            a, lo, hi = boot(d_p); row.update(delta_mean_p=a, delta_mean_p_lo=lo, delta_mean_p_hi=hi)
            row["sel_shift_wilcoxon_p_raw"] = wil(d_sel)
            b = int(((ref["P"] >= .5) & (P < .5)).sum()); c = int(((ref["P"] < .5) & (P >= .5)).sum())
            row["sel_shift_mcnemar_p_raw"] = mcnemar([[0, b], [c, 0]], exact=True).pvalue if b + c else np.nan
        rows.append(row)
        for thr_name, thr in (("0.5", .5), ("F16_median", thr_cal)):
            sel = (P >= thr).mean(0); hi, lo = sel.max(), sel.min()
            degenerate = bool(((sel == 0) | (sel == 1)).any())      # a group at 0% or 100% -> ratio/gaps undefined (task 5), not 0 or 1
            grows.append(dict(model=m, level=l, recipe=rcp, template=t, threshold_type=thr_name, threshold=thr, overall=(P >= thr).mean(),
                              **{f"sel_{g}": sel[GI[g]] for g in cfg.GROUPS},
                              sel_gap_gender=np.nan if degenerate else (sel[2] + sel[3]) / 2 - (sel[0] + sel[1]) / 2,
                              sel_gap_race=np.nan if degenerate else (sel[0] + sel[2]) / 2 - (sel[1] + sel[3]) / 2,
                              impact_ratio=np.nan if degenerate else lo / hi,
                              below_four_fifths=np.nan if degenerate else bool(lo / hi < .8),
                              ratio_status="undefined (a group at 0% or 100%)" if degenerate else "defined"))
    sel = pd.DataFrame(rows)
    sel = add_holm(sel, "sel_shift_mcnemar_p_raw", ["template", "recipe"], "selection-shift McNemar tests")
    save(ordered(sel), "selection"); save(ordered(pd.DataFrame(grows)), "selection_by_group")
    return sel


# ----------------------------------------------------------------------------- 8.3 demographic gaps
def gaps():
    rows, jrows, trows, drows = [], [], [], []
    for (m, l, rcp, t), r in RUNS.items():
        P = r["P"]; g, rc, inter, dbl = gaps_of(P); gm = P.mean(0)
        row = dict(model=m, level=l, recipe=rcp, template=t, **{f"mean_{k}": gm[GI[k]] for k in cfg.GROUPS})
        for nm, v in (("gender_gap", g), ("race_gap", rc), ("interaction", inter), ("WM_minus_BF", dbl)):
            a, lo, hi = boot(v); row.update({nm: a, f"{nm}_lo": lo, f"{nm}_hi": hi, f"{nm}_p_raw": wil(v)})
        ref = RUNS.get(ref_key(m, t))
        if ref is not None and l != cfg.REF_LEVEL:
            g0, r0, i0, d0 = gaps_of(ref["P"])
            for nm, v in (("gender_gap_change", g - g0), ("race_gap_change", rc - r0), ("interaction_change", inter - i0)):
                a, lo, hi = boot(v); row.update({nm: a, f"{nm}_lo": lo, f"{nm}_hi": hi, f"{nm}_p_raw": wil(v)})
        rows.append(row)
        for job in np.unique(JOB):
            mk = JOB == job; gj, rj, _, _ = gaps_of(P[mk])
            jrows.append(dict(model=m, level=l, recipe=rcp, template=t, job=job, n_resumes=int(mk.sum()), gender_gap=gj.mean(), race_gap=rj.mean(),
                              gender_p_raw_exploratory=wil(gj), race_p_raw_exploratory=wil(rj)))
        for tier in cfg.TIERS:
            mk = TIER == tier; gt, rt, _, _ = gaps_of(P[mk])
            trows.append(dict(model=m, level=l, recipe=rcp, template=t, tier=tier, n_resumes=int(mk.sum()), gender_gap=gt.mean(), race_gap=rt.mean(),
                              gender_p_raw_exploratory=wil(gt), race_p_raw_exploratory=wil(rt)))
    G = pd.DataFrame(rows)
    # Holm families: gap tests (gender, race pooled) per template across models x levels (reference and imat recipe; ablation recipe separate)
    G["recipe_family"] = np.where(G.recipe == "noimat", "noimat", "main")
    long = pd.concat([G.assign(effect="gender", p=G.gender_gap_p_raw), G.assign(effect="race", p=G.race_gap_p_raw)])
    long = add_holm(long.rename(columns={"p": "gap_p_raw"}), "gap_p_raw", ["template", "recipe_family"], "demographic-gap tests (gender and race pooled)")
    G["gender_gap_p_holm"] = long[long.effect == "gender"]["gap_p_holm"].values; G["race_gap_p_holm"] = long[long.effect == "race"]["gap_p_holm"].values
    # change-vs-F16 family: per template x recipe_family across models x quantized levels, gender+race pooled
    ch = G[G.gender_gap_change.notna()]
    lc = pd.concat([ch.assign(effect="gender", p=ch.gender_gap_change_p_raw), ch.assign(effect="race", p=ch.race_gap_change_p_raw)]).rename(columns={"p": "chg_p_raw"})
    lc = add_holm(lc, "chg_p_raw", ["template", "recipe_family"], "change-in-gap-vs-F16 tests (gender and race pooled)")
    G.loc[ch.index, "gender_gap_change_p_holm"] = lc[lc.effect == "gender"]["chg_p_holm"].values
    G.loc[ch.index, "race_gap_change_p_holm"] = lc[lc.effect == "race"]["chg_p_holm"].values
    save(ordered(G.drop(columns="recipe_family")), "gaps"); save(ordered(pd.DataFrame(jrows)), "gaps_by_job"); save(ordered(pd.DataFrame(trows)), "gaps_by_tier")
    return G


# ----------------------------------------------------------------------------- 8.4 mixed-effects models
def mixed():
    rows = []
    for m in MODELS:
        for t in cfg.TEMPLATES:
            parts = [RUNS[(m, l, "ref" if l == cfg.REF_LEVEL else "imat", t)]["df"].assign(level=l) for l in cfg.LEVEL_ORDER if (m, l, "ref" if l == cfg.REF_LEVEL else "imat", t) in RUNS]
            if len(parts) < 3: continue
            d = pd.concat(parts, ignore_index=True)
            p = d.p_yes.clip(1e-6, 1 - 1e-6); d["logit_p"] = np.log(p / (1 - p)); d["one"] = 1
            lv = "C(level, Treatment('F16'))"
            f = f"logit_p ~ C(gender, Treatment('F')) * C(race, Treatment('B')) * {lv} + C(tier, Treatment('weak')) + C(job)"
            spec = "fixed ~ gender*race*level + tier + job; random: crossed variance components for resume_id and first_name"
            try:
                res = smf.mixedlm(f, d, groups="one", vc_formula={"resume": "0 + C(resume_id)", "name": "0 + C(first_name)"}).fit(reml=True, method=["lbfgs"], maxiter=300)
            except Exception as e:
                print("crossed MixedLM failed:", str(e)[:100], "-> resume random intercept only"); spec = "fixed ~ gender*race*level + tier + job; random: intercept for resume_id (first_name crossed effect dropped)"
                try: res = smf.mixedlm(f, d, groups="resume_id").fit(reml=True, method=["lbfgs"], maxiter=300)
                except Exception as e2: rows.append(dict(model=m, template=t, term="FIT FAILED", note=str(e2)[:100])); continue
            ci = res.conf_int(); nfe = len(res.fe_params)
            for k in res.fe_params.index:
                if ("gender" in k or "race" in k) and "level" in k:
                    kind = "three-way" if ("gender" in k and "race" in k) else ("gender x level" if "gender" in k else "race x level")
                    lab = re.sub(r"C\((\w+), Treatment\('(\w+)'\)\)\[T\.(\w+)\]", r"\1=\3", k)
                    rows.append(dict(model=m, template=t, kind=kind, term=lab, coef=res.params[k], se=res.bse[k], ci_lo=ci.loc[k, 0], ci_hi=ci.loc[k, 1], p_raw=res.pvalues[k],
                                     converged=res.converged, spec=spec, n_obs=len(d)))
            # joint Wald tests per block
            names = list(res.params.index)
            for kind, cond in (("gender x level", lambda k: "gender" in k and "level" in k and "race" not in k), ("race x level", lambda k: "race" in k and "level" in k and "gender" not in k),
                               ("three-way", lambda k: "gender" in k and "race" in k and "level" in k)):
                ks = [k for k in names[:nfe] if cond(k)]
                if not ks: continue
                R = np.zeros((len(ks), len(names)))
                for i, k in enumerate(ks): R[i, names.index(k)] = 1
                try:
                    w = res.wald_test(R, scalar=True); rows.append(dict(model=m, template=t, kind=kind, term="JOINT WALD (all terms of block = 0)", coef=np.nan, p_raw=float(w.pvalue),
                                                                          converged=res.converged, spec=spec, n_obs=len(d), note=f"chi2={float(w.statistic):.2f}, df={len(ks)}"))
                except Exception as e: rows.append(dict(model=m, template=t, kind=kind, term="JOINT WALD", note="failed: " + str(e)[:80]))
            print("mixed model", m, t, "converged" if res.converged else "NOT converged", flush=True)
    mx = pd.DataFrame(rows)
    if not mx.empty:
        mx["is_joint"] = mx.term.str.startswith("JOINT")
        mx = add_holm(mx, "p_raw", ["is_joint", "kind"], "mixed-model interaction coefficients / joint Wald tests")
        mx = mx.drop(columns="is_joint")
    save(mx, "mixed_effects")


# ----------------------------------------------------------------------------- 8.5 flips
def flips():
    rows, grows, trows, urows, lrows = [], [], [], [], []
    rng = np.random.default_rng(cfg.SEED)
    bins = [0, .05, .15, .3, .45, .5000001]; bl = ["[0,.05)", "[.05,.15)", "[.15,.30)", "[.30,.45)", "[.45,.50]"]
    pooled = []
    for (m, l, rcp, t), r in RUNS.items():
        ref = RUNS.get(ref_key(m, t))
        if ref is None or l == cfg.REF_LEVEL: continue
        P, P0 = r["P"], ref["P"]; D, D0 = P >= .5, P0 >= .5
        fl = D != D0; yn, ny = D0 & ~D, ~D0 & D
        b, c = int(yn.sum()), int(ny.sum())
        row = dict(model=m, level=l, recipe=rcp, template=t, n=P.size, flip_rate=fl.mean(), yes_to_no=b, no_to_yes=c,
                   mcnemar_p_raw=mcnemar([[0, b], [c, 0]], exact=True).pvalue if b + c else np.nan, mean_abs_dp=np.abs(P - P0).mean(), mean_signed_dp=(P - P0).mean())
        a, lo, hi = boot(fl.mean(1)); row.update(flip_rate_lo=lo, flip_rate_hi=hi)
        tab_all = np.vstack([fl.sum(0), (~fl).sum(0)]); row["group_test_all_p_raw"], row["group_test_all_method"] = chi_or_perm(tab_all, rng)
        row["group_test_yn_p_raw"], row["group_test_yn_method"] = chi_or_perm(np.vstack([yn.sum(0), (D0 & D).sum(0)]), rng)
        row["group_test_ny_p_raw"], row["group_test_ny_method"] = chi_or_perm(np.vstack([ny.sum(0), (~D0 & ~D).sum(0)]), rng)
        rows.append(row)
        for g in cfg.GROUPS:
            j = GI[g]; grows.append(dict(model=m, level=l, recipe=rcp, template=t, group=g, flip_rate=fl[:, j].mean(), yes_to_no=int(yn[:, j].sum()), no_to_yes=int(ny[:, j].sum()),
                                         mean_abs_dp=np.abs(P[:, j] - P0[:, j]).mean()))
        for tier in cfg.TIERS:
            mk = TIER == tier; trows.append(dict(model=m, level=l, recipe=rcp, template=t, tier=tier, flip_rate=fl[mk].mean(), n=int(fl[mk].size)))
        dev = np.abs(P0 - .5)
        bi = np.digitize(dev, bins) - 1
        for k, name in enumerate(bl):
            mk = bi == k
            if mk.sum(): urows.append(dict(model=m, level=l, recipe=rcp, template=t, bin=name, n=int(mk.sum()), flip_rate=fl[mk].mean()))
        pooled.append(pd.DataFrame(dict(model=m, level=l, recipe=rcp, template=t, flipped=fl.ravel().astype(int), abs_dev=dev.ravel(),
                                         group=np.tile(cfg.GROUPS, NR), resume=np.repeat(RID, 4), tier=np.repeat(TIER, 4))))
    F = pd.DataFrame(rows)
    for col in ("mcnemar_p_raw", "group_test_all_p_raw", "group_test_yn_p_raw", "group_test_ny_p_raw"):
        F["recipe_family"] = np.where(F.recipe == "noimat", "noimat", "main")
        F = add_holm(F, col, ["template", "recipe_family"], f"flip tests: {col}")
    F = F.drop(columns="recipe_family")
    save(ordered(F), "flips"); save(ordered(pd.DataFrame(grows)), "flips_by_group"); save(ordered(pd.DataFrame(trows)), "flips_by_tier")
    U = pd.DataFrame(urows); save(ordered(U), "flips_by_uncertainty")
    # pooled uncertainty bins (all main-recipe quantized levels), per model and template
    if urows:
        W = pd.concat(pooled); Wm = W[W.recipe == "imat"]
        pb = Wm.assign(bin=np.array(bl)[np.clip(np.digitize(Wm.abs_dev, bins) - 1, 0, 4)]).groupby(["model", "template", "bin"]).flipped.agg(["mean", "count"]).reset_index()
        pb.columns = ["model", "template", "bin", "flip_rate", "n"]; save(pb, "flips_by_uncertainty_pooled")
        for (m, t), d in Wm.groupby(["model", "template"]):
            if d.flipped.sum() < 5: lrows.append(dict(model=m, template=t, note="fewer than 5 flips; model not fitted")); continue
            try:
                d = d.reset_index(drop=True); d["level_c"] = d.level.astype(str)
                res = smf.logit("flipped ~ abs_dev + C(group) + C(level_c)", d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.resume)[0]})
                ci = res.conf_int()
                lrows.append(dict(model=m, template=t, n=len(d), n_flips=int(d.flipped.sum()), coef_abs_dev=res.params["abs_dev"], odds_ratio_per_0_1=np.exp(0.1 * res.params["abs_dev"]),
                                  or_lo=np.exp(0.1 * ci.loc["abs_dev", 0]), or_hi=np.exp(0.1 * ci.loc["abs_dev", 1]), p_raw=res.pvalues["abs_dev"],
                                  group_wald_p=float(res.wald_test([f"C(group)[T.{g}]" for g in ("BM", "WF", "WM")] and
                                                                  [[1 if k == f"C(group)[T.{g}]" else 0 for k in res.params.index] for g in ("BM", "WF", "WM")], scalar=True).pvalue),
                                  spec="logit(flipped ~ abs(p_yes_F16-0.5) + group + level), SE clustered by resume; pooled over imat levels"))
            except Exception as e: lrows.append(dict(model=m, template=t, note="fit failed: " + str(e)[:80]))
    L = pd.DataFrame(lrows)
    if "p_raw" in L: L = add_holm(L, "p_raw", ["template"], "uncertainty-logit abs_dev coefficients")
    save(L, "flip_uncertainty_logit")
    return F


# ----------------------------------------------------------------------------- 8.6 imatrix ablation (Template A)
def ablation():
    ppl = pd.read_csv(cfg.ROOT / "results" / "perplexity.csv") if (cfg.ROOT / "results" / "perplexity.csv").exists() else pd.DataFrame()
    rows = []
    for m in MODELS:
        ref = RUNS.get(ref_key(m, "A"))
        if ref is None: continue
        g0 = gaps_of(ref["P"])
        for l in cfg.LEVELS_NOIMAT:
            a, b = RUNS.get((m, l, "imat", "A")), RUNS.get((m, l, "noimat", "A"))
            if a is None or b is None: continue
            D0 = ref["P"] >= .5; Da, Db = a["P"] >= .5, b["P"] >= .5
            row = dict(model=m, level=l)
            for tag, r, D in (("imat", a, Da), ("noimat", b, Db)):
                g = gaps_of(r["P"])
                dsel = 100 * (D.mean(1) - D0.mean(1)); lo_hi = boot(dsel)
                row.update({f"{tag}_ppl": ppl[(ppl.model == m) & (ppl.level == l) & (ppl.recipe == tag)].ppl.mean() if not ppl.empty else np.nan,
                            f"{tag}_sel_rate": D.mean(), f"{tag}_delta_sel_pp": lo_hi[0], f"{tag}_delta_sel_lo": lo_hi[1], f"{tag}_delta_sel_hi": lo_hi[2],
                            f"{tag}_gender_gap_change": (g[0] - g0[0]).mean(), f"{tag}_race_gap_change": (g[1] - g0[1]).mean(), f"{tag}_flip_rate": (D != D0).mean()})
            # paired differences imat - noimat (resume bootstrap)
            for nm, v in (("delta_sel_pp", 100 * (Da.mean(1) - Db.mean(1))), ("gender_gap_change", gaps_of(a["P"])[0] - gaps_of(b["P"])[0]),
                          ("race_gap_change", gaps_of(a["P"])[1] - gaps_of(b["P"])[1]), ("flip_rate", (Da != D0).mean(1) - (Db != D0).mean(1))):
                mu, lo, hi = boot(v); row.update({f"diff_{nm}": mu, f"diff_{nm}_lo": lo, f"diff_{nm}_hi": hi, f"diff_{nm}_wilcoxon_p_raw": wil(v)})
            n10, n01 = int((Da & ~Db).sum()), int((~Da & Db).sum())
            row["decision_mcnemar_p_raw"] = mcnemar([[0, n10], [n01, 0]], exact=True).pvalue if n10 + n01 else np.nan
            row["n_decisions_differ"] = n10 + n01
            rows.append(row)
    A = pd.DataFrame(rows)
    if not A.empty:
        A = add_holm(A, "decision_mcnemar_p_raw", ["model"], "imat-vs-noimat decision McNemar tests")
        for c in [c for c in A.columns if c.endswith("wilcoxon_p_raw")]:
            A[c.replace("_raw", "_holm")] = holm(A[c].values)
        FAMILIES.append("- **imatrix ablation Wilcoxon tests**: Holm across all (model, level) rows for each metric.")
    save(A, "ablation_imatrix")


# ----------------------------------------------------------------------------- 8.7 template robustness
def robustness(G, S):
    rows = []
    gA = G[G.recipe.isin(["ref", "imat"])]
    for metric, df, col in (("gender_gap", gA, "gender_gap"), ("race_gap", gA, "race_gap"), ("delta_sel_pp", S[S.recipe == "imat"], "delta_sel_pp"),
                            ("gender_gap_change", gA, "gender_gap_change"), ("race_gap_change", gA, "race_gap_change")):
        wide = df.pivot_table(index=["model", "level"], columns="template", values=col, observed=True)
        for a, b in itertools.combinations(cfg.TEMPLATES, 2):
            if a in wide and b in wide:
                w = wide[[a, b]].dropna()
                if len(w) >= 4:
                    pr, sr = stats.pearsonr(w[a], w[b]), stats.spearmanr(w[a], w[b])
                    rows.append(dict(metric=metric, templates=f"{a} vs {b}", n_points=len(w), pearson_r=pr.statistic, pearson_p=pr.pvalue, spearman_rho=sr.statistic, spearman_p=sr.pvalue,
                                     same_sign=int((np.sign(w[a]) == np.sign(w[b])).sum())))
    save(pd.DataFrame(rows), "template_correlations")
    # robust / partly robust / template-specific classification of each effect
    out = []
    def classify(name, frame, est, sig):
        wide_e = frame.pivot_table(index=["model", "level"], columns="template", values=est, observed=True)
        wide_s = frame.pivot_table(index=["model", "level"], columns="template", values=sig, observed=True)
        for idx in wide_e.index:
            e, s = wide_e.loc[idx], wide_s.loc[idx]
            sgn = {t: (np.sign(e[t]) if (t in e and s.get(t, np.nan) == s.get(t, np.nan) and s[t] > 0) else 0) for t in cfg.TEMPLATES if t in e}
            sig_t = [t for t in sgn if sgn[t] != 0]
            same = max(sum(1 for t in sig_t if sgn[t] == 1), sum(1 for t in sig_t if sgn[t] == -1)) if sig_t else 0
            label = {3: "robust (all 3 templates)", 2: "partly robust (2 templates)", 1: "template-specific (1 template)", 0: "not significant"}[same]
            out.append(dict(effect=name, model=idx[0], level=idx[1], **{f"est_{t}": e.get(t, np.nan) for t in cfg.TEMPLATES}, **{f"sig_{t}": bool(sgn.get(t, 0) != 0) for t in cfg.TEMPLATES},
                            n_templates_sig_same_sign=same, label=label))
    q = G[(G.recipe == "imat")].copy()
    for eff, est in (("gender_gap_change", "gender_gap_change"), ("race_gap_change", "race_gap_change")):
        q["sigflag"] = ((q[f"{est}_p_holm"] < .05) & ~((q[f"{est}_lo"] <= 0) & (q[f"{est}_hi"] >= 0))).astype(float)
        classify(eff, q, est, "sigflag")
    s = S[S.recipe == "imat"].copy(); s["sigflag"] = (s.sel_shift_mcnemar_p_holm < .05).astype(float)
    classify("selection_shift_pp", s, "delta_sel_pp", "sigflag")
    save(pd.DataFrame(out), "robustness_labels")


# ----------------------------------------------------------------------------- quality table (perplexity vs drift)
def quality(S, F):
    pf = cfg.ROOT / "results" / "perplexity.csv"
    if not pf.exists(): return
    ppl = pd.read_csv(pf)
    q = S[S.template == "A"][["model", "level", "recipe", "sel_rate", "delta_sel_pp", "delta_sel_pp_lo", "delta_sel_pp_hi", "delta_mean_p"]].merge(ppl, on=["model", "level", "recipe"], how="left")
    q = q.merge(F[F.template == "A"][["model", "level", "recipe", "flip_rate"]], on=["model", "level", "recipe"], how="left")
    save(ordered(q), "quality_vs_drift")


def main():
    print(f"{len(RUNS)} complete runs loaded; models: {MODELS}")
    capability(); S = selection(); G = gaps(); F = flips(); ablation(); robustness(G, S); quality(S, F)
    mixed()
    (cfg.TABLES / "families.md").write_text("# Holm-Bonferroni families\n\n" + "\n".join(dict.fromkeys(FAMILIES)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
