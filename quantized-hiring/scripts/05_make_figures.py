"""Round 2 figures (PNG 300 dpi + PDF) from results/tables. Colour-blind-safe (Okabe-Ito) with distinct markers/linestyles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.titlesize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
STYLE = {"llama32_3b": ("#0072B2", "o", "-", "Llama-3.2-3B"), "qwen25_3b": ("#D55E00", "s", "--", "Qwen2.5-3B"), "phi35_mini": ("#009E73", "^", "-.", "Phi-3.5-mini")}
T = C.TABLES; OUT = C.FIGS
XPOS = {l: i for i, l in enumerate(C.LEVEL_ORDER)}
LAB = [l.replace("_", "\\_") if False else l for l in C.LEVEL_ORDER]
OFF = {"llama32_3b": -.08, "qwen25_3b": 0.0, "phi35_mini": .08}


def rd(n):
    return pd.read_csv(T / f"{n}.csv")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main_runs(df, tmpl="A"):
    d = df[(df.template == tmpl) & (df.recipe.isin(["ref", "imat"]))].copy(); d["x"] = d.level.map(XPOS); return d.sort_values("x")


def xaxis(ax, show_label=True):
    ax.set_xticks(range(len(C.LEVEL_ORDER))); ax.set_xticklabels(C.LEVEL_ORDER, rotation=45, ha="right")
    if show_label: ax.set_xlabel("Quantization level")


def fig1_selection():
    s = main_runs(rd("selection")); fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for m, (c, mk, ls, nm) in STYLE.items():
        d = s[s.model == m]
        if d.empty: continue
        ax.errorbar(d.x + OFF[m], d.sel_rate, yerr=[d.sel_rate - d.sel_rate_lo, d.sel_rate_hi - d.sel_rate], color=c, marker=mk, ls=ls, ms=4, lw=1.1, capsize=2, label=nm)
    ax.set_ylabel("Selection rate (P(Yes) $\\geq$ 0.5)"); xaxis(ax); ax.legend(frameon=False); ax.set_ylim(bottom=0)
    save(fig, "fig1_selection_vs_level")


def fig2_gap_change():
    g = main_runs(rd("gaps")); g = g[g.level != C.REF_LEVEL]; fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, nm, ttl in zip(axs, ("gender", "race"), ("Change in gender gap vs F16 (M $-$ F)", "Change in race gap vs F16 (White $-$ Black)")):
        for m, (c, mk, ls, lab) in STYLE.items():
            d = g[g.model == m]
            if d.empty: continue
            y = d[f"{nm}_gap_change"]; ax.errorbar(d.x + OFF[m], y, yerr=[y - d[f"{nm}_gap_change_lo"], d[f"{nm}_gap_change_hi"] - y], color=c, marker=mk, ls=ls, ms=4, lw=1.1, capsize=2, label=lab)
        ax.axhline(0, color="grey", lw=.6); xaxis(ax); ax.set_title(ttl); ax.set_ylabel("Change in paired gap in P(Yes)")
        ax.set_xlim(.4, len(C.LEVEL_ORDER) - .6)
    axs[0].legend(frameon=False); fig.tight_layout(); save(fig, "fig2_gap_change_vs_f16")


def fig3_flips():
    f = rd("flips"); f = f[(f.template == "A") & (f.recipe == "imat")].copy(); f["x"] = f.level.map(XPOS); models = [m for m in STYLE if m in f.model.values]
    fig, axs = plt.subplots(1, len(models), figsize=(7.0, 2.4), sharey=True, squeeze=False)
    for ax, m in zip(axs[0], models):
        d = f[f.model == m].sort_values("x")
        ax.bar(d.x, d.yes_to_no / d.n, .6, color="#0072B2", hatch="//", edgecolor="white", label="Yes$\\to$No")
        ax.bar(d.x, -d.no_to_yes / d.n, .6, color="#E69F00", hatch="..", edgecolor="white", label="No$\\to$Yes")
        ax.axhline(0, color="k", lw=.6); xaxis(ax); ax.set_title(STYLE[m][3]); ax.set_xlim(.4, len(C.LEVEL_ORDER) - .4)
    axs[0][0].set_ylabel("Share of decisions flipped vs F16"); axs[0][0].legend(frameon=False); fig.tight_layout(); save(fig, "fig3_flips_vs_level")


def fig4_uncertainty():
    u = rd("flips_by_uncertainty_pooled"); u = u[u.template == "A"]; bins = ["[0,.05)", "[.05,.15)", "[.15,.30)", "[.30,.45)", "[.45,.50]"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for m, (c, mk, ls, nm) in STYLE.items():
        d = u[u.model == m].set_index("bin").reindex(bins)
        if d.flip_rate.notna().any(): ax.plot(range(len(bins)), d.flip_rate, color=c, marker=mk, ls=ls, ms=4, lw=1.1, label=nm)
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(bins, rotation=30, ha="right"); ax.set_xlabel("|P(Yes) $-$ 0.5| under F16"); ax.set_ylabel("Flip rate vs F16 (pooled quantized levels)")
    ax.legend(frameon=False); save(fig, "fig4_flip_vs_uncertainty")


def fig5_auc():
    c = main_runs(rd("capability")); pairs = [("strong", "borderline_high"), ("borderline_high", "borderline_low"), ("borderline_low", "weak")]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, (a, b) in zip(axs, pairs):
        for m, (col, mk, ls, nm) in STYLE.items():
            d = c[c.model == m]
            if not d.empty: ax.plot(d.x, d[f"auc_{a}_vs_{b}"], color=col, marker=mk, ls=ls, ms=3.5, lw=1.1, label=nm)
        ax.axhline(.5, color="grey", lw=.6, ls=":"); ax.set_title(f"{a.replace('_', ' ')} vs {b.replace('_', ' ')}", fontsize=7.5); xaxis(ax, False)
    axs[0].set_ylabel("AUC (adjacent tiers)"); axs[0].legend(frameon=False); fig.tight_layout(); save(fig, "fig5_adjacent_tier_auc")


def fig6_imatrix():
    a = rd("ablation_imatrix")
    if a.empty: return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5)); a = a.copy(); a["lab"] = a.model.map({m: STYLE[m][3].split("-")[0] for m in STYLE}) + "\n" + a.level
    x = np.arange(len(a)); w = .38
    for ax, (ci, cn, yl) in zip(axs, (("imat_delta_sel_pp", "noimat_delta_sel_pp", "Selection-rate shift vs F16 (pp)"), ("imat_flip_rate", "noimat_flip_rate", "Flip rate vs F16"))):
        ax.bar(x - w / 2, a[ci], w, color="#0072B2", label="with imatrix"); ax.bar(x + w / 2, a[cn], w, color="#E69F00", hatch="//", edgecolor="white", label="no imatrix")
        ax.set_xticks(x); ax.set_xticklabels(a.lab, fontsize=6.5); ax.set_ylabel(yl); ax.axhline(0, color="k", lw=.6)
    axs[0].legend(frameon=False); fig.tight_layout(); save(fig, "fig6_imatrix_ablation")


def fig7_ppl():
    q = rd("quality_vs_drift"); q = q[q.recipe.isin(["imat", "noimat"])].copy()
    ref = pd.read_csv(C.ROOT / "results" / "perplexity.csv"); ref = ref[ref.recipe == "ref"].set_index("model").ppl
    q["ppl_ratio"] = q.ppl / q.model.map(ref); fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for m, (c, mk, ls, nm) in STYLE.items():
        d = q[q.model == m]
        ax.scatter(d[d.recipe == "imat"].ppl_ratio, d[d.recipe == "imat"].delta_sel_pp, color=c, marker=mk, s=22, label=nm)
        ax.scatter(d[d.recipe == "noimat"].ppl_ratio, d[d.recipe == "noimat"].delta_sel_pp, facecolors="none", edgecolors=c, marker=mk, s=22)
    ax.axhline(0, color="grey", lw=.6); ax.set_xscale("log"); ax.set_xlabel("Perplexity ratio vs F16 (WikiText-2 test)"); ax.set_ylabel("Selection-rate shift vs F16 (pp)")
    try:
        sp = rd("perplexity_vs_drift_spearman"); r = sp[(sp.subset == "imatrix levels, all models") & (sp.target == "abs_delta_sel_pp")].iloc[0]
        ax.text(.03, .04, r"Spearman $\rho$ = %.2f (n = %d, imatrix)" % (r.spearman_rho, int(r.n)), transform=ax.transAxes, fontsize=6.5)
    except Exception: pass
    ax.legend(frameon=False, title="filled: imatrix, open: none", title_fontsize=6.5); save(fig, "fig7_perplexity_vs_drift")


def fig8_heatmap():
    g = main_runs(rd("gaps")); models = [m for m in STYLE if m in g.model.values]; fig, axs = plt.subplots(1, len(models), figsize=(7.0, 2.3), squeeze=False)
    vmax = g[[f"mean_{k}" for k in C.GROUPS]].values.max()
    for ax, m in zip(axs[0], models):
        d = g[g.model == m]; M = d[[f"mean_{k}" for k in C.GROUPS]].values.T
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=vmax); ax.set_xticks(range(len(d))); ax.set_xticklabels(d.level, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(4)); ax.set_yticklabels(C.GROUPS); ax.set_title(STYLE[m][3])
        for i in range(4):
            for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="w" if M[i, j] < vmax / 2 else "k")
    fig.colorbar(im, ax=axs[0].tolist(), shrink=.8, label="Mean P(Yes)"); save(fig, "fig8_group_heatmap")


if __name__ == "__main__":
    for f in (fig1_selection, fig2_gap_change, fig3_flips, fig4_uncertainty, fig5_auc, fig6_imatrix, fig7_ppl, fig8_heatmap):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, repr(e)[:200])
