"""Build compact booktabs tables for the paper (paper/tables/*.tex) directly from results/tables/*.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import config as C

OUT = C.ROOT / "paper" / "tables"; OUT.mkdir(parents=True, exist_ok=True)
T = C.TABLES
SHORT = {"llama32_3b": "Llama", "qwen25_3b": "Qwen"}
ORDER = {l: i for i, l in enumerate(C.LEVELS)}


def lv(l): return l.replace("_", r"\_")
def f3(x): return "--" if pd.isna(x) else f"{x:.3f}"
def p(x):
    if pd.isna(x): return "--"
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def srt(d):
    d = d.copy(); d["o"] = d.level.map(ORDER); return d.sort_values(["model", "o"], key=lambda s: s if s.name == "o" else s.map({"llama32_3b": 0, "qwen25_3b": 1}))


def tex2md(s):
    """Strip the small amount of LaTeX used in the paper tables for the Markdown copy."""
    import re
    for a, b in ((r"\_", "_"), ("$<$", "<"), (r"$\to$", "->"), (r"$^{\dagger}$", "†"), (r"\%", "%"), (r"\chi", "χ"),
                 (r"\Delta", "Δ"), ("``", '"'), ("''", '"'), ("$-$", "-"), ("$<0.8$", "<0.8"), (r"$\rho$", "rho"), (r"\dagger", "†")):
        s = s.replace(a, b)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    return s.replace("$", "").replace("{", "").replace("}", "").replace("|", "\\|")


def write(name, header, rows, align, caption, label, env="table"):
    cols = tex2md(header).split(" & ")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(tex2md(c) for c in r) + " |" for r in rows]
    (OUT / f"{name}.md").write_text(f"*{tex2md(caption)}*\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    tex = (f"\\begin{{{env}}}[t]\n\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}\n\\caption{{{caption}}}\\label{{{label}}}\n"
           f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")
    (OUT / f"{name}.tex").write_text(tex, encoding="utf-8")
    # tabular-only body, included by paper/main.tex inside its own table environment
    (OUT / f"{name}_body.tex").write_text(
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n", encoding="utf-8")


def main():
    gaps = pd.read_csv(T / "gaps.csv"); flips = pd.read_csv(T / "flips.csv"); cap = pd.read_csv(T / "capability.csv"); sel = pd.read_csv(T / "selection.csv")

    for tm in ("A", "B"):
        g = srt(gaps[gaps.template == tm]); rows = []
        for _, r in g.iterrows():
            rows.append([SHORT[r.model], lv(r.level),
                         f"{r.gender_gap:+.3f} [{r.gender_ci_lo:+.3f}, {r.gender_ci_hi:+.3f}]", p(r.gender_p_holm),
                         f"{r.race_gap:+.3f} [{r.race_ci_lo:+.3f}, {r.race_ci_hi:+.3f}]", p(r.race_p_holm)])
        write(f"tab_gaps_{tm}", "Model & Level & Gender gap [95\\% CI] & $p_{\\mathrm{Holm}}$ & Race gap [95\\% CI] & $p_{\\mathrm{Holm}}$", rows, "llcccc",
              f"Paired gaps in P(Yes), Template {tm}. Gender gap = men $-$ women; race gap = White-named $-$ Black-named. Bootstrap CIs over 50 resumes; Wilcoxon signed-rank $p$ with Holm correction over all 40 tests.",
              f"tab:gaps{tm}", env="table*")

    f = srt(flips[(flips.template == "A") & (flips.threshold_type == "0.5")]); rows = []
    for _, r in f.iterrows():
        rows.append([SHORT[r.model], lv(r.level), f3(r.flip_rate), f"{int(r.yes_to_no)}/{int(r.no_to_yes)}", p(r.mcnemar_p_holm), p(r.chi2_yn_groups_p_raw), f3(r.mean_abs_dp)])
    write("tab_flips_A", "Model & Level & Flip rate & Y$\\to$N / N$\\to$Y & $p_{\\mathrm{McNemar}}^{\\mathrm{Holm}}$ & $p_{\\chi^2}$ (Y$\\to$N by group) & Mean $|\\Delta P|$", rows, "llccccc",
          "Decision flips versus Q8\\_0 (Template A, threshold 0.5, 200 prompts per run). ``--'': undefined (no discordant pairs) or fewer than 4 Y$\\to$N flips for the group test.", "tab:flips")

    c = srt(cap[cap.template == "A"]); s = sel[(sel.template == "A") & (sel.threshold_type == "0.5")].set_index(["model", "level"]); rows = []
    for _, r in c.iterrows():
        ss = s.loc[(r.model, r.level)]
        rows.append([SHORT[r.model], lv(r.level), f3(r.mean_strong), f3(r.mean_borderline), f3(r.mean_weak), f3(r.spearman), f3(r.auc_strong_vs_weak),
                     f3(ss.overall), f3(ss.impact_ratio) + (r"$^{\dagger}$" if ss["below_0.8"] else "")])
    write("tab_capability_A", "Model & Level & Strong & Border. & Weak & Spearman & AUC & Sel. rate & IR", rows, "llccccccc",
          "Capability and selection (Template A). Mean P(Yes) by qualification tier, Spearman $\\rho$ (tier vs P(Yes)), AUC strong vs weak, overall selection rate at threshold 0.5 and impact ratio (IR); $^{\\dagger}$: IR $<0.8$.",
          "tab:cap")
    print("paper tables written")


if __name__ == "__main__":
    main()
