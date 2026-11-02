"""Task 1 c+d: re-created Qwen Q3_K_M/Q2_K must reproduce round-2 p_yes on 50 prompts; then greedy 30-token free-text samples for 60 template-A prompts.
Resumable (skips outputs that exist). Seed 42. Outputs: results/qwen_recreate_check.csv, results/qwen_freetext_samples.csv, results/qwen_freetext_summary.md"""
import sys, re, gc, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import config as C
C.add_cuda_dll_dirs()
spec = importlib.util.spec_from_file_location("inf", C.ROOT / "scripts" / "03_run_inference.py"); inf = importlib.util.module_from_spec(spec); spec.loader.exec_module(inf)

KEY = "qwen25_3b"; ALL = [("F16", "ref"), ("Q3_K_M", "imat"), ("Q2_K", "imat")]
LEVELS = [x for x in ALL if len(sys.argv) < 2 or x[0] == sys.argv[1]]   # optional: one level per process
PART = lambda lv, k: C.ROOT / "results" / "logs" / f"qwen_part_{lv}_{k}.csv"
prompts = {p["prompt_id"]: p for p in inf.load_prompts(["A"])}
raw = {lv: pd.read_csv(C.RAW / f"{KEY}_{lv}_{rc}_A.csv").set_index("prompt_id") for lv, rc in LEVELS}
rng = np.random.default_rng(C.SEED)

ids50 = sorted(rng.choice(sorted(prompts), 50, replace=False))
# 60 prompts balanced over tiers (15 each) and name groups (15 each): group order rotates by tier
meta = raw["F16"][["tier", "group", "resume_id"]]
ids60 = []
for ti, tier in enumerate(C.TIERS):
    gs = [C.GROUPS[(ti + k) % 4] for k in range(15)]
    for g in dict.fromkeys(gs):
        n = gs.count(g); pool = sorted(meta[(meta.tier == tier) & (meta.group == g)].index)
        ids60 += list(rng.choice(pool, n, replace=False))
ids60 = sorted(ids60)
assert len(ids60) == 60

CLS = re.compile(r"^[\W_]*(yes|no)\b", re.I)
def classify(t):
    m = CLS.match(t.strip())
    return {"yes": "clear Yes", "no": "clear No"}[m.group(1).lower()] if m else "other"

rec_out, ft_rows = [], []
for lv, rc in LEVELS:
    path = C.gguf_path(KEY, lv, rc)
    if not path.exists(): print("missing", path); continue
    llm = inf.load_llm(path); sc = inf.Scorer(llm, KEY)
    if lv != "F16":
        for pid in ids50:
            py, pyr, pnr, top = sc.score(prompts[pid]["messages"]); old = raw[lv].loc[pid, "p_yes"]
            rec_out.append(dict(level=lv, prompt_id=pid, p_yes_new=py, p_yes_round2=old, abs_diff=abs(py - old), same_decision=(py >= .5) == (old >= .5)))
    for pid in ids60:
        toks = sc.tokens(prompts[pid]["messages"]); llm.reset(); out = []
        for t in llm.generate(toks, top_k=1, top_p=1.0, temp=0.0, reset=True):
            if t == llm.token_eos() or len(out) >= 30: break
            out.append(t)
        txt = llm.detokenize(out).decode("utf-8", errors="replace")
        r = raw[lv].loc[pid]
        ft_rows.append(dict(level=lv, prompt_id=pid, tier=r.tier, group=r.group, p_yes_round2=r.p_yes, mass_yesno_round2=r.mass_yesno, output=txt, cls=classify(txt)))
    del sc, llm; gc.collect(); print("done", lv, flush=True)
    pd.DataFrame([r for r in ft_rows if r["level"] == lv]).to_csv(PART(lv, "ft"), index=False)
    if lv != "F16": pd.DataFrame([r for r in rec_out if r["level"] == lv]).to_csv(PART(lv, "rec"), index=False)
if len(sys.argv) >= 2: sys.exit(0)
# aggregate (run with no argument after all levels are done)
rec_out = [r for lv, _ in ALL[1:] for r in pd.read_csv(PART(lv, "rec")).to_dict("records")]
ft_rows = [r for lv, _ in ALL for r in pd.read_csv(PART(lv, "ft")).to_dict("records")]

if rec_out:
    rd = pd.DataFrame(rec_out); rd.to_csv(C.ROOT / "results" / "qwen_recreate_check.csv", index=False)
    print(rd.groupby("level").agg(n=("abs_diff", "size"), max_abs_diff=("abs_diff", "max"), mean_abs_diff=("abs_diff", "mean"), same_decision=("same_decision", "mean")))
ft = pd.DataFrame(ft_rows); ft.to_csv(C.ROOT / "results" / "qwen_freetext_samples.csv", index=False)
ft["greedy_dec"] = ft.cls.map({"clear Yes": "Yes", "clear No": "No"}).fillna("other"); ft["p_dec"] = np.where(ft.p_yes_round2 >= .5, "Yes", "No")
lines = ["# Qwen free-text check (60 template-A prompts, greedy, <=30 tokens)\n"]
for lv, g in ft.groupby("level", sort=False):
    c = g.cls.value_counts(); n = len(g)
    agree = ((g.greedy_dec == g.p_dec)[g.cls != "other"]).mean()
    lines.append(f"- **{lv}**: clear Yes {c.get('clear Yes',0)} ({100*c.get('clear Yes',0)/n:.0f}%), clear No {c.get('clear No',0)} ({100*c.get('clear No',0)/n:.0f}%), other {c.get('other',0)} ({100*c.get('other',0)/n:.0f}%); greedy answer agrees with the P(Yes)>=0.5 decision on {100*agree:.0f}% of clear answers")
lines.append("\n## All 'other' outputs (full text)\n")
for _, r in ft[ft.cls == "other"].iterrows(): lines.append(f"- {r.level} {r.prompt_id} ({r.tier}/{r.group}, p_yes={r.p_yes_round2:.3f}): {r.output!r}")
if not (ft.cls == "other").any(): lines.append("(none)")
(C.ROOT / "results" / "qwen_freetext_summary.md").write_text("\n".join(lines), encoding="utf-8"); print("\n".join(lines))
