"""Streamlit front-end for the quantized-hiring audit.
Run from the project root:  .venv\\Scripts\\streamlit run app\\streamlit_app.py
Upload (or paste) a resume, pick a model and quantization level, and see P(Yes), the decision, and an optional name-swap counterfactual.
Uses exactly the same prompt templates, chat template and P(Yes) computation as the experiments (scripts/02_generate_data.py, scripts/03_run_inference.py).
"""
import sys, re, gc, io, subprocess, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import streamlit as st
import config as C

st.set_page_config(page_title="Quantized hiring audit", layout="wide")
LABELS = {"llama32_3b": "Llama-3.2-3B-Instruct", "qwen25_3b": "Qwen2.5-3B-Instruct", "phi35_mini": "Phi-3.5-mini-instruct"}
LEVELS = [C.REF_LEVEL] + C.LEVELS_IMAT


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@st.cache_resource
def modules():
    C.add_cuda_dll_dirs()
    return _load_module("gen", ROOT / "scripts" / "02_generate_data.py"), _load_module("inf", ROOT / "scripts" / "03_run_inference.py")


gen, inf = modules()


def model_file(model, level):
    return C.gguf_path(model, level, "ref" if level == C.REF_LEVEL else "imat")


# ------------------------------------------------------------------ model handling (one model in memory at a time: 6 GB VRAM)
if "loaded" not in st.session_state:
    st.session_state.loaded = None      # (path, llm, scorer)


def get_scorer(model, level):
    path = str(model_file(model, level))
    cur = st.session_state.loaded
    if cur and cur[0] == path:
        return cur[2]
    if cur:
        st.session_state.loaded = None; del cur; gc.collect()
    llm = inf.load_llm(path)
    sc = inf.Scorer(llm, model)
    st.session_state.loaded = (path, llm, sc)
    return sc


def build_level(model, level):
    """Re-create a quantized file from the kept F16 + imatrix (same command as scripts/01_quantize.py). CPU only, ~2 min."""
    out = model_file(model, level); tmp = out.with_suffix(".part")
    cmd = [str(C.llama_bin("llama-quantize", "cpu"))]
    if level != "Q8_0":
        cmd += ["--imatrix", str(C.MODELS_DIR / model / "imatrix.dat")]
    r = subprocess.run(cmd + [str(model_file(model, C.REF_LEVEL)), str(tmp), level], capture_output=True, text=True)
    if r.returncode == 0 and tmp.exists():
        tmp.rename(out); return True, ""
    return False, (r.stderr or r.stdout)[-500:]


# ------------------------------------------------------------------ resume input
def read_upload(f):
    name = f.name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(f).pages)
    if name.endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(f).paragraphs)
    return f.read().decode("utf-8", errors="replace")


def build_messages(template, job, resume_text):
    sysm, usr = gen.TEMPLATES[template]
    u = usr.replace("{JOB_TITLE}", job).replace("{JOB_DESCRIPTION}", gen.job_desc(job)).replace("{RESUME_TEXT}", resume_text)
    return [{"role": "system", "content": sysm}, {"role": "user", "content": u}]


def swap_name(text, old_name, new_first, new_sur):
    """Replace the candidate's full name, e-mail and bare first/last name occurrences."""
    parts = old_name.split()
    new_full = f"{new_first} {new_sur}".strip()
    t = text
    email_old = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
    t = email_old.sub(f"{new_first.lower()}.{new_sur.lower()}@email.com" if new_sur else f"{new_first.lower()}@email.com", t, count=1)
    t = t.replace(old_name, new_full)
    if len(parts) >= 2:
        t = re.sub(rf"\b{re.escape(parts[0])}\b", new_first, t); t = re.sub(rf"\b{re.escape(parts[-1])}\b", new_sur, t)
    elif parts:
        t = re.sub(rf"\b{re.escape(parts[0])}\b", new_first, t)
    return t


def score(sc, template, job, text):
    msgs = build_messages(template, job, text)
    n_tok = len(sc.tokens(msgs))
    if n_tok > 1900:
        st.warning(f"Prompt is {n_tok} tokens; the context window is 2048, so very long resumes may fail. Shorten the resume.")
    py, pyr, pnr, top = sc.score(msgs)
    return dict(p_yes=py, mass=pyr + pnr, top=top.strip(), tokens=n_tok)


# ------------------------------------------------------------------ UI
st.title("Does quantization change a hiring decision?")
st.caption("Counterfactual audit tool from the quantized-hiring project: screens a resume with a local GGUF model and reports P(Yes).")

tab_run, tab_res = st.tabs(["Screen a resume", "Project results"])

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", list(LABELS), format_func=LABELS.get)
    avail = [l for l in LEVELS if model_file(model, l).exists()]
    missing = [l for l in LEVELS if l not in avail]
    level = st.selectbox("Quantization level", LEVELS, index=0, format_func=lambda l: f"{l}" + ("" if l in avail else "  (not built)"))
    if level not in avail:
        if not model_file(model, C.REF_LEVEL).exists():
            st.error("The F16 source file for this model is missing, so this level cannot be built.")
        elif st.button(f"Build {level} from F16 (~2 min, CPU)"):
            with st.spinner(f"Quantizing {model} to {level} ..."):
                ok, err = build_level(model, level)
            st.success("Built.") if ok else st.error(err)
            st.rerun()
    template = st.selectbox("Prompt template", list(gen.TEMPLATES), help="The three wordings used in the experiments")
    job = st.selectbox("Job", list(gen.OLD_JOBS))
    st.markdown(f"**Job description**\n\n{gen.job_desc(job)}")
    st.info("F16 files do not fit in 6 GB VRAM and run with partial offload (slower, first load ~1 min).")

with tab_run:
    up = st.file_uploader("Upload a resume (.txt, .pdf, .docx)", type=["txt", "pdf", "docx", "md"])
    default = ""
    if up is not None:
        try:
            default = read_upload(up)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
    text = st.text_area("Resume text (edit if needed)", value=default, height=300, key=f"resume_{up.name if up else 'paste'}")
    m = re.search(r"(?im)^\s*name\s*[:\-]\s*(.+)$", text)
    guess = m.group(1).strip() if m else (text.strip().splitlines()[0].strip() if text.strip() else "")
    cand_name = st.text_input("Candidate name as written in the resume (used for the name-swap test)", value=guess)
    c1, c2 = st.columns(2)
    do_swap = c1.checkbox("Name-swap counterfactual (WF / BF / WM / BM)", value=True)
    n_names = c2.slider("Names per group", 1, 5, 2, disabled=not do_swap)
    compare = st.checkbox("Also score at every built quantization level of this model (slower)")
    run = st.button("Run", type="primary", disabled=not (text.strip() and level in avail))

    if run:
        levels = [l for l in LEVELS if l in avail] if compare else [level]
        results, rows_cf = [], []
        bar = st.progress(0.0)
        for li, lv in enumerate(levels):
            with st.spinner(f"Loading {LABELS[model]} {lv} ..."):
                sc = get_scorer(model, lv)
            r = score(sc, template, job, text); r.update(level=lv)
            results.append(r)
            if do_swap and cand_name.strip():
                sur_pool = gen.SURNAMES
                for grp in C.GROUPS:
                    for first in gen.NAMES[grp][:n_names]:
                        t2 = swap_name(text, cand_name.strip(), first, sur_pool[0])
                        rr = score(sc, template, job, t2); rows_cf.append(dict(level=lv, group=grp, first_name=first, p_yes=rr["p_yes"]))
            bar.progress((li + 1) / len(levels))
        main = pd.DataFrame(results)
        sel = main[main.level == level].iloc[0] if (main.level == level).any() else main.iloc[0]
        st.subheader("Result")
        a, b, c, d = st.columns(4)
        a.metric("P(Yes)", f"{sel.p_yes:.3f}")
        b.metric("Decision (threshold 0.5)", "Shortlist (Yes)" if sel.p_yes >= .5 else "Reject (No)")
        c.metric("Yes/No probability mass", f"{sel.mass:.3f}", help="Near 1 means the model's next token is Yes or No; low values indicate a format failure.")
        d.metric("Top next token", sel.top)
        if sel.mass < 0.5:
            st.warning("Low Yes/No mass: the model is not answering Yes/No here, so P(Yes) is unreliable (seen with the broken no-imatrix Q2_K Qwen file).")
        if compare:
            st.markdown("**Across quantization levels**")
            st.bar_chart(main.set_index("level").p_yes)
            st.dataframe(main[["level", "p_yes", "mass", "top", "tokens"]].round(4), hide_index=True)
        if rows_cf:
            cf = pd.DataFrame(rows_cf)
            st.markdown("**Name-swap counterfactual** (same resume, only the name and e-mail change)")
            g = cf.groupby(["level", "group"]).p_yes.mean().unstack()[C.GROUPS]
            g["gender gap (M-F)"] = (g.WM + g.BM) / 2 - (g.WF + g.BF) / 2
            g["race gap (W-B)"] = (g.WF + g.WM) / 2 - (g.BF + g.BM) / 2
            g["decision flips across names"] = cf.groupby("level").apply(lambda x: bool((x.p_yes >= .5).nunique() > 1))
            st.dataframe(g.round(4))
            st.caption("Gaps are means over the sampled names; the project found them to be tiny (|gap| < ~0.02) and decision changes across names rare. Compare this resume with the stats in the Project results tab before drawing conclusions from a single resume.")
            with st.expander("Per-name scores"):
                st.dataframe(cf.round(4), hide_index=True)

with tab_res:
    rpt = ROOT / "RESULTS_REPORT.md"
    t = ROOT / "results" / "tables"
    if (t / "selection.csv").exists():
        sel = pd.read_csv(t / "selection.csv"); sel = sel[(sel.template == "A") & sel.recipe.isin(["ref", "imat"])]
        st.subheader("Selection rate by model and level (template A)")
        piv = sel.pivot(index="level", columns="model", values="sel_rate").reindex(LEVELS)
        st.line_chart(piv)
        fl = pd.read_csv(t / "flips.csv"); fl = fl[(fl.template == "A") & (fl.recipe == "imat")]
        st.subheader("Flip rate vs F16 (template A)")
        st.line_chart(fl.pivot(index="level", columns="model", values="flip_rate").reindex(LEVELS[1:]))
    for f, cap in (("fig2_gap_change_vs_f16", "Change in demographic gaps vs F16"), ("fig7_perplexity_vs_drift", "Perplexity vs decision drift")):
        p = ROOT / "results" / "figures" / f"{f}.png"
        if p.exists(): st.image(str(p), caption=cap)
    if rpt.exists():
        with st.expander("Full results report"):
            st.markdown(rpt.read_text(encoding="utf-8"))
