"""Round 2 inference: P(Yes) from full next-token logits for every model x level x recipe x template. Resumable.

  python scripts/03_run_inference.py --sanity                 # 20 prompts per model at F16
  python scripts/03_run_inference.py --models llama32_3b      # full run for one model (levels per config.run_spec())
  python scripts/03_run_inference.py --determinism            # repeat 100 prompts on Q4_K_M imat and F16 and compare
Each prompt is evaluated from a clean context (llm.reset() + direct eval; no prompt/KV cache reuse).
"""
import sys, gc, json, time, math, argparse, datetime, random, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from tqdm import tqdm
import config as C

C.add_cuda_dll_dirs()
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

FIELDS = ["prompt_id", "resume_id", "job", "tier", "group", "gender", "race", "first_name", "template"]
OUT_COLS = FIELDS + ["p_yes", "p_yes_raw", "p_no_raw", "mass_yesno", "decision", "top_token", "latency_s"]
NGL_CACHE = {}


def status(line):
    with open(C.ROOT / "STATUS.md", "a", encoding="utf-8") as f:
        f.write(f"- [{datetime.datetime.now():%Y-%m-%d %H:%M}] {line}\n")


def decisions(line):
    with open(C.ROOT / "DECISIONS.md", "a", encoding="utf-8") as f:
        f.write(f"- {line}\n")


def load_prompts(templates=None):
    ps = [json.loads(l) for l in open(C.DATA / "prompts.jsonl", encoding="utf-8")]
    return [p for p in ps if templates is None or p["template"] in templates]


class Scorer:
    def __init__(self, llm, model_key):
        self.llm = llm
        md = llm.metadata
        tok = lambda i: llm._model.token_get_text(i)
        self.bos, self.eos = tok(llm.token_bos()), tok(llm.token_eos())
        self.fmt = Jinja2ChatFormatter(template=md["tokenizer.chat_template"], eos_token=self.eos, bos_token=self.bos, add_generation_prompt=True)
        self.add_bos_meta = str(md.get("tokenizer.ggml.add_bos_token", "false")).lower() == "true"
        self.yes_ids, self.no_ids = self._ids(model_key)

    def _ids(self, key):
        """Token ids whose decoded, stripped, lowercased text is exactly 'yes' / 'no' (cached per model; vocab is identical across levels)."""
        f = C.ROOT / "results" / f"yesno_token_ids_{key}.json"
        if f.exists():
            d = json.load(open(f)); return np.array(d["yes"]), np.array(d["no"])
        yes, no = [], []
        for i in range(self.llm.n_vocab()):
            try:
                s = self.llm.detokenize([i]).decode("utf-8", errors="ignore").strip().lower()
            except Exception:
                continue
            if s == "yes": yes.append(i)
            elif s == "no": no.append(i)
        f.parent.mkdir(exist_ok=True)
        json.dump({"yes": yes, "no": no, "yes_text": [self.llm.detokenize([i]).decode("utf-8", "ignore") for i in yes],
                   "no_text": [self.llm.detokenize([i]).decode("utf-8", "ignore") for i in no]}, open(f, "w"))
        return np.array(yes), np.array(no)

    def tokens(self, messages):
        text = self.fmt(messages=messages).prompt
        add_bos = (not text.startswith(self.bos)) and self.add_bos_meta
        return self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True)

    def score(self, messages):
        toks = self.tokens(messages)
        self.llm.reset()                                   # clean state: no KV/prompt cache reuse between prompts
        self.llm.eval(toks)
        lg = np.ctypeslib.as_array(self.llm._ctx.get_logits(), shape=(self.llm.n_vocab(),)).astype(np.float64)
        lg -= lg.max(); pr = np.exp(lg); pr /= pr.sum()
        py, pn = float(pr[self.yes_ids].sum()), float(pr[self.no_ids].sum())
        top = self.llm.detokenize([int(pr.argmax())]).decode("utf-8", errors="ignore")
        return (py / (py + pn) if py + pn > 0 else float("nan")), py, pn, top


NGL_FILE = C.ROOT / "results" / "ngl_cache.json"


def _try_load(path, ngl):
    llm = Llama(model_path=str(path), n_ctx=2048, n_gpu_layers=ngl, seed=C.SEED, logits_all=False, verbose=False)
    toks = llm.tokenize(b"warm up " * 225, add_bos=True)
    llm.eval(toks); llm.reset()
    return llm, toks


def load_llm(path):
    """Largest working n_gpu_layers. Files larger than ~5 GB (the F16 references) do not fit in 6 GB VRAM: full offload makes the driver page
    weights and is ~10x slower, so for them the best partial offload is chosen by timing (cached in results/ngl_cache.json)."""
    key = str(path); big = Path(path).stat().st_size > 5.0e9
    cache = json.load(open(NGL_FILE)) if NGL_FILE.exists() else {}
    if key in cache: NGL_CACHE[key] = cache[key]
    if big and key not in NGL_CACHE:
        best = None
        for ngl in (28, 26, 24, 22, 20, 18, 16, 12):
            try:
                llm, toks = _try_load(path, ngl)
                t0 = time.time()
                for _ in range(4):
                    llm.reset(); llm.eval(toks)
                dt = (time.time() - t0) / 4; del llm; gc.collect()
                print(f"  autotune {Path(path).name}: n_gpu_layers={ngl} -> {dt:.2f} s per 450-token prompt", flush=True)
                if best is None or dt < best[1]: best = (ngl, dt)
            except Exception as e:
                print(f"  autotune n_gpu_layers={ngl} failed: {str(e)[:80]}", flush=True)
        if best is None: raise RuntimeError(f"cannot load {path}")
        NGL_CACHE[key] = best[0]; cache[key] = best[0]; json.dump(cache, open(NGL_FILE, "w"))
        decisions(f"n_gpu_layers={best[0]} for {Path(path).name} (auto-tuned by timing: {best[1]:.2f} s per 450-token prompt; full offload spills VRAM and is ~10x slower)")
    ladder = [NGL_CACHE[key]] if key in NGL_CACHE else [-1, 26, 22, 18, 14, 10, 6, 0]
    for ngl in ladder:
        try:
            llm, _ = _try_load(path, ngl)
            if key not in NGL_CACHE:
                NGL_CACHE[key] = ngl; decisions(f"n_gpu_layers={ngl} for {Path(path).name}")
            return llm
        except Exception as e:
            print(f"load failed n_gpu_layers={ngl}: {str(e)[:100]}", flush=True); gc.collect()
    raise RuntimeError(f"cannot load {path}")


def run_one(sc, prompts, out_csv, desc):
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=OUT_COLS)
    seen = set(done["prompt_id"]); rows = done.to_dict("records")
    todo = [p for p in prompts if p["prompt_id"] not in seen]
    for i, p in enumerate(tqdm(todo, desc=desc, mininterval=30)):
        row = {k: p[k] for k in FIELDS}
        t0 = time.time()
        for attempt in range(2):
            try:
                py, pyr, pnr, top = sc.score(p["messages"]); break
            except Exception as e:
                print("error:", e, flush=True); py = pyr = pnr = float("nan"); top = ""
        row.update(p_yes=py, p_yes_raw=pyr, p_no_raw=pnr, mass_yesno=pyr + pnr, decision=(1 if py >= 0.5 else 0) if not math.isnan(py) else float("nan"),
                   top_token=top, latency_s=round(time.time() - t0, 4))
        rows.append(row)
        if (i + 1) % 50 == 0:
            pd.DataFrame(rows, columns=OUT_COLS).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows, columns=OUT_COLS); df.to_csv(out_csv, index=False)
    return df


def log_template(key, sc):
    f = C.ROOT / "results" / "chat_templates.txt"
    with open(f, "a", encoding="utf-8") as fh:
        ex = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
        fh.write(f"== {key}: template source = GGUF metadata tokenizer.chat_template; add_bos_token={sc.add_bos_meta}; bos={sc.bos!r}; eos={sc.eos!r}\n"
                 f"rendered example: {sc.fmt(messages=ex).prompt!r}\nyes ids: {len(sc.yes_ids)}, no ids: {len(sc.no_ids)}\n")


def run_model(key, only=None):
    C.RAW.mkdir(parents=True, exist_ok=True)
    allp = load_prompts()
    for level, recipe, templates in C.run_spec():
        if only and (level, recipe) not in only:
            continue
        path = C.gguf_path(key, level, recipe)
        if not path.exists():
            print(f"[{key}] missing {path.name}, skipping", flush=True); continue
        todo_t = [t for t in templates if not ((C.RAW / f"{key}_{level}_{recipe}_{t}.csv").exists() and
                  len(pd.read_csv(C.RAW / f"{key}_{level}_{recipe}_{t}.csv")) >= sum(p["template"] == t for p in allp))]
        if not todo_t:
            continue
        llm = load_llm(path); sc = Scorer(llm, key)
        if not (C.ROOT / "results" / "chat_templates.txt").exists() or key not in (C.ROOT / "results" / "chat_templates.txt").read_text(encoding="utf-8"):
            log_template(key, sc)
        for t in todo_t:
            t0 = time.time()
            df = run_one(sc, [p for p in allp if p["template"] == t], C.RAW / f"{key}_{level}_{recipe}_{t}.csv", f"{key} {level} {recipe} {t}")
            n_nan = int(df["p_yes"].isna().sum()); low = float((df["mass_yesno"] < 0.5).mean())
            status(f"inference {key} {level} {recipe} template {t}: {len(df)} prompts in {(time.time()-t0)/60:.1f} min "
                   f"({(time.time()-t0)/max(len(df),1):.2f} s/prompt), NaN={n_nan}, share with P(yes)+P(no)<0.5: {low:.3f}, ngl={NGL_CACHE.get(str(path))}")
        del sc, llm; gc.collect()


def sanity(keys):
    for key in keys:
        llm = load_llm(C.gguf_path(key, C.REF_LEVEL, "ref")); sc = Scorer(llm, key); log_template(key, sc)
        ps = load_prompts(["A"]); rng = random.Random(C.SEED)
        sel = [p for t in C.TIERS for p in rng.sample([q for q in ps if q["tier"] == t], 5)]
        res = []
        for p in sel:
            t0 = time.time(); py, pyr, pnr, top = sc.score(p["messages"]); res.append((p["tier"], p["group"], round(py, 4), round(pyr + pnr, 3), top, round(time.time() - t0, 2)))
        for r in res: print(r)
        m = pd.DataFrame(res, columns=["tier", "g", "p", "mass", "top", "t"]).groupby("tier").p.mean()
        print(key, m.to_dict(), "OK" if (m["strong"] > m["weak"] and len(set(r[2] for r in res)) > 3) else "SANITY FAILED", flush=True)
        status(f"sanity {key} F16: tier means {m.round(3).to_dict()}, mean s/prompt {np.mean([r[5] for r in res]):.2f}, ngl={NGL_CACHE.get(str(C.gguf_path(key, C.REF_LEVEL, 'ref')))}")
        del sc, llm; gc.collect()


def determinism(keys):
    f = C.ROOT / "results" / "determinism_check.csv"
    rows = []
    for key in keys:
        for level, recipe in (("Q4_K_M", "imat"), (C.REF_LEVEL, "ref")):
            path = C.gguf_path(key, level, recipe)
            if not path.exists(): continue
            ref = pd.concat([pd.read_csv(C.RAW / f"{key}_{level}_{recipe}_{t}.csv") for t in C.TEMPLATES if (C.RAW / f"{key}_{level}_{recipe}_{t}.csv").exists()])
            ids = random.Random(C.SEED).sample(list(ref.prompt_id), 100); byid = {p["prompt_id"]: p for p in load_prompts()}
            llm = load_llm(path); sc = Scorer(llm, key)
            new = {i: sc.score(byid[i]["messages"])[0] for i in ids}
            old = ref.set_index("prompt_id").p_yes
            d = np.array([abs(new[i] - old[i]) for i in ids])
            rows.append(dict(model=key, level=level, recipe=recipe, n=100, max_abs_diff=d.max(), n_over_1e6=int((d > 1e-6).sum())))
            print(rows[-1], flush=True)
            del sc, llm; gc.collect()
    pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true"); ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--models", default=",".join(C.MODELS))
    a = ap.parse_args(); keys = a.models.split(",")
    if a.sanity: sanity(keys)
    elif a.determinism: determinism(keys)
    else:
        for k in keys: run_model(k)
