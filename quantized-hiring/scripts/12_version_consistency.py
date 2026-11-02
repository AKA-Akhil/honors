"""Task 3: score 100 seeded template-A prompts on Qwen F16 and Q2_K with the b11972 llama-server (CUDA build) and compare with round-2 p_yes
(llama-cpp-python 0.3.36, bundled ggml 0.25.3 commit 0c1e570). Prompts are pre-tokenised with the SAME tokenizer/chat template as round 2 and sent as token ids,
so only the inference engine differs. p_yes = P(yes ids)/(P(yes ids)+P(no ids)) over the server's top-N logprobs (N=n_probs). Output: results/version_consistency.csv"""
import sys, json, time, subprocess, importlib.util, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import config as C
C.add_cuda_dll_dirs()
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
spec = importlib.util.spec_from_file_location("inf", C.ROOT / "scripts" / "03_run_inference.py"); inf = importlib.util.module_from_spec(spec); spec.loader.exec_module(inf)
KEY = "qwen25_3b"; PORT = 8089; NPROBS = 300
prompts = {p["prompt_id"]: p for p in inf.load_prompts(["A"])}
ids = sorted(np.random.default_rng(C.SEED).choice(sorted(prompts), 100, replace=False))
yn = json.load(open(C.ROOT / "results" / f"yesno_token_ids_{KEY}.json")); YES, NO = set(yn["yes"]), set(yn["no"])

# tokenise exactly like round 2 (vocab-only load: no weights)
vocab = Llama(model_path=str(C.gguf_path(KEY, "Q2_K", "imat")), vocab_only=True, verbose=False)
sc = inf.Scorer.__new__(inf.Scorer); sc.llm = vocab; md = vocab.metadata
tok = lambda i: vocab._model.token_get_text(i)
sc.bos, sc.eos = tok(vocab.token_bos()), tok(vocab.token_eos())
sc.fmt = Jinja2ChatFormatter(template=md["tokenizer.chat_template"], eos_token=sc.eos, bos_token=sc.bos, add_generation_prompt=True)
sc.add_bos_meta = str(md.get("tokenizer.ggml.add_bos_token", "false")).lower() == "true"
TOK = {pid: [int(x) for x in sc.tokens(prompts[pid]["messages"])] for pid in ids}
del vocab


def post(path, body, timeout=600):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


rows = []
for lv, rc, ngl in (("F16", "ref", 24), ("Q2_K", "imat", 99)):
    raw = pd.read_csv(C.RAW / f"{KEY}_{lv}_{rc}_A.csv").set_index("prompt_id")
    srv = subprocess.Popen([str(C.llama_bin("llama-server", "cuda")), "-m", str(C.gguf_path(KEY, lv, rc)), "-ngl", str(ngl), "-c", "2048", "--port", str(PORT), "-np", "1", "--no-webui", "--seed", "42"],
                           stdout=open(C.ROOT / "results" / "logs" / f"server_{lv}.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            time.sleep(2)
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).status == 200: break
            except Exception: pass
        for pid in ids:
            r = post("/completion", {"prompt": TOK[pid], "n_predict": 1, "temperature": 0, "n_probs": NPROBS, "cache_prompt": False, "post_sampling_probs": False})
            cp = r["completion_probabilities"][0]; tl = cp.get("top_logprobs") or cp.get("top_probs")
            py = pn = 0.0
            for e in tl:
                p = np.exp(e["logprob"]) if "logprob" in e else e["prob"]
                if e["id"] in YES: py += p
                elif e["id"] in NO: pn += p
            top_id = max(tl, key=lambda e: e.get("logprob", e.get("prob")))["id"]
            p_new = py / (py + pn) if py + pn > 0 else np.nan
            old = raw.loc[pid, "p_yes"]
            rows.append(dict(level=lv, prompt_id=pid, p_yes_b11972=p_new, p_yes_round2=old, abs_diff=abs(p_new - old), same_decision=(p_new >= .5) == (old >= .5),
                             mass_yesno_b11972=py + pn, top_is_yes_or_no=(top_id in YES) or (top_id in NO)))
    finally:
        srv.terminate(); srv.wait(); time.sleep(3)
    print("done", lv, flush=True)
df = pd.DataFrame(rows); df.to_csv(C.ROOT / "results" / "version_consistency.csv", index=False)
s = df.groupby("level").agg(n=("abs_diff", "size"), max_abs_diff=("abs_diff", "max"), mean_abs_diff=("abs_diff", "mean"), identical_decisions=("same_decision", "mean"), mean_mass=("mass_yesno_b11972", "mean"))
print(s)
