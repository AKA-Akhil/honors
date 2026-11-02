"""Make every quantization locally from the F16 source with one recipe, plus perplexity per file. Resumable.

  python scripts/01_quantize.py --models llama32_3b --stage all
Stages: provenance (SHA256 of sources), imatrix, quantize, perplexity.
Recipe: llama-imatrix on WikiText-2 *train* (100 chunks, ctx 512); llama-quantize [--imatrix] F16 -> LEVEL; llama-perplexity on WikiText-2 *test* (50 chunks, ctx 512).
"""
import sys, re, csv, argparse, subprocess, hashlib, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

CAL = C.DATA / "calibration"
LOGS = C.ROOT / "results" / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
NGL_LADDER = [99, 24, 16, 8, 0]      # GPU layers to try (6 GB VRAM; F16 3-4B models do not fit fully)


def sh(cmd, log):
    t0 = time.time()
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
    Path(log).write_text("CMD: " + " ".join(map(str, cmd)) + "\n" + p.stdout + "\n" + p.stderr, encoding="utf-8")
    return p.returncode, p.stdout + p.stderr, time.time() - t0


def with_ngl(build, log, ok):
    """Run build(ngl) down the ladder until it succeeds and ok(text) is true. Returns (ngl, text)."""
    for ngl in NGL_LADDER:
        rc, txt, _ = sh(build(ngl), log)
        if rc == 0 and ok(txt):
            return ngl, txt
        print(f"  failed with -ngl {ngl} (rc={rc}); trying lower", flush=True)
    raise RuntimeError(f"all -ngl settings failed, see {log}")


def sha256(p, buf=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


def stage_imatrix(key):
    out = C.MODELS_DIR / key / "imatrix.dat"
    if out.exists():
        print(f"[{key}] imatrix exists"); return
    ref = C.gguf_path(key, C.REF_LEVEL, "ref")
    print(f"[{key}] llama-imatrix ...", flush=True)
    ngl, _ = with_ngl(lambda n: [C.llama_bin("llama-imatrix"), "-m", ref, "-f", CAL / "wikitext2_train.txt", "-o", out, "--chunks", 100, "-c", 512, "-ngl", n],
                      LOGS / f"imatrix_{key}.log", lambda t: out.exists())
    print(f"[{key}] imatrix done (-ngl {ngl})", flush=True)
    with open(LOGS / "imatrix_settings.txt", "a") as f:
        f.write(f"{key}: llama-imatrix -f wikitext2_train.txt --chunks 100 -c 512 -ngl {ngl}\n")


def stage_quantize(key):
    ref = C.gguf_path(key, C.REF_LEVEL, "ref")
    imat = C.MODELS_DIR / key / "imatrix.dat"
    for level, recipe in [(l, "imat") for l in C.LEVELS_IMAT] + [(l, "noimat") for l in C.LEVELS_NOIMAT]:
        out = C.gguf_path(key, level, recipe)
        if out.exists():
            continue
        cmd = [C.llama_bin("llama-quantize", "cpu")]          # quantize is CPU-bound; the CPU build is used for every model
        if recipe == "imat" and level != "Q8_0":              # Q8_0 does not use the imatrix; file still carries the -imat tag
            cmd += ["--imatrix", imat]
        tmp = out.with_suffix(".part")                        # write to .part, rename when complete (resumable)
        rc, txt, dt = sh(cmd + [ref, tmp, level], LOGS / f"quantize_{key}_{level}_{recipe}.log")
        if rc != 0 or not tmp.exists():
            print(f"[{key}] quantize {level} {recipe} FAILED, see log", flush=True); continue
        tmp.rename(out)
        print(f"[{key}] quantized {level} {recipe} in {dt:.0f}s -> {out.stat().st_size/1e9:.2f} GB", flush=True)


def stage_perplexity(key, only_existing=True):
    pfile = C.ROOT / "results" / "perplexity.csv"
    done = set()
    if pfile.exists():
        done = {(r["model"], r["level"], r["recipe"]) for r in csv.DictReader(open(pfile))}
    for level, recipe, _ in C.run_spec():
        p = C.gguf_path(key, level, recipe)
        if (key, level, recipe) in done or not p.exists():
            continue
        print(f"[{key}] perplexity {level} {recipe} ...", flush=True)
        ngl, txt = with_ngl(lambda n: [C.llama_bin("llama-perplexity"), "-m", p, "-f", CAL / "wikitext2_test.txt", "--chunks", 50, "-c", 512, "-ngl", n],
                            LOGS / f"ppl_{key}_{level}_{recipe}.log", lambda t: "Final estimate" in t)
        m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", txt)
        new = not pfile.exists()
        with open(pfile, "a", newline="") as f:
            w = csv.writer(f)
            if new: w.writerow(["model", "level", "recipe", "ppl", "ppl_se", "size_gb", "ngl"])
            w.writerow([key, level, recipe, m.group(1), m.group(2), round(p.stat().st_size / 1e9, 3), ngl])
        print(f"[{key}] {level} {recipe}: PPL {m.group(1)} +/- {m.group(2)}", flush=True)


PROV = {"llama32_3b": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-f16.gguf", "downloaded F16 (file renamed to <key>-F16.gguf)"),
        "qwen25_3b": ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-f16.gguf", "downloaded F16 (file renamed to <key>-F16.gguf)"),
        "phi35_mini": ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-f32.gguf", "no F16 GGUF published: F32 GGUF downloaded and converted to F16 with llama-quantize")}


def stage_convert(key):
    """Phi-3.5-mini only: bartowski publishes F32 but no F16. F32 -> F16 rounding is identical to converting the bf16 source to F16 directly."""
    out = C.gguf_path(key, C.REF_LEVEL, "ref")
    src = C.MODELS_DIR / key / PROV[key][1]
    if out.exists() or not src.exists() or key != "phi35_mini":
        return
    rc, txt, dt = sh([C.llama_bin("llama-quantize", "cpu"), src, out.with_suffix(".part"), "F16"], LOGS / f"convert_{key}.log")
    if rc == 0: out.with_suffix(".part").rename(out); print(f"[{key}] converted F32 -> F16 in {dt:.0f}s", flush=True)
    else: print(f"[{key}] conversion FAILED, see log", flush=True)


def stage_provenance(key):
    repo, fname, how = PROV[key]
    pf = C.ROOT / "results" / "model_provenance.csv"
    rows = list(csv.DictReader(open(pf))) if pf.exists() else []
    if any(r["model"] == key for r in rows):
        return
    p = C.gguf_path(key, C.REF_LEVEL, "ref"); src = C.MODELS_DIR / key / fname
    rows.append(dict(model=key, source_repo=repo, source_file=fname, how=how, source_sha256=sha256(src) if src.exists() else "", source_size_bytes=src.stat().st_size if src.exists() else "",
                     f16_file=p.name, f16_sha256=sha256(p), f16_size_bytes=p.stat().st_size))
    with open(pf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(C.MODELS)); ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    for k in a.models.split(","):
        st = a.stage
        if st in ("convert", "all"): stage_convert(k)
        if st in ("provenance", "all"): stage_provenance(k)
        if st in ("imatrix", "all"): stage_imatrix(k)
        if st in ("quantize", "all"): stage_quantize(k)
        if st in ("perplexity", "all"): stage_perplexity(k)
