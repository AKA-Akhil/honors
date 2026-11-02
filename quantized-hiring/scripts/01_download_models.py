"""Download GGUF files for each model x level (resumable; uses actual repo filenames, never guessed ones)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import hf_hub_download, list_repo_files
import config as C


def match_level(files, level):
    """Case-insensitive substring match on real repo filenames; prefer exact '-LEVEL.gguf' suffix."""
    cands = [f for f in files if f.lower().endswith(".gguf") and level.lower() in f.lower()]
    exact = [f for f in cands if f.lower().endswith(f"-{level.lower()}.gguf")]
    return (exact or cands or [None])[0]


def main():
    # Priority order (slow link): small file first to test the pipeline, then both Q8_0 references,
    # then the remaining levels level-major so both models always have the same levels available.
    order = [("llama32_3b", "Q2_K"), ("llama32_3b", "Q8_0"), ("qwen25_3b", "Q8_0"), ("qwen25_3b", "Q2_K")]
    for lvl in ("Q4_K_M", "Q3_K_M", "Q6_K"):
        order += [("llama32_3b", lvl), ("qwen25_3b", lvl)]
    listing = {k: list_repo_files(r) for k, r in C.MODELS.items()}
    mpath = C.MODELS_DIR / "manifest.json"
    for key, lvl in order:
        repo = C.MODELS[key]; f = match_level(listing[key], lvl)
        (C.MODELS_DIR / key).mkdir(parents=True, exist_ok=True)
        if f is None:
            print(f"[MISSING] {key} {lvl}", flush=True); continue
        dest = C.MODELS_DIR / key / Path(f).name
        for attempt in range(50):
            if dest.exists():
                print(f"[have] {dest.name}", flush=True); break
            try:
                print(f"[download] {repo}/{f} (attempt {attempt + 1})", flush=True)
                hf_hub_download(repo, f, local_dir=C.MODELS_DIR / key)
            except Exception as e:
                print("retry after error:", e, flush=True)
        if dest.exists():
            mp = json.loads(mpath.read_text()) if mpath.exists() else {}
            mp.setdefault(key, {})[lvl] = Path(f).name
            mpath.write_text(json.dumps(mp, indent=2))
    print(mpath.read_text())


if __name__ == "__main__":
    main()
