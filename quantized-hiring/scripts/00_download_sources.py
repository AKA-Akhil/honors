"""Download full-precision sources for the three models (resumable). Prefers an F16 GGUF; Phi-3.5-mini has none, see DECISIONS.md.

Usage: python scripts/00_download_sources.py [llama32_3b qwen25_3b phi35_mini]
"""
import os, sys, hashlib, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if os.environ.get("USE_HF_TRANSFER", "1") == "1":
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
import config as C

# key -> (repo, kind, pattern). kind 'gguf' = single full-precision GGUF; 'f32gguf' / 'hf' described in DECISIONS.md
SOURCES = {
    "llama32_3b": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "gguf", "f16"),
    "qwen25_3b": ("bartowski/Qwen2.5-3B-Instruct-GGUF", "gguf", "f16"),
    "phi35_mini": (os.environ.get("PHI_REPO", "bartowski/Phi-3.5-mini-instruct-GGUF"), "gguf", os.environ.get("PHI_PAT", "f32")),
}


def sha256(p, buf=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


def main():
    keys = sys.argv[1:] or list(SOURCES)
    for k in keys:
        repo, kind, pat = SOURCES[k]
        d = C.MODELS_DIR / k; d.mkdir(parents=True, exist_ok=True)
        files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf") and pat in f.lower()]
        assert files, f"no {pat} file in {repo}"
        f = files[0]
        print(f"[{k}] {repo}/{f}", flush=True)
        for attempt in range(50):
            try:
                hf_hub_download(repo, f, local_dir=d); break
            except Exception as e:
                print("retry:", str(e)[:200], flush=True)
                os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        print(f"[{k}] done {f}", flush=True)


if __name__ == "__main__":
    main()
