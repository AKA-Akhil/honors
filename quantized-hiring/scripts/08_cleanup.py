"""Delete a model's quantized GGUFs (not the F16 or the imatrix) once all of its raw result CSVs are complete.

  python scripts/08_cleanup.py llama32_3b          # quantized files only
  python scripts/08_cleanup.py --all-f16           # at the very end: delete F16 files too if > 30 GB would remain free... (see DECISIONS.md)
"""
import sys, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import config as C

n_prompts = {t: 0 for t in C.TEMPLATES}
for l in open(C.DATA / "prompts.jsonl", encoding="utf-8"):
    n_prompts[json.loads(l)["template"]] += 1


def complete(key):
    for level, recipe, templates in C.run_spec():
        for t in templates:
            f = C.RAW / f"{key}_{level}_{recipe}_{t}.csv"
            if not f.exists() or len(pd.read_csv(f)) < n_prompts[t]:
                print("incomplete:", f.name); return False
    return True


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--all-f16":
        free = shutil.disk_usage(C.ROOT).free / 1e9
        print(f"free {free:.1f} GB")
        if free > 30:
            print("more than 30 GB free: keeping F16 files (per spec)")
        else:
            for k in C.MODELS:
                p = C.gguf_path(k, C.REF_LEVEL, "ref")
                if p.exists(): p.unlink(); print("deleted", p.name)
        sys.exit(0)
    for key in args:
        if not complete(key):
            print(f"{key}: runs incomplete, NOT deleting"); continue
        for level, recipe, _ in C.run_spec():
            if recipe == "ref": continue
            p = C.gguf_path(key, level, recipe)
            if p.exists(): p.unlink(); print("deleted", p.name)
        print(f"{key}: quantized files deleted; F16 and imatrix kept")
