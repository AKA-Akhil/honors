"""Fetch WikiText-2 raw (Salesforce/wikitext) train/test and write plain text files for imatrix calibration and perplexity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from huggingface_hub import hf_hub_download
import config as C

out = C.DATA / "calibration"; out.mkdir(parents=True, exist_ok=True)
for split in ("train", "test"):
    p = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    txt = "".join(df["text"].tolist())  # raw lines already carry newlines
    (out / f"wikitext2_{split}.txt").write_text(txt, encoding="utf-8")
    print(split, len(df), "rows,", len(txt), "chars")
