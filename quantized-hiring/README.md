# Does Quantization Change Hiring Decisions? (Round 2)

Counterfactual audit of GGUF-quantized LLMs in resume screening: Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, Phi-3.5-mini-instruct.
**Round 2** quantizes every model locally from one F16 source with one recipe (llama.cpp `llama-quantize`, WikiText-2 importance matrix), uses **F16 as the reference**,
150 synthetic resumes x 4 name-swapped variants x 3 prompt templates, plus an imatrix-vs-no-imatrix ablation, mixed-effects models, flip/uncertainty analysis and an independent verification script.

> `paper/` still reflects **round 1** (Q8_0 reference, 2 models, 50 resumes, differently-sourced GGUFs) and is **outdated**; it also contains citation errors listed in `results/citation_check.md`. Round 1 is archived in `archive/round1/`.
> Read `RESULTS_REPORT.md` for the round-2 findings, `STATUS.md` for the run log and `DECISIONS.md` for every deviation.

## Reproduce (Windows, project root; Python 3.12)
```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install llama-cpp-python==0.3.36 --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
.venv\Scripts\python -m pip install -r requirements.txt
set HF_HUB_DISABLE_XET=1
# llama.cpp release b11972 tools: download llama-b11972-bin-win-cpu-x64.zip, llama-b11972-bin-win-cuda-12.4-x64.zip and cudart-llama-bin-win-cuda-12.4-x64.zip
# from https://github.com/ggml-org/llama.cpp/releases/tag/b11972 and unzip into tools\llama.cpp\cpu and tools\llama.cpp\cuda (cuda zip + cudart zip together)
.venv\Scripts\python scripts\00_get_wikitext.py                  # calibration (train) + perplexity (test) text
.venv\Scripts\python scripts\00_download_sources.py              # F16 sources (Phi: F32, converted by 01_quantize.py)
# rename each downloaded F16 GGUF to models\<key>\<key>-F16.gguf (llama32_3b, qwen25_3b); Phi stays as Phi-3.5-mini-instruct-f32.gguf
.venv\Scripts\python scripts\02_generate_data.py                 # 150 resumes, 600 variants, 1,800 prompts, data\data_card.md
bash scripts\run_all.sh llama32_3b qwen25_3b phi35_mini          # per model: 01_quantize (imatrix, quantize, perplexity) -> sanity -> 03_run_inference -> determinism -> cleanup
.venv\Scripts\python scripts\04_analyze.py                       # tables (CSV + LaTeX) in results\tables
.venv\Scripts\python scripts\05_make_figures.py                  # figures in results\figures
.venv\Scripts\python scripts\06_verify.py                        # independent recomputation of headline numbers
```
All scripts are resumable (existing outputs are skipped). Seed 42 everywhere. Test the analysis on synthetic data by setting `QH_RAW`, `QH_TABLES`, `QH_FIGS`.

## Layout
`config.py` constants; `data/` generated data + `data_card.md` + calibration text; `models/` (large, gitignored); `tools/llama.cpp/` binaries;
`scripts/` pipeline; `results/raw/` one CSV per model x level x recipe x template; `results/tables|figures`; `results/perplexity.csv`, `model_provenance.csv`,
`determinism_check.csv`, `verification_log.txt`, `citation_check.md`; `archive/round1/` the finished round 1.

## Hardware
RTX 4050 Laptop GPU (6 GB), 14-core CPU, 16 GB RAM, Windows 11. F16 files (6.4-7.6 GB) are run with auto-tuned partial GPU offload (see DECISIONS.md).

## Demo app
`.venv\Scripts\streamlit run app\streamlit_app.py` - upload (.txt/.pdf/.docx) or paste a resume, pick model, quantization level, prompt template and job; it reports P(Yes), the decision, and an optional name-swap counterfactual (WF/BF/WM/BM), and can score every built level. Missing quantized levels are rebuilt from the kept F16 + imatrix with one click (CPU, ~2 min). A "Project results" tab shows the study's tables and figures.
