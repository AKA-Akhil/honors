# STATUS
- [2026-10-28 23:52] Env set up (py3.12 venv, llama-cpp-python CUDA OK), data generated (50 resumes, 400 prompts), scripts 00-05 written. Downloading models (slow link ~1 MB/s).
- [2026-10-29 03:33] Sanity check passed after fixing logits readout (see DECISIONS.md). Projected inference: ~0.1 s/prompt => ~40 s per 400-prompt run, ~15 min total for 4,000 prompts. No reductions needed. Downloads still in progress (Q8_0, Q2_K of both models present).
- [2026-10-29 03:34] inference llama32_3b Q8_0 template A: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:34] inference llama32_3b Q2_K template A: 200 prompts in 0.5 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:35] inference llama32_3b Q8_0 template B: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:35] inference llama32_3b Q2_K template B: 200 prompts in 0.5 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:35] inference llama32_3b Q2_K template B: 200 prompts in 0.1 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:37] inference qwen25_3b Q8_0 template A: 200 prompts in 1.8 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:37] inference qwen25_3b Q8_0 template A: 200 prompts in 1.8 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:39] inference qwen25_3b Q2_K template A: 200 prompts in 1.0 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:39] inference qwen25_3b Q2_K template A: 200 prompts in 1.0 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:40] inference qwen25_3b Q8_0 template B: 200 prompts in 1.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:40] inference qwen25_3b Q8_0 template B: 200 prompts in 1.5 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:41] inference qwen25_3b Q2_K template B: 200 prompts in 0.9 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:41] inference qwen25_3b Q2_K template B: 200 prompts in 0.9 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:44] inference llama32_3b Q4_K_M template A: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:45] inference llama32_3b Q4_K_M template B: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:45] inference qwen25_3b Q4_K_M template A: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:46] inference qwen25_3b Q4_K_M template B: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:49] inference llama32_3b Q3_K_M template A: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:49] inference llama32_3b Q3_K_M template B: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:53] inference qwen25_3b Q3_K_M template A: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 03:54] inference qwen25_3b Q3_K_M template B: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 04:05] inference llama32_3b Q6_K template A: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 04:06] inference llama32_3b Q6_K template B: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 04:06] inference qwen25_3b Q6_K template A: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 04:07] inference qwen25_3b Q6_K template B: 200 prompts in 0.4 min, mode=logits, NaN p_yes=0 (0.0%)
- [2026-10-29 04:07] ALL inference complete: 20 runs, 4000 prompts, 0 NaN. Analysis + figures generated.
- [2026-10-29 04:09] Paper tables generated (scripts/06_paper_tables.py). Writing paper.
- [2026-10-29 04:12] Paper written (paper/main.tex, main.md, references.bib), figures copied to paper/figures. Final cross-check done: scripts/check_numbers.py ALL PASS (18 headline numbers vs results/tables), scripts/check_tex.py OK. Re-running inference skips all 20 existing runs (resumable). Models kept in models/ (22 GB). Total inference time ~15 min; wall-clock dominated by ~1 MB/s model downloads (several hours).

## For the user when you return

**What finished:** everything in the checklist: environment (Python 3.12 venv, CUDA llama-cpp-python), all 20 runs (2 models x 5 levels x 2 templates, 4,000 prompts, 0 NaN), analysis tables (CSV + LaTeX), 5 figures, paper (IEEEtran .tex + .md + .bib), README, DECISIONS, requirements.
**What didn't / caveats:** main.tex was **not compiled** (no LaTeX on this machine), so page count (target 5-6) is unchecked; Llama files come from unsloth (bartowski lacks Q2_K/Q3_K_M), Qwen from bartowski.

**Three main findings**
1. Demographic gaps in P(Yes) stay tiny (|gender| <= 0.020, |race| <= 0.036) with no trend toward larger gaps at lower precision; the only Holm-significant gaps are small gender gaps (Llama, women slightly favoured, already present at Q8_0; Qwen at Q2_K); no race gap is significant.
2. Individual decisions change a lot: up to 26.5% flipped vs Q8_0 (Qwen Q3_K_M, almost all Yes->No; selection rate 32.5% -> 6.0%), Llama drifts toward Yes; flips are balanced across name groups where testable.
3. Qualification ranking survives (AUC 1.0 everywhere), but calibration shifts (non-monotone for Qwen) and effects depend on prompt wording (gender-gap correlation between templates r = 0.52, p = 0.13).

**Three things to check/improve before submitting**
1. Compile main.tex (e.g. Overleaf), fix any LaTeX errors and trim/extend to the target page count (3 wide figures + 4 tables may exceed 6 pages; fig4/fig5 are not in the paper); verify every citation's details and the related-work descriptions (written from titles only) against the actual papers.
2. Decide how to handle the weaknesses reviewers will raise: AUC saturates (resumes too easy), only 50 resumes/two 3B models, Llama and Qwen quantized files come from different repos (recipes unverified), reference is Q8_0 not FP16; consider harder borderline resumes, more resumes, and an FP16 reference.
3. Pick a target venue and add the author email, and consider re-framing the abstract around the "stable averages, unstable decisions" finding.
