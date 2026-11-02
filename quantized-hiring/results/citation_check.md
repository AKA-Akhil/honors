# Citation check (round 2)

**Method.** Each reference was looked up on arXiv, the ACL Anthology, the AEA, the MLSys/PMLR proceedings or GitHub by two web-fetch helpers; I compared title, authors, venue and abstract with the details in the original spec (Section 11) / `paper/references.bib`. Summaries below are my own words based on the fetched abstracts only (not full texts). Web access was available. Items marked **FIX** need correcting in the paper (the paper was not touched this round).

## Corrections needed (summary)

| # | Reference | Problem | Fix |
|---|---|---|---|
| 2 | Hua, Lotfi, Chen | **Wrong title, first-author name and venue in round 1.** arXiv 2602.06181 is titled "Investigating Social Bias Changes in Quantized Language Models"; authors Stanley Z. Hua, Sanae Lotfi, Irene Y. Chen; the page says "Accepted to COLM 2026", **not ICLR**. | Use the actual title/authors/venue (COLM 2026). Do not call it "Uncertainty Drives ...". |
| 4 | Rath, Maliakkal | Authors are Plawan Kumar Rath and Rahul Maliakkal; accepted at **IEEE Cloud Summit 2026** (DOI 10.1109/CloudSummit68932.2026.00016); arXiv 2605.15208. | Add venue; full first names optional. |
| 1 | Marcuzzi et al. | arXiv page (2508.18088) shows no venue; "EACL 2026 (Long Papers), pp. 363-404" given in the spec could **not** be confirmed from the pages read. | Verify venue/pages on the EACL 2026 proceedings before keeping them; otherwise cite as arXiv. |
| 3 | Al Hakim et al. | Details match, but the abstract does **not** say hiring is future work. The paper's claim "explicitly lists hiring-decision scenarios as future work" (round-1 intro) is **unverified**. | Check the full text (conclusion/limitations); otherwise drop or soften the claim. |
| 9 | GPTQ | Page lists ICLR 2023. | Optionally cite as ICLR 2023. |
| 11 | LLM.int8() | Page lists NeurIPS 2022. | Optionally cite as NeurIPS 2022. |
| 12 | Kurt | Single author (Uygar Kurt). Spec's "U. Kurt" is fine. | none |

All other details (authors, titles, venues, pages) matched.

## Per-reference notes

**[1] Marcuzzi, Ning, Schwartz, Gurevych — How Quantization Shapes Bias in Large Language Models (arXiv 2508.18088).** *Details:* title/authors correct; venue not confirmed (see above). *What it studied:* a broad evaluation of how weight and activation quantization changes bias, across 13 benchmarks (stereotypes, fairness, toxicity, sentiment), probability- and generation-based metrics, several architectures and reasoning abilities. *Finding:* the effect is nuanced: toxicity can fall, sentiment barely moves, but stereotypes and unfairness in generative tasks tend to rise slightly under aggressive compression, similarly across demographic subgroups. *Relation:* **contrasts/supports** (benchmark-based, not decisions; our results on hiring decisions show small gap changes but large decision instability).

**[2] Hua, Lotfi, Chen — Investigating Social Bias Changes in Quantized Language Models (arXiv 2602.06181, COLM 2026).** *Studied:* 50 quantized models on a 13-dataset bias benchmark. *Found:* "bias flipping": individual responses change between biased/unbiased up to 21% of the time even when aggregate bias is unchanged; high-uncertainty responses are 3-11x more likely to flip; 4-bit models change 4-6x more than 8-bit; group-level effects are asymmetric and hidden by aggregates. *Relation:* **closest prior work and direct motivation** for our flip and uncertainty analyses (Sections 8.5); our design tests whether the same holds for hiring decisions.

**[3] Al Hakim, Wicaksono, Koto — Preserving Fairness and Safety in Quantized LLMs Through Critical Weight Protection (arXiv 2601.12033).** *Studied:* static vs dynamic quantization effects on fairness (English, French, Dutch, Spanish, Turkish) and safety (English, Korean, Arabic), then proposes Critical Weight Protection. *Found:* quantization consistently degrades fairness and safety, dynamic methods more stable; degradation varies by language; the proposed method mitigates it. *Relation:* **contrasts** (benchmarks, other quantizers, multilingual). The "hiring as future work" claim is not in the abstract: unverified.

**[4] Rath, Maliakkal — Quantization Undoes Alignment (arXiv 2605.15208, IEEE Cloud Summit 2026).** *Studied:* Qwen2.5-7B, Mistral-7B, Phi-3.5-mini at five precision levels (BF16 to 3-bit) on 12,148 BBQ items, 5 seeds. *Found:* 3-bit quantization makes 6-21% of previously unbiased items stereotypical, a dose-response pattern; "unknown" answers drop 17.4%; perplexity barely moves while item-level bias changes already appear at 4-bit. *Relation:* **supports/contrasts**: same message (aggregate quality metrics miss item-level change), different task (BBQ), and it also uses Phi-3.5-mini. Our perplexity-vs-drift analysis tests the same point for hiring decisions.

**[5] Gonçalves, Strubell — Understanding the Effect of Model Compression on Social Bias in LLMs (EMNLP 2023, pp. 2663-2675).** *Studied:* quantization and knowledge distillation of pretraining-stage models and effects on social-bias measures. *Found:* longer pretraining and larger models gave higher bias; quantization acted like a regularizer with a best trade-off around 20% of original pretraining time. *Relation:* **background**; earlier, smaller-scale, benchmark-style.

**[6] Ramesh, Chavan, Pandit, Sitaram — A Comparative Study on the Impact of Model Compression Techniques on Fairness in Language Models (ACL 2023, pp. 15762-15782).** *Studied:* pruned, distilled and quantized language models on intrinsic/extrinsic bias metrics in text classification, incl. multilingual models. *Found:* the effect depends on both the pretrained model and the compression strategy, and compression can hurt fairness. *Relation:* **background**.

**[7] Hong et al. — Decoding Compressed Trust (ICML 2024; 15 authors).** *Studied:* 3 LLMs x 5 compression methods x 8 trustworthiness dimensions. *Found:* quantization generally beats pruning for keeping trustworthiness; moderate bit-widths can even improve ethics/fairness dimensions, while very low (3-bit) quantization hurts; benign performance does not reveal the risk. *Relation:* **supports** (low-bit risks hidden from utility metrics) and **contrasts** (task is not hiring).

**[8] Kirsten, Habernal, Nanda, Zafar — The Impact of Inference Acceleration on Bias of LLMs (NAACL 2025, pp. 1834-1853).** *Studied:* demographic bias in generations before/after inference-acceleration (quantization, pruning, caching). *Found:* significant, complex and unpredictable bias changes that depend on the model and the acceleration/bias-type combination; case-by-case evaluation is needed. *Relation:* **supports** (model-specific, unpredictable effects) — it also mentions caching, relevant to our determinism check.

**[9] Frantar et al. — GPTQ (arXiv 2210.17323; ICLR 2023).** One-shot weight quantization using approximate second-order information, down to 3-4 bits (and 2-bit/ternary with some loss) for very large GPT models. *Relation:* **method source** (a quantizer family not used here).

**[10] Lin et al. — AWQ (MLSys 2024).** Weight-only low-bit quantization that protects the ~1% salient weights identified by activation statistics (via per-channel scaling), with an on-device inference framework. *Relation:* **method source**; note that its calibration-based idea parallels llama.cpp's importance matrix.

**[11] Dettmers et al. — LLM.int8() (NeurIPS 2022).** Int8 matrix multiplication for transformers using vector-wise quantization plus mixed-precision handling of emergent outlier features. *Relation:* **method source**.

**[12] Kurt — Which Quantization Should I Use? (arXiv 2601.14277).** Unified evaluation of llama.cpp K-quant and legacy formats (3-8 bit) on Llama-3.1-8B-Instruct: benchmarks, perplexity, CPU throughput, size. *Relation:* **supports** (same format family; no fairness analysis).

**[13] Bertrand, Mullainathan — Are Emily and Greg More Employable Than Lakisha and Jamal? (AER 94(4), 991-1013, 2004).** Field experiment sending fictitious resumes with randomly assigned White- or African-American-sounding names: White names got ~50% more callbacks, and callbacks responded more to resume quality for White names. *Relation:* **method source** for our name lists and counterfactual design (note the page capitalises "Than").

**[14] Parrish et al. — BBQ (Findings of ACL 2022, pp. 2086-2105).** Hand-built question-answering benchmark probing social biases along nine dimensions, in under-informative and informative contexts. *Relation:* **background** (benchmark used by Rath & Maliakkal-style studies).

**[15] Koh, Plata, Chai — BAD (arXiv 2305.10407).** Identifies and quantifies social bias in ChatGPT/OpenAI models in the candidate-screening setting. *Relation:* **supports** (LLM hiring bias; API models, no quantization).

**[16] llama.cpp (github.com/ggml-org/llama.cpp).** Public MIT-licensed repository, described as "LLM inference in C/C++". Round 2 used release tag **b11972** for quantization/imatrix/perplexity tools. *Relation:* **software/tool.**

## Re-check (final pre-paper checks, 2026-10-29)
- **[1] Marcuzzi et al.: now confirmed.** ACL Anthology 2026.eacl-long.17, EACL 2026 Long Papers, pp. 363-404 (the arXiv v2 revised Jan 2026; the abstract was updated from 9 to 13 benchmarks, so cite the EACL version's numbers).
- **[3] Al Hakim et al.: venue now confirmed** as Findings of ACL 2026 (2026.findings-acl.993, pp. 19831-19855). The "hiring" statement: a web-fetch helper reported one sentence in the Limitations ("We did not extend our analysis to complex use cases, such as the application of LLMs for assisting in hiring decisions"), but it came from an extracted summary, not verbatim page text, and there is no "Future Work" heading. **Still unverified verbatim**: write "mentions hiring decisions as an unexplored use case in its limitations" only after checking the PDF; do not say "lists hiring as future work".
- Items [2], [4], [9], [11] corrected in `results/references_verified.bib`. Still unverified: Hua et al. COLM 2026 page numbers (none published in what was read); Rath & Maliakkal pages.
