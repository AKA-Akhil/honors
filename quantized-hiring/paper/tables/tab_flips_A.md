*Decision flips versus Q8_0 (Template A, threshold 0.5, 200 prompts per run). "--": undefined (no discordant pairs) or fewer than 4 Y->N flips for the group test.*

| Model | Level | Flip rate | Y->N / N->Y | p_McNemar^Holm | p_χ^2 (Y->N by group) | Mean \|Δ P\| |
|---|---|---|---|---|---|---|
| Llama | Q6_K | 0.000 | 0/0 | -- | -- | 0.006 |
| Llama | Q4_K_M | 0.010 | 0/2 | 1.000 | -- | 0.020 |
| Llama | Q3_K_M | 0.055 | 0/11 | 0.009 | -- | 0.050 |
| Llama | Q2_K | 0.065 | 1/12 | 0.027 | -- | 0.073 |
| Qwen | Q6_K | 0.005 | 1/0 | 1.000 | -- | 0.007 |
| Qwen | Q4_K_M | 0.045 | 9/0 | 0.027 | 0.550 | 0.061 |
| Qwen | Q3_K_M | 0.265 | 53/0 | <0.001 | 0.949 | 0.259 |
| Qwen | Q2_K | 0.140 | 26/2 | <0.001 | 0.709 | 0.136 |
