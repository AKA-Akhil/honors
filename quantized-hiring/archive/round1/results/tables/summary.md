# Summary of key results (auto-generated from results/tables)

Runs analysed: 20; prompts scored: 4000; NaN p_yes: 0.

## Capability (RQ3), Template A

- llama32_3b Q8_0: mean p_yes strong/borderline/weak = 0.985/0.380/0.000; Spearman=0.923; AUC=1.000
- llama32_3b Q6_K: mean p_yes strong/borderline/weak = 0.978/0.372/0.000; Spearman=0.924; AUC=1.000
- llama32_3b Q4_K_M: mean p_yes strong/borderline/weak = 0.983/0.430/0.001; Spearman=0.926; AUC=1.000
- llama32_3b Q3_K_M: mean p_yes strong/borderline/weak = 0.985/0.528/0.001; Spearman=0.912; AUC=1.000
- llama32_3b Q2_K: mean p_yes strong/borderline/weak = 0.964/0.503/0.010; Spearman=0.911; AUC=1.000
- qwen25_3b Q8_0: mean p_yes strong/borderline/weak = 0.839/0.062/0.000; Spearman=0.826; AUC=1.000
- qwen25_3b Q6_K: mean p_yes strong/borderline/weak = 0.827/0.062/0.000; Spearman=0.848; AUC=1.000
- qwen25_3b Q4_K_M: mean p_yes strong/borderline/weak = 0.677/0.064/0.000; Spearman=0.842; AUC=1.000
- qwen25_3b Q3_K_M: mean p_yes strong/borderline/weak = 0.172/0.003/0.000; Spearman=0.849; AUC=1.000
- qwen25_3b Q2_K: mean p_yes strong/borderline/weak = 0.522/0.064/0.000; Spearman=0.900; AUC=1.000

## Paired gaps (RQ1), Template A (positive gender gap = men favoured; positive race gap = White-named favoured)

- llama32_3b Q8_0: gender gap=-0.0108 [-0.0186,-0.0051] p_raw=6.59e-13 p_holm=2.64e-11; race gap=+0.0022 [-0.0005,+0.0055] p_raw=0.074 p_holm=1
- llama32_3b Q6_K: gender gap=-0.0126 [-0.0201,-0.0063] p_raw=3.86e-11 p_holm=1.51e-09; race gap=+0.0031 [+0.0002,+0.0067] p_raw=0.0141 p_holm=0.422
- llama32_3b Q4_K_M: gender gap=-0.0191 [-0.0415,-0.0054] p_raw=1.6e-09 p_holm=6.07e-08; race gap=-0.0013 [-0.0059,+0.0028] p_raw=0.323 p_holm=1
- llama32_3b Q3_K_M: gender gap=-0.0120 [-0.0234,-0.0039] p_raw=6.71e-08 p_holm=2.42e-06; race gap=-0.0045 [-0.0172,+0.0024] p_raw=0.426 p_holm=1
- llama32_3b Q2_K: gender gap=-0.0035 [-0.0133,+0.0037] p_raw=0.393 p_holm=1; race gap=+0.0024 [-0.0034,+0.0108] p_raw=0.362 p_holm=1
- qwen25_3b Q8_0: gender gap=+0.0041 [-0.0017,+0.0143] p_raw=0.249 p_holm=1; race gap=-0.0041 [-0.0121,+0.0005] p_raw=0.954 p_holm=1
- qwen25_3b Q6_K: gender gap=+0.0007 [-0.0011,+0.0026] p_raw=0.201 p_holm=1; race gap=-0.0029 [-0.0106,+0.0012] p_raw=0.534 p_holm=1
- qwen25_3b Q4_K_M: gender gap=+0.0198 [+0.0018,+0.0420] p_raw=0.138 p_holm=1; race gap=+0.0150 [-0.0055,+0.0359] p_raw=0.0756 p_holm=1
- qwen25_3b Q3_K_M: gender gap=-0.0173 [-0.0503,+0.0091] p_raw=0.194 p_holm=1; race gap=-0.0026 [-0.0218,+0.0161] p_raw=0.0141 p_holm=0.422
- qwen25_3b Q2_K: gender gap=-0.0203 [-0.0439,+0.0027] p_raw=0.000156 p_holm=0.00499; race gap=+0.0362 [+0.0011,+0.0770] p_raw=0.107 p_holm=1

## Flips vs Q8_0 (RQ2), threshold 0.5

- llama32_3b Q6_K TA: flip rate=0.000 (Y->N 0, N->Y 0); McNemar p_raw=nan; chi2 Y->N across groups p_raw=nan; mean|dp|=0.006
- llama32_3b Q6_K TB: flip rate=0.030 (Y->N 5, N->Y 1); McNemar p_raw=0.219; chi2 Y->N across groups p_raw=0.532; mean|dp|=0.024
- llama32_3b Q4_K_M TA: flip rate=0.010 (Y->N 0, N->Y 2); McNemar p_raw=0.5; chi2 Y->N across groups p_raw=nan; mean|dp|=0.020
- llama32_3b Q4_K_M TB: flip rate=0.025 (Y->N 1, N->Y 4); McNemar p_raw=0.375; chi2 Y->N across groups p_raw=nan; mean|dp|=0.020
- llama32_3b Q3_K_M TA: flip rate=0.055 (Y->N 0, N->Y 11); McNemar p_raw=0.000977; chi2 Y->N across groups p_raw=nan; mean|dp|=0.050
- llama32_3b Q3_K_M TB: flip rate=0.045 (Y->N 0, N->Y 9); McNemar p_raw=0.00391; chi2 Y->N across groups p_raw=nan; mean|dp|=0.043
- llama32_3b Q2_K TA: flip rate=0.065 (Y->N 1, N->Y 12); McNemar p_raw=0.00342; chi2 Y->N across groups p_raw=nan; mean|dp|=0.073
- llama32_3b Q2_K TB: flip rate=0.085 (Y->N 0, N->Y 17); McNemar p_raw=1.53e-05; chi2 Y->N across groups p_raw=nan; mean|dp|=0.095
- qwen25_3b Q6_K TA: flip rate=0.005 (Y->N 1, N->Y 0); McNemar p_raw=1; chi2 Y->N across groups p_raw=nan; mean|dp|=0.007
- qwen25_3b Q6_K TB: flip rate=0.015 (Y->N 0, N->Y 3); McNemar p_raw=0.25; chi2 Y->N across groups p_raw=nan; mean|dp|=0.019
- qwen25_3b Q4_K_M TA: flip rate=0.045 (Y->N 9, N->Y 0); McNemar p_raw=0.00391; chi2 Y->N across groups p_raw=0.55; mean|dp|=0.061
- qwen25_3b Q4_K_M TB: flip rate=0.150 (Y->N 0, N->Y 30); McNemar p_raw=1.86e-09; chi2 Y->N across groups p_raw=nan; mean|dp|=0.131
- qwen25_3b Q3_K_M TA: flip rate=0.265 (Y->N 53, N->Y 0); McNemar p_raw=2.22e-16; chi2 Y->N across groups p_raw=0.949; mean|dp|=0.259
- qwen25_3b Q3_K_M TB: flip rate=0.195 (Y->N 39, N->Y 0); McNemar p_raw=3.64e-12; chi2 Y->N across groups p_raw=0.826; mean|dp|=0.175
- qwen25_3b Q2_K TA: flip rate=0.140 (Y->N 26, N->Y 2); McNemar p_raw=3.03e-06; chi2 Y->N across groups p_raw=0.709; mean|dp|=0.136
- qwen25_3b Q2_K TB: flip rate=0.195 (Y->N 39, N->Y 0); McNemar p_raw=3.64e-12; chi2 Y->N across groups p_raw=0.922; mean|dp|=0.180

## Flips vs Q8_0, reference-calibrated threshold

- llama32_3b Q6_K TA: flip rate=0.015 (Y->N 2, N->Y 1); McNemar p_raw=1
- llama32_3b Q6_K TB: flip rate=0.030 (Y->N 4, N->Y 2); McNemar p_raw=0.688
- llama32_3b Q4_K_M TA: flip rate=0.030 (Y->N 0, N->Y 6); McNemar p_raw=0.0312
- llama32_3b Q4_K_M TB: flip rate=0.015 (Y->N 0, N->Y 3); McNemar p_raw=0.25
- llama32_3b Q3_K_M TA: flip rate=0.050 (Y->N 0, N->Y 10); McNemar p_raw=0.00195
- llama32_3b Q3_K_M TB: flip rate=0.025 (Y->N 0, N->Y 5); McNemar p_raw=0.0625
- llama32_3b Q2_K TA: flip rate=0.075 (Y->N 1, N->Y 14); McNemar p_raw=0.000977
- llama32_3b Q2_K TB: flip rate=0.080 (Y->N 0, N->Y 16); McNemar p_raw=3.05e-05
- qwen25_3b Q6_K TA: flip rate=0.000 (Y->N 0, N->Y 0); McNemar p_raw=nan
- qwen25_3b Q6_K TB: flip rate=0.010 (Y->N 2, N->Y 0); McNemar p_raw=0.5
- qwen25_3b Q4_K_M TA: flip rate=0.035 (Y->N 0, N->Y 7); McNemar p_raw=0.0156
- qwen25_3b Q4_K_M TB: flip rate=0.110 (Y->N 0, N->Y 22); McNemar p_raw=4.77e-07
- qwen25_3b Q3_K_M TA: flip rate=0.060 (Y->N 12, N->Y 0); McNemar p_raw=0.000488
- qwen25_3b Q3_K_M TB: flip rate=0.105 (Y->N 17, N->Y 4); McNemar p_raw=0.0072
- qwen25_3b Q2_K TA: flip rate=0.500 (Y->N 0, N->Y 100); McNemar p_raw=1.58e-30
- qwen25_3b Q2_K TB: flip rate=0.500 (Y->N 0, N->Y 100); McNemar p_raw=1.58e-30

## Template robustness (RQ4)

- gap=gender_gap, n_pairs=10, pearson_r=0.516, pearson_p=0.127, spearman_rho=0.479, spearman_p=0.162, sign_agree=8
- gap=race_gap, n_pairs=10, pearson_r=-0.128, pearson_p=0.725, spearman_rho=0.055, spearman_p=0.881, sign_agree=7
- gap=gender_significant_either, n_pairs=5, pearson_r=nan, pearson_p=nan, spearman_rho=nan, spearman_p=nan, sign_agree=5
- gap=race_significant_either, n_pairs=4, pearson_r=nan, pearson_p=nan, spearman_rho=nan, spearman_p=nan, sign_agree=3