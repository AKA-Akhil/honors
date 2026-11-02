# Holm-Bonferroni families

- **selection-shift McNemar tests**: Holm-Bonferroni on `sel_shift_mcnemar_p_raw` within each combination of ['template', 'recipe'].
- **demographic-gap tests (gender and race pooled)**: Holm-Bonferroni on `gap_p_raw` within each combination of ['template', 'recipe_family'].
- **change-in-gap-vs-F16 tests (gender and race pooled)**: Holm-Bonferroni on `chg_p_raw` within each combination of ['template', 'recipe_family'].
- **flip tests: mcnemar_p_raw**: Holm-Bonferroni on `mcnemar_p_raw` within each combination of ['template', 'recipe_family'].
- **flip tests: group_test_all_p_raw**: Holm-Bonferroni on `group_test_all_p_raw` within each combination of ['template', 'recipe_family'].
- **flip tests: group_test_yn_p_raw**: Holm-Bonferroni on `group_test_yn_p_raw` within each combination of ['template', 'recipe_family'].
- **flip tests: group_test_ny_p_raw**: Holm-Bonferroni on `group_test_ny_p_raw` within each combination of ['template', 'recipe_family'].
- **uncertainty-logit abs_dev coefficients**: Holm-Bonferroni on `p_raw` within each combination of ['template'].
- **imat-vs-noimat decision McNemar tests**: Holm-Bonferroni on `decision_mcnemar_p_raw` within each combination of ['model'].
- **imatrix ablation Wilcoxon tests**: Holm across all (model, level) rows for each metric.
- **mixed-model interaction coefficients / joint Wald tests**: Holm-Bonferroni on `p_raw` within each combination of ['is_joint', 'kind'].
