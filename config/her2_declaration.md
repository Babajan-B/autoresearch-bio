# HER2 benchmark: locked decisions (written before any HER2 model was trained)

Declared: 2026-10-10T00:10:45Z

- Cohort: GSE96058 primary tumors (titles ending "repl" excluded), clinical "her2 status" recorded as 0 or 1.
- Partition: one stratified 60/20/20 split on HER2 status, seed 42, created once by `prepare`.
- Audit log: reports/her2/results_log_her2.json (separate from the ER log).
- Baseline: logistic regression, k = 1,000 genes (required first).
- Training runs (same configurations as the ER benchmark): random forest k = 2,000; histogram gradient boosting k = 2,000;
  Extra Trees k = 2,000 with seeds 7, 42, 123; Extra Trees seed 42 with k = 500, 1,000, 5,000.
- Promotion rule: unchanged (validation AUROC; ties Macro-F1, Brier, MCC). Test metrics never read by the rule.
- Declared ablation of the baseline: ERBB2, GRB7, STARD3 (core genes of the 17q12 HER2 amplicon).
