# Results for the manuscript

These are the outputs behind the LockLoop-Bio manuscript. The expression data are not included; download them from GEO (GSE96058, GSE81538). Model files (`*.pkl`) and plots are also left out, because the pipeline regenerates them.

| Folder | Contents |
|---|---|
| `audit_logs/audit_log_er.json` | The append-only audit log of the ER benchmark as the pipeline wrote it (11–12 April 2026). It also holds one earlier smoke-test entry (`local_expression`). The `git_commit_hash` field is empty for the GSE96058 runs because they were executed from a copy of the code outside a git working tree. |
| `audit_logs/audit_log_her2.json` | Audit log of the HER2 benchmark (10 October 2026). The decisions for it were declared beforehand in `config/her2_declaration.md`. |
| `splits/` | Frozen train/validation/test partitions (sample IDs) for GSE96058 ER, GSE96058 HER2, and the GSE81538 companion cohort. |
| `er/<run>/` | One folder per ER run, from the re-execution on the second computing environment that the manuscript reports: validation and test predictions per sample, metrics, the selected gene list and the SHAP ranking. |
| `er_external/<run>/` | Predictions and metrics of the frozen ER models applied to GSE81538. |
| `her2/<run>/` | The same files for every HER2 run, including the declared ablation (`ablation_17q12`). |

The statistics, tables and figures in the manuscript are recomputed from these per-sample predictions; no model is refitted.
