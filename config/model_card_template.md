# Model Card: {model_name}

## Objective
{objective}

## Training Data
- **Source**: {data_source}
- **Size**: {n_samples} samples, {n_features} features
- **Time Range**: {start_date} to {end_date}
- **Split Method**: Time-aware split by reporting_month
  - Train: months ≤ {train_end_month}
  - Validation: months {val_start_month}–{val_end_month}
  - Test: months ≥ {test_start_month}

## Features Used
{feature_list}

## Model Type & Hyperparameters
- **Algorithm**: {algorithm}
- **Hyperparameters**: {hyperparameters}
- **Class Imbalance Handling**: {imbalance_method}
- **Calibration**: {calibration_method}

## Validation Method
Time-aware split ensuring no loan_id appears in both train and validation sets inappropriately. Future months strictly held out for validation/testing.

## Metrics
| Dataset | ROC-AUC | PR-AUC | F1 | Recall@Precision=0.8 | Brier Score | Macro-F1 |
|---------|---------|--------|-----|---------------------|-------------|----------|
| Train   | {train_roc_auc} | {train_pr_auc} | {train_f1} | {train_recall_at_prec} | {train_brier} | {train_macro_f1} |
| Val     | {val_roc_auc} | {val_pr_auc} | {val_f1} | {val_recall_at_prec} | {val_brier} | {val_macro_f1} |
| Test    | {test_roc_auc} | {test_pr_auc} | {test_f1} | {test_recall_at_prec} | {test_brier} | {test_macro_f1} |

## Calibration Approach
{calibration_details}

## Known Limitations & Failure Modes
{limitations}

## Leakage Controls Applied
- No target-derived features in training set
- Time-aware split by reporting_month
- Group-aware split ensuring loan_id doesn't straddle train/val
- Feature engineering uses only lagged/rolling features available at prediction time

## Fairness/Bias Notes
{fairness_notes}

## Intended Use
{intended_use}

## Out-of-Scope Use
{out_of_scope_use}

## Model Version
- **Version**: {version}
- **Training Date**: {training_date}
- **Git Commit**: {git_commit}