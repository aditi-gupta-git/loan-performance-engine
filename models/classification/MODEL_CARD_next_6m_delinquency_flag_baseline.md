# Model Card: next_6m_delinquency_flag_baseline

## Objective
Predict next_6m_delinquency_flag for loan performance monitoring

## Training Data
- **Source**: Synthetic loan panel data
- **Size**: ~40,000 training rows, 146 features
- **Time Range**: 2020-01 to 2023-12
- **Split Method**: Time-aware split by reporting_month
  - Train: months ≤ 24
  - Validation: months 25–30
  - Test: months ≥ 31

## Features Used
- remaining_term_months: 0.0013
- days_past_due: 0.0005
- days_past_due_rollmax_3: 0.0003
- days_past_due_lag1: 0.0003
- days_past_due_rollmean_3: 0.0002
- days_past_due_rollmax_6: 0.0002
- days_past_due_lag2: 0.0002
- servicer_days_past_due: 0.0001
- days_past_due_rollmean_6: 0.0001
- days_past_due_rollstd_3: 0.0001
- months_since_dq: 0.0001
- days_past_due_lag3: 0.0001
- days_past_due_rollmean_12: 0.0001
- days_past_due_rollmax_12: 0.0001
- days_past_due_rollstd_6: 0.0001
- log_dpd: 0.0000
- days_past_due_lag6: 0.0000
- loan_age_months: 0.0000
- days_past_due_rollstd_12: 0.0000
- dq_streak: 0.0000

## Model Type & Hyperparameters
- **Algorithm**: logistic_regression
- **Hyperparameters**: {
  "class_weight": "balanced",
  "max_iter": 1000
}
- **Class Imbalance Handling**: class_weight=balanced
- **Calibration**: isotonic

## Validation Method
Time-aware split ensuring no loan_id appears in both train and validation sets inappropriately. Future months strictly held out for validation/testing.

## Metrics
| Dataset | ROC-AUC | PR-AUC | F1 | Recall@P80 | Brier | Macro-F1 |
|---------|---------|--------|-----|------------|-------|----------|
| Train   | 0.5832 | 0.2641 | 0.0007 | N/A | 0.1652 | 0.0007 |
| Val     | 0.5886 | 0.2385 | 0.0000 | N/A | 0.1538 | 0.0000 |
| Test    | 0.5804 | 0.2321 | 0.0000 | N/A | 0.1531 | 0.0000 |
## Calibration Approach
Calibrated using isotonic on training data

## Known Limitations & Failure Modes
Trained on synthetic data; may not generalize to real portfolio

## Leakage Controls Applied
- No target-derived features in training set
- Time-aware split by reporting_month
- Group-aware split ensuring loan_id doesn't straddle train/val
- Feature engineering uses only lagged/rolling features available at prediction time

## Fairness/Bias Notes
No protected attributes used; geographic features may proxy for demographics

## Intended Use
Portfolio risk monitoring and reviewer prioritization

## Out-of-Scope Use
Individual loan approval/denial decisions

## Model Version
- **Version**: 1.0
- **Training Date**: 2026-09-01
- **Git Commit**: unknown