# Model Card: next_6m_delinquency_flag_improved

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
- interest_rate: 581.0000
- original_balance: 374.0000
- current_balance: 253.0000
- balance_ratio_rollmax_12: 229.0000
- dti_band_encoded: 204.0000
- balance_3m_change: 193.0000
- current_balance_lag6: 158.0000
- balance_3m_pct_change: 157.0000
- current_balance_rollstd_3: 154.0000
- current_balance_rollmax_12: 150.0000
- current_balance_lag3: 139.0000
- balance_ratio_rollmax_6: 135.0000
- current_balance_rollmean_12: 125.0000
- balance_ratio_rollstd_3: 115.0000
- current_balance_rollstd_6: 112.0000
- current_status_encoded: 93.0000
- current_balance_lag2: 90.0000
- current_balance_rollstd_12: 83.0000
- month_of_year: 75.0000
- remaining_term_months: 72.0000

## Model Type & Hyperparameters
- **Algorithm**: lightgbm
- **Hyperparameters**: {
  "objective": "binary",
  "metric": "binary_logloss",
  "boosting_type": "gbdt",
  "num_leaves": 63,
  "learning_rate": 0.05,
  "feature_fraction": 0.8,
  "bagging_fraction": 0.8,
  "bagging_freq": 5,
  "min_child_samples": 50,
  "class_weight": "balanced",
  "random_state": 42,
  "verbosity": -1,
  "n_jobs": -1
}
- **Class Imbalance Handling**: class_weight=balanced
- **Calibration**: isotonic

## Validation Method
Time-aware split ensuring no loan_id appears in both train and validation sets inappropriately. Future months strictly held out for validation/testing.

## Metrics
| Dataset | ROC-AUC | PR-AUC | F1 | Recall@P80 | Brier | Macro-F1 |
|---------|---------|--------|-----|------------|-------|----------|
| Train   | 0.9144 | 0.7292 | 0.5229 | N/A | 0.1021 | 0.5229 |
| Val     | 0.7456 | 0.4889 | 0.4159 | N/A | 0.1285 | 0.4159 |
| Test    | 0.6851 | 0.4420 | 0.4126 | N/A | 0.1348 | 0.4126 |
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