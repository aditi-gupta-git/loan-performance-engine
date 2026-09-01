# Model Card: next_state_improved

## Objective
Predict next_state for loan performance monitoring

## Training Data
- **Source**: Synthetic loan panel data
- **Size**: ~40,000 training rows, 146 features
- **Time Range**: 2020-01 to 2023-12
- **Split Method**: Time-aware split by reporting_month
  - Train: months ≤ 24
  - Validation: months 25–30
  - Test: months ≥ 31

## Features Used
- interest_rate: 2298.0000
- month_of_year: 1544.0000
- days_past_due: 1373.0000
- current_balance: 1299.0000
- balance_ratio_rollmax_12: 1241.0000
- original_balance: 1092.0000
- servicer_current_balance: 973.0000
- balance_ratio_rollmax_6: 966.0000
- dti_band_encoded: 964.0000
- current_balance_rollstd_12: 950.0000
- balance_ratio: 925.0000
- balance_ratio_lag6: 893.0000
- current_balance_rollstd_3: 890.0000
- balance_3m_change: 869.0000
- current_status_encoded: 864.0000
- balance_paid_pct: 858.0000
- balance_ratio_lag3: 836.0000
- current_balance_rollstd_6: 801.0000
- balance_ratio_rollstd_3: 776.0000
- balance_3m_pct_change: 757.0000

## Model Type & Hyperparameters
- **Algorithm**: lightgbm
- **Hyperparameters**: {
  "objective": "multiclass",
  "metric": "multi_logloss",
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
  "n_jobs": -1,
  "num_class": 7
}
- **Class Imbalance Handling**: class_weight=balanced
- **Calibration**: isotonic

## Validation Method
Time-aware split ensuring no loan_id appears in both train and validation sets inappropriately. Future months strictly held out for validation/testing.

## Metrics
| Dataset | ROC-AUC | PR-AUC | F1 | Recall@P80 | Brier | Macro-F1 |
|---------|---------|--------|-----|------------|-------|----------|
| Train   | 0.9904 | 0.9080 | 0.6877 | N/A | 0.1188 | 0.6877 |
| Val     | 0.8579 | 0.5126 | 0.5252 | N/A | 0.1463 | 0.5252 |
| Test    | 0.8549 | 0.5173 | 0.5330 | N/A | 0.1515 | 0.5330 |
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