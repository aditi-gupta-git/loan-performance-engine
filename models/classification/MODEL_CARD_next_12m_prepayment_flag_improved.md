# Model Card: next_12m_prepayment_flag_improved

## Objective
Predict next_12m_prepayment_flag for loan performance monitoring

## Training Data
- **Source**: Synthetic loan panel data
- **Size**: ~40,000 training rows, 146 features
- **Time Range**: 2020-01 to 2023-12
- **Split Method**: Time-aware split by reporting_month
  - Train: months ≤ 24
  - Validation: months 25–30
  - Test: months ≥ 31

## Features Used
- interest_rate: 709.0000
- original_balance: 565.0000
- balance_ratio_rollmax_12: 301.0000
- dti_band_encoded: 246.0000
- current_balance: 245.0000
- current_balance_rollmax_12: 162.0000
- balance_3m_change: 156.0000
- current_balance_rollstd_3: 137.0000
- current_balance_rollmean_12: 122.0000
- balance_3m_pct_change: 109.0000
- balance_ratio_rollstd_3: 107.0000
- balance_ratio_rollmax_6: 97.0000
- current_balance_lag6: 95.0000
- current_balance_lag1: 90.0000
- current_balance_lag3: 89.0000
- current_balance_rollstd_6: 86.0000
- servicer_name_Servicer_D: 84.0000
- loan_purpose_Refinance: 83.0000
- property_type_SFR: 77.0000
- days_past_due_rollmax_12: 77.0000

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
| Train   | 0.9497 | 0.7818 | 0.6091 | N/A | 0.0635 | 0.6091 |
| Val     | 0.7704 | 0.4416 | 0.2872 | N/A | 0.1157 | 0.2872 |
| Test    | 0.6444 | 0.2569 | 0.1292 | N/A | 0.1321 | 0.1292 |
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