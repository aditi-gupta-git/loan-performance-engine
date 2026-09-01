# Model Card: next_12m_default_flag_improved

## Objective
Predict next_12m_default_flag for loan performance monitoring

## Training Data
- **Source**: Synthetic loan panel data
- **Size**: ~40,000 training rows, 146 features
- **Time Range**: 2020-01 to 2023-12
- **Split Method**: Time-aware split by reporting_month
  - Train: months ≤ 24
  - Validation: months 25–30
  - Test: months ≥ 31

## Features Used
- interest_rate: 758.0000
- original_balance: 497.0000
- dti_band_encoded: 338.0000
- current_balance: 318.0000
- balance_ratio_rollmax_12: 231.0000
- balance_3m_change: 148.0000
- current_balance_rollmax_12: 146.0000
- credit_score_band_encoded: 109.0000
- balance_ratio_rollmax_6: 105.0000
- current_balance_rollstd_3: 101.0000
- current_balance_lag1: 92.0000
- current_balance_rollmean_12: 91.0000
- balance_3m_pct_change: 89.0000
- servicer_name_Servicer_D: 88.0000
- remaining_term_months: 87.0000
- days_past_due_rollmax_12: 87.0000
- current_balance_lag3: 85.0000
- current_status_encoded: 82.0000
- current_balance_lag6: 80.0000
- current_balance_lag2: 77.0000

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
| Train   | 0.9532 | 0.7503 | 0.5917 | N/A | 0.0600 | 0.5917 |
| Val     | 0.8538 | 0.5608 | 0.4829 | N/A | 0.0756 | 0.4829 |
| Test    | 0.8025 | 0.4701 | 0.4464 | N/A | 0.0808 | 0.4464 |
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