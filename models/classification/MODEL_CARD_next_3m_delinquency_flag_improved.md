# Model Card: next_3m_delinquency_flag_improved

## Objective
Predict next_3m_delinquency_flag for loan performance monitoring

## Training Data
- **Source**: Synthetic loan panel data
- **Size**: ~40,000 training rows, 146 features
- **Time Range**: 2020-01 to 2023-12
- **Split Method**: Time-aware split by reporting_month
  - Train: months ≤ 24
  - Validation: months 25–30
  - Test: months ≥ 31

## Features Used
- interest_rate: 429.0000
- current_balance: 218.0000
- original_balance: 199.0000
- dti_band_encoded: 195.0000
- balance_ratio_rollmax_12: 194.0000
- balance_3m_change: 186.0000
- current_balance_rollstd_3: 167.0000
- current_balance_lag6: 145.0000
- balance_3m_pct_change: 143.0000
- month_of_year: 137.0000
- current_balance_rollstd_12: 136.0000
- balance_ratio_rollstd_3: 135.0000
- current_balance_rollstd_6: 131.0000
- balance_ratio_rollmax_6: 128.0000
- balance_ratio_rollstd_6: 116.0000
- current_balance_rollmax_12: 110.0000
- current_balance_lag1: 106.0000
- current_balance_lag2: 100.0000
- balance_ratio_lag6: 100.0000
- servicer_current_balance: 99.0000

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
| Train   | 0.9353 | 0.7156 | 0.4804 | N/A | 0.0808 | 0.4804 |
| Val     | 0.7423 | 0.4542 | 0.4626 | N/A | 0.0905 | 0.4626 |
| Test    | 0.7154 | 0.4545 | 0.4655 | N/A | 0.0910 | 0.4655 |
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