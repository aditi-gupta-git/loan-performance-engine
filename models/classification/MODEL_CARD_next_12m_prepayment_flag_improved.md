# Model Card: next_12m_prepayment_flag_improved

## Objective
Predict next_12m_prepayment_flag for loan performance monitoring

## Training Data
- **Source**: Synthetic loan panel data
- **Size**: see training logs samples, 147 features
- **Time Range**: 2020-01 to 2023-12
- **Split Method**: Time-aware split by reporting_month
  - Train: months ≤ 24
  - Validation: months 25–30
  - Test: months ≥ 31

## Features Used
- interest_rate: 781.0000
- original_balance: 492.0000
- balance_ratio_rollmax_12: 288.0000
- dti_band_encoded: 285.0000
- current_balance: 245.0000
- balance_3m_change: 157.0000
- current_balance_rollmax_12: 148.0000
- current_balance_rollstd_3: 141.0000
- current_balance_lag6: 113.0000
- current_balance_rollmean_12: 110.0000
- days_past_due_rollmax_12: 106.0000
- current_status_encoded: 102.0000
- balance_3m_pct_change: 96.0000
- balance_ratio_rollstd_3: 93.0000
- current_balance_rollstd_6: 87.0000
- current_balance_rollmean_3: 85.0000
- current_balance_lag2: 83.0000
- current_balance_lag1: 82.0000
- loan_purpose_Purchase: 81.0000
- balance_ratio_rollstd_6: 80.0000

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
| Dataset | ROC-AUC | PR-AUC | F1 | Recall@Precision=0.8 | Brier Score | Macro-F1 |
|---------|---------|--------|-----|---------------------|-------------|----------|
| Train   | 0.9420 | 0.7339 | 0.5817 | 0.3951 | 0.0680 | 0.5817 |
| Val     | 0.7691 | 0.4305 | 0.2830 | 0.0498 | 0.1151 | 0.2830 |
| Test    | 0.6363 | 0.2259 | 0.1501 | 0.0000 | 0.1207 | 0.1501 |

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