# Model Card: next_6m_delinquency_flag_improved

## Objective
Predict next_6m_delinquency_flag for loan performance monitoring

## Training Data
- **Source**: Synthetic loan panel data
- **Size**: see training logs samples, 147 features
- **Time Range**: 2020-01 to 2023-12
- **Split Method**: Time-aware split by reporting_month
  - Train: months ≤ 24
  - Validation: months 25–30
  - Test: months ≥ 31

## Features Used
- interest_rate: 641.0000
- original_balance: 333.0000
- balance_ratio_rollmax_12: 269.0000
- current_balance: 243.0000
- dti_band_encoded: 229.0000
- balance_3m_change: 191.0000
- current_balance_rollstd_3: 180.0000
- balance_3m_pct_change: 142.0000
- current_balance_lag6: 134.0000
- balance_ratio_rollstd_3: 130.0000
- current_balance_rollmean_12: 123.0000
- data_quality_flag: 117.0000
- current_balance_lag3: 114.0000
- balance_ratio_rollmax_6: 111.0000
- current_balance_lag2: 107.0000
- balance_ratio_rollstd_6: 106.0000
- current_balance_rollmax_12: 103.0000
- current_balance_lag1: 94.0000
- current_balance_rollstd_6: 92.0000
- current_status_encoded: 85.0000

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
| Train   | 0.9103 | 0.7150 | 0.5050 | 0.2397 | 0.1027 | 0.5050 |
| Val     | 0.7326 | 0.4728 | 0.4060 | 0.0233 | 0.1305 | 0.4060 |
| Test    | 0.6990 | 0.4440 | 0.4243 | 0.0995 | 0.1289 | 0.4243 |

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