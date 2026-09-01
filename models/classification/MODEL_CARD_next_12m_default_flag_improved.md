# Model Card: next_12m_default_flag_improved

## Objective
Predict next_12m_default_flag for loan performance monitoring

## Training Data
- **Source**: Synthetic loan panel data
- **Size**: see training logs samples, 147 features
- **Time Range**: 2020-01 to 2023-12
- **Split Method**: Time-aware split by reporting_month
  - Train: months ≤ 24
  - Validation: months 25–30
  - Test: months ≥ 31

## Features Used
- interest_rate: 727.0000
- original_balance: 512.0000
- dti_band_encoded: 349.0000
- balance_ratio_rollmax_12: 288.0000
- current_balance: 240.0000
- data_quality_flag: 146.0000
- balance_3m_change: 135.0000
- current_balance_rollmax_12: 131.0000
- current_balance_lag6: 111.0000
- balance_3m_pct_change: 108.0000
- current_balance_rollstd_3: 105.0000
- credit_score_band_encoded: 101.0000
- current_balance_rollmean_12: 94.0000
- balance_ratio_rollstd_3: 93.0000
- days_past_due_rollmax_12: 88.0000
- balance_ratio_rollmax_6: 78.0000
- current_balance_rollstd_6: 76.0000
- current_balance_lag3: 76.0000
- servicer_name_Servicer_B: 75.0000
- servicer_name_Servicer_D: 72.0000

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
| Train   | 0.9569 | 0.7663 | 0.6053 | 0.3539 | 0.0582 | 0.6053 |
| Val     | 0.8593 | 0.5654 | 0.4564 | 0.2572 | 0.0738 | 0.4564 |
| Test    | 0.8083 | 0.4868 | 0.4944 | 0.1817 | 0.0676 | 0.4944 |

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