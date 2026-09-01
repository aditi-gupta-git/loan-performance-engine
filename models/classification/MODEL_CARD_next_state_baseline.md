# Model Card: next_state_baseline

## Objective
Predict next_state for loan performance monitoring

## Training Data
- **Source**: Synthetic loan panel data
- **Size**: see training logs samples, 147 features
- **Time Range**: 2020-01 to 2023-12
- **Split Method**: Time-aware split by reporting_month
  - Train: months ≤ 24
  - Validation: months 25–30
  - Test: months ≥ 31

## Features Used
- current_balance_rollmean_12: 0.0000
- current_balance_rollmean_6: 0.0000
- current_balance_rollmax_3: 0.0000
- current_balance_rollmean_3: 0.0000
- days_past_due: 0.0000
- days_past_due_rollmax_12: 0.0000
- current_balance_rollstd_3: 0.0000
- days_past_due_rollmax_6: 0.0000
- days_past_due_rollmax_3: 0.0000
- days_past_due_lag1: 0.0000
- days_past_due_rollmean_3: 0.0000
- days_past_due_lag2: 0.0000
- current_balance_rollmax_6: 0.0000
- servicer_days_past_due: 0.0000
- days_past_due_rollmean_6: 0.0000
- current_balance: 0.0000
- current_balance_rollmax_12: 0.0000
- days_past_due_lag3: 0.0000
- days_past_due_rollmean_12: 0.0000
- days_past_due_rollstd_12: 0.0000

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
| Dataset | ROC-AUC | PR-AUC | F1 | Recall@Precision=0.8 | Brier Score | Macro-F1 |
|---------|---------|--------|-----|---------------------|-------------|----------|
| Train   | 0.6867 | 0.2214 | 0.1495 | 0.1452 | 0.2376 | 0.1495 |
| Val     | 0.6685 | 0.2151 | 0.1485 | 0.1429 | 0.2370 | 0.1485 |
| Test    | 0.6940 | 0.2215 | 0.1508 | 0.1456 | 0.2375 | 0.1508 |

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