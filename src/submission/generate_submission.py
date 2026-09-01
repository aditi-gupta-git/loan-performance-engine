"""
Submission Generator
====================
Produces submission.csv in the required format:
  loan_id, reporting_month, month_index, next_3m_delinquency_prob,
  next_6m_delinquency_prob, next_12m_default_prob, next_12m_prepayment_prob,
  next_state_pred, next_state_prob, exception_type, anomaly_score,
  top_drivers, recommended_action, confidence

Usage (CLI):
    python -m src.submission.generate_submission \
        --test-data data/loan_monthly_performance_test.csv \
        --output submission.csv
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "loan_id", "reporting_month", "month_index",
    "next_3m_delinquency_prob", "next_6m_delinquency_prob",
    "next_12m_default_prob", "next_12m_prepayment_prob",
    "next_state_pred", "next_state_prob",
    "exception_type", "anomaly_score", "top_drivers",
    "recommended_action", "confidence",
]


def build_submission(
    test_df: pd.DataFrame,
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    anomaly_result: Any,
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """
    Build submission DataFrame and save to CSV.

    Parameters
    ----------
    test_df      : raw test panel DataFrame (for loan_id, reporting_month, month_index)
    models       : dict mapping target name → fitted sklearn model
    X_test       : feature matrix aligned with test_df rows
    anomaly_result : AnomalyResult object with .scores, .flags, .drivers, .exception_predictions
    output_path  : where to save the CSV

    Returns
    -------
    submission DataFrame
    """
    targets = [
        "next_3m_delinquency_flag",
        "next_6m_delinquency_flag",
        "next_12m_default_flag",
        "next_12m_prepayment_flag",
        "next_state",
    ]

    # Build probability arrays
    probas: Dict[str, np.ndarray] = {}
    for t in targets:
        if t in models:
            probas[t] = models[t].predict_proba(X_test)
        else:
            logger.warning(f"Model for {t} not found — filling with zeros.")
            probas[t] = np.zeros((len(test_df), 2))

    # next_state class labels
    ns_classes = getattr(models.get("next_state"), "classes_", None)
    exc_types = ["data_quality", "servicer_conflict", "stale_record", "document_gap", "balance_anomaly"]

    rows = []
    for idx, (_, row) in enumerate(test_df.iterrows()):
        # --- probabilities ---
        p3  = float(probas["next_3m_delinquency_flag"][idx, 1]) if probas["next_3m_delinquency_flag"].shape[1] > 1 else 0.0
        p6  = float(probas["next_6m_delinquency_flag"][idx, 1]) if probas["next_6m_delinquency_flag"].shape[1] > 1 else 0.0
        pd_ = float(probas["next_12m_default_flag"][idx, 1]) if probas["next_12m_default_flag"].shape[1] > 1 else 0.0
        pp  = float(probas["next_12m_prepayment_flag"][idx, 1]) if probas["next_12m_prepayment_flag"].shape[1] > 1 else 0.0

        ns_proba  = probas["next_state"][idx]
        ns_idx    = int(np.argmax(ns_proba))
        ns_pred   = str(ns_classes[ns_idx]) if ns_classes is not None else str(ns_idx)
        ns_conf   = float(ns_proba[ns_idx])

        # --- anomaly ---
        a_score = float(anomaly_result.scores[idx]) if idx < len(anomaly_result.scores) else 0.0
        a_flag  = int(anomaly_result.flags[idx]) if idx < len(anomaly_result.flags) else 0

        # --- drivers ---
        drivers = anomaly_result.drivers[idx] if idx < len(anomaly_result.drivers) else []
        if drivers:
            driver_str = "; ".join([f"{d['feature']}:{d['contribution']:.3f}" for d in drivers[:3]])
        else:
            dpd    = float(row.get("days_past_due", 0))
            status = str(row.get("current_status", "Current"))
            mod    = int(row.get("modification_flag", 0))
            parts: List[str] = []
            if dpd > 0:      parts.append(f"days_past_due:{dpd:.0f}")
            if status != "Current": parts.append(f"current_status:{status}")
            if mod:          parts.append("modification_flag:1")
            if a_score > 0.1: parts.append(f"anomaly_score:{a_score:.3f}")
            if not parts:
                br = float(row.get("current_balance", 0)) / max(float(row.get("original_balance", 1)), 1)
                parts.append(f"balance_ratio:{br:.3f}")
                parts.append(f"loan_age:{int(row.get('loan_age_months', 0))}m")
            driver_str = "; ".join(parts[:3])

        # --- exception type ---
        exc_pred = int(anomaly_result.exception_predictions[idx]) if (
            anomaly_result.exception_predictions is not None and idx < len(anomaly_result.exception_predictions)
        ) else 0
        exc_type = exc_types[exc_pred % len(exc_types)]

        # --- action ---
        if a_flag:
            action = "Review - anomaly detected"
        elif pd_ > 0.5:
            action = "Review - high default risk"
        elif p3 > 0.5:
            action = "Monitor - delinquency risk"
        else:
            action = "No action - within normal range"

        rows.append({
            "loan_id":                   row["loan_id"],
            "reporting_month":           str(row["reporting_month"]),
            "month_index":               int(row["month_index"]),
            "next_3m_delinquency_prob":  round(p3,  6),
            "next_6m_delinquency_prob":  round(p6,  6),
            "next_12m_default_prob":     round(pd_, 6),
            "next_12m_prepayment_prob":  round(pp,  6),
            "next_state_pred":           ns_pred,
            "next_state_prob":           round(ns_conf, 6),
            "exception_type":            exc_type,
            "anomaly_score":             round(a_score, 6),
            "top_drivers":               driver_str,
            "recommended_action":        action,
            "confidence":                round(ns_conf, 6),
        })

    submission = pd.DataFrame(rows)

    # Validate columns
    missing = set(REQUIRED_COLUMNS) - set(submission.columns)
    if missing:
        raise ValueError(f"Submission missing required columns: {missing}")

    submission.to_csv(output_path, index=False)
    logger.info(f"Saved submission.csv → {output_path} ({len(submission):,} rows)")
    return submission


def validate_submission(path: str) -> bool:
    """Validate that submission.csv has correct columns and no nulls in key fields."""
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        print(f"❌ Missing columns: {missing}")
        return False
    nulls = df[REQUIRED_COLUMNS].isnull().sum()
    if nulls.any():
        print(f"❌ Null values found:\n{nulls[nulls > 0]}")
        return False
    print(f"✅ submission.csv valid — {len(df):,} rows, {len(df.columns)} columns")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate submission.csv")
    parser.add_argument("--path", default="submission.csv")
    args = parser.parse_args()
    validate_submission(args.path)
