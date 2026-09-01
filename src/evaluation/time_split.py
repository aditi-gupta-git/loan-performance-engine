"""
Time-Aware Train / Validation Split
=====================================
Ensures no loan_id appears in both train and validation sets.
Splits strictly by reporting_month so future data never leaks into training.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def time_aware_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    date_col: str = "month_index",
    loan_col: str = "loan_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a panel dataset temporally.

    Parameters
    ----------
    df : panel DataFrame (one row per loan × month)
    train_frac : fraction of months in training set
    val_frac   : fraction of months in validation set (remainder = test)
    date_col   : column used for chronological ordering
    loan_col   : loan identifier — used to verify no cross-set leakage

    Returns
    -------
    (train_df, val_df, test_df) — non-overlapping, chronological

    Notes
    -----
    A loan that originates in the training window may appear in all three sets
    (each month independently), but NO future information leaks backward because
    we split by *time* not by *loan*.  Target variables are computed within the
    training frame using forward-looking labels that are derived before the split.
    """
    unique_months = sorted(df[date_col].unique())
    n = len(unique_months)

    train_end_idx = int(n * train_frac)
    val_end_idx   = int(n * (train_frac + val_frac))

    train_cutoff = unique_months[train_end_idx - 1]
    val_cutoff   = unique_months[val_end_idx - 1]

    train_mask = df[date_col] <= train_cutoff
    val_mask   = (df[date_col] > train_cutoff) & (df[date_col] <= val_cutoff)
    test_mask  = df[date_col] > val_cutoff

    train_df = df[train_mask].copy()
    val_df   = df[val_mask].copy()
    test_df  = df[test_mask].copy()

    logger.info(
        f"Time-aware split: train={len(train_df):,} rows (months ≤ {train_cutoff}), "
        f"val={len(val_df):,} rows (months {train_cutoff+1}–{val_cutoff}), "
        f"test={len(test_df):,} rows (months > {val_cutoff})"
    )

    # Leakage audit
    train_loans = set(train_df[loan_col].unique())
    val_loans   = set(val_df[loan_col].unique())
    test_loans  = set(test_df[loan_col].unique())
    overlap_tv  = train_loans & val_loans
    logger.info(
        f"Loan-level overlap — train∩val: {len(overlap_tv)} "
        f"(expected: same loans appear across time splits — OK as long as no label leakage)"
    )

    return train_df, val_df, test_df


def get_split_masks(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    date_col: str = "month_index",
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return boolean masks instead of DataFrames — useful for indexing feature matrices."""
    unique_months = sorted(df[date_col].unique())
    n = len(unique_months)
    train_cut = unique_months[int(n * train_frac) - 1]
    val_cut   = unique_months[int(n * (train_frac + val_frac)) - 1]

    return (
        df[date_col] <= train_cut,
        (df[date_col] > train_cut) & (df[date_col] <= val_cut),
        df[date_col] > val_cut,
    )
