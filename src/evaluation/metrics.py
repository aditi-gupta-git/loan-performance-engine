"""
Evaluation Metrics
==================
Compute ROC-AUC, PR-AUC, F1, Brier score, calibration, and recall@precision.
Standalone module — can be imported independently of the training pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        brier_score_loss, precision_recall_curve, confusion_matrix,
        classification_report
    )
    from sklearn.calibration import calibration_curve
except ImportError:
    raise ImportError("scikit-learn is required for evaluation metrics.")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    target_precision: float = 0.8,
) -> Dict[str, float]:
    """
    Full binary classification metrics suite.

    Parameters
    ----------
    y_true : array of true labels (0/1)
    y_proba : array of predicted probabilities for the positive class
    y_pred : optional hard predictions; derived from threshold if not supplied
    threshold : classification threshold
    target_precision : precision level at which to report recall

    Returns
    -------
    dict with roc_auc, pr_auc, f1, macro_f1, brier_score, recall_at_p80, etc.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_pred is None:
        y_pred = (y_proba >= threshold).astype(int)

    metrics: Dict[str, float] = {}

    # Guard: degenerate target
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true — metrics undefined.")
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "f1": float("nan"), "brier_score": float("nan"),
                "recall_at_p80": float("nan"), "macro_f1": float("nan")}

    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["brier_score"] = brier_score_loss(y_true, y_proba)

    # Recall at target precision
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_proba)
    idx = np.searchsorted(-prec_arr, -target_precision)
    metrics[f"recall_at_p{int(target_precision*100)}"] = float(rec_arr[idx]) if idx < len(rec_arr) else 0.0

    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Multiclass metrics (one-vs-rest macro averages).
    """
    y_true = np.asarray(y_true)
    metrics: Dict[str, float] = {}

    try:
        metrics["roc_auc"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro",
            labels=classes if classes is not None else np.unique(y_true)
        )
    except Exception as e:
        logger.warning(f"Multiclass ROC-AUC failed: {e}")
        metrics["roc_auc"] = float("nan")

    # One-vs-rest PR-AUC
    unique_classes = classes if classes is not None else np.unique(y_true)
    pr_scores = []
    for i, cls in enumerate(unique_classes):
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() > 0 and y_bin.sum() < len(y_true) and i < y_proba.shape[1]:
            try:
                pr_scores.append(average_precision_score(y_bin, y_proba[:, i]))
            except Exception:
                pass
    metrics["pr_auc"] = float(np.mean(pr_scores)) if pr_scores else float("nan")

    return metrics


def calibration_summary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Return a calibration curve as a DataFrame for plotting.
    Columns: mean_predicted_prob, fraction_of_positives, count
    """
    fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({
        "mean_predicted_prob": mean_pred,
        "fraction_of_positives": fraction_pos,
    })


def print_classification_report(y_true, y_pred, target_names=None):
    """Print sklearn classification report."""
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
