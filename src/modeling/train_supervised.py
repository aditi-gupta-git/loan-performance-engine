"""Classification modeling module for loan performance prediction."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import joblib
from dataclasses import dataclass, asdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    recall_score, precision_score, brier_score_loss,
    classification_report, confusion_matrix, precision_recall_curve
)
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""
    roc_auc: float
    pr_auc: float
    f1: float
    recall_at_precision_80: float
    brier_score: float
    macro_f1: float
    precision: float
    recall: float
    accuracy: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelResult:
    """Complete model training result."""
    model_name: str
    target: str
    model_type: str
    hyperparameters: Dict
    train_metrics: ModelMetrics
    val_metrics: ModelMetrics
    test_metrics: Optional[ModelMetrics]
    feature_importance: Dict[str, float]
    calibration_method: str
    training_time_seconds: float
    model_path: str


class TimeAwareSplitter:
    """Time-aware train/validation/test splitter."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.split_config = self.config.get('split', {})
        self.train_end_month = self.split_config.get('train_end_month', 24)
        self.val_start_month = self.split_config.get('val_start_month', 25)
        self.val_end_month = self.split_config.get('val_end_month', 30)
        self.test_start_month = self.split_config.get('test_start_month', 31)
    
    def split(
        self, df: pd.DataFrame, month_col: str = 'month_index'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by time periods."""
        train = df[df[month_col] <= self.train_end_month].copy()
        val = df[(df[month_col] >= self.val_start_month) & 
                 (df[month_col] <= self.val_end_month)].copy()
        test = df[df[month_col] >= self.test_start_month].copy()
        
        # Verify no loan_id leakage
        train_ids = set(train['loan_id'].unique()) if 'loan_id' in train.columns else set()
        val_ids = set(val['loan_id'].unique()) if 'loan_id' in val.columns else set()
        test_ids = set(test['loan_id'].unique()) if 'loan_id' in test.columns else set()
        
        overlap_train_val = train_ids & val_ids
        overlap_val_test = val_ids & test_ids
        overlap_train_test = train_ids & test_ids
        
        if overlap_train_val:
            logger.warning(f"Loan IDs in both train and val: {len(overlap_train_val)}")
        if overlap_val_test:
            logger.warning(f"Loan IDs in both val and test: {len(overlap_val_test)}")
        if overlap_train_test:
            logger.warning(f"Loan IDs in both train and test: {len(overlap_train_test)}")
        
        logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test
    
    def get_split_indices(self, df: pd.DataFrame, month_col: str = 'month_index'):
        """Get boolean masks for train/val/test splits.

        Uses configured month thresholds when they fall within the data range.
        Falls back to 60/20/20 percentile split when the data has fewer months
        than the configured thresholds (e.g. small synthetic runs in tests).
        """
        unique_months = sorted(df[month_col].unique())
        n = len(unique_months)

        if self.train_end_month in unique_months and self.val_end_month in unique_months:
            # Happy path: configured thresholds fit the data
            train_end  = self.train_end_month
            val_start  = self.val_start_month
            val_end    = self.val_end_month
            test_start = self.test_start_month
        else:
            # Fallback: percentile split so any month range produces 3 non-empty sets
            t60 = max(0, int(n * 0.60) - 1)
            t80 = min(n - 1, int(n * 0.80))
            train_end  = unique_months[t60]
            val_start  = unique_months[min(t60 + 1, n - 1)]
            val_end    = unique_months[t80]
            test_start = unique_months[min(t80 + 1, n - 1)]
            logger.info(
                "TimeAwareSplitter: configured thresholds outside data range; "
                f"using 60/20/20 percentile split: train≤{train_end}, "
                f"val {val_start}–{val_end}, test≥{test_start}"
            )

        train_mask = df[month_col] <= train_end
        val_mask   = (df[month_col] >= val_start) & (df[month_col] <= val_end)
        test_mask  = df[month_col] >= test_start
        return train_mask, val_mask, test_mask


class ClassifierTrainer:
    """Train and evaluate classification models for each target."""
    
    TARGETS = [
        'next_3m_delinquency_flag',
        'next_6m_delinquency_flag', 
        'next_12m_default_flag',
        'next_12m_prepayment_flag',
        'next_state'
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.model_config = self.config.get('modeling', {}).get('classification', {})
        self.splitter = TimeAwareSplitter(config)
        self.results: Dict[str, List[ModelResult]] = {}
        self.models: Dict[str, Any] = {}
        self.calibrators: Dict[str, Any] = {}
    
    def train_all_targets(
        self, 
        X_train: pd.DataFrame, y_train: pd.DataFrame,
        X_val: pd.DataFrame, y_val: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.DataFrame] = None
    ) -> Dict[str, List[ModelResult]]:
        """Train baseline and improved models for all targets."""
        results = {}
        
        for target in self.TARGETS:
            if target not in y_train.columns:
                logger.warning(f"Target {target} not in training data, skipping")
                continue
            
            logger.info(f"Training models for target: {target}")
            target_results = self._train_target(
                target, X_train, y_train[target], X_val, y_val[target],
                X_test, y_test[target] if y_test is not None and target in y_test.columns else None
            )
            results[target] = target_results
            
            # Store best model
            best_result = max(target_results, key=lambda r: r.val_metrics.pr_auc)
            self.models[target] = best_result
            self.calibrators[target] = self._get_calibrator(best_result)
        
        self.results = results
        return results
    
    def _train_target(
        self, target: str,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        X_test: Optional[pd.DataFrame], y_test: Optional[pd.Series]
    ) -> List[ModelResult]:
        """Train baseline and improved models for a single target."""
        results = []
        
        # Determine if multiclass
        is_multiclass = target == 'next_state' or y_train.nunique() > 2
        
        # Baseline: Logistic Regression
        logger.info(f"  Training baseline (LogisticRegression) for {target}")
        baseline_result = self._train_model(
            'baseline', target, 'logistic_regression',
            X_train, y_train, X_val, y_val, X_test, y_test, is_multiclass
        )
        results.append(baseline_result)
        
        # Improved: LightGBM
        logger.info(f"  Training improved (LightGBM) for {target}")
        improved_result = self._train_model(
            'improved', target, 'lightgbm',
            X_train, y_train, X_val, y_val, X_test, y_test, is_multiclass
        )
        results.append(improved_result)
        
        # Compare
        self._log_comparison(target, baseline_result, improved_result)
        
        return results
    
    def _train_model(
        self, model_name: str, target: str, model_type: str,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        X_test: Optional[pd.DataFrame], y_test: Optional[pd.Series],
        is_multiclass: bool
    ) -> ModelResult:
        """Train a single model with calibration."""
        import time
        start_time = time.time()
        
        # Get hyperparameters
        if model_type == 'logistic_regression':
            # sklearn 1.9+ no longer needs multi_class param; handled automatically
            model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.config.get('reproducibility', {}).get('global_seed', 42),
                n_jobs=-1,
                solver='lbfgs'
            )
            hyperparams = {'class_weight': 'balanced', 'max_iter': 1000}
        elif model_type == 'lightgbm':
            params = {
                'objective': 'multiclass' if is_multiclass else 'binary',
                'metric': 'multi_logloss' if is_multiclass else 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 63,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 50,
                'class_weight': 'balanced',
                'random_state': self.config.get('reproducibility', {}).get('global_seed', 42),
                'verbosity': -1,
                'n_jobs': -1
            }
            if is_multiclass:
                params['num_class'] = y_train.nunique()
            model = lgb.LGBMClassifier(**params)
            hyperparams = params
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Calibrate
        calibration_method = self.model_config.get('calibration_method', 'isotonic')
        if calibration_method == 'isotonic':
            calibrator = CalibratedClassifierCV(model, method='isotonic', cv=3)
        elif calibration_method == 'platt':
            calibrator = CalibratedClassifierCV(model, method='sigmoid', cv=3)
        else:
            calibrator = model
        
        calibrator.fit(X_train, y_train)
        
        # Evaluate
        train_metrics = self._evaluate(calibrator, X_train, y_train, is_multiclass)
        val_metrics = self._evaluate(calibrator, X_val, y_val, is_multiclass)
        test_metrics = None
        if X_test is not None and y_test is not None:
            test_metrics = self._evaluate(calibrator, X_test, y_test, is_multiclass)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(X_train.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            importance = dict(zip(X_train.columns, np.abs(model.coef_).flatten()))
        else:
            importance = {}
        
        # Save model
        model_path = f"models/classification/{target}_{model_name}.pkl"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrator, model_path)
        
        training_time = time.time() - start_time
        
        return ModelResult(
            model_name=model_name,
            target=target,
            model_type=model_type,
            hyperparameters=hyperparams,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_importance=importance,
            calibration_method=calibration_method,
            training_time_seconds=training_time,
            model_path=model_path
        )
    
    def _evaluate(
        self, model: Any, X: pd.DataFrame, y: pd.Series, is_multiclass: bool
    ) -> ModelMetrics:
        """Evaluate model on a dataset."""
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        if is_multiclass:
            # For multiclass, use macro averages
            try:
                roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='macro', labels=getattr(model, 'classes_', np.unique(y)))
            except Exception as _e:
                logger.warning(f"ROC-AUC failed for multiclass {y.name}: {_e}, using NaN")
                roc_auc = float('nan')
            
            # PR-AUC for multiclass: convert to one-vs-rest binary and average
            try:
                unique_classes = np.unique(y)
                if len(unique_classes) > 2:
                    # One-vs-rest macro average
                    pr_scores = []
                    for i, class_label in enumerate(unique_classes):
                        y_binary = (y == class_label).astype(int)
                        if y_binary.sum() > 0 and y_binary.sum() < len(y):  # Only if both classes present
                            try:
                                # Use class probability column i if available, else max prob
                                class_prob = y_proba[:, i] if i < y_proba.shape[1] else y_proba.max(axis=1)
                                pr = average_precision_score(y_binary, class_prob)
                                pr_scores.append(pr)
                            except:
                                pass
                    pr_auc = np.mean(pr_scores) if pr_scores else float('nan')
                else:
                    pr_auc = average_precision_score(y, y_proba[:, 1])
            except Exception as _e:
                logger.warning(f"PR-AUC failed for multiclass {y.name}: {_e}, using NaN")
                pr_auc = float('nan')
            
            f1 = f1_score(y, y_pred, average='macro', zero_division=0)
            macro_f1 = f1
            precision = precision_score(y, y_pred, average='macro', zero_division=0)
            recall = recall_score(y, y_pred, average='macro', zero_division=0)
            
            # Recall at precision=0.8 (per class, then average)
            recall_at_prec = 0
            classes = getattr(model, 'classes_', np.unique(y))
            for i in range(y_proba.shape[1]):
                class_label = classes[i] if i < len(classes) else i
                # y is categorical string, compare to class label
                y_binary = (y == class_label).astype(int)
                if y_binary.sum() == 0 or y_binary.sum() == len(y):
                    continue
                prec, rec, _ = precision_recall_curve(y_binary, y_proba[:, i])
                idx = np.where(prec >= 0.8)[0]
                if len(idx) > 0:
                    recall_at_prec += rec[idx[0]]
            # Average over classes that had valid precision-recall
            recall_at_prec = recall_at_prec / y_proba.shape[1] if y_proba.shape[1] > 0 else 0
            
            # Brier score for multiclass
            try:
                y_dummy = pd.get_dummies(y, dtype=float).values
                # If y_proba has different number of classes, align them
                if y_proba.shape[1] != y_dummy.shape[1]:
                    # Pad or trim to match
                    if y_proba.shape[1] > y_dummy.shape[1]:
                        y_dummy = np.pad(y_dummy, ((0, 0), (0, y_proba.shape[1] - y_dummy.shape[1])), mode='constant')
                    else:
                        y_proba = y_proba[:, :y_dummy.shape[1]]
                brier = np.mean(np.sum((y_proba - y_dummy)**2, axis=1))
            except Exception as _e:
                logger.warning(f"Brier score failed for multiclass {y.name}: {_e}, using NaN")
                brier = float('nan')
        else:
            # Binary
            y_proba_pos = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
            roc_auc = roc_auc_score(y, y_proba_pos)
            pr_auc = average_precision_score(y, y_proba_pos)
            f1 = f1_score(y, y_pred)
            macro_f1 = f1
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            
            # Recall at precision=0.8
            prec, rec, _ = precision_recall_curve(y, y_proba_pos)
            idx = np.where(prec >= 0.8)[0]
            recall_at_prec = rec[idx[0]] if len(idx) > 0 else 0
            
            brier = brier_score_loss(y, y_proba_pos)
        
        accuracy = (y_pred == y).mean()
        
        return ModelMetrics(
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            f1=f1,
            recall_at_precision_80=recall_at_prec,
            brier_score=brier,
            macro_f1=macro_f1,
            precision=precision,
            recall=recall,
            accuracy=accuracy
        )
    
    def _log_comparison(self, target: str, baseline: ModelResult, improved: ModelResult):
        """Log comparison between baseline and improved."""
        logger.info(f"  {target} Comparison:")
        logger.info(f"    Baseline Val PR-AUC: {baseline.val_metrics.pr_auc:.4f}, ROC-AUC: {baseline.val_metrics.roc_auc:.4f}")
        logger.info(f"    Improved Val PR-AUC: {improved.val_metrics.pr_auc:.4f}, ROC-AUC: {improved.val_metrics.roc_auc:.4f}")
        logger.info(f"    Improvement: PR-AUC +{improved.val_metrics.pr_auc - baseline.val_metrics.pr_auc:.4f}")
    
    def _get_calibrator(self, result: ModelResult) -> Any:
        """Load calibrator from saved model."""
        return joblib.load(result.model_path)
    
    def predict(
        self, X: pd.DataFrame, target: str, return_proba: bool = True
    ) -> np.ndarray:
        """Make predictions using best model for target."""
        if target not in self.models:
            raise ValueError(f"No trained model for target: {target}")
        
        model = joblib.load(self.models[target].model_path)
        if return_proba:
            return model.predict_proba(X)
        return model.predict(X)
    
    def save_results(self, output_dir: str = "reports/modeling"):
        """Save all training results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary table
        summary_rows = []
        for target, results in self.results.items():
            for r in results:
                row = {
                    'target': target,
                    'model': r.model_name,
                    'type': r.model_type,
                    'train_roc_auc': r.train_metrics.roc_auc,
                    'train_pr_auc': r.train_metrics.pr_auc,
                    'val_roc_auc': r.val_metrics.roc_auc,
                    'val_pr_auc': r.val_metrics.pr_auc,
                    'val_f1': r.val_metrics.f1,
                    'val_recall@prec80': r.val_metrics.recall_at_precision_80,
                    'val_brier': r.val_metrics.brier_score,
                    'test_roc_auc': r.test_metrics.roc_auc if r.test_metrics else None,
                    'test_pr_auc': r.test_metrics.pr_auc if r.test_metrics else None,
                    'calibration': r.calibration_method,
                    'train_time_sec': r.training_time_seconds
                }
                summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "model_comparison.csv", index=False)
        
        # Detailed results
        detailed = {}
        for target, results in self.results.items():
            detailed[target] = [asdict(r) for r in results]
        
        with open(output_dir / "detailed_results.json", 'w') as f:
            json.dump(detailed, f, indent=2, default=str)
        
        logger.info(f"Saved modeling results to {output_dir}")
    
    def generate_model_cards(self, output_dir: str = "models/classification"):
        """Generate model cards for all trained models."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        template_path = Path("config/model_card_template.md")
        if template_path.exists():
            with open(template_path) as f:
                template = f.read()
        else:
            template = "# Model Card: {model_name}\n\n## Objective\n{objective}\n..."
        
        for target, results in self.results.items():
            for r in results:
                card = template.format(
                    model_name=f"{target}_{r.model_name}",
                    objective=f"Predict {target} for loan performance monitoring",
                    data_source="Synthetic loan panel data",
                    n_samples=len(self.X_train) if hasattr(self, "X_train") else "see training logs",
                    n_features=len(r.feature_importance),
                    start_date="2020-01", end_date="2023-12",
                    train_end_month=self.splitter.train_end_month,
                    val_start_month=self.splitter.val_start_month,
                    val_end_month=self.splitter.val_end_month,
                    test_start_month=self.splitter.test_start_month,
                    feature_list="\n".join([f"- {k}: {v:.4f}" for k, v in 
                          sorted(r.feature_importance.items(), key=lambda x: -x[1])[:20]]),
                    algorithm=r.model_type,
                    hyperparameters=json.dumps(r.hyperparameters, indent=2),
                    imbalance_method="class_weight=balanced",
                    calibration_method=r.calibration_method,
                    train_roc_auc=f"{r.train_metrics.roc_auc:.4f}",
                    train_pr_auc=f"{r.train_metrics.pr_auc:.4f}",
                    train_f1=f"{r.train_metrics.f1:.4f}",
                    train_recall_at_prec=f"{r.train_metrics.recall_at_precision_80:.4f}",
                    train_brier=f"{r.train_metrics.brier_score:.4f}",
                    train_macro_f1=f"{r.train_metrics.macro_f1:.4f}",
                    val_roc_auc=f"{r.val_metrics.roc_auc:.4f}",
                    val_pr_auc=f"{r.val_metrics.pr_auc:.4f}",
                    val_f1=f"{r.val_metrics.f1:.4f}",
                    val_recall_at_prec=f"{r.val_metrics.recall_at_precision_80:.4f}",
                    val_brier=f"{r.val_metrics.brier_score:.4f}",
                    val_macro_f1=f"{r.val_metrics.macro_f1:.4f}",
                    test_roc_auc=f"{r.test_metrics.roc_auc:.4f}" if r.test_metrics else "N/A",
                    test_pr_auc=f"{r.test_metrics.pr_auc:.4f}" if r.test_metrics else "N/A",
                    test_f1=f"{r.test_metrics.f1:.4f}" if r.test_metrics else "N/A",
                    test_recall_at_prec=f"{r.test_metrics.recall_at_precision_80:.4f}" if r.test_metrics else "N/A",
                    test_brier=f"{r.test_metrics.brier_score:.4f}" if r.test_metrics else "N/A",
                    test_macro_f1=f"{r.test_metrics.macro_f1:.4f}" if r.test_metrics else "N/A",
                    calibration_details=f"Calibrated using {r.calibration_method} on training data",
                    limitations="Trained on synthetic data; may not generalize to real portfolio",
                    fairness_notes="No protected attributes used; geographic features may proxy for demographics",
                    intended_use="Portfolio risk monitoring and reviewer prioritization",
                    out_of_scope_use="Individual loan approval/denial decisions",
                    version="1.0",
                    training_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                    git_commit="unknown"
                )
                
                with open(output_dir / f"MODEL_CARD_{target}_{r.model_name}.md", 'w') as f:
                    f.write(card)


def train_classification_models(
    X_train: pd.DataFrame, y_train: pd.DataFrame,
    X_val: pd.DataFrame, y_val: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None
) -> ClassifierTrainer:
    """Train all classification models."""
    trainer = ClassifierTrainer(config)
    trainer.train_all_targets(X_train, y_train, X_val, y_val, X_test, y_test)
    trainer.save_results()
    trainer.generate_model_cards()
    return trainer