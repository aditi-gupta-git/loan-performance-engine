"""Explainability layer with SHAP and local explanations."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

import shap
import matplotlib.pyplot as plt

from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


class ExplainabilityEngine:
    """Generate global and local explanations for models."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.explainers = {}
        self.shap_values = {}
        self.feature_names = None
    
    def _unwrap_model(self, model: Any) -> Any:
        """Unwrap CalibratedClassifierCV and other wrappers to get base estimator."""
        # Handle CalibratedClassifierCV
        if hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
            base_model = model.calibrated_classifiers_[0].estimator
            logger.info(f"Unwrapped CalibratedClassifierCV to get base model: {type(base_model).__name__}")
            return base_model
        return model
    
    def fit_explainers(
        self, models: Dict[str, Any], X_train: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> 'ExplainabilityEngine':
        """Fit SHAP explainers for all models."""
        self.feature_names = feature_names or X_train.columns.tolist()
        
        # Sample background data for SHAP (use small sample for efficiency)
        background_size = min(100, len(X_train))
        background = X_train.sample(n=background_size, random_state=42)
        
        for target, model_info in models.items():
            logger.info(f"Fitting SHAP explainer for {target}...")
            
            try:
                if hasattr(model_info, 'model_path'):
                    import joblib
                    model = joblib.load(model_info.model_path)
                else:
                    model = model_info
                
                # Unwrap calibrated classifiers to access base estimator
                base_model = self._unwrap_model(model)
                model_type_str = str(type(base_model)).lower()
                
                # Determine model type and create appropriate explainer
                # Prefer TreeExplainer for tree-based models (much faster than KernelExplainer)
                if ('lightgbm' in model_type_str or 'lgbm' in model_type_str or
                    'xgboost' in model_type_str or 'xgb' in model_type_str or
                    'randomforest' in model_type_str or 'gradientboosting' in model_type_str):
                    logger.info(f"  Using TreeExplainer for {type(base_model).__name__}")
                    explainer = shap.TreeExplainer(base_model)
                else:
                    # For linear models or unknowns, use KernelExplainer with small background
                    logger.info(f"  Using KernelExplainer for {type(base_model).__name__}")
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                
                self.explainers[target] = explainer
                
            except Exception as e:
                logger.error(f"Failed to fit explainer for {target}: {e}", exc_info=True)
        
        return self
    
    def compute_global_shap(
        self, X: pd.DataFrame, max_samples: int = 1000
    ) -> Dict[str, Any]:
        """Compute global SHAP values for all models."""
        global_results = {}
        
        # Sample for efficiency
        X_sample = X.sample(n=min(max_samples, len(X)), random_state=42)
        
        for target, explainer in self.explainers.items():
            logger.info(f"Computing global SHAP for {target}...")
            
            try:
                shap_vals = explainer.shap_values(X_sample)
                
                # Handle all SHAP output shapes:
                # - Binary TreeExplainer: 2D array (n_samples, n_features)
                # - Multiclass TreeExplainer (new SHAP): 3D (n_samples, n_features, n_classes)
                # - Old-style multiclass: list of 2D arrays
                if isinstance(shap_vals, list):
                    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    # New-style 3D: average over samples and classes
                    mean_abs_shap = np.abs(shap_vals).mean(axis=(0, 2))
                else:
                    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                
                # Feature importance
                importance = dict(zip(self.feature_names, mean_abs_shap))
                importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
                
                self.shap_values[target] = {
                    'values': shap_vals,
                    'data': X_sample.values,
                    'feature_names': self.feature_names
                }
                
                global_results[target] = {
                    'importance': importance,
                    'top_20': dict(list(importance.items())[:20])
                }
                
            except Exception as e:
                logger.warning(f"Global SHAP failed for {target}: {e}")
                global_results[target] = {'error': str(e)}
        
        return global_results
    
    def explain_local(
        self, X: pd.DataFrame, loan_indices: List[int], target: str
    ) -> List[Dict[str, Any]]:
        """Generate local explanations for specific loans."""
        if target not in self.explainers:
            raise ValueError(f"No explainer for target: {target}")
        
        explainer = self.explainers[target]
        X_subset = X.iloc[loan_indices]
        
        try:
            shap_vals = explainer.shap_values(X_subset)
            
            explanations = []
            for i, idx in enumerate(loan_indices):
                if isinstance(shap_vals, list):
                    # Multiclass - take class with highest probability
                    class_idx = 1 if len(shap_vals) > 1 else 0
                    vals = shap_vals[class_idx][i]
                    base_value = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, list) else explainer.expected_value
                else:
                    vals = shap_vals[i]
                    base_value = explainer.expected_value
                
                # Top contributing features
                contributions = list(zip(self.feature_names, vals))
                contributions.sort(key=lambda x: -abs(x[1]))
                
                explanations.append({
                    'loan_index': int(idx),
                    'base_value': float(base_value),
                    'prediction': float(base_value + vals.sum()),
                    'contributions': [
                        {'feature': f, 'shap_value': float(v), 'feature_value': float(X_subset.iloc[i][f])}
                        for f, v in contributions[:10]
                    ]
                })
            
            return explanations
            
        except Exception as e:
            logger.warning(f"Local explanation failed: {e}")
            return []
    
    def generate_error_analysis(
        self, X: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray,
        y_proba: np.ndarray, target: str, n_examples: int = 5
    ) -> Dict[str, List[Dict]]:
        """Generate false positive/false negative case studies."""
        results = {'false_positives': [], 'false_negatives': []}
        
        # Handle multiclass case
        if y_proba.ndim == 2 and y_proba.shape[1] > 2:
            # Multiclass: convert to binary per class or skip detailed analysis
            logger.info(f"Skipping detailed error analysis for multiclass target {target}")
            return results
        
        # Binary classification
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            y_pred_binary = (y_proba[:, 1] > 0.5).astype(int)
        elif isinstance(y_pred, np.ndarray) and y_pred.ndim == 1:
            y_pred_binary = y_pred
        else:
            # Fallback: can't determine binary predictions
            logger.warning(f"Could not determine binary predictions for {target}, skipping error analysis")
            return results
        
        # False positives: predicted 1, actual 0
        fp_mask = (y_pred_binary == 1) & (y_true == 0)
        fp_indices = np.where(fp_mask)[0]
        
        # False negatives: predicted 0, actual 1
        fn_mask = (y_pred_binary == 0) & (y_true == 1)
        fn_indices = np.where(fn_mask)[0]
        
        # Sample examples
        fp_sample = np.random.choice(fp_indices, size=min(n_examples, len(fp_indices)), replace=False) if len(fp_indices) > 0 else []
        fn_sample = np.random.choice(fn_indices, size=min(n_examples, len(fn_indices)), replace=False) if len(fn_indices) > 0 else []
        
        for idx in fp_sample:
            local_exp = self.explain_local(X, [idx], target)
            if local_exp:
                exp = local_exp[0]
                exp['actual'] = int(y_true.iloc[idx])
                exp['predicted'] = int(y_pred_binary[idx])
                exp['probability'] = float(y_proba[idx, 1] if y_proba.ndim == 2 else y_proba[idx])
                results['false_positives'].append(exp)
        
        for idx in fn_sample:
            local_exp = self.explain_local(X, [idx], target)
            if local_exp:
                exp = local_exp[0]
                exp['actual'] = int(y_true.iloc[idx])
                exp['predicted'] = int(y_pred_binary[idx])
                exp['probability'] = float(y_proba[idx, 1] if y_proba.ndim == 2 else y_proba[idx])
                results['false_negatives'].append(exp)
        
        return results
    
    def plot_summary(
        self, target: str, output_path: str, max_display: int = 20
    ):
        """Generate SHAP summary plot."""
        if target not in self.shap_values:
            logger.warning(f"No SHAP values for {target}")
            return
        
        sv = self.shap_values[target]
        
        plt.figure(figsize=(10, 8))
        if isinstance(sv['values'], list):
            # Multiclass
            shap.summary_plot(sv['values'][1], sv['data'], 
                            feature_names=sv['feature_names'],
                            max_display=max_display, show=False)
        else:
            shap.summary_plot(sv['values'], sv['data'],
                            feature_names=sv['feature_names'],
                            max_display=max_display, show=False)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved SHAP summary plot to {output_path}")
    
    def plot_waterfall(
        self, target: str, loan_index: int, X: pd.DataFrame,
        output_path: str
    ):
        """Generate SHAP waterfall plot for single prediction."""
        if target not in self.explainers:
            return
        
        explainer = self.explainers[target]
        x = X.iloc[[loan_index]]
        
        try:
            shap_vals = explainer.shap_values(x)
            
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # Positive class
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                base_value = explainer.expected_value
            
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[0],
                    base_values=base_value,
                    data=x.values[0],
                    feature_names=self.feature_names
                ),
                show=False
            )
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved waterfall plot to {output_path}")
            
        except Exception as e:
            logger.warning(f"Waterfall plot failed: {e}")
    
    def compute_uncertainty(
        self, models: Dict[str, Any], X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Compute prediction uncertainty (variance across trees for ensembles)."""
        uncertainties = {}
        
        for target, model_info in models.items():
            if hasattr(model_info, 'model_path'):
                import joblib
                model = joblib.load(model_info.model_path)
            else:
                model = model_info
            
            # For tree ensembles, use prediction variance across trees
            if hasattr(model, 'estimators_'):
                # RandomForest or similar
                preds = np.array([est.predict_proba(X)[:, 1] for est in model.estimators_])
                uncertainties[target] = preds.std(axis=0)
            elif 'lgbm' in str(type(model)).lower() or 'lightgbm' in str(type(model)).lower():
                # LightGBM - use early stopping iterations or boostrap
                # Approximate: use raw scores variance if available
                try:
                    raw_scores = model.predict(X, raw_score=True)
                    # Can't easily get per-tree variance without modification
                    uncertainties[target] = np.ones(len(X)) * 0.1  # Placeholder
                except:
                    uncertainties[target] = np.ones(len(X)) * 0.1
            else:
                # Calibrated classifier or other
                uncertainties[target] = np.ones(len(X)) * 0.1
        
        return uncertainties
    
    def save_explanations(
        self, global_results: Dict, error_analysis: Dict,
        output_dir: str = "reports/explainability"
    ):
        """Save all explanations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Global importance
        with open(output_dir / "global_importance.json", 'w') as f:
            json.dump(global_results, f, indent=2, default=str)
        
        # Error analysis
        with open(output_dir / "error_analysis.json", 'w') as f:
            json.dump(error_analysis, f, indent=2, default=str)
        
        # SHAP values (sample)
        shap_sample = {}
        for target, vals in self.shap_values.items():
            shap_sample[target] = {
                'values': vals['values'][0].tolist() if isinstance(vals['values'], list) else vals['values'].tolist(),
                'feature_names': vals['feature_names']
            }
        
        with open(output_dir / "shap_values_sample.json", 'w') as f:
            json.dump(shap_sample, f, indent=2, default=str)
        
        logger.info(f"Saved explanations to {output_dir}")


def run_explainability(
    models: Dict[str, Any], X_train: pd.DataFrame, X_val: pd.DataFrame,
    y_val: Dict[str, pd.Series], predictions: Dict[str, np.ndarray],
    config: Optional[Dict] = None
) -> ExplainabilityEngine:
    """Run complete explainability analysis."""
    engine = ExplainabilityEngine(config)
    
    # Fit explainers
    engine.fit_explainers(models, X_train)
    
    # Global SHAP
    global_results = engine.compute_global_shap(X_val)
    
    # Error analysis for each target
    error_analysis = {}
    for target in models.keys():
        if target in y_val and target in predictions:
            error_analysis[target] = engine.generate_error_analysis(
                X_val, y_val[target], predictions[target],
                predictions[target], target
            )
    
    # Save
    engine.save_explanations(global_results, error_analysis)
    
    # Generate plots
    plots_dir = Path("reports/explainability/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for target in models.keys():
        engine.plot_summary(target, str(plots_dir / f"shap_summary_{target}.png"))
    
    return engine