"""Anomaly and exception detection module."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import joblib
from dataclasses import dataclass, asdict
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
try:
    from pyod.models.auto_encoder import AutoEncoder
except (ImportError, ModuleNotFoundError):
    AutoEncoder = None
try:
    from pyod.models.lof import LOF
except (ImportError, ModuleNotFoundError):
    LOF = None
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


@dataclass
class AnomalyResult:
    """Anomaly detection result for a dataset."""
    scores: np.ndarray
    flags: np.ndarray
    drivers: List[Dict[str, Any]]
    rule_violations: pd.DataFrame
    exception_predictions: Optional[np.ndarray] = None
    exception_probabilities: Optional[np.ndarray] = None


@dataclass
class ReviewerExample:
    """Reviewer-ready anomaly example."""
    loan_id: str
    month_index: int
    reporting_month: str
    anomaly_score: float
    rule_flags: List[str]
    top_drivers: List[Dict[str, Any]]
    exception_type: str
    suggested_action: str
    narrative: str


class RuleBasedChecker:
    """Apply validation rules from config."""
    
    def __init__(self, rules_path: str = "config/validation_rules.json"):
        self.rules_path = Path(rules_path)
        self.rules = self._load_rules()
    
    def _load_rules(self) -> List[Dict]:
        if self.rules_path.exists():
            with open(self.rules_path) as f:
                return json.load(f).get("rules", [])
        return []
    
    def check(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply all rules, return (violations_df, summary_df)."""
        all_violations = []
        
        for rule in self.rules:
            try:
                mask = df.eval(rule["check"])
                violations = df[~mask].copy()
                if len(violations) > 0:
                    violations['rule_id'] = rule['rule_id']
                    violations['rule_name'] = rule['name']
                    violations['severity'] = rule['severity']
                    violations['description'] = rule['description']
                    all_violations.append(violations)
            except Exception as e:
                logger.warning(f"Rule {rule['rule_id']} failed: {e}")
        
        if all_violations:
            violations_df = pd.concat(all_violations, ignore_index=True)
        else:
            violations_df = pd.DataFrame()
        
        # Summary by rule
        if len(violations_df) > 0:
            summary = violations_df.groupby(['rule_id', 'rule_name', 'severity']).size().reset_index(name='count')
        else:
            summary = pd.DataFrame(columns=['rule_id', 'rule_name', 'severity', 'count'])
        
        return violations_df, summary


class MLAnomalyDetector:
    """Unsupervised ML anomaly detection ensemble."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.anomaly_config = self.config.get('anomaly', {})
        self.contamination = self.anomaly_config.get('contamination', 0.05)
        self.methods = self.anomaly_config.get('methods', ['isolation_forest', 'rule_based'])
        self.weights = self.anomaly_config.get('ensemble_weights', [0.6, 0.4])
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> 'MLAnomalyDetector':
        """Fit anomaly detectors."""
        self.feature_columns = feature_cols
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        if 'isolation_forest' in self.methods:
            logger.info("Fitting Isolation Forest...")
            iso = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
                n_estimators=200
            )
            iso.fit(X_scaled)
            self.models['isolation_forest'] = iso
        
        if 'lof' in self.methods:
            logger.info("Fitting Local Outlier Factor...")
            lof = LocalOutlierFactor(
                contamination=self.contamination,
                n_jobs=-1,
                novelty=True
            )
            lof.fit(X_scaled)
            self.models['lof'] = lof
        
        if 'autoencoder' in self.methods:
            logger.info("Fitting AutoEncoder...")
            try:
                ae = AutoEncoder(
                    hidden_neurons=[64, 32, 64],
                    contamination=self.contamination,
                    epochs=50,
                    batch_size=256,
                    random_state=42,
                    verbose=0
                )
                ae.fit(X_scaled)
                self.models['autoencoder'] = ae
            except Exception as e:
                logger.warning(f"AutoEncoder failed: {e}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Predict anomaly scores and flags."""
        # Handle missing features: use only those that exist in df
        available_features = [c for c in self.feature_columns if c in df.columns]
        if len(available_features) == 0:
            # Fallback if no features available
            logger.warning("No anomaly features available in dataframe, returning zero scores")
            return np.zeros(len(df)), np.zeros(len(df), dtype=int), {}
        
        if len(available_features) < len(self.feature_columns):
            logger.warning(f"Using {len(available_features)}/{len(self.feature_columns)} available anomaly features")
        
        X = df[available_features].fillna(0).values
        
        # Pad with zeros if we have fewer features than the models expect
        n_expected = getattr(self.scaler, 'n_features_in_', len(available_features))
        if X.shape[1] < n_expected:
            X_padded = np.zeros((X.shape[0], n_expected))
            X_padded[:, :X.shape[1]] = X
            X = X_padded
            logger.info(f"Padded features from {X.shape[1]-n_expected} to {n_expected}")
        
        # Scale features
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            logger.warning(f"Scaler transform failed: {e}, using raw values")
            X_scaled = X
        
        scores_dict = {}
        
        if 'isolation_forest' in self.models:
            # Isolation Forest: lower score = more anomalous
            iso_scores = -self.models['isolation_forest'].score_samples(X_scaled)
            # Normalize to 0-1
            iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
            scores_dict['isolation_forest'] = iso_scores
        
        if 'lof' in self.models:
            # LOF: negative outlier factor, lower = more anomalous
            lof_scores = -self.models['lof'].decision_function(X_scaled)
            lof_scores = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-8)
            scores_dict['lof'] = lof_scores
        
        if 'autoencoder' in self.models:
            # AutoEncoder: reconstruction error
            ae_scores = self.models['autoencoder'].decision_function(X_scaled)
            ae_scores = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-8)
            scores_dict['autoencoder'] = ae_scores
        
        # Ensemble
        if scores_dict:
            weighted_scores = np.zeros(len(df))
            total_weight = 0
            for i, (method, weight) in enumerate(zip(self.methods, self.weights)):
                if method in scores_dict:
                    weighted_scores += weight * scores_dict[method]
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_scores = weighted_scores / total_weight
            else:
                ensemble_scores = np.mean(list(scores_dict.values()), axis=0)
            
            # Flag top contamination% as anomalies
            threshold = np.percentile(ensemble_scores, 100 * (1 - self.contamination))
            flags = (ensemble_scores >= threshold).astype(int)
        else:
            ensemble_scores = np.zeros(len(df))
            flags = np.zeros(len(df), dtype=int)
        
        return ensemble_scores, flags, scores_dict
    
    def get_feature_contributions(self, df: pd.DataFrame, indices: np.ndarray) -> List[Dict]:
        """Get feature contributions for anomalous records using reconstruction error."""
        contributions = []
        
        # Handle missing features
        available_features = [c for c in self.feature_columns if c in df.columns]
        if len(available_features) == 0:
            return [[] for _ in indices]
        
        X = df[available_features].fillna(0).values
        n_expected = getattr(self.scaler, 'n_features_in_', len(available_features))
        if X.shape[1] < n_expected:
            X_padded = np.zeros((X.shape[0], n_expected))
            X_padded[:, :X.shape[1]] = X
            X = X_padded
        
        try:
            X_scaled = self.scaler.transform(X)
        except:
            X_scaled = X
        
        if 'autoencoder' in self.models:
            # Use autoencoder reconstruction error per feature
            ae = self.models['autoencoder']
            reconstructions = ae.model_.predict(X_scaled[indices])
            errors = (X_scaled[indices] - reconstructions) ** 2
            
            for i, idx in enumerate(indices):
                feat_errors = errors[i]
                top_idx = np.argsort(feat_errors)[-5:][::-1]
                drivers = [
                    {'feature': available_features[j] if j < len(available_features) else f'feature_{j}', 
                     'contribution': float(feat_errors[j])}
                    for j in top_idx
                ]
                contributions.append(drivers)
        else:
            # Fallback: use isolation forest feature importance (approximate)
            for idx in indices:
                contributions.append([
                    {'feature': 'composite', 'contribution': 1.0}
                ])
        
        return contributions


class ExceptionClassifier:
    """Supervised exception type classifier."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.anomaly_config = self.config.get('anomaly', {})
        self.exception_types = self.anomaly_config.get('exception_types', 
            ['data_quality', 'business_rule', 'pattern_anomaly', 'servicer_conflict'])
        self.model = None
        self.feature_columns = None
    
    def fit(
        self, df: pd.DataFrame, feature_cols: List[str], 
        labels: Optional[pd.Series] = None
    ) -> 'ExceptionClassifier':
        """Train exception type classifier."""
        self.feature_columns = feature_cols
        
        if labels is not None and labels.nunique() > 1:
            # Supervised training
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(
                objective='multiclass',
                num_class=len(self.exception_types),
                random_state=42,
                verbosity=-1,
                n_jobs=-1
            )
            X = df[feature_cols].fillna(0)
            # Map labels to indices
            label_map = {t: i for i, t in enumerate(self.exception_types)}
            y = labels.map(label_map).fillna(0).astype(int)
            self.model.fit(X, y)
        else:
            # Unsupervised: use rule-based pseudo-labels
            logger.info("No exception labels, using rule-based pseudo-labels")
            self.model = None
        
        return self
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict exception types."""
        if self.model is None:
            # Return default
            return np.zeros(len(df), dtype=int), np.ones((len(df), len(self.exception_types))) / len(self.exception_types)
        
        # Handle missing features
        available_features = [c for c in self.feature_columns if c in df.columns]
        if len(available_features) == 0:
            # Return default if no features available
            return np.zeros(len(df), dtype=int), np.ones((len(df), len(self.exception_types))) / len(self.exception_types)
        
        X = df[available_features].fillna(0)
        try:
            preds = self.model.predict(X)
            probas = self.model.predict_proba(X)
        except Exception as e:
            logger.warning(f"Exception classifier prediction failed: {e}, returning defaults")
            preds = np.zeros(len(df), dtype=int)
            probas = np.ones((len(df), len(self.exception_types))) / len(self.exception_types)
        
        return preds, probas


class AnomalyDetector:
    """Main anomaly detection orchestrator."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.rule_checker = RuleBasedChecker()
        self.ml_detector = MLAnomalyDetector(config)
        self.exception_classifier = ExceptionClassifier(config)
        self.feature_columns = None
    
    def fit(self, df: pd.DataFrame) -> 'AnomalyDetector':
        """Fit all detectors."""
        logger.info("Fitting anomaly detectors...")
        
        # Rule-based checks
        violations_df, rule_summary = self.rule_checker.check(df)
        
        # Select features for ML
        self.feature_columns = self._select_features(df)
        
        # Fit ML detectors
        self.ml_detector.fit(df, self.feature_columns)
        
        # Fit exception classifier if labels available
        if 'exception_type' in df.columns:
            self.exception_classifier.fit(df, self.feature_columns, df['exception_type'])
        
        return self
    
    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select numeric features for anomaly detection."""
        exclude = ['loan_id', 'month_index', 'reporting_month', 'origination_month',
                  'last_updated_at', 'source_system', 'credit_score_band', 'ltv_band',
                  'dti_band', 'state', 'loan_purpose', 'occupancy_type', 'property_type',
                  'servicer_name', 'current_status', 'loss_severity_band', 'document_status',
                  'next_3m_delinquency_flag', 'next_6m_delinquency_flag',
                  'next_12m_default_flag', 'next_12m_prepayment_flag', 'next_state',
                  'exception_required', 'exception_type']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in exclude]
        
        # Remove constant/near-constant columns
        feature_cols = [c for c in feature_cols if df[c].nunique() > 1]
        
        logger.info(f"Selected {len(feature_cols)} features for anomaly detection")
        return feature_cols
    
    def detect(self, df: pd.DataFrame) -> AnomalyResult:
        """Run anomaly detection on dataset."""
        logger.info(f"Detecting anomalies in {len(df)} records...")
        
        # Rule-based violations
        violations_df, rule_summary = self.rule_checker.check(df)
        
        # ML anomaly scores
        ml_scores, ml_flags, method_scores = self.ml_detector.predict(df)
        
        # Combine rule-based and ML scores
        rule_scores = np.zeros(len(df))
        if len(violations_df) > 0:
            # Count violations per record
            violation_counts = violations_df.groupby(level=0).size()
            rule_scores = violation_counts.reindex(df.index, fill_value=0).values
            rule_scores = np.clip(rule_scores / 5, 0, 1)  # Normalize
        
        # Ensemble: 60% ML, 40% rules
        combined_scores = 0.6 * ml_scores + 0.4 * rule_scores
        combined_flags = (combined_scores > np.percentile(combined_scores, 95)).astype(int)
        
        # Exception classification
        exc_preds, exc_probas = self.exception_classifier.predict(df)
        exc_labels = [self.exception_classifier.exception_types[p] for p in exc_preds]
        
        # Get drivers for flagged records
        flagged_indices = np.where(combined_flags)[0]
        drivers = self.ml_detector.get_feature_contributions(df, flagged_indices)
        
        # Build driver list for all records
        all_drivers = []
        driver_idx = 0
        for i in range(len(df)):
            if combined_flags[i] and driver_idx < len(drivers):
                all_drivers.append(drivers[driver_idx])
                driver_idx += 1
            else:
                all_drivers.append([])
        
        return AnomalyResult(
            scores=combined_scores,
            flags=combined_flags,
            drivers=all_drivers,
            rule_violations=violations_df,
            exception_predictions=exc_preds,
            exception_probabilities=exc_probas
        )
    
    def generate_reviewer_examples(
        self, df: pd.DataFrame, result: AnomalyResult, n_examples: int = 20
    ) -> List[ReviewerExample]:
        """Generate reviewer-ready anomaly examples."""
        logger.info(f"Generating {n_examples} reviewer examples...")
        
        # Get flagged records with scores
        flagged = df[result.flags == 1].copy()
        flagged['anomaly_score'] = result.scores[result.flags == 1]
        flagged['drivers'] = [result.drivers[i] for i in np.where(result.flags == 1)[0]]
        
        # Sort by anomaly score
        flagged = flagged.sort_values('anomaly_score', ascending=False)
        
        # Take top examples, diversified by exception type
        examples = []
        for exc_type in self.exception_classifier.exception_types:
            exc_examples = flagged[flagged.get('exception_type', '') == exc_type] if 'exception_type' in flagged.columns else flagged
            if len(exc_examples) == 0:
                exc_examples = flagged
            
            for _, row in exc_examples.head(n_examples // len(self.exception_classifier.exception_types) + 1).iterrows():
                if len(examples) >= n_examples:
                    break
                
                drivers = row.get('drivers', [])
                rule_flags = []
                if len(result.rule_violations) > 0:
                    loan_violations = result.rule_violations[result.rule_violations['loan_id'] == row['loan_id']]
                    if len(loan_violations) > 0:
                        rule_flags = loan_violations['rule_name'].unique().tolist()
                
                narrative = self._generate_narrative(row, drivers, rule_flags, exc_type)
                action = self._suggest_action(exc_type, drivers, rule_flags)
                
                examples.append(ReviewerExample(
                    loan_id=row['loan_id'],
                    month_index=row.get('month_index', 0),
                    reporting_month=str(row.get('reporting_month', '')),
                    anomaly_score=float(row['anomaly_score']),
                    rule_flags=rule_flags,
                    top_drivers=drivers,
                    exception_type=exc_type,
                    suggested_action=action,
                    narrative=narrative
                ))
        
        # Fill remaining with highest scoring
        if len(examples) < n_examples:
            for _, row in flagged.head(n_examples).iterrows():
                if len(examples) >= n_examples:
                    break
                if row['loan_id'] not in [e.loan_id for e in examples]:
                    drivers = row.get('drivers', [])
                    rule_flags = []
                    exc_type = self.exception_classifier.exception_types[0]
                    narrative = self._generate_narrative(row, drivers, rule_flags, exc_type)
                    action = self._suggest_action(exc_type, drivers, rule_flags)
                    examples.append(ReviewerExample(
                        loan_id=row['loan_id'],
                        month_index=row.get('month_index', 0),
                        reporting_month=str(row.get('reporting_month', '')),
                        anomaly_score=float(row['anomaly_score']),
                        rule_flags=rule_flags,
                        top_drivers=drivers,
                        exception_type=exc_type,
                        suggested_action=action,
                        narrative=narrative
                    ))
        
        return examples[:n_examples]
    
    def _generate_narrative(self, row: pd.Series, drivers: List[Dict], 
                           rule_flags: List[str], exc_type: str) -> str:
        """Generate human-readable narrative for anomaly."""
        parts = []
        
        if rule_flags:
            parts.append(f"Rule violations: {', '.join(rule_flags[:3])}")
        
        if drivers:
            driver_strs = [f"{d['feature']} (score: {d['contribution']:.3f})" for d in drivers[:3]]
            parts.append(f"Key drivers: {', '.join(driver_strs)}")
        
        if exc_type == 'data_quality':
            parts.append("Data quality issue detected - verify record accuracy")
        elif exc_type == 'business_rule':
            parts.append("Business rule violation - review loan terms and status")
        elif exc_type == 'pattern_anomaly':
            parts.append("Unusual pattern detected - investigate loan behavior")
        elif exc_type == 'servicer_conflict':
            parts.append("Servicer data conflict - reconcile with primary source")
        
        return ". ".join(parts) + "."
    
    def _suggest_action(self, exc_type: str, drivers: List[Dict], rule_flags: List[str]) -> str:
        """Suggest reviewer action."""
        actions = {
            'data_quality': "Verify data source and correct record",
            'business_rule': "Review loan terms and servicer compliance",
            'pattern_anomaly': "Investigate borrower behavior and payment history",
            'servicer_conflict': "Contact servicer to reconcile discrepancies"
        }
        return actions.get(exc_type, "Review loan file for anomalies")
    
    def save_artifacts(self, output_dir: str = "models/anomaly"):
        """Save fitted detectors."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.ml_detector, output_dir / "ml_detector.pkl")
        joblib.dump(self.exception_classifier, output_dir / "exception_classifier.pkl")
        joblib.dump(self.feature_columns, output_dir / "feature_columns.pkl")
        
        logger.info(f"Saved anomaly detection artifacts to {output_dir}")
    
    def load_artifacts(self, input_dir: str = "models/anomaly"):
        """Load fitted detectors.
        
        Handles backward compatibility when pkl files were saved under the old
        src.models.anomaly.anomaly module path (before restructure).
        """
        import sys, types
        # Register module aliases so old pickle paths resolve correctly
        import src.modeling.anomaly as _this_mod
        import src.modeling.train_supervised as _trainer_mod
        _aliases = {
            'src.models': types.ModuleType('src.models'),
            'src.models.anomaly': types.ModuleType('src.models.anomaly'),
            'src.models.anomaly.anomaly': _this_mod,
            'src.models.classification': types.ModuleType('src.models.classification'),
            'src.models.classification.trainer': _trainer_mod,
        }
        for k, v in _aliases.items():
            if k not in sys.modules:
                sys.modules[k] = v
        # Ensure class attributes are reachable from the alias module
        for attr in dir(_this_mod):
            setattr(_aliases['src.models.anomaly.anomaly'], attr, getattr(_this_mod, attr))

        input_dir = Path(input_dir)
        self.ml_detector = joblib.load(input_dir / "ml_detector.pkl")
        self.exception_classifier = joblib.load(input_dir / "exception_classifier.pkl")
        self.feature_columns = joblib.load(input_dir / "feature_columns.pkl")
        logger.info(f"Loaded anomaly detection artifacts from {input_dir}")


def run_anomaly_detection(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    config: Optional[Dict] = None
) -> Tuple[AnomalyDetector, AnomalyResult, AnomalyResult, List[ReviewerExample]]:
    """Run complete anomaly detection pipeline."""
    detector = AnomalyDetector(config)
    detector.fit(train_df)
    
    train_result = detector.detect(train_df)
    test_result = detector.detect(test_df)
    
    reviewer_examples = detector.generate_reviewer_examples(train_df, train_result, n_examples=20)
    
    detector.save_artifacts()
    
    return detector, train_result, test_result, reviewer_examples