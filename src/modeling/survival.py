"""Time-to-event / Survival modeling module."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import joblib
from dataclasses import dataclass, asdict
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


@dataclass
class SurvivalMetrics:
    """Survival model metrics."""
    concordance_index: float
    brier_score: float
    log_likelihood: float
    aic: float


@dataclass
class SurvivalResult:
    """Survival model training result."""
    model_name: str
    model_type: str
    event_type: str
    metrics: SurvivalMetrics
    survival_curves: Dict[str, Any]
    feature_importance: Dict[str, float]
    training_time_seconds: float
    model_path: str


class SurvivalDataBuilder:
    """Build survival analysis dataset from panel data."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.survival_config = self.config.get('modeling', {}).get('survival', {})
    
    def build_survival_dataset(
        self, df: pd.DataFrame, event_col: str = 'event_type'
    ) -> pd.DataFrame:
        """
        Build survival dataset with one row per loan.
        event_type: 'default', 'prepayment', 'censored'
        duration: months to event or censoring
        """
        logger.info("Building survival dataset...")
        
        # For each loan, determine event and duration
        survival_rows = []
        
        for loan_id, loan_df in df.groupby('loan_id'):
            loan_df = loan_df.sort_values('month_index')
            
            # Find first event
            default_idx = loan_df[loan_df['default_flag'] == 1].index
            prepay_idx = loan_df[loan_df['prepayment_flag'] == 1].index
            
            if len(default_idx) > 0 and (len(prepay_idx) == 0 or default_idx[0] < prepay_idx[0]):
                # Default occurred first
                event_row = loan_df.loc[default_idx[0]]
                duration = event_row['loan_age_months']
                event_type = 'default'
                event = 1
            elif len(prepay_idx) > 0:
                # Prepayment occurred first
                event_row = loan_df.loc[prepay_idx[0]]
                duration = event_row['loan_age_months']
                event_type = 'prepayment'
                event = 2  # Competing risk
            else:
                # Censored - still active at end
                event_row = loan_df.iloc[-1]
                duration = event_row['loan_age_months']
                event_type = 'censored'
                event = 0
            
            # Get baseline features (at origination or first observation)
            baseline_row = loan_df.iloc[0]
            
            row = {
                'loan_id': loan_id,
                'duration': duration,
                'event': event,
                'event_type': event_type,
                'origination_month': baseline_row.get('origination_month'),
                'original_balance': baseline_row.get('original_balance'),
                'interest_rate': baseline_row.get('interest_rate'),
                'credit_score_band': baseline_row.get('credit_score_band'),
                'ltv_band': baseline_row.get('ltv_band'),
                'dti_band': baseline_row.get('dti_band'),
                'state': baseline_row.get('state'),
                'loan_purpose': baseline_row.get('loan_purpose'),
                'occupancy_type': baseline_row.get('occupancy_type'),
                'property_type': baseline_row.get('property_type'),
                'servicer_name': baseline_row.get('servicer_name'),
                'term_months': baseline_row.get('term_months'),
            }
            
            # Add time-varying covariates at event/censoring time
            if 'current_balance' in event_row:
                row['balance_at_event'] = event_row['current_balance']
                row['balance_ratio_at_event'] = event_row['current_balance'] / baseline_row['original_balance']
            if 'days_past_due' in event_row:
                row['dpd_at_event'] = event_row['days_past_due']
            if 'current_status' in event_row:
                row['status_at_event'] = event_row['current_status']
            if 'modification_flag' in event_row:
                row['ever_modified'] = loan_df['modification_flag'].max()
            
            survival_rows.append(row)
        
        survival_df = pd.DataFrame(survival_rows)
        
        # Encode categoricals
        survival_df = self._encode_categorical(survival_df)
        
        logger.info(f"Survival dataset: {len(survival_df)} loans")
        logger.info(f"Event distribution: {survival_df['event_type'].value_counts().to_dict()}")
        
        return survival_df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for survival modeling."""
        df = df.copy()
        
        # Ordinal encoding for bands
        credit_order = ['<620', '620-659', '660-699', '700-739', '740-779', '780+']
        ltv_order = ['<60%', '60-70%', '70-80%', '80-90%', '90-100%', '>100%']
        dti_order = ['<20%', '20-30%', '30-36%', '36-43%', '43-50%', '>50%']
        
        for col, order in [('credit_score_band', credit_order), ('ltv_band', ltv_order), ('dti_band', dti_order)]:
            if col in df.columns:
                mapping = {v: i for i, v in enumerate(order)}
                df[f'{col}_encoded'] = df[col].map(mapping).fillna(-1)
        
        # One-hot for others
        for col in ['state', 'loan_purpose', 'occupancy_type', 'property_type', 'servicer_name']:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = pd.concat([df, dummies], axis=1)
        
        return df


class SurvivalModeler:
    """Train and evaluate survival models."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.survival_config = self.config.get('modeling', {}).get('survival', {})
        self.model_type = self.survival_config.get('model_type', 'discrete_time')
        self.results: Dict[str, SurvivalResult] = {}
    
    def train_all(
        self, survival_df: pd.DataFrame,
        train_mask: pd.Series, val_mask: pd.Series, test_mask: Optional[pd.Series] = None
    ) -> Dict[str, SurvivalResult]:
        """Train survival models for each event type."""
        results = {}
        
        # Split data
        train_df = survival_df[train_mask].copy()
        val_df = survival_df[val_mask].copy()
        test_df = survival_df[test_mask].copy() if test_mask is not None else None
        
        # Overall survival (any event)
        logger.info("Training overall survival model...")
        results['overall'] = self._train_kaplan_meier(train_df, val_df, test_df)
        
        # Competing risks: default vs prepayment
        for event_type in ['default', 'prepayment']:
            logger.info(f"Training {event_type} model...")
            results[event_type] = self._train_cox_ph(train_df, val_df, test_df, event_type)
        
        # Discrete-time transition model
        logger.info("Training discrete-time transition model...")
        results['discrete_time'] = self._train_discrete_time(train_df, val_df, test_df)
        
        self.results = results
        return results
    
    def _train_kaplan_meier(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
        test_df: Optional[pd.DataFrame]
    ) -> SurvivalResult:
        """Train Kaplan-Meier baseline."""
        import time
        start_time = time.time()
        
        kmf = KaplanMeierFitter()
        kmf.fit(train_df['duration'], event_observed=(train_df['event'] > 0).astype(int))
        
        # Evaluate on validation
        val_concordance = self._eval_concordance(kmf, val_df)
        val_brier = self._eval_brier(kmf, val_df)
        
        # Survival curves by segment
        curves = self._compute_segment_curves(kmf, train_df, 'credit_score_band')
        
        training_time = time.time() - start_time
        
        # Handle lifelines version without log_likelihood_ on KM
        try:
            ll = float(kmf.log_likelihood_)
        except Exception:
            ll = 0.0
        return SurvivalResult(
            model_name='kaplan_meier_overall',
            model_type='kaplan_meier',
            event_type='any',
            metrics=SurvivalMetrics(
                concordance_index=val_concordance,
                brier_score=val_brier,
                log_likelihood=ll,
                aic=-2 * ll + 2  # 1 parameter
            ),
            survival_curves=curves,
            feature_importance={},
            training_time_seconds=training_time,
            model_path='models/survival/kaplan_meier_overall.pkl'
        )
    
    def _train_cox_ph(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame], event_type: str
    ) -> SurvivalResult:
        """Train Cox Proportional Hazards model."""
        import time
        start_time = time.time()
        
        # Prepare data for specific event (competing risk: treat other event as censored)
        train_event = train_df.copy()
        train_event['event_observed'] = (train_event['event_type'] == event_type).astype(int)
        
        val_event = val_df.copy()
        val_event['event_observed'] = (val_event['event_type'] == event_type).astype(int)
        
        # Select features
        feature_cols = [c for c in train_event.columns if c.endswith('_encoded') or 
                       c in ['original_balance', 'interest_rate', 'term_months',
                             'balance_ratio_at_event', 'dpd_at_event', 'ever_modified'] or
                       c.startswith(('state_', 'loan_purpose_', 'occupancy_type_', 
                                   'property_type_', 'servicer_name_'))]
        
        feature_cols = [c for c in feature_cols if c in train_event.columns]
        
        # Clean NaNs/Infs for lifelines (strict)
        for df_ in [train_event, val_event]:
            if feature_cols:
                df_[feature_cols] = df_[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            df_['duration'] = df_['duration'].replace([np.inf, -np.inf], np.nan).fillna(df_['duration'].median() if df_['duration'].notna().any() else 12)
            # Ensure duration >0
            df_['duration'] = df_['duration'].clip(lower=1)
        
        # Drop constant columns which cause lifelines issues
        if feature_cols:
            nunique = train_event[feature_cols].nunique()
            feature_cols = [c for c in feature_cols if nunique.get(c, 0) > 1]
            if not feature_cols:
                # Fallback dummy
                logger.warning(f"No varying features for Cox {event_type}, returning dummy")
                training_time = time.time() - start_time
                return SurvivalResult(
                    model_name=f'cox_{event_type}',
                    model_type='cox_ph',
                    event_type=event_type,
                    metrics=SurvivalMetrics(concordance_index=0.5, brier_score=0.25, log_likelihood=0.0, aic=0.0),
                    survival_curves={},
                    feature_importance={},
                    training_time_seconds=training_time,
                    model_path=f'models/survival/cox_{event_type}_dummy.pkl'
                )
        
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(train_event[feature_cols + ['duration', 'event_observed']], 
                    duration_col='duration', event_col='event_observed')
        except Exception as e:
            logger.warning(f"CoxPH {event_type} failed: {e}, returning dummy result")
            training_time = time.time() - start_time
            return SurvivalResult(
                model_name=f'cox_{event_type}',
                model_type='cox_ph',
                event_type=event_type,
                metrics=SurvivalMetrics(concordance_index=0.5, brier_score=0.25, log_likelihood=0.0, aic=0.0),
                survival_curves={},
                feature_importance={},
                training_time_seconds=training_time,
                model_path=f'models/survival/cox_{event_type}_dummy.pkl'
            )
        
        # Evaluate
        val_concordance = self._eval_concordance(cph, val_event, feature_cols)
        val_brier = self._eval_brier(cph, val_event, feature_cols)
        
        # Feature importance - handle lifelines version differences (hazards_ vs params_)
        try:
            if hasattr(cph, 'hazards_'):
                importance = dict(zip(feature_cols, cph.hazards_.iloc[0].abs().values))
            elif hasattr(cph, 'params_'):
                # params_ is DataFrame with coef per feature
                coef = cph.params_.iloc[:, 0] if isinstance(cph.params_, pd.DataFrame) else cph.params_
                importance = dict(zip(feature_cols, np.abs(coef.values).flatten()[:len(feature_cols)]))
            elif hasattr(cph, 'summary'):
                importance = dict(zip(feature_cols, np.abs(cph.summary['coef'].values).flatten()[:len(feature_cols)]))
            else:
                importance = {c: 0.0 for c in feature_cols}
        except Exception:
            importance = {c: 0.0 for c in feature_cols}
        
        # Save
        model_path = f'models/survival/cox_{event_type}.pkl'
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cph, model_path)
        
        training_time = time.time() - start_time
        
        # Robust log_likelihood / AIC
        try:
            ll = float(cph.log_likelihood_)
        except Exception:
            ll = 0.0
        try:
            aic = float(cph.AIC_)
        except Exception:
            aic = -2 * ll + 2 * len(feature_cols)
        return SurvivalResult(
            model_name=f'cox_{event_type}',
            model_type='cox_ph',
            event_type=event_type,
            metrics=SurvivalMetrics(
                concordance_index=val_concordance,
                brier_score=val_brier,
                log_likelihood=ll,
                aic=aic
            ),
            survival_curves=self._compute_segment_curves(cph, train_event, 'credit_score_band', feature_cols),
            feature_importance=importance,
            training_time_seconds=training_time,
            model_path=model_path
        )
    
    def _train_discrete_time(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame]
    ) -> SurvivalResult:
        """Train discrete-time monthly transition model (logistic regression per month)."""
        import time
        from sklearn.linear_model import LogisticRegression
        start_time = time.time()
        
        # Build monthly transition dataset - may raise NotImplementedError for stub
        try:
            train_trans = self._build_transition_data(train_df)
            val_trans = self._build_transition_data(val_df)
        except NotImplementedError as e:
            logger.warning(f"Discrete-time training skipped (stub data): {e}")
            training_time = time.time() - start_time
            return SurvivalResult(
                model_name='discrete_time_transition',
                model_type='discrete_time',
                event_type='transition',
                metrics=SurvivalMetrics(
                    concordance_index=0.5,
                    brier_score=0.25,
                    log_likelihood=0.0,
                    aic=0.0
                ),
                survival_curves={},
                feature_importance={},
                training_time_seconds=training_time,
                model_path='models/survival/discrete_time_dummy.pkl'
            )
        except KeyError as e:
            logger.warning(f"Discrete-time training skipped (missing column): {e}")
            training_time = time.time() - start_time
            return SurvivalResult(
                model_name='discrete_time_transition',
                model_type='discrete_time',
                event_type='transition',
                metrics=SurvivalMetrics(
                    concordance_index=0.5,
                    brier_score=0.25,
                    log_likelihood=0.0,
                    aic=0.0
                ),
                survival_curves={},
                feature_importance={},
                training_time_seconds=training_time,
                model_path='models/survival/discrete_time_dummy.pkl'
            )
        
        # Train discrete-time hazard model: predict event_at_t ~ covariates + time
        # This is binary logistic regression on the expanded dataset
        target_col = 'event_at_t'
        feature_cols = [c for c in train_trans.columns if c not in [
            'loan_id', 'duration', 'event', 'event_type', 'event_at_t',
            'to_default', 'to_prepayment', 'to_current', 'censored', 'next_state'
        ]]
        # Keep only numeric features
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_trans[c])]
        
        X_train = train_trans[feature_cols].fillna(0)
        y_train = train_trans[target_col].astype(int)
        
        if y_train.nunique() < 2:
            raise ValueError("Discrete-time target is constant (no events in training set)")
        
        model = LogisticRegression(max_iter=500,
                                   class_weight='balanced', random_state=42, n_jobs=-1, solver='lbfgs')
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        X_val = val_trans[feature_cols].fillna(0)
        y_val = val_trans['event_at_t'].astype(int)
        
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)
        
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_logloss = log_loss(y_val, y_val_proba) if y_val.nunique() > 1 else 0.5
        try:
            val_auc = roc_auc_score(y_val, y_val_proba[:, 1])
        except Exception:
            val_auc = 0.5
        
        # Feature importance (absolute logistic regression coefficients)
        importance = dict(zip(feature_cols, np.abs(model.coef_[0])))
        
        training_time = time.time() - start_time
        
        # Save model
        import joblib
        model_path = Path('models/survival/discrete_time.pkl')
        joblib.dump(model, model_path)
        
        logger.info(f"Discrete-time model: val AUC={val_auc:.4f}, val accuracy={val_accuracy:.4f}")
        
        return SurvivalResult(
            model_name='discrete_time_transition',
            model_type='discrete_time',
            event_type='transition',
            metrics=SurvivalMetrics(
                concordance_index=val_auc,
                brier_score=val_logloss,
                log_likelihood=-val_logloss * len(val_trans),
                aic=2 * len(feature_cols) - 2 * (-val_logloss * len(val_trans))
            ),
            survival_curves={},
            feature_importance=dict(sorted(importance.items(), key=lambda x: -x[1])[:20]),
            training_time_seconds=training_time,
            model_path=str(model_path)
        )
    
    def _build_transition_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build monthly transition dataset from loan-level survival data.
        
        Since we have loan-level survival_df (one row per loan), we construct a
        discrete-time expanded dataset where each loan contributes one row per
        month until the event or censoring, with a binary 'event_at_t' outcome.
        This is the standard discrete-time hazard model approach.
        """
        rows = []
        feature_cols = [c for c in df.columns if c not in [
            'loan_id', 'duration', 'event', 'event_type'
        ]]
        
        for _, loan in df.iterrows():
            duration = int(max(1, loan['duration']))
            event = int(loan['event'])
            
            for t in range(1, duration + 1):
                # At each time-step, did the event occur at exactly this month?
                event_at_t = 1 if (t == duration and event == 1) else 0
                
                row = {'t': t, 'log_t': np.log(t), 'event_at_t': event_at_t}
                
                # Add all feature columns (static covariates from survival_df)
                for col in feature_cols:
                    row[col] = loan[col]
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _eval_concordance(self, model: Any, df: pd.DataFrame, feature_cols: List[str] = None) -> float:
        """Evaluate concordance index."""
        try:
            if hasattr(model, 'predict_partial_hazard'):
                # Cox model
                hazard = model.predict_partial_hazard(df[feature_cols])
                return concordance_index(df['duration'], -hazard, df['event_observed'])
            else:
                # Kaplan-Meier
                surv_probs = model.survival_function_at_times(df['duration'].values)
                return concordance_index(df['duration'], -surv_probs.values.flatten(), df['event_observed'] if 'event_observed' in df.columns else (df['event'] > 0))
        except Exception:
            return 0.5
    
    def _eval_brier(self, model: Any, df: pd.DataFrame, feature_cols: List[str] = None) -> float:
        """Evaluate Brier score for survival."""
        try:
            if hasattr(model, 'predict_survival_function'):
                # Cox model
                surv_funcs = model.predict_survival_function(df[feature_cols])
                times = df['duration'].values
                event = (df['event_observed'] > 0).astype(int) if 'event_observed' in df.columns else (df['event'] > 0).astype(int)
                
                # Brier score at each time point
                scores = []
                for i, t in enumerate(times):
                    surv_prob = surv_funcs.iloc[i].get(t, surv_funcs.iloc[i].iloc[-1])
                    scores.append((surv_prob - (1 - event[i]))**2)
                return np.mean(scores)
            else:
                # Kaplan-Meier
                surv_probs = model.survival_function_at_times(df['duration'].values)
                event = (df['event'] > 0).astype(int)
                return np.mean((surv_probs.values.flatten() - (1 - event))**2)
        except Exception:
            return 0.25
    
    def _compute_segment_curves(
        self, model: Any, df: pd.DataFrame, segment_col: str, feature_cols: List[str] = None
    ) -> Dict[str, Any]:
        """Compute survival curves by segment."""
        curves = {}
        
        if segment_col not in df.columns:
            return curves
        
        for segment in df[segment_col].unique():
            seg_df = df[df[segment_col] == segment]
            if len(seg_df) < 10:
                continue
            
            if hasattr(model, 'predict_survival_function') and feature_cols:
                try:
                    surv_funcs = model.predict_survival_function(seg_df[feature_cols])
                    # surv_funcs is DataFrame with time index, columns = patients
                    # Compute median survival: time where survival ~0.5
                    # Use median of durations where survival crosses 0.5, or fallback to mean
                    if hasattr(surv_funcs, 'median_'):
                        median_surv = surv_funcs.median_
                    else:
                        # DataFrame case: median across patients
                        try:
                            # Median survival time per patient, then median across patients
                            median_per_patient = surv_funcs.apply(lambda col: col[col <= 0.5].index.min() if (col <= 0.5).any() else col.index.max(), axis=0)
                            median_surv = float(median_per_patient.median())
                        except Exception:
                            median_surv = float(seg_df['duration'].median())
                    curves[str(segment)] = {
                        'median_survival': float(median_surv) if median_surv is not None and not (isinstance(median_surv, float) and np.isnan(median_surv)) else None,
                        'n_loans': len(seg_df)
                    }
                except Exception as e:
                    logger.warning(f"Survival curve median failed for {segment}: {e}")
                    curves[str(segment)] = {
                        'median_survival': float(seg_df['duration'].median()),
                        'n_loans': len(seg_df)
                    }
            else:
                kmf = KaplanMeierFitter()
                kmf.fit(seg_df['duration'], event_observed=(seg_df['event'] > 0).astype(int))
                curves[str(segment)] = {
                    'median_survival': float(kmf.median_survival_time_) if kmf.median_survival_time_ else None,
                    'n_loans': len(seg_df)
                }
        
        return curves
    
    def save_results(self, output_dir: str = "reports/survival"):
        """Save survival modeling results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {}
        for name, result in self.results.items():
            summary[name] = asdict(result)
            # Convert curves to serializable
            summary[name]['survival_curves'] = result.survival_curves
        
        with open(output_dir / "survival_results.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved survival results to {output_dir}")


def train_survival_models(
    survival_df: pd.DataFrame,
    train_mask: pd.Series, val_mask: pd.Series, test_mask: Optional[pd.Series] = None,
    config: Optional[Dict] = None
) -> SurvivalModeler:
    """Train all survival models."""
    builder = SurvivalDataBuilder(config)
    modeler = SurvivalModeler(config)
    modeler.train_all(survival_df, train_mask, val_mask, test_mask)
    modeler.save_results()
    return modeler