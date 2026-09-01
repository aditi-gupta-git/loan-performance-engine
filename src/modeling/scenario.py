"""Scenario and stress simulation module."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from dataclasses import dataclass, asdict
from copy import deepcopy

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScenarioResult:
    """Results of scenario simulation."""
    scenario_name: str
    aggregate_projections: Dict[str, float]
    segment_projections: Dict[str, Dict[str, float]]
    driver_explanations: Dict[str, Any]
    feature_shifts: Dict[str, float]


class ScenarioSimulator:
    """Apply macro scenarios and re-score portfolio."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.scenarios_config = self.config.get('scenarios', {})
        self.scenarios = self._load_scenarios()
    
    def _load_scenarios(self) -> Dict[str, Dict]:
        """Load scenario definitions from config."""
        scenarios = {}
        for name, params in self.scenarios_config.items():
            if isinstance(params, dict) and 'rate_shift' in params:
                scenarios[name] = params
        return scenarios
    
    def apply_scenario(
        self, df: pd.DataFrame, scenario_name: str,
        models: Dict[str, Any], feature_engineer: Any
    ) -> pd.DataFrame:
        """
        Apply scenario to features and re-score.
        Returns dataframe with scenario predictions.
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.scenarios[scenario_name]
        df_scenario = df.copy()
        
        # Apply feature perturbations
        df_scenario = self._perturb_features(df_scenario, scenario)
        
        # Re-engineer features (rolling stats will change)
        df_scenario = feature_engineer.transform(df_scenario)
        
        # Select only features the models were trained on (handle new perturbed columns)
        expected_features = feature_engineer.fitted_encoders.get('feature_columns', [])
        logger.info(f"Scenario apply: expected_features={len(expected_features)}, df_scenario cols={len(df_scenario.columns)}")
        if expected_features:
            missing = [c for c in expected_features if c not in df_scenario.columns]
            if missing:
                logger.warning(f"Missing {len(missing)} expected features, filling with 0")
                for c in missing:
                    df_scenario[c] = 0
            extra = [c for c in df_scenario.columns if c not in expected_features]
            if extra:
                logger.warning(f"Dropping {len(extra)} extra features: {extra[:5]}...")
            df_scenario = df_scenario[expected_features]
            logger.info(f"After selection: {len(df_scenario.columns)} features")
        else:
            logger.warning("No expected_features found in fitted_encoders")
        
        # Re-score with all models
        for target, model_info in models.items():
            if hasattr(model_info, 'model_path'):
                # Load model
                import joblib
                model = joblib.load(model_info.model_path)
            else:
                model = model_info
            
            # Disable LightGBM feature shape check for scenario data (features match but LightGBM is strict)
            if hasattr(model, 'set_params'):
                try:
                    model.set_params(predict_disable_shape_check=True)
                except Exception:
                    pass
            # Also try on base estimator if calibrated
            if hasattr(model, 'calibrated_classifiers_'):
                for cal in model.calibrated_classifiers_:
                    if hasattr(cal.estimator, 'set_params'):
                        try:
                            cal.estimator.set_params(predict_disable_shape_check=True)
                        except Exception:
                            pass
            
            # Re-select expected_features each iteration so that scenario prediction columns
            # added in prior loop iterations do not reach predict_proba (which validates
            # feature names strictly for CalibratedClassifierCV wrappers).
            predict_input = df_scenario[expected_features] if expected_features else df_scenario
            try:
                proba = model.predict_proba(predict_input)
            except Exception as _e:
                logger.warning(f"predict_proba failed for {target} in scenario {scenario_name}: {_e}")
                continue
            if proba.shape[1] > 1:
                df_scenario[f'{target}_scenario_{scenario_name}'] = proba[:, 1]
            else:
                df_scenario[f'{target}_scenario_{scenario_name}'] = proba[:, 0]
        
        return df_scenario
    
    def _perturb_features(self, df: pd.DataFrame, scenario: Dict) -> pd.DataFrame:
        """Apply scenario-specific feature perturbations."""
        df = df.copy()
        
        rate_shift = scenario.get('rate_shift', 0.0)
        credit_deterioration = scenario.get('credit_deterioration', 0.0)
        prepayment_multiplier = scenario.get('prepayment_multiplier', 1.0)
        default_multiplier = scenario.get('default_multiplier', 1.0)
        hpi_delta = scenario.get('hpi_delta', 0.0)
        unemployment_delta = scenario.get('unemployment_delta', 0.0)
        
        # Interest rate shift affects:
        # - Current rate (for adjustable rate mortgages, but we'll assume fixed)
        # - Prepayment incentive (lower rates -> higher prepayment)
        # - Refinance incentive
        
        if 'interest_rate' in df.columns:
            # For scenario analysis, shift the rate environment
            # This affects prepayment probability
            df['rate_environment'] = df['interest_rate'] + rate_shift
        
        # Credit deterioration: increase delinquency probability
        # Proxy: increase days_past_due, worsen status
        if credit_deterioration > 0:
            # Shift some current loans to delinquent
            current_mask = df['current_status'] == 'Current'
            n_shift = int(current_mask.sum() * credit_deterioration)
            if n_shift > 0:
                shift_indices = df[current_mask].sample(n=n_shift, random_state=42).index
                new_statuses = np.random.choice(
                    ['30-59 DPD', '60-89 DPD'], 
                    size=n_shift, p=[0.7, 0.3]
                )
                df.loc[shift_indices, 'current_status'] = new_statuses
                df.loc[shift_indices, 'days_past_due'] = np.random.randint(30, 90, n_shift)
                df.loc[shift_indices, 'is_delinquent'] = 1
                df.loc[shift_indices, 'is_severe_dq'] = (df.loc[shift_indices, 'current_status'] == '60-89 DPD').astype(int)
        
        # HPI delta affects LTV and prepayment
        if hpi_delta != 0 and 'ltv_band' in df.columns:
            # Negative HPI -> higher effective LTV
            # This is complex; simplified: adjust balance ratio
            if 'balance_ratio' in df.columns:
                df['balance_ratio'] = df['balance_ratio'] / (1 + hpi_delta)
                df['balance_ratio'] = df['balance_ratio'].clip(0, 2)
        
        # Unemployment delta affects default probability
        # Proxy: increase modification flags, worsen status
        if unemployment_delta > 0:
            mod_mask = df['modification_flag'] == 0
            n_mod = int(mod_mask.sum() * unemployment_delta * 0.5)
            if n_mod > 0:
                mod_indices = df[mod_mask].sample(n=n_mod, random_state=42).index
                df.loc[mod_indices, 'modification_flag'] = 1
                df.loc[mod_indices, 'ever_modified'] = 1
        
        # Prepayment multiplier: adjust features that drive prepayment
        if prepayment_multiplier != 1.0:
            # Lower rate environment -> higher prepayment
            # This is captured via rate_environment feature
            pass
        
        # Default multiplier: similar to credit deterioration
        if default_multiplier != 1.0 and default_multiplier > 1.0:
            extra_deterioration = (default_multiplier - 1.0) * 0.1
            current_mask = df['current_status'] == 'Current'
            n_shift = int(current_mask.sum() * extra_deterioration)
            if n_shift > 0:
                shift_indices = df[current_mask].sample(n=n_shift, random_state=42).index
                df.loc[shift_indices, 'current_status'] = '30-59 DPD'
                df.loc[shift_indices, 'days_past_due'] = np.random.randint(30, 60, n_shift)
                df.loc[shift_indices, 'is_delinquent'] = 1
        
        return df
    
    def run_all_scenarios(
        self, df: pd.DataFrame, models: Dict[str, Any], 
        feature_engineer: Any
    ) -> Dict[str, ScenarioResult]:
        """Run all scenarios and compute projections."""
        results = {}
        
        for scenario_name in self.scenarios:
            logger.info(f"Running scenario: {scenario_name}")
            df_scenario = self.apply_scenario(df, scenario_name, models, feature_engineer)
            
            # Compute aggregate projections
            aggregate = self._compute_aggregate_projections(df_scenario, scenario_name)
            
            # Compute segment projections
            segments = self._compute_segment_projections(df_scenario, scenario_name, original_df=df)
            
            # Driver explanations
            drivers = self._compute_driver_explanations(df, df_scenario, scenario_name)
            
            # Feature shifts
            shifts = self._compute_feature_shifts(df, df_scenario)
            
            results[scenario_name] = ScenarioResult(
                scenario_name=scenario_name,
                aggregate_projections=aggregate,
                segment_projections=segments,
                driver_explanations=drivers,
                feature_shifts=shifts
            )
        
        return results
    
    def _compute_aggregate_projections(
        self, df: pd.DataFrame, scenario_name: str
    ) -> Dict[str, float]:
        """Compute portfolio-level projections."""
        projections = {}
        
        # Delinquency rates
        for target in ['next_3m_delinquency_flag', 'next_6m_delinquency_flag', 
                       'next_12m_default_flag', 'next_12m_prepayment_flag']:
            col = f'{target}_scenario_{scenario_name}'
            if col in df.columns:
                projections[f'{target}_rate'] = float(df[col].mean())
                projections[f'{target}_count'] = int((df[col] > 0.5).sum())
        
        # Average probabilities
        for target in ['next_3m_delinquency_flag', 'next_6m_delinquency_flag',
                       'next_12m_default_flag', 'next_12m_prepayment_flag']:
            col = f'{target}_scenario_{scenario_name}'
            if col in df.columns:
                projections[f'avg_{target}_prob'] = float(df[col].mean())
        
        return projections
    
    def _compute_segment_projections(
        self, df: pd.DataFrame, scenario_name: str, original_df: pd.DataFrame = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute segment-level projections by joining scenario predictions back to segment columns."""
        segments = {}
        
        # Scenario df has predictions but not the original categorical segment columns
        # (they got replaced by encoded numerics during feature engineering).
        # If original_df is supplied, join on index to recover segment labels.
        if original_df is not None and len(original_df) == len(df):
            work_df = df.copy()
            for col in ['credit_score_band', 'ltv_band', 'dti_band', 'state',
                       'loan_purpose', 'servicer_name']:
                if col in original_df.columns:
                    work_df[col] = original_df[col].values
        else:
            work_df = df
        
        segment_cols = ['credit_score_band', 'ltv_band', 'dti_band', 'state',
                       'loan_purpose', 'servicer_name']
        
        for seg_col in segment_cols:
            if seg_col not in work_df.columns:
                continue
            
            segments[seg_col] = {}
            for segment_val, seg_df in work_df.groupby(seg_col):
                seg_projections = {}
                for target in ['next_3m_delinquency_flag', 'next_6m_delinquency_flag',
                              'next_12m_default_flag', 'next_12m_prepayment_flag']:
                    col = f'{target}_scenario_{scenario_name}'
                    if col in seg_df.columns:
                        seg_projections[f'{target}_rate'] = float(seg_df[col].mean())
                        seg_projections[f'{target}_count'] = int((seg_df[col] > 0.5).sum())
                segments[seg_col][str(segment_val)] = seg_projections
        
        return segments
    
    def _compute_driver_explanations(
        self, base_df: pd.DataFrame, scenario_df: pd.DataFrame, scenario_name: str
    ) -> Dict[str, Any]:
        """Explain which features drove scenario changes."""
        drivers = {}
        
        # Compare feature distributions
        numeric_cols = base_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in scenario_df.columns:
                base_mean = base_df[col].mean()
                scen_mean = scenario_df[col].mean()
                if base_mean != 0:
                    pct_change = (scen_mean - base_mean) / abs(base_mean) * 100
                else:
                    pct_change = 0
                if abs(pct_change) > 1:  # Only significant changes
                    drivers[col] = {
                        'base_mean': float(base_mean),
                        'scenario_mean': float(scen_mean),
                        'pct_change': float(pct_change)
                    }
        
        # Prediction changes
        pred_drivers = {}
        for target in ['next_3m_delinquency_flag', 'next_6m_delinquency_flag',
                      'next_12m_default_flag', 'next_12m_prepayment_flag']:
            base_col = target
            scen_col = f'{target}_scenario_{scenario_name}'
            if base_col in base_df.columns and scen_col in scenario_df.columns:
                base_pred = base_df[base_col].mean() if base_df[base_col].dtype in [np.float64, np.int64] else 0
                scen_pred = scenario_df[scen_col].mean()
                pred_drivers[target] = {
                    'base_rate': float(base_pred),
                    'scenario_rate': float(scen_pred),
                    'absolute_change': float(scen_pred - base_pred),
                    'relative_change': float((scen_pred - base_pred) / base_pred * 100) if base_pred != 0 else 0
                }
        
        drivers['predictions'] = pred_drivers
        
        return drivers
    
    def _compute_feature_shifts(
        self, base_df: pd.DataFrame, scenario_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute feature distribution shifts (PSI-like)."""
        shifts = {}
        numeric_cols = base_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in scenario_df.columns:
                # Simple mean shift
                base_mean = base_df[col].mean()
                scen_mean = scenario_df[col].mean()
                if base_mean != 0:
                    shifts[col] = float((scen_mean - base_mean) / base_mean * 100)
                else:
                    shifts[col] = 0.0
        
        return shifts
    
    def compare_scenarios(
        self, results: Dict[str, ScenarioResult]
    ) -> pd.DataFrame:
        """Create comparison table across scenarios."""
        comparison_rows = []
        
        for scenario_name, result in results.items():
            row = {'scenario': scenario_name}
            row.update(result.aggregate_projections)
            comparison_rows.append(row)
        
        return pd.DataFrame(comparison_rows)
    
    def save_results(
        self, results: Dict[str, ScenarioResult], output_dir: str = "reports/scenario"
    ):
        """Save scenario results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        serializable = {}
        for name, result in results.items():
            serializable[name] = asdict(result)
        
        with open(output_dir / "scenario_results.json", 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        
        # Save comparison table
        comparison_df = self.compare_scenarios(results)
        comparison_df.to_csv(output_dir / "scenario_comparison.csv", index=False)
        
        logger.info(f"Saved scenario results to {output_dir}")


def run_scenario_simulation(
    df: pd.DataFrame, models: Dict[str, Any], feature_engineer: Any,
    config: Optional[Dict] = None
) -> Dict[str, ScenarioResult]:
    """Run complete scenario simulation."""
    simulator = ScenarioSimulator(config)
    results = simulator.run_all_scenarios(df, models, feature_engineer)
    simulator.save_results(results)
    return results