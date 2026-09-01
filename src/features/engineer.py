"""Feature engineering pipeline for loan performance modeling."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import json

from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


@dataclass
class FeatureManifest:
    """Manifest documenting all features."""
    features: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_feature(self, name: str, source: str, transformation: str, 
                    as_of_logic: str, dtype: str, description: str = ""):
        self.features.append({
            "name": name,
            "source": source,
            "transformation": transformation,
            "as_of_logic": as_of_logic,
            "dtype": dtype,
            "description": description
        })
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.features)
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)


class LeakageSafeFeatureEngineer:
    """
    Feature engineering pipeline with strict leakage prevention.
    All features computed using only data available at prediction time.
    """
    
    # Column groups
    STATIC_COLS = [
        'original_balance', 'interest_rate', 'credit_score_band', 'ltv_band',
        'dti_band', 'state', 'loan_purpose', 'occupancy_type', 'property_type',
        'servicer_name', 'origination_month', 'term_months'
    ]
    
    DYNAMIC_COLS = [
        'loan_age_months', 'remaining_term_months', 'current_balance',
        'current_status', 'days_past_due', 'modification_flag',
        'prepayment_flag', 'default_flag', 'loss_severity_band',
        'document_status'
    ]
    
    TARGET_COLS = [
        'next_3m_delinquency_flag', 'next_6m_delinquency_flag',
        'next_12m_default_flag', 'next_12m_prepayment_flag',
        'next_state', 'exception_required', 'exception_type'
    ]
    
    ID_COLS = ['loan_id', 'month_index', 'reporting_month', 'last_updated_at', 'source_system']
    
    # Categorical encodings (band-aware ordinal)
    CREDIT_BAND_ORDER = ['<620', '620-659', '660-699', '700-739', '740-779', '780+']
    LTV_BAND_ORDER = ['<60%', '60-70%', '70-80%', '80-90%', '90-100%', '>100%']
    DTI_BAND_ORDER = ['<20%', '20-30%', '30-36%', '36-43%', '43-50%', '>50%']
    STATUS_ORDER = ['Current', '30-59 DPD', '60-89 DPD', '90+ DPD', 'Prepaid', 'Closed', 'Defaulted']
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.feature_config = self.config.get('features', {})
        self.manifest = FeatureManifest()
        self.fitted_encoders = {}
        self.fitted_scalers = {}
        self._initialize_manifest()
    
    def _initialize_manifest(self):
        """Initialize feature manifest with base features."""
        # Static features
        for col in self.STATIC_COLS:
            self.manifest.add_feature(
                name=col, source="static", transformation="passthrough",
                as_of_logic="origination", dtype="original", description=f"Original {col}"
            )
        
        # Dynamic base features
        for col in self.DYNAMIC_COLS:
            self.manifest.add_feature(
                name=col, source="panel", transformation="passthrough",
                as_of_logic="reporting_month", dtype="original", description=f"Current {col}"
            )
    
    def fit_transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Fit and transform training data."""
        logger.info(f"Fitting feature pipeline on {len(df)} rows")
        df = df.copy()
        df = self._ensure_sorted(df)
        
        # Encode categorical
        df = self._encode_categorical(df, fit=is_train)
        
        # Create engineered features
        df = self._create_engineered_features(df)
        
        # Create rolling/lag features (groupby loan_id)
        df = self._create_temporal_features(df)
        
        # Select final feature columns
        feature_cols = self._get_feature_columns(df)
        
        if is_train:
            self._save_artifacts(feature_cols)
        
        logger.info(f"Generated {len(feature_cols)} features")
        return df[feature_cols]
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted transformers."""
        logger.info(f"Transforming {len(df)} rows")
        if len(df) == 0:
            # Return empty with expected feature columns
            feature_cols = self.fitted_encoders.get('feature_columns', [])
            return pd.DataFrame(columns=feature_cols)
        df = df.copy()
        df = self._ensure_sorted(df)
        
        df = self._encode_categorical(df, fit=False)
        df = self._create_engineered_features(df)
        df = self._create_temporal_features(df)
        
        feature_cols = self._get_feature_columns(df)
        # Ensure all train features present
        for col in self.fitted_encoders.get('feature_columns', feature_cols):
            if col not in df.columns:
                df[col] = 0  # or appropriate default
        
        return df[self.fitted_encoders.get('feature_columns', feature_cols)]
    
    def _ensure_sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data is sorted by loan_id and month_index for temporal features."""
        if 'loan_id' in df.columns and 'month_index' in df.columns:
            return df.sort_values(['loan_id', 'month_index']).reset_index(drop=True)
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables with band-aware ordinal encoding."""
        df = df.copy()
        
        # Ordinal encodings for bands
        band_mappings = {
            'credit_score_band': self.CREDIT_BAND_ORDER,
            'ltv_band': self.LTV_BAND_ORDER,
            'dti_band': self.DTI_BAND_ORDER,
            'current_status': self.STATUS_ORDER
        }
        
        for col, order in band_mappings.items():
            if col in df.columns:
                mapping = {v: i for i, v in enumerate(order)}
                if fit:
                    self.fitted_encoders[col] = mapping
                else:
                    mapping = self.fitted_encoders.get(col, mapping)
                
                encoded_col = f"{col}_encoded"
                df[encoded_col] = df[col].map(mapping).fillna(-1).astype(int)
                
                if fit:
                    self.manifest.add_feature(
                        name=encoded_col, source=f"{col}", transformation="ordinal_encoding",
                        as_of_logic="origination" if 'band' in col else "reporting_month",
                        dtype="int", description=f"Ordinal encoded {col}"
                    )
        
        # One-hot for other categoricals (low cardinality)
        nominal_cols = ['state', 'loan_purpose', 'occupancy_type', 'property_type', 'servicer_name']
        for col in nominal_cols:
            if col in df.columns:
                if fit:
                    dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                    self.fitted_encoders[f"{col}_categories"] = dummies.columns.tolist()
                else:
                    expected = self.fitted_encoders.get(f"{col}_categories", [])
                    dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                    # Align with training columns
                    for exp_col in expected:
                        if exp_col not in dummies.columns:
                            dummies[exp_col] = 0
                    dummies = dummies[expected]
                
                df = pd.concat([df, dummies], axis=1)
                
                if fit:
                    for dummy_col in dummies.columns:
                        self.manifest.add_feature(
                            name=dummy_col, source=col, transformation="onehot",
                            as_of_logic="origination", dtype="int",
                            description=f"One-hot for {col}"
                        )
        
        return df
    
    def _create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from base columns."""
        df = df.copy()
        
        # Balance ratios
        if 'current_balance' in df.columns and 'original_balance' in df.columns:
            df['balance_ratio'] = df['current_balance'] / df['original_balance'].replace(0, np.nan)
            df['balance_paid_pct'] = 1 - df['balance_ratio']
            self._add_manifest('balance_ratio', 'current_balance/original_balance', 'ratio', 'reporting_month', 'float', 'Current/original balance')
            self._add_manifest('balance_paid_pct', 'current_balance/original_balance', 'ratio', 'reporting_month', 'float', 'Pct of balance paid')
        
        # Amortization progress
        if 'loan_age_months' in df.columns and 'term_months' in df.columns:
            df['amortization_progress'] = df['loan_age_months'] / df['term_months'].replace(0, np.nan)
            self._add_manifest('amortization_progress', 'loan_age/term', 'ratio', 'reporting_month', 'float', 'Loan age as fraction of term')
        
        # Rate environment proxy
        if 'interest_rate' in df.columns:
            # This would ideally use market rate at reporting_month
            df['rate_level'] = pd.qcut(df['interest_rate'], q=5, labels=False, duplicates='drop')
            self._add_manifest('rate_level', 'interest_rate', 'quantile_bin', 'origination', 'int', 'Interest rate quintile')
        
        # Delinquency indicators
        if 'current_status' in df.columns:
            df['is_delinquent'] = df['current_status'].isin(['30-59 DPD', '60-89 DPD', '90+ DPD']).astype(int)
            df['is_severe_dq'] = df['current_status'].isin(['60-89 DPD', '90+ DPD']).astype(int)
            df['is_terminal'] = df['current_status'].isin(['Prepaid', 'Closed', 'Defaulted']).astype(int)
            for c in ['is_delinquent', 'is_severe_dq', 'is_terminal']:
                self._add_manifest(c, 'current_status', 'indicator', 'reporting_month', 'int', f'Flag for {c}')
        
        # DPD buckets
        if 'days_past_due' in df.columns:
            df['dpd_30_plus'] = (df['days_past_due'] >= 30).astype(int)
            df['dpd_60_plus'] = (df['days_past_due'] >= 60).astype(int)
            df['dpd_90_plus'] = (df['days_past_due'] >= 90).astype(int)
            df['log_dpd'] = np.log1p(df['days_past_due'].clip(lower=0))
            for c in ['dpd_30_plus', 'dpd_60_plus', 'dpd_90_plus', 'log_dpd']:
                self._add_manifest(c, 'days_past_due', 'bucket/log', 'reporting_month', 'float' if c == 'log_dpd' else 'int', f'DPD feature: {c}')
        
        # Modification history
        if 'modification_flag' in df.columns:
            df['ever_modified'] = df.groupby('loan_id')['modification_flag'].transform(lambda x: x.astype(int).cummax())
            self._add_manifest('ever_modified', 'modification_flag', 'cummax', 'reporting_month', 'int', 'Ever modified flag')
        
        # Seasonality
        if 'reporting_month' in df.columns:
            # Handle Period dtype (convert to timestamp for dt accessor)
            reporting_month_series = df['reporting_month']
            if isinstance(reporting_month_series.dtype, pd.PeriodDtype):
                reporting_month_series = reporting_month_series.dt.to_timestamp()
            else:
                try:
                    reporting_month_series = pd.to_datetime(reporting_month_series)
                except Exception:
                    pass
            df['month_of_year'] = reporting_month_series.dt.month
            df['quarter'] = reporting_month_series.dt.quarter
            df['is_year_end'] = df['month_of_year'].isin([11, 12]).astype(int)
            for c in ['month_of_year', 'quarter', 'is_year_end']:
                self._add_manifest(c, 'reporting_month', 'datetime_extract', 'reporting_month', 'int', f'Seasonality: {c}')
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling and lag features within each loan (vectorized, fast)."""
        df = df.copy()
        
        rolling_windows = self.feature_config.get('rolling_windows', [3, 6, 12])
        lag_periods = self.feature_config.get('lag_periods', [1, 2, 3, 6])
        
        # Features to create rolling stats for
        roll_features = [
            'current_balance', 'days_past_due', 'is_delinquent',
            'is_severe_dq', 'modification_flag', 'balance_ratio'
        ]
        roll_features = [f for f in roll_features if f in df.columns]
        
        # Build all new columns in a dict first to avoid repeated assignment
        new_cols: dict = {}
        
        # Vectorized approach: sort once, use pandas groupby on the sorted frame
        # All operations below are groupby-transform which pandas executes efficiently
        loan_groups = df.groupby('loan_id', sort=False)
        
        for feat in roll_features:
            if feat not in df.columns:
                continue
            # Lag 1 (shifted series for rolling)
            shifted = loan_groups[feat].shift(1)
            
            for window in rolling_windows:
                # Rolling mean, std, max on the already-shifted series
                # Using Series.groupby().transform avoids re-grouping per window
                grouped_shifted = shifted.groupby(df['loan_id'])
                new_cols[f'{feat}_rollmean_{window}'] = grouped_shifted.transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                new_cols[f'{feat}_rollstd_{window}'] = grouped_shifted.transform(
                    lambda x: x.rolling(window, min_periods=2).std()
                ).fillna(0)
                new_cols[f'{feat}_rollmax_{window}'] = grouped_shifted.transform(
                    lambda x: x.rolling(window, min_periods=1).max()
                )
        
        # Lag features
        lag_features = ['current_balance', 'days_past_due', 'current_status_encoded',
                       'is_delinquent', 'modification_flag', 'balance_ratio']
        lag_features = [f for f in lag_features if f in df.columns]
        
        for feat in lag_features:
            for lag in lag_periods:
                new_cols[f'{feat}_lag{lag}'] = loan_groups[feat].shift(lag)
        
        # Trend: balance change over 3 months
        if 'current_balance' in df.columns:
            bal_3m_ago = loan_groups['current_balance'].shift(3)
            new_cols['balance_3m_change'] = df['current_balance'] - bal_3m_ago
            new_cols['balance_3m_pct_change'] = new_cols['balance_3m_change'] / bal_3m_ago.replace(0, np.nan)
        
        # Delinquency streak — vectorized via cumsum trick
        if 'is_delinquent' in df.columns:
            is_dq = df['is_delinquent'].astype(float)
            # Group changes within each loan
            group_key = df['loan_id']
            changed = is_dq.groupby(group_key).transform(lambda x: (x != x.shift()).cumsum())
            new_cols['dq_streak'] = (
                is_dq * is_dq.groupby([group_key, changed]).cumcount().add(1)
            )
            new_cols['months_since_dq'] = (
                is_dq.eq(0).groupby([group_key, is_dq.groupby(group_key).transform(
                    lambda x: x.eq(1).cumsum()
                )]).cumcount()
            )
        
        # Add manifest entries for all new features
        for col_name in new_cols:
            parts = col_name.split('_')
            self._add_manifest(col_name, parts[0], col_name, 'reporting_month', 'float', col_name)
        
        # Assign all columns at once (much faster than repeated df[col] = )
        new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        # Fill NaN from lag/rolling
        new_df = new_df.fillna(0)
        
        return new_df
    
    def _add_manifest(self, name: str, source: str, transformation: str, 
                      as_of_logic: str, dtype: str, description: str):
        """Add feature to manifest if not already present."""
        if not any(f['name'] == name for f in self.manifest.features):
            self.manifest.add_feature(name, source, transformation, as_of_logic, dtype, description)
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (exclude IDs and targets)."""
        exclude = set(self.ID_COLS + self.TARGET_COLS + 
                     ['origination_month', 'credit_score_band', 'ltv_band', 'dti_band', 'current_status',
                      'state', 'loan_purpose', 'occupancy_type', 'property_type', 'servicer_name',
                      'loss_severity_band', 'document_status'])
        # Only keep numeric columns, drop Period/datetime/object remnants
        feature_cols = []
        for c in df.columns:
            if c in exclude:
                continue
            dtype = df[c].dtype
            # Exclude period, datetime, and object types (except encoded numeric)
            if isinstance(dtype, pd.PeriodDtype) or pd.api.types.is_datetime64_any_dtype(dtype) or dtype == object:
                continue
            if isinstance(dtype, pd.CategoricalDtype):
                continue
            feature_cols.append(c)
        return feature_cols
    
    def _save_artifacts(self, feature_cols: Optional[List[str]] = None):
        """Save fitted transformers and manifest."""
        artifacts_dir = Path("models/feature_engineering")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Store feature columns if provided
        if feature_cols is not None:
            self.fitted_encoders['feature_columns'] = feature_cols
        elif 'feature_columns' not in self.fitted_encoders:
            # Fallback: try to infer, but prefer explicit
            self.fitted_encoders['feature_columns'] = []
        
        joblib.dump(self.fitted_encoders, artifacts_dir / "encoders.pkl")
        joblib.dump(self.fitted_scalers, artifacts_dir / "scalers.pkl")
        self.manifest.save(artifacts_dir / "feature_manifest.csv")
        
        logger.info(f"Saved feature engineering artifacts to {artifacts_dir}")
    
    def load_artifacts(self, artifacts_dir: str = "models/feature_engineering"):
        """Load fitted transformers and manifest."""
        artifacts_dir = Path(artifacts_dir)
        self.fitted_encoders = joblib.load(artifacts_dir / "encoders.pkl")
        self.fitted_scalers = joblib.load(artifacts_dir / "scalers.pkl")
        logger.info(f"Loaded feature engineering artifacts from {artifacts_dir}")


def build_feature_pipeline(config: Optional[Dict] = None) -> LeakageSafeFeatureEngineer:
    """Factory function to create feature engineer."""
    return LeakageSafeFeatureEngineer(config)