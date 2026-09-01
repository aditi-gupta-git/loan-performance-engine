"""Synthetic data generator for Loan Performance Intelligence Engine."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import json

from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed, get_rng

logger = get_logger(__name__)


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    n_loans: int = 50000
    n_months: int = 36
    missing_rate: float = 0.05
    outlier_rate: float = 0.02
    conflict_rate: float = 0.03
    label_noise_rate: float = 0.01
    random_seed: int = 42
    
    # Distributions
    credit_score_bands: List[str] = None
    ltv_bands: List[str] = None
    dti_bands: List[str] = None
    states: List[str] = None
    loan_purposes: List[str] = None
    occupancy_types: List[str] = None
    property_types: List[str] = None
    servicers: List[str] = None
    statuses: List[str] = None
    doc_statuses: List[str] = None
    
    def __post_init__(self):
        if self.credit_score_bands is None:
            self.credit_score_bands = ['<620', '620-659', '660-699', '700-739', '740-779', '780+']
        if self.ltv_bands is None:
            self.ltv_bands = ['<60%', '60-70%', '70-80%', '80-90%', '90-100%', '>100%']
        if self.dti_bands is None:
            self.dti_bands = ['<20%', '20-30%', '30-36%', '36-43%', '43-50%', '>50%']
        if self.states is None:
            self.states = ['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI',
                          'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI']
        if self.loan_purposes is None:
            self.loan_purposes = ['Purchase', 'Refinance', 'Cash-out Refinance']
        if self.occupancy_types is None:
            self.occupancy_types = ['Primary', 'Second Home', 'Investment']
        if self.property_types is None:
            self.property_types = ['SFR', 'Condo', '2-4 Unit', 'Manufactured']
        if self.servicers is None:
            self.servicers = ['Servicer_A', 'Servicer_B', 'Servicer_C', 'Servicer_D', 'Servicer_E']
        if self.statuses is None:
            self.statuses = ['Current', '30-59 DPD', '60-89 DPD', '90+ DPD', 'Prepaid', 'Closed', 'Defaulted']
        if self.doc_statuses is None:
            self.doc_statuses = ['Complete', 'Incomplete', 'Missing']


class SyntheticDataGenerator:
    """Generates schema-faithful synthetic loan performance data."""
    
    def __init__(self, config: Optional[SyntheticConfig] = None):
        self.config = config or SyntheticConfig()
        self.rng = get_rng(self.config.random_seed)
        set_global_seed(self.config.random_seed)
        
        # Load validation rules for reference
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> List[Dict]:
        """Load validation rules to understand constraints."""
        rules_path = Path("config/validation_rules.json")
        if rules_path.exists():
            with open(rules_path) as f:
                return json.load(f).get("rules", [])
        return []
    
    def generate_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate all required datasets."""
        logger.info(f"Generating synthetic data: {self.config.n_loans} loans, {self.config.n_months} months")
        
        # Generate static attributes first
        static_df = self._generate_static_attributes()
        
        # Generate monthly panel data
        train_df, test_df = self._generate_panel_data(static_df)
        
        # Generate servicer updates
        servicer_df = self._generate_servicer_updates(train_df)
        
        # Inject data quality issues
        train_df = self._inject_quality_issues(train_df)
        test_df = self._inject_quality_issues(test_df)
        servicer_df = self._inject_servicer_conflicts(servicer_df, train_df)
        
        # Save to disk
        self._save_synthetic_data(train_df, test_df, static_df, servicer_df)
        
        return train_df, test_df, static_df, servicer_df
    
    def _generate_static_attributes(self) -> pd.DataFrame:
        """Generate loan-level static attributes."""
        n = self.config.n_loans
        
        # Loan IDs
        loan_ids = [f"LOAN_{i:08d}" for i in range(n)]
        
        # Origination dates - spread over 3 years before observation window
        origination_months = pd.period_range('2020-01', periods=36, freq='M')
        origination_month = self.rng.choice(origination_months, size=n)
        
        # Term: 15, 20, 30 years
        term_months = self.rng.choice([180, 240, 360], size=n, p=[0.15, 0.10, 0.75])
        
        # Original balance: log-normal around $300k
        original_balance = self.rng.lognormal(mean=12.6, sigma=0.5, size=n).astype(int)
        original_balance = np.clip(original_balance, 50000, 1500000)
        
        # Interest rate: normal around 5.5%
        interest_rate = self.rng.normal(0.055, 0.012, size=n)
        interest_rate = np.clip(interest_rate, 0.02, 0.12)
        
        # Credit score band - correlated with rate
        credit_score_band = self._assign_credit_band(interest_rate, n)
        
        # LTV band - correlated with balance
        ltv_band = self._assign_ltv_band(original_balance, n)
        
        # DTI band
        dti_band = self.rng.choice(self.config.dti_bands, size=n, 
                                   p=[0.15, 0.20, 0.20, 0.20, 0.15, 0.10])
        
        # Geography
        state = self.rng.choice(self.config.states, size=n)
        
        # Loan purpose
        loan_purpose = self.rng.choice(self.config.loan_purposes, size=n, 
                                       p=[0.50, 0.30, 0.20])
        
        # Occupancy
        occupancy_type = self.rng.choice(self.config.occupancy_types, size=n,
                                         p=[0.80, 0.10, 0.10])
        
        # Property type
        property_type = self.rng.choice(self.config.property_types, size=n,
                                        p=[0.70, 0.15, 0.10, 0.05])
        
        # Servicer
        servicer_name = self.rng.choice(self.config.servicers, size=n)
        
        df = pd.DataFrame({
            'loan_id': loan_ids,
            'original_balance': original_balance,
            'interest_rate': interest_rate,
            'credit_score_band': credit_score_band,
            'ltv_band': ltv_band,
            'dti_band': dti_band,
            'state': state,
            'loan_purpose': loan_purpose,
            'occupancy_type': occupancy_type,
            'property_type': property_type,
            'servicer_name': servicer_name,
            'origination_month': origination_month,
            'term_months': term_months
        })
        
        return df
    
    def _assign_credit_band(self, rates: np.ndarray, n: int) -> np.ndarray:
        """Assign credit score bands correlated with interest rates."""
        # Higher rate -> lower credit score
        rate_percentiles = pd.Series(rates).rank(pct=True)
        
        bands = []
        for pct in rate_percentiles:
            if pct < 0.10:
                bands.append('780+')
            elif pct < 0.25:
                bands.append('740-779')
            elif pct < 0.40:
                bands.append('700-739')
            elif pct < 0.55:
                bands.append('660-699')
            elif pct < 0.75:
                bands.append('620-659')
            else:
                bands.append('<620')
        return np.array(bands)
    
    def _assign_ltv_band(self, balances: np.ndarray, n: int) -> np.ndarray:
        """Assign LTV bands."""
        # Higher balance tends to have higher LTV
        balance_percentiles = pd.Series(balances).rank(pct=True)
        
        bands = []
        for pct in balance_percentiles:
            if pct < 0.15:
                bands.append('<60%')
            elif pct < 0.30:
                bands.append('60-70%')
            elif pct < 0.50:
                bands.append('70-80%')
            elif pct < 0.70:
                bands.append('80-90%')
            elif pct < 0.90:
                bands.append('90-100%')
            else:
                bands.append('>100%')
        return np.array(bands)
    
    def _generate_panel_data(self, static_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate monthly panel data for each loan."""
        rows = []
        
        for _, loan in static_df.iterrows():
            loan_rows = self._generate_loan_history(loan)
            rows.extend(loan_rows)
        
        panel_df = pd.DataFrame(rows)
        
        # Split into train/test by month_index (80/20 of the actual month range).
        # Using a fixed threshold (e.g. month 24) would produce an empty test set
        # whenever n_months < 24, so we always use a percentile-based cutoff.
        unique_months = sorted(panel_df['month_index'].unique())
        n_unique = len(unique_months)
        cutoff_idx = max(0, int(n_unique * 0.80) - 1)
        train_end = unique_months[cutoff_idx]

        train_mask = panel_df['month_index'] <= train_end
        test_mask  = panel_df['month_index'] > train_end
        
        train_df = panel_df[train_mask].copy()
        test_df = panel_df[test_mask].copy()
        
        # Remove target columns from test
        target_cols = ['next_3m_delinquency_flag', 'next_6m_delinquency_flag',
                       'next_12m_default_flag', 'next_12m_prepayment_flag',
                       'next_state', 'exception_required', 'exception_type']
        for col in target_cols:
            if col in test_df.columns:
                test_df = test_df.drop(columns=[col])
        
        logger.info(f"Generated panel: train={len(train_df)}, test={len(test_df)}")
        return train_df, test_df
    
    def _generate_loan_history(self, loan: pd.Series) -> List[Dict]:
        """Generate monthly history for a single loan."""
        loan_id = loan['loan_id']
        orig_month = loan['origination_month']
        term = loan['term_months']
        orig_balance = loan['original_balance']
        rate = loan['interest_rate']
        credit_band = loan['credit_score_band']
        ltv_band = loan['ltv_band']
        dti_band = loan['dti_band']
        state = loan['state']
        purpose = loan['loan_purpose']
        occupancy = loan['occupancy_type']
        prop_type = loan['property_type']
        servicer = loan['servicer_name']
        
        # Determine loan lifetime
        max_age = min(term, self.config.n_months)
        
        # Current balance starts at original
        current_balance = orig_balance
        
        # Monthly payment (simplified)
        monthly_rate = rate / 12
        if monthly_rate > 0:
            monthly_payment = orig_balance * (monthly_rate * (1 + monthly_rate)**term) / ((1 + monthly_rate)**term - 1)
        else:
            monthly_payment = orig_balance / term
        
        rows = []
        status = 'Current'
        days_past_due = 0
        ever_delinquent = False
        ever_modified = False
        
        for month_idx in range(1, max_age + 1):
            reporting_month = orig_month + month_idx - 1
            loan_age = month_idx
            remaining_term = term - month_idx + 1
            
            # Simulate balance amortization
            interest_portion = current_balance * monthly_rate
            principal_portion = min(monthly_payment - interest_portion, current_balance)
            current_balance = max(0, current_balance - principal_portion)
            
            # Transition probabilities based on credit band, LTV, DTI
            trans_probs = self._get_transition_probs(
                credit_band, ltv_band, dti_band, loan_age, status
            )
            
            # Add seasonality
            if reporting_month.month in [1, 2]:  # Post-holiday stress
                trans_probs['30-59 DPD'] *= 1.2
                trans_probs['60-89 DPD'] *= 1.1
            
            # Normalize
            total = sum(trans_probs.values())
            trans_probs = {k: v/total for k, v in trans_probs.items()}
            
            # Sample next status
            new_status = self.rng.choice(list(trans_probs.keys()), p=list(trans_probs.values()))
            
            # Update DPD
            if new_status == 'Current':
                days_past_due = 0
            elif new_status == '30-59 DPD':
                days_past_due = self.rng.integers(30, 60)
            elif new_status == '60-89 DPD':
                days_past_due = self.rng.integers(60, 90)
            elif new_status == '90+ DPD':
                days_past_due = self.rng.integers(90, 180)
            elif new_status in ['Prepaid', 'Closed']:
                days_past_due = 0
                current_balance = 0
            elif new_status == 'Defaulted':
                days_past_due = self.rng.integers(90, 365)
                current_balance = 0
            
            # Modification flag
            modification_flag = 0
            if status in ['60-89 DPD', '90+ DPD'] and self.rng.random() < 0.15:
                modification_flag = 1
                ever_modified = True
            
            # Prepayment flag
            prepayment_flag = 1 if new_status == 'Prepaid' else 0
            
            # Default flag
            default_flag = 1 if new_status == 'Defaulted' else 0
            
            # Loss severity
            if default_flag:
                loss_severity = self.rng.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2])
            else:
                loss_severity = 'NA'
            
            # Document status
            doc_status = self.rng.choice(self.config.doc_statuses, p=[0.7, 0.2, 0.1])
            
            # Source system
            source_system = self.rng.choice(['Primary', 'Servicer'], p=[0.9, 0.1])
            
            # Last updated
            last_updated = pd.Timestamp(reporting_month.to_timestamp()) + pd.Timedelta(days=self.rng.integers(0, 28))
            
            row = {
                'loan_id': loan_id,
                'month_index': month_idx,
                'reporting_month': reporting_month,
                'origination_month': orig_month,
                'loan_age_months': loan_age,
                'remaining_term_months': remaining_term,
                'original_balance': orig_balance,
                'current_balance': round(current_balance, 2),
                'interest_rate': round(rate, 4),
                'credit_score_band': credit_band,
                'ltv_band': ltv_band,
                'dti_band': dti_band,
                'state': state,
                'loan_purpose': purpose,
                'occupancy_type': occupancy,
                'property_type': prop_type,
                'servicer_name': servicer,
                'current_status': new_status,
                'days_past_due': days_past_due,
                'modification_flag': modification_flag,
                'prepayment_flag': prepayment_flag,
                'default_flag': default_flag,
                'loss_severity_band': loss_severity,
                'last_updated_at': last_updated,
                'source_system': source_system,
                'document_status': doc_status
            }
            
            # Compute forward-looking targets (for train only)
            # Look ahead 3, 6, 12 months
            row['next_3m_delinquency_flag'] = 0
            row['next_6m_delinquency_flag'] = 0
            row['next_12m_default_flag'] = 0
            row['next_12m_prepayment_flag'] = 0
            row['next_state'] = new_status
            row['exception_required'] = 0
            row['exception_type'] = 'none'
            
            rows.append(row)
            status = new_status
            
            # Stop if terminal state
            if status in ['Prepaid', 'Closed', 'Defaulted']:
                break
        
        # Compute forward targets by looking ahead in generated rows
        self._compute_forward_targets(rows)
        
        return rows
    
    def _get_transition_probs(
        self, credit_band: str, ltv_band: str, dti_band: str,
        loan_age: int, current_status: str
    ) -> Dict[str, float]:
        """Get transition probabilities based on loan characteristics."""
        # Base probabilities
        credit_risk = {'<620': 3.0, '620-659': 2.0, '660-699': 1.5, 
                       '700-739': 1.0, '740-779': 0.7, '780+': 0.5}[credit_band]
        ltv_risk = {'<60%': 0.5, '60-70%': 0.7, '70-80%': 1.0, 
                    '80-90%': 1.5, '90-100%': 2.0, '>100%': 3.0}[ltv_band]
        dti_risk = {'<20%': 0.5, '20-30%': 0.7, '30-36%': 1.0,
                    '36-43%': 1.3, '43-50%': 1.8, '>50%': 2.5}[dti_band]
        
        risk_factor = (credit_risk + ltv_risk + dti_risk) / 3
        
        # Age effect: higher risk in first few years, then stabilizes
        if loan_age <= 12:
            age_factor = 1.2
        elif loan_age <= 36:
            age_factor = 1.0
        else:
            age_factor = 0.8
        
        risk = risk_factor * age_factor
        
        if current_status == 'Current':
            return {
                'Current': max(0.85, 0.98 - risk * 0.05),
                '30-59 DPD': min(0.10, 0.015 * risk),
                '60-89 DPD': min(0.03, 0.003 * risk),
                '90+ DPD': min(0.01, 0.001 * risk),
                'Prepaid': 0.02 / risk,
                'Defaulted': 0.0001 * risk,
                'Closed': 0.001
            }
        elif current_status == '30-59 DPD':
            return {
                'Current': 0.40 / risk,
                '30-59 DPD': 0.30,
                '60-89 DPD': 0.20 * risk,
                '90+ DPD': 0.05 * risk,
                'Prepaid': 0.01 / risk,
                'Defaulted': 0.01 * risk,
                'Closed': 0.01
            }
        elif current_status == '60-89 DPD':
            return {
                'Current': 0.20 / risk,
                '30-59 DPD': 0.20,
                '60-89 DPD': 0.30,
                '90+ DPD': 0.20 * risk,
                'Prepaid': 0.01 / risk,
                'Defaulted': 0.05 * risk,
                'Closed': 0.01
            }
        elif current_status == '90+ DPD':
            return {
                'Current': 0.05 / risk,
                '30-59 DPD': 0.05,
                '60-89 DPD': 0.10,
                '90+ DPD': 0.40,
                'Prepaid': 0.005 / risk,
                'Defaulted': 0.35 * risk,
                'Closed': 0.01
            }
        else:
            # Terminal states stay terminal
            return {current_status: 1.0}
    
    def _compute_forward_targets(self, rows: List[Dict]) -> None:
        """Compute forward-looking target variables."""
        n = len(rows)
        for i in range(n):
            # Look ahead 3 months
            for horizon, target_col in [(3, 'next_3m_delinquency_flag'),
                                         (6, 'next_6m_delinquency_flag'),
                                         (12, 'next_12m_default_flag'),
                                         (12, 'next_12m_prepayment_flag')]:
                for j in range(i + 1, min(i + horizon + 1, n)):
                    future_status = rows[j]['current_status']
                    if target_col == 'next_3m_delinquency_flag' and future_status in ['30-59 DPD', '60-89 DPD', '90+ DPD']:
                        rows[i][target_col] = 1
                        break
                    elif target_col == 'next_6m_delinquency_flag' and future_status in ['30-59 DPD', '60-89 DPD', '90+ DPD']:
                        rows[i][target_col] = 1
                        break
                    elif target_col == 'next_12m_default_flag' and future_status == 'Defaulted':
                        rows[i][target_col] = 1
                        break
                    elif target_col == 'next_12m_prepayment_flag' and future_status == 'Prepaid':
                        rows[i][target_col] = 1
                        break
            
            # Next state (1 month ahead)
            if i + 1 < n:
                rows[i]['next_state'] = rows[i + 1]['current_status']
            else:
                rows[i]['next_state'] = rows[i]['current_status']
            
            # Exception flags
            rows[i]['exception_required'] = 1 if rows[i]['next_3m_delinquency_flag'] or rows[i]['next_12m_default_flag'] else 0
            if rows[i]['exception_required']:
                rows[i]['exception_type'] = self.rng.choice(
                    ['data_quality', 'business_rule', 'pattern_anomaly', 'servicer_conflict'],
                    p=[0.3, 0.3, 0.2, 0.2]
                )
    
    def _inject_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject realistic data quality issues."""
        df = df.copy()
        n = len(df)
        
        # Missing values
        n_missing = int(n * self.config.missing_rate)
        missing_indices = self.rng.choice(n, size=n_missing, replace=False)
        missing_cols = ['current_balance', 'days_past_due', 'document_status', 'interest_rate']
        for idx in missing_indices:
            col = self.rng.choice(missing_cols)
            df.iloc[idx, df.columns.get_loc(col)] = np.nan
        
        # Outliers
        n_outliers = int(n * self.config.outlier_rate)
        outlier_indices = self.rng.choice(n, size=n_outliers, replace=False)
        for idx in outlier_indices:
            col = self.rng.choice(['current_balance', 'interest_rate', 'days_past_due'])
            if col == 'current_balance':
                df.iloc[idx, df.columns.get_loc(col)] *= self.rng.uniform(5, 20)
            elif col == 'interest_rate':
                df.iloc[idx, df.columns.get_loc(col)] = self.rng.uniform(0.15, 0.30)
            elif col == 'days_past_due':
                df.iloc[idx, df.columns.get_loc(col)] = self.rng.integers(365, 1000)
        
        # Label noise
        n_noise = int(n * self.config.label_noise_rate)
        noise_indices = self.rng.choice(n, size=n_noise, replace=False)
        for idx in noise_indices:
            if 'next_3m_delinquency_flag' in df.columns:
                df.iloc[idx, df.columns.get_loc('next_3m_delinquency_flag')] = \
                    1 - df.iloc[idx, df.columns.get_loc('next_3m_delinquency_flag')]
        
        return df
    
    def _generate_servicer_updates(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Generate servicer updates with some conflicts."""
        # Sample a subset of loans/months
        sample_frac = 0.3
        servicer_df = train_df.sample(frac=sample_frac, random_state=self.config.random_seed)[
            ['loan_id', 'reporting_month', 'current_balance', 'current_status', 'days_past_due', 'last_updated_at']
        ].copy()
        
        servicer_df = servicer_df.rename(columns={
            'current_balance': 'servicer_current_balance',
            'current_status': 'servicer_current_status',
            'days_past_due': 'servicer_days_past_due',
            'last_updated_at': 'servicer_last_updated'
        })
        
        return servicer_df
    
    def _inject_servicer_conflicts(
        self, servicer_df: pd.DataFrame, primary_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Inject conflicts between servicer and primary data."""
        servicer_df = servicer_df.copy()
        n = len(servicer_df)
        n_conflicts = int(n * self.config.conflict_rate)
        conflict_indices = self.rng.choice(n, size=n_conflicts, replace=False)
        
        for idx in conflict_indices:
            conflict_type = self.rng.choice(['balance', 'status', 'dpd'])
            if conflict_type == 'balance':
                servicer_df.iloc[idx, servicer_df.columns.get_loc('servicer_current_balance')] *= \
                    self.rng.uniform(0.5, 2.0)
            elif conflict_type == 'status':
                servicer_df.iloc[idx, servicer_df.columns.get_loc('servicer_current_status')] = \
                    self.rng.choice(['Current', '30-59 DPD', '60-89 DPD'])
            elif conflict_type == 'dpd':
                servicer_df.iloc[idx, servicer_df.columns.get_loc('servicer_days_past_due')] = \
                    self.rng.integers(0, 180)
            
            # Make servicer update newer sometimes
            if self.rng.random() < 0.5:
                servicer_df.iloc[idx, servicer_df.columns.get_loc('servicer_last_updated')] = \
                    servicer_df.iloc[idx]['servicer_last_updated'] + pd.Timedelta(days=self.rng.integers(1, 30))
        
        return servicer_df
    
    def _save_synthetic_data(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
        static_df: pd.DataFrame, servicer_df: pd.DataFrame
    ) -> None:
        """Save synthetic data to disk."""
        out_dir = Path("data/synthetic")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(out_dir / "loan_monthly_performance_train.csv", index=False)
        test_df.to_csv(out_dir / "loan_monthly_performance_test.csv", index=False)
        static_df.to_csv(out_dir / "loan_static_attributes.csv", index=False)
        servicer_df.to_csv(out_dir / "servicer_updates.csv", index=False)
        
        logger.info(f"Saved synthetic data to {out_dir}")


def generate_synthetic_data(
    n_loans: int = 50000,
    n_months: int = 36,
    output_dir: str = "data/synthetic",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to generate synthetic data."""
    config = SyntheticConfig(
        n_loans=n_loans,
        n_months=n_months,
        random_seed=seed
    )
    generator = SyntheticDataGenerator(config)
    return generator.generate_all()