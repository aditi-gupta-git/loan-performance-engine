"""Data ingestion and validation module."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass

from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Result of data ingestion."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    static_df: pd.DataFrame
    servicer_df: pd.DataFrame
    validation_results: Dict[str, Any]
    schema_info: Dict[str, Any]


class DataIngestor:
    """Handles loading and validation of all input data files."""
    
    REQUIRED_FILES = [
        "loan_monthly_performance_train.csv",
        "loan_monthly_performance_test.csv",
        "loan_static_attributes.csv",
        "servicer_updates.csv",
    ]
    
    EXPECTED_COLUMNS = {
        "loan_monthly_performance_train.csv": [
            "loan_id", "month_index", "reporting_month", "origination_month",
            "loan_age_months", "remaining_term_months", "original_balance",
            "current_balance", "interest_rate", "credit_score_band", "ltv_band",
            "dti_band", "state", "loan_purpose", "occupancy_type", "property_type",
            "servicer_name", "current_status", "days_past_due", "modification_flag",
            "prepayment_flag", "default_flag", "loss_severity_band",
            "last_updated_at", "source_system", "document_status",
            "next_3m_delinquency_flag", "next_6m_delinquency_flag",
            "next_12m_default_flag", "next_12m_prepayment_flag",
            "next_state", "exception_required", "exception_type"
        ],
        "loan_monthly_performance_test.csv": [
            "loan_id", "month_index", "reporting_month", "origination_month",
            "loan_age_months", "remaining_term_months", "original_balance",
            "current_balance", "interest_rate", "credit_score_band", "ltv_band",
            "dti_band", "state", "loan_purpose", "occupancy_type", "property_type",
            "servicer_name", "current_status", "days_past_due", "modification_flag",
            "prepayment_flag", "default_flag", "loss_severity_band",
            "last_updated_at", "source_system", "document_status"
        ],
        "loan_static_attributes.csv": [
            "loan_id", "original_balance", "interest_rate", "credit_score_band",
            "ltv_band", "dti_band", "state", "loan_purpose", "occupancy_type",
            "property_type", "servicer_name", "origination_month", "term_months"
        ],
        "servicer_updates.csv": [
            "loan_id", "reporting_month", "servicer_current_balance",
            "servicer_current_status", "servicer_days_past_due",
            "servicer_last_updated"
        ]
    }
    
    def __init__(self, data_dir: str = "data/raw", config: Optional[Dict] = None):
        self.data_dir = Path(data_dir)
        self.config = config or get_settings().data
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> List[Dict]:
        """Load validation rules from JSON."""
        rules_path = Path("config/validation_rules.json")
        if rules_path.exists():
            with open(rules_path) as f:
                return json.load(f).get("rules", [])
        return []
    
    def load_all(self, use_synthetic: bool = False) -> IngestionResult:
        """Load all required data files."""
        if use_synthetic:
            return self._load_synthetic()
        
        logger.info(f"Loading data from {self.data_dir}")
        
        # Check all required files exist
        missing = [f for f in self.REQUIRED_FILES if not (self.data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required files: {missing}")
        
        # Load each file
        train_df = self._load_csv("loan_monthly_performance_train.csv")
        test_df = self._load_csv("loan_monthly_performance_test.csv")
        static_df = self._load_csv("loan_static_attributes.csv")
        servicer_df = self._load_csv("servicer_updates.csv")
        
        # Validate schemas
        validation_results = self._validate_schemas(
            train_df, test_df, static_df, servicer_df
        )
        
        # Apply validation rules
        rule_results = self._apply_validation_rules(train_df)
        validation_results["rule_checks"] = rule_results
        
        schema_info = self._get_schema_info(train_df, test_df, static_df, servicer_df)
        
        return IngestionResult(
            train_df=train_df,
            test_df=test_df,
            static_df=static_df,
            servicer_df=servicer_df,
            validation_results=validation_results,
            schema_info=schema_info
        )
    
    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV with proper dtype handling."""
        path = self.data_dir / filename
        logger.info(f"Loading {filename}...")
        
        # Parse dates
        parse_dates = []
        if "month" in filename or "date" in filename or "updated" in filename:
            parse_dates = [c for c in ["reporting_month", "origination_month", "last_updated_at", "servicer_last_updated"] 
                          if c in self.EXPECTED_COLUMNS.get(filename, [])]
        
        df = pd.read_csv(path, parse_dates=parse_dates if parse_dates else False)
        
        # Convert month columns to period for easier handling
        for col in ["reporting_month", "origination_month"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.to_period('M')
        
        logger.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} cols")
        return df
    
    def _validate_schemas(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
        static_df: pd.DataFrame, servicer_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate data schemas against expected columns."""
        results = {"passed": True, "errors": [], "warnings": []}
        
        for name, df in [
            ("train", train_df), ("test", test_df),
            ("static", static_df), ("servicer", servicer_df)
        ]:
            expected = self.EXPECTED_COLUMNS.get(f"loan_monthly_performance_{name}.csv", [])
            if name == "static":
                expected = self.EXPECTED_COLUMNS["loan_static_attributes.csv"]
            elif name == "servicer":
                expected = self.EXPECTED_COLUMNS["servicer_updates.csv"]
            
            missing = set(expected) - set(df.columns)
            extra = set(df.columns) - set(expected)
            
            if missing:
                results["errors"].append(f"{name}: missing columns {missing}")
                results["passed"] = False
            if extra:
                results["warnings"].append(f"{name}: extra columns {extra}")
        
        return results
    
    def _apply_validation_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Apply validation rules from JSON config."""
        results = {"total_checks": 0, "passed": 0, "failed": 0, "violations": []}
        
        for rule in self.validation_rules:
            results["total_checks"] += 1
            try:
                # Evaluate rule expression
                mask = df.eval(rule["check"])
                violations = df[~mask]
                
                if len(violations) > 0:
                    results["failed"] += 1
                    results["violations"].append({
                        "rule_id": rule["rule_id"],
                        "name": rule["name"],
                        "severity": rule["severity"],
                        "count": len(violations),
                        "sample_ids": violations["loan_id"].head(10).tolist() if "loan_id" in violations.columns else []
                    })
                else:
                    results["passed"] += 1
            except Exception as e:
                results["warnings"] = results.get("warnings", [])
                results["warnings"].append(f"Rule {rule['rule_id']} evaluation failed: {e}")
        
        return results
    
    def _get_schema_info(self, *dfs) -> Dict[str, Any]:
        """Get schema information for all dataframes."""
        info = {}
        for i, df in enumerate(dfs):
            name = ["train", "test", "static", "servicer"][i]
            info[name] = {
                "shape": df.shape,
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing": df.isnull().sum().to_dict(),
                "memory_mb": df.memory_usage(deep=True).sum() / 1024**2
            }
        return info
    
    def _load_synthetic(self) -> IngestionResult:
        """Load synthetic data (placeholder - will use generator)."""
        from src.pipeline.synthetic_generator import SyntheticDataGenerator
        
        logger.info("Generating synthetic data...")
        generator = SyntheticDataGenerator()
        train_df, test_df, static_df, servicer_df = generator.generate_all()
        
        validation_results = {"passed": True, "source": "synthetic", "errors": [], "warnings": []}
        schema_info = self._get_schema_info(train_df, test_df, static_df, servicer_df)
        
        return IngestionResult(
            train_df=train_df,
            test_df=test_df,
            static_df=static_df,
            servicer_df=servicer_df,
            validation_results=validation_results,
            schema_info=schema_info
        )


def reconcile_servicer_updates(
    primary_df: pd.DataFrame,
    servicer_df: pd.DataFrame,
    on: List[str] = ["loan_id", "reporting_month"]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconcile servicer updates against primary data.
    Returns (reconciled_df, conflicts_df).
    """
    logger.info("Reconciling servicer updates...")
    
    # Normalise reporting_month to str so Period and object types merge cleanly
    def _norm_month(df: pd.DataFrame, col: str = "reporting_month") -> pd.DataFrame:
        if col in df.columns:
            df = df.copy()
            df[col] = df[col].astype(str)
        return df

    primary_norm   = _norm_month(primary_df)
    servicer_norm  = _norm_month(servicer_df)
    merged = primary_norm.merge(
        servicer_norm, on=on, how="left", suffixes=("", "_servicer")
    )
    # Restore Period type for reporting_month if it was Period before
    if "reporting_month" in primary_df.columns and hasattr(primary_df["reporting_month"].dtype, "freq"):
        col = merged["reporting_month"]
        if not hasattr(col.dtype, "freq"):
            merged["reporting_month"] = pd.to_datetime(col.astype(str)).dt.to_period("M")
    
    # Identify conflicts
    conflict_cols = [
        ("current_balance", "servicer_current_balance"),
        ("current_status", "servicer_current_status"),
        ("days_past_due", "servicer_days_past_due")
    ]
    
    conflicts = []
    for primary_col, servicer_col in conflict_cols:
        if primary_col in merged.columns and servicer_col in merged.columns:
            # Numeric comparison with tolerance
            if merged[primary_col].dtype in ['float64', 'int64']:
                diff = (merged[primary_col] - merged[servicer_col]).abs()
                mask = diff > (merged[primary_col] * 0.01).clip(lower=1.0)
            else:
                mask = merged[primary_col] != merged[servicer_col]
            
            conflict_rows = merged[mask & merged[servicer_col].notna()].copy()
            if len(conflict_rows) > 0:
                conflict_rows["conflict_field"] = primary_col
                conflict_rows["primary_value"] = conflict_rows[primary_col]
                conflict_rows["servicer_value"] = conflict_rows[servicer_col]
                conflicts.append(conflict_rows)
    
    conflicts_df = pd.concat(conflicts, ignore_index=True) if conflicts else pd.DataFrame()
    
    # Resolution: last_updated_at wins
    if "last_updated_at" in merged.columns and "servicer_last_updated" in merged.columns:
        servicer_newer = merged["servicer_last_updated"] > merged["last_updated_at"]
        for primary_col, servicer_col in conflict_cols:
            if primary_col in merged.columns and servicer_col in merged.columns:
                merged.loc[servicer_newer & merged[servicer_col].notna(), primary_col] = \
                    merged.loc[servicer_newer & merged[servicer_col].notna(), servicer_col]
    
    # Drop servicer columns
    servicer_cols = [c for c in merged.columns if c.endswith("_servicer") or c == "servicer_last_updated"]
    reconciled = merged.drop(columns=servicer_cols)
    
    logger.info(f"Reconciliation complete: {len(conflicts_df)} conflicts found")
    return reconciled, conflicts_df