"""Data profiling and quality intelligence module."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


@dataclass
class ColumnProfile:
    """Profile for a single column."""
    column: str
    dtype: str
    missing_count: int
    missing_pct: float
    cardinality: int
    # Numeric stats
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    # Categorical stats
    top_categories: Optional[Dict[str, int]] = None
    # Data quality
    outlier_count: int = 0
    outlier_pct: float = 0.0
    # Drift
    psi: Optional[float] = None
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    drift_flag: bool = False


@dataclass
class ProfilingReport:
    """Complete profiling report."""
    dataset_name: str
    n_rows: int
    n_cols: int
    column_profiles: List[ColumnProfile]
    correlation_matrix: Optional[pd.DataFrame] = None
    association_matrix: Optional[pd.DataFrame] = None
    record_quality_scores: Optional[pd.Series] = None
    batch_quality_score: float = 0.0
    drift_summary: Optional[Dict] = None
    validation_summary: Optional[Dict] = None


class DataProfiler:
    """Comprehensive data profiling and quality intelligence."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.profiling_config = self.config.get('profiling', {})
    
    def profile_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        reference_df: Optional[pd.DataFrame] = None,
        validation_results: Optional[Dict] = None
    ) -> ProfilingReport:
        """Generate comprehensive profiling report."""
        logger.info(f"Profiling {dataset_name}: {df.shape[0]} rows, {df.shape[1]} cols")
        
        profiles = []
        for col in df.columns:
            profile = self._profile_column(df[col], col, reference_df[col] if reference_df is not None and col in reference_df.columns else None)
            profiles.append(profile)
        
        # Correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = None
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
        
        # Association matrix for categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        assoc_matrix = None
        if len(cat_cols) > 1:
            assoc_matrix = self._compute_associations(df[cat_cols])
        
        # Record-level quality scores
        record_scores = self._compute_record_quality_scores(df, profiles)
        batch_score = record_scores.mean()
        
        # Drift summary
        drift_summary = None
        if reference_df is not None:
            drift_summary = self._compute_drift_summary(df, reference_df, profiles)
        
        report = ProfilingReport(
            dataset_name=dataset_name,
            n_rows=df.shape[0],
            n_cols=df.shape[1],
            column_profiles=profiles,
            correlation_matrix=corr_matrix,
            association_matrix=assoc_matrix,
            record_quality_scores=record_scores,
            batch_quality_score=batch_score,
            drift_summary=drift_summary,
            validation_summary=validation_results
        )
        
        return report
    
    def _profile_column(
        self,
        series: pd.Series,
        col_name: str,
        reference_series: Optional[pd.Series] = None
    ) -> ColumnProfile:
        """Profile a single column."""
        dtype = str(series.dtype)
        missing_count = series.isnull().sum()
        missing_pct = missing_count / len(series) * 100
        cardinality = series.nunique()
        
        profile = ColumnProfile(
            column=col_name,
            dtype=dtype,
            missing_count=int(missing_count),
            missing_pct=missing_pct,
            cardinality=cardinality
        )
        
        # Numeric profiling
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                profile.mean = float(clean.mean())
                profile.std = float(clean.std())
                profile.min = float(clean.min())
                profile.max = float(clean.max())
                profile.median = float(clean.median())
                profile.q25 = float(clean.quantile(0.25))
                profile.q75 = float(clean.quantile(0.75))
                profile.skewness = float(stats.skew(clean))
                profile.kurtosis = float(stats.kurtosis(clean))
                
                # Outliers using IQR
                iqr = profile.q75 - profile.q25
                lower = profile.q25 - 1.5 * iqr
                upper = profile.q75 + 1.5 * iqr
                outliers = clean[(clean < lower) | (clean > upper)]
                profile.outlier_count = len(outliers)
                profile.outlier_pct = len(outliers) / len(clean) * 100
        
        # Categorical profiling
        elif isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                value_counts = clean.value_counts().head(20)
                profile.top_categories = value_counts.to_dict()
        
        # Drift detection
        if reference_series is not None:
            profile.psi = self._compute_psi(series, reference_series)
            profile.ks_statistic, profile.ks_pvalue = self._compute_ks(series, reference_series)
            profile.drift_flag = (profile.psi is not None and profile.psi > 0.2) or \
                                (profile.ks_pvalue is not None and profile.ks_pvalue < 0.05)
        
        return profile
    
    def _compute_psi(self, current: pd.Series, reference: pd.Series, bins: int = 10) -> Optional[float]:
        """Compute Population Stability Index."""
        try:
            # For numeric, use quantile bins
            if pd.api.types.is_numeric_dtype(current):
                _, bin_edges = pd.qcut(reference.dropna(), q=bins, retbins=True, duplicates='drop')
                if len(bin_edges) < 2:
                    return None
                current_binned = pd.cut(current.dropna(), bins=bin_edges, include_lowest=True)
                reference_binned = pd.cut(reference.dropna(), bins=bin_edges, include_lowest=True)
            else:
                # For categorical, use value counts
                all_cats = set(current.dropna().unique()) | set(reference.dropna().unique())
                current_binned = current.dropna()
                reference_binned = reference.dropna()
                bin_edges = list(all_cats)
            
            current_counts = current_binned.value_counts().reindex(bin_edges[:-1] if isinstance(bin_edges, list) else pd.IntervalIndex.from_breaks(bin_edges), fill_value=0)
            reference_counts = reference_binned.value_counts().reindex(bin_edges[:-1] if isinstance(bin_edges, list) else pd.IntervalIndex.from_breaks(bin_edges), fill_value=0)
            
            current_perc = current_counts / current_counts.sum()
            reference_perc = reference_counts / reference_counts.sum()
            
            # Avoid division by zero
            current_perc = current_perc.replace(0, 0.0001)
            reference_perc = reference_perc.replace(0, 0.0001)
            
            psi = ((current_perc - reference_perc) * np.log(current_perc / reference_perc)).sum()
            return float(psi)
        except Exception:
            return None
    
    def _compute_ks(self, current: pd.Series, reference: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        """Compute Kolmogorov-Smirnov test statistic."""
        try:
            if pd.api.types.is_numeric_dtype(current):
                stat, pval = ks_2samp(current.dropna(), reference.dropna())
                return float(stat), float(pval)
        except Exception:
            pass
        return None, None
    
    def _compute_associations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Cramér's V for categorical associations."""
        cols = df.columns
        n = len(cols)
        assoc_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
        
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i <= j:
                    if i == j:
                        assoc_matrix.loc[col1, col2] = 1.0
                    else:
                        v = self._cramers_v(df[col1], df[col2])
                        assoc_matrix.loc[col1, col2] = v
                        assoc_matrix.loc[col2, col1] = v
        
        return assoc_matrix.astype(float)
    
    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Compute Cramér's V statistic."""
        try:
            confusion = pd.crosstab(x, y)
            chi2 = stats.chi2_contingency(confusion, correction=False)[0]
            n = confusion.sum().sum()
            phi2 = chi2 / n
            r, k = confusion.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        except Exception:
            return 0.0
    
    def _compute_record_quality_scores(
        self, df: pd.DataFrame, profiles: List[ColumnProfile]
    ) -> pd.Series:
        """Compute record-level data quality scores (0-100)."""
        n_rows = len(df)
        scores = np.ones(n_rows) * 100.0
        
        # Penalize missing values
        for profile in profiles:
            if profile.missing_count > 0:
                missing_mask = df[profile.column].isnull()
                penalty = 100 * (profile.missing_pct / 100) * 0.5  # Max 50% penalty per column
                scores[missing_mask] -= penalty
        
        # Penalize outliers
        for profile in profiles:
            if profile.outlier_count > 0 and pd.api.types.is_numeric_dtype(df[profile.column]):
                clean = df[profile.column].dropna()
                if len(clean) > 0:
                    iqr = profile.q75 - profile.q25
                    lower = profile.q25 - 1.5 * iqr
                    upper = profile.q75 + 1.5 * iqr
                    outlier_mask = (df[profile.column] < lower) | (df[profile.column] > upper)
                    penalty = 100 * (profile.outlier_pct / 100) * 0.3  # Max 30% penalty
                    scores[outlier_mask] -= penalty
        
        # Business rule violations (from validation)
        # This would be enhanced with actual rule check results
        
        return pd.Series(np.clip(scores, 0, 100), index=df.index)
    
    def _compute_drift_summary(
        self, current_df: pd.DataFrame, reference_df: pd.DataFrame,
        profiles: List[ColumnProfile]
    ) -> Dict[str, Any]:
        """Summarize drift detection results."""
        drift_cols = [p.column for p in profiles if p.drift_flag]
        high_psi_cols = [p.column for p in profiles if p.psi is not None and p.psi > 0.2]
        high_ks_cols = [p.column for p in profiles if p.ks_pvalue is not None and p.ks_pvalue < 0.05]
        
        return {
            "drift_detected_columns": drift_cols,
            "high_psi_columns": high_psi_cols,
            "high_ks_columns": high_ks_cols,
            "n_drift_columns": len(drift_cols),
            "max_psi": max([p.psi for p in profiles if p.psi is not None], default=0),
            "avg_psi": np.mean([p.psi for p in profiles if p.psi is not None]) if any(p.psi is not None for p in profiles) else 0
        }
    
    def save_report(self, report: ProfilingReport, output_path: str) -> None:
        """Save profiling report to JSON and HTML."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_path.with_suffix('.json')
        self._save_json(report, json_path)
        
        # Save HTML
        html_path = output_path.with_suffix('.html')
        self._save_html(report, html_path)
        
        logger.info(f"Saved profiling report to {json_path} and {html_path}")
    
    def _save_json(self, report: ProfilingReport, path: Path) -> None:
        """Save report as JSON."""
        data = {
            "dataset_name": report.dataset_name,
            "n_rows": report.n_rows,
            "n_cols": report.n_cols,
            "batch_quality_score": report.batch_quality_score,
            "column_profiles": [asdict(p) for p in report.column_profiles],
            "drift_summary": report.drift_summary,
            "validation_summary": report.validation_summary
        }
        
        # Convert DataFrames to dict
        if report.correlation_matrix is not None:
            data["correlation_matrix"] = report.correlation_matrix.to_dict()
        if report.association_matrix is not None:
            data["association_matrix"] = report.association_matrix.to_dict()
        if report.record_quality_scores is not None:
            data["record_quality_scores"] = report.record_quality_scores.to_dict()
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_html(self, report: ProfilingReport, path: Path) -> None:
        """Save report as HTML."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profiling Report: {report.dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
                .warning {{ color: #d9534f; }}
                .success {{ color: #5cb85c; }}
            </style>
        </head>
        <body>
            <h1>Data Profiling Report: {report.dataset_name}</h1>
            
            <h2>Dataset Overview</h2>
            <div class="metric">Rows: {report.n_rows:,}</div>
            <div class="metric">Columns: {report.n_cols}</div>
            <div class="metric">Batch Quality Score: {report.batch_quality_score:.1f}/100</div>
            
            <h2>Column Profiles</h2>
            <table>
                <tr>
                    <th>Column</th><th>Type</th><th>Missing %</th><th>Cardinality</th>
                    <th>Mean</th><th>Std</th><th>Min</th><th>Max</th>
                    <th>Outlier %</th><th>PSI</th><th>Drift</th>
                </tr>
        """
        
        for p in report.column_profiles:
            drift_class = "warning" if p.drift_flag else "success"
            mean_str = f"{p.mean:.2f}" if p.mean is not None else 'N/A'
            std_str = f"{p.std:.2f}" if p.std is not None else 'N/A'
            min_str = f"{p.min:.2f}" if p.min is not None else 'N/A'
            max_str = f"{p.max:.2f}" if p.max is not None else 'N/A'
            psi_str = f"{p.psi:.3f}" if p.psi is not None else 'N/A'
            html += f"""
                <tr>
                    <td>{p.column}</td>
                    <td>{p.dtype}</td>
                    <td>{p.missing_pct:.1f}%</td>
                    <td>{p.cardinality}</td>
                    <td>{mean_str}</td>
                    <td>{std_str}</td>
                    <td>{min_str}</td>
                    <td>{max_str}</td>
                    <td>{p.outlier_pct:.1f}%</td>
                    <td>{psi_str}</td>
                    <td class="{drift_class}">{'Yes' if p.drift_flag else 'No'}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Drift Summary</h2>
        """
        
        if report.drift_summary:
            ds = report.drift_summary
            html += f"""
            <div class="metric">Drift Columns: {ds.get('n_drift_columns', 0)}</div>
            <div class="metric">Max PSI: {ds.get('max_psi', 0):.3f}</div>
            <div class="metric">Avg PSI: {ds.get('avg_psi', 0):.3f}</div>
            <p>High PSI Columns: {', '.join(ds.get('high_psi_columns', [])) or 'None'}</p>
            <p>High KS Columns: {', '.join(ds.get('high_ks_columns', [])) or 'None'}</p>
            """
        
        html += """
        </body>
        </html>
        """
        
        with open(path, 'w') as f:
            f.write(html)


def profile_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "reports/profiling"
) -> Tuple[ProfilingReport, ProfilingReport]:
    """Profile train and test datasets."""
    profiler = DataProfiler()
    
    train_report = profiler.profile_dataset(train_df, "train", reference_df=test_df)
    test_report = profiler.profile_dataset(test_df, "test", reference_df=train_df)
    
    profiler.save_report(train_report, Path(output_dir) / "train_profile")
    profiler.save_report(test_report, Path(output_dir) / "test_profile")
    
    return train_report, test_report