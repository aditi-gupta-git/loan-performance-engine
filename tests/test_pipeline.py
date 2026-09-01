"""Unit tests for Loan Performance Intelligence Engine."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.reproducibility import set_global_seed, get_rng, verify_deterministic
from src.utils.config import Settings
from src.pipeline.synthetic_generator import SyntheticDataGenerator, SyntheticConfig
from src.pipeline.loader import DataIngestor, reconcile_servicer_updates
from src.pipeline.validation import load_rules, apply_rules
from src.profiling.profile import DataProfiler, ColumnProfile
from src.features.engineer import LeakageSafeFeatureEngineer
from src.modeling.train_supervised import TimeAwareSplitter, ClassifierTrainer
from src.modeling.survival import SurvivalDataBuilder
from src.modeling.anomaly import RuleBasedChecker, MLAnomalyDetector
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.time_split import get_split_masks


class TestReproducibility:
    """Test deterministic execution."""

    def test_global_seed(self):
        set_global_seed(42)
        a = np.random.rand(5)
        set_global_seed(42)
        b = np.random.rand(5)
        assert np.array_equal(a, b)

    def test_rng(self):
        rng1 = get_rng(42)
        rng2 = get_rng(42)
        assert np.array_equal(rng1.random(10), rng2.random(10))

    def test_verify_deterministic(self):
        def deterministic_func(x):
            return x * 2

        assert verify_deterministic(deterministic_func, 5, n_runs=3)

        # np.random.rand() with same seed IS deterministic (seed is reset each run).
        # Use a function with external state to test non-determinism detection.
        counter = [0]
        def stateful_func(x):
            counter[0] += 1
            return counter[0]  # Returns different value each call regardless of seed

        assert not verify_deterministic(stateful_func, 5, n_runs=3)


class TestConfig:
    """Test configuration loading."""

    def test_settings_load(self):
        settings = Settings.load("config/settings.yaml")
        # Settings stores nested dicts; access via dict keys
        assert settings.synthetic["n_loans"] == 50000
        assert settings.split["train_end_month"] == 24
        assert settings.llm["provider"] == "deepseek"

    def test_validation_rules_load(self):
        rules = load_rules("config/validation_rules.json")
        assert len(rules) == 14
        assert rules[0]["rule_id"] == "R001"
        assert rules[-1]["rule_id"] == "R014"


class TestSyntheticGenerator:
    """Test synthetic data generation."""

    @pytest.fixture
    def generator(self):
        config = SyntheticConfig(n_loans=200, n_months=12, random_seed=42)
        return SyntheticDataGenerator(config)

    def test_generate_all(self, generator):
        train_df, test_df, static_df, servicer_df = generator.generate_all()

        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(static_df) == 200
        assert len(servicer_df) > 0

        required_train = [
            "loan_id", "month_index", "reporting_month", "origination_month",
            "next_3m_delinquency_flag", "next_12m_default_flag",
            "next_12m_prepayment_flag", "next_state",
        ]
        for col in required_train:
            assert col in train_df.columns, f"Missing column: {col}"

        # Test set must NOT have target labels
        for col in ["next_3m_delinquency_flag", "next_12m_default_flag"]:
            assert col not in test_df.columns

    def test_static_attributes(self, generator):
        _, _, static_df, _ = generator.generate_all()
        assert "loan_id" in static_df.columns
        assert "credit_score_band" in static_df.columns
        valid_bands = {"<620", "620-659", "660-699", "700-739", "740-779", "780+"}
        assert set(static_df["credit_score_band"].unique()).issubset(valid_bands)

    def test_servicer_updates(self, generator):
        train_df, _, _, servicer_df = generator.generate_all()
        assert "loan_id" in servicer_df.columns
        assert "servicer_current_balance" in servicer_df.columns
        assert set(servicer_df["loan_id"].unique()).issubset(set(train_df["loan_id"].unique()))


class TestDataIngestion:
    """Test data loading and servicer reconciliation."""

    @pytest.fixture
    def sample_data(self):
        generator = SyntheticDataGenerator(SyntheticConfig(n_loans=100, n_months=12, random_seed=42))
        return generator.generate_all()

    def test_servicer_reconciliation(self, sample_data):
        train_df, _, _, servicer_df = sample_data
        # reporting_month is already Period[M] from the generator
        reconciled, conflicts = reconcile_servicer_updates(train_df, servicer_df)
        assert isinstance(reconciled, pd.DataFrame)
        assert isinstance(conflicts, pd.DataFrame)
        assert len(reconciled) == len(train_df)

    def test_validation_rules(self, sample_data):
        train_df, _, _, _ = sample_data
        rules = load_rules()
        violations_df, summary_df = apply_rules(train_df, rules)
        assert isinstance(violations_df, pd.DataFrame)
        assert isinstance(summary_df, pd.DataFrame)


class TestProfiling:
    """Test data profiling."""

    @pytest.fixture
    def sample_df(self):
        generator = SyntheticDataGenerator(SyntheticConfig(n_loans=300, n_months=12, random_seed=42))
        train_df, _, _, _ = generator.generate_all()
        return train_df

    def test_column_profiling(self, sample_df):
        profiler = DataProfiler()
        report = profiler.profile_dataset(sample_df, "test")

        assert report.n_rows == len(sample_df)
        assert report.n_cols == len(sample_df.columns)
        assert len(report.column_profiles) == len(sample_df.columns)
        assert 0 <= report.batch_quality_score <= 100

    def test_drift_detection(self, sample_df):
        profiler = DataProfiler()
        drifted = sample_df.copy()
        drifted["current_balance"] = drifted["current_balance"] * 2

        report = profiler.profile_dataset(sample_df, "base", reference_df=drifted)
        balance_profile = next(
            p for p in report.column_profiles if p.column == "current_balance"
        )
        assert balance_profile.psi is not None
        assert balance_profile.psi > 0.2  # Significant drift


class TestFeatureEngineering:
    """Test feature engineering pipeline."""

    @pytest.fixture
    def sample_df(self):
        generator = SyntheticDataGenerator(SyntheticConfig(n_loans=200, n_months=12, random_seed=42))
        train_df, _, _, _ = generator.generate_all()
        for col in ["reporting_month", "origination_month"]:
            if col in train_df.columns and not hasattr(train_df[col].dtype, "freq"):
                train_df[col] = pd.to_datetime(train_df[col]).dt.to_period("M")
        return train_df

    def test_fit_transform(self, sample_df):
        fe = LeakageSafeFeatureEngineer()
        X = fe.fit_transform(sample_df, is_train=True)
        assert X.shape[1] > sample_df.shape[1]

    def test_no_target_leakage(self, sample_df):
        fe = LeakageSafeFeatureEngineer()
        X = fe.fit_transform(sample_df, is_train=True)
        target_cols = [
            "next_3m_delinquency_flag", "next_6m_delinquency_flag",
            "next_12m_default_flag", "next_12m_prepayment_flag", "next_state",
        ]
        for col in target_cols:
            assert col not in X.columns, f"Target column leaked: {col}"

    def test_temporal_features_exist(self, sample_df):
        fe = LeakageSafeFeatureEngineer()
        X = fe.fit_transform(sample_df, is_train=True)
        lag_cols = [c for c in X.columns if "_lag" in c]
        roll_cols = [c for c in X.columns if "_rollmean_" in c]
        assert len(lag_cols) > 0, "No lag features created"
        assert len(roll_cols) > 0, "No rolling features created"

    def test_manifest_created(self, sample_df):
        fe = LeakageSafeFeatureEngineer()
        fe.fit_transform(sample_df, is_train=True)
        manifest_path = Path("models/feature_engineering/feature_manifest.csv")
        assert manifest_path.exists()
        manifest = pd.read_csv(manifest_path)
        assert "name" in manifest.columns
        assert "transformation" in manifest.columns
        assert len(manifest) > 0


class TestTimeAwareSplit:
    """Test time-aware splitting — no temporal leakage."""

    @pytest.fixture
    def sample_df(self):
        generator = SyntheticDataGenerator(SyntheticConfig(n_loans=300, n_months=36, random_seed=42))
        train_df, _, _, _ = generator.generate_all()
        return train_df

    def test_temporal_ordering(self, sample_df):
        train_mask, val_mask, test_mask = get_split_masks(sample_df)
        train_months = sample_df[train_mask]["month_index"].unique()
        val_months = sample_df[val_mask]["month_index"].unique()
        test_months = sample_df[test_mask]["month_index"].unique()
        # Strict temporal ordering — no month overlap
        assert max(train_months) < min(val_months)
        assert max(val_months) < min(test_months)

    def test_splitter_class(self, sample_df):
        splitter = TimeAwareSplitter()
        train_mask, val_mask, test_mask = splitter.get_split_indices(sample_df)
        assert train_mask.sum() > 0
        assert val_mask.sum() > 0
        assert test_mask.sum() > 0


class TestMetrics:
    """Test evaluation metrics."""

    def test_binary_metrics(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.4, 0.85, 0.15, 0.75])
        m = compute_binary_metrics(y_true, y_proba)
        assert "roc_auc" in m
        assert "pr_auc" in m
        assert "f1" in m
        assert "brier_score" in m
        assert m["roc_auc"] > 0.5
        assert 0 <= m["brier_score"] <= 1

    def test_degenerate_target(self):
        y_true = np.zeros(10)
        y_proba = np.random.rand(10)
        m = compute_binary_metrics(y_true, y_proba)
        assert np.isnan(m["roc_auc"])  # Undefined for single class


class TestAnomalyDetection:
    """Test anomaly detection components."""

    @pytest.fixture
    def sample_data(self):
        generator = SyntheticDataGenerator(SyntheticConfig(n_loans=300, n_months=12, random_seed=42))
        train_df, test_df, _, _ = generator.generate_all()
        return train_df, test_df

    def test_rule_checker(self, sample_data):
        train_df, _ = sample_data
        checker = RuleBasedChecker()
        violations, summary = checker.check(train_df)
        assert isinstance(violations, pd.DataFrame)
        assert isinstance(summary, pd.DataFrame)

    def test_ml_detector_fit_predict(self, sample_data):
        train_df, test_df = sample_data
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in ["loan_id", "month_index"]][:8]
        detector = MLAnomalyDetector()
        detector.fit(train_df, feature_cols)
        scores, flags, _ = detector.predict(test_df)
        assert len(scores) == len(test_df)
        assert len(flags) == len(test_df)
        assert scores.min() >= 0


class TestSurvivalData:
    """Test survival dataset construction."""

    @pytest.fixture
    def sample_df(self):
        generator = SyntheticDataGenerator(SyntheticConfig(n_loans=200, n_months=24, random_seed=42))
        train_df, _, _, _ = generator.generate_all()
        for col in ["reporting_month", "origination_month"]:
            if col in train_df.columns and not hasattr(train_df[col].dtype, "freq"):
                train_df[col] = pd.to_datetime(train_df[col]).dt.to_period("M")
        return train_df

    def test_build_survival_dataset(self, sample_df):
        builder = SurvivalDataBuilder()
        survival_df = builder.build_survival_dataset(sample_df)

        assert len(survival_df) > 0
        assert "duration" in survival_df.columns
        assert "event" in survival_df.columns
        assert "event_type" in survival_df.columns

        valid_events = {"default", "prepayment", "censored"}
        assert set(survival_df["event_type"].unique()).issubset(valid_events)
        assert (survival_df["duration"] > 0).all()

    def test_censoring_logic(self, sample_df):
        builder = SurvivalDataBuilder()
        survival_df = builder.build_survival_dataset(sample_df)
        # Event encoding: 0=censored, 1=default, 2=prepayment (competing risk)
        assert survival_df["event"].isin([0, 1, 2]).all()
        # Should have some censored and some events
        assert (survival_df["event"] > 0).sum() > 0   # at least one event
        assert (survival_df["event"] == 0).sum() > 0  # at least one censored


class TestSubmissionSchema:
    """Test submission.csv matches the required template."""

    def test_template_exists(self):
        assert Path("config/submission_template.csv").exists()

    def test_submission_matches_template(self):
        template = pd.read_csv("config/submission_template.csv")
        submission = pd.read_csv("submission.csv")

        assert set(template.columns) == set(submission.columns), (
            f"Schema mismatch. Missing: {set(template.columns) - set(submission.columns)}, "
            f"Extra: {set(submission.columns) - set(template.columns)}"
        )
        # No nulls in submission
        assert submission.isnull().sum().sum() == 0
        # Probabilities in [0, 1]
        prob_cols = [c for c in submission.columns if c.endswith("_prob")]
        for col in prob_cols:
            assert submission[col].between(0, 1).all(), f"{col} has out-of-range values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
