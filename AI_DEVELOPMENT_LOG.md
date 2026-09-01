# AI Development Log - Loan Performance Intelligence Engine

## Project Overview
Building a Loan Performance Intelligence Engine for the Intain Campus FinTech Challenge 2026 AI Track. This log documents the agentic development process using opencode.

## Tools Used
- **opencode**: Primary AI coding agent for autonomous implementation
- **AI coding assistant**: LLM-assisted code generation, debugging, and iteration
- **Python 3.11+**: Primary development language
- **Git**: Version control

## Development Sessions

### Session 1: Project Setup & Configuration (2026-08-31)
**Objective**: Initialize project structure, configuration, and synthetic data generator

**Prompts Used**:
1. "Create the complete project structure for a Loan Performance Intelligence Engine based on the PRD"
2. "Build a schema-faithful synthetic data generator that produces realistic loan panel data with injected quality issues"
3. "Create configuration files for all modules: settings.yaml, validation_rules.json, data_dictionary.md, macro_scenarios.csv"

**Accepted Outputs**:
- Complete directory structure under `/Volumes/Data/Hackathon/`
- `config/settings.yaml` with all pipeline parameters
- `config/validation_rules.json` with 14 business rules
- `config/data_dictionary.md` documenting all 30+ fields
- `config/macro_scenarios.csv` with 3 scenarios
- `src/ingest/synthetic_generator.py` - 400+ line generator with realistic loan dynamics
- `src/utils/config.py`, `logging.py`, `reproducibility.py`

**Rejected Outputs**:
- Initial synthetic generator used too simplistic transition probabilities - revised to include credit/LTV/DTI risk factors and seasonality
- First validation rules missed servicer conflict detection - added R012-R014

**Human Review**: Verified synthetic data matches schema exactly, includes all required target variables, and has realistic distributions.

### Session 2: Data Ingestion & Profiling (2026-08-31)
**Objective**: Build data ingestion with schema validation and comprehensive profiling

**Prompts Used**:
1. "Create DataIngestor class that loads all 4 CSV files, validates schemas, applies validation rules, and reconciles servicer updates"
2. "Build DataProfiler with column-level distributions, missingness, outliers, correlations, categorical associations (Cramér's V), train/test drift (PSI/KS), and record-level quality scores"

**Accepted Outputs**:
- `src/ingest/ingest.py` - DataIngestor with schema validation, rule evaluation, servicer reconciliation
- `src/profiling/profiler.py` - DataProfiler with ColumnProfile/ProfilingReport dataclasses, HTML/JSON output

**Rejected Outputs**:
- Initial profiler didn't handle Period dtype for month columns - fixed with proper datetime handling
- PSI calculation failed on constant columns - added robustness checks

**Human Review**: Confirmed profiling outputs include all required metrics per PRD Section 5.1 Module 1.

### Session 3: Feature Engineering (2026-08-31)
**Objective**: Build leakage-safe feature pipeline with temporal features

**Prompts Used**:
1. "Create LeakageSafeFeatureEngineer with band-aware ordinal encoding, rolling/lag features, delinquency streaks, and feature manifest for auditability"
2. "Ensure no target leakage: only use data available at reporting_month, groupby loan_id for temporal features"

**Accepted Outputs**:
- `src/features/feature_engineering.py` - Complete pipeline with 50+ engineered features
- Band-aware encoding for credit_score_band, ltv_band, dti_band, current_status
- Rolling windows (3,6,12) and lags (1,2,3,6) for key variables
- Delinquency streak and months-since-dq features
- FeatureManifest saved as CSV for auditability

**Rejected Outputs**:
- First version computed rolling stats globally instead of per-loan - fixed with groupby transform
- Missing feature manifest - added comprehensive tracking

**Human Review**: Verified no future leakage by checking all features use only lagged/shifted values.

### Session 4: Classification Modeling (2026-08-31)
**Objective**: Train baseline and improved models for 5 targets with time-aware splits

**Prompts Used**:
1. "Build ClassifierTrainer with LogisticRegression baseline and LightGBM improved models for all 5 targets"
2. "Implement time-aware split by month_index, calibration (isotonic), class imbalance handling, and comprehensive metrics (ROC-AUC, PR-AUC, F1, recall@precision=0.8, Brier, macro-F1)"

**Accepted Outputs**:
- `src/models/classification/trainer.py` - Full training pipeline with ModelResult/Metrics dataclasses
- TimeAwareSplitter ensuring no loan_id leakage across splits
- Isotonic calibration with reliability diagrams
- Model comparison tables and Model Card generation

**Rejected Outputs**:
- Initial split used random sampling - replaced with strict time-based split
- LightGBM multiclass objective misconfiguration - fixed num_class parameter

**Human Review**: Verified all 5 targets trained with baseline vs improved comparison, calibration applied, metrics logged.

### Session 5: Survival Modeling (2026-08-31)
**Objective**: Implement time-to-event modeling with competing risks

**Prompts Used**:
1. "Build SurvivalDataBuilder to create duration/event dataset from panel data with competing risks (default vs prepayment vs censored)"
2. "Train Kaplan-Meier baseline, Cox PH for each event type, and discrete-time transition model"

**Accepted Outputs**:
- `src/models/survival/survival.py` - SurvivalDataBuilder, SurvivalModeler
- Competing risks handling: default and prepayment as separate events
- Concordance index and Brier score evaluation
- Segment-level survival curves by credit band

**Rejected Outputs**:
- Cox PH convergence issues with high-dimensional features - added penalizer and feature selection
- Discrete-time model required panel reconstruction - simplified to month-level transitions

**Human Review**: Confirmed censoring treatment documented, baseline comparisons included.

### Session 6: Anomaly Detection (2026-08-31)
**Objective**: Build ensemble anomaly detection with reviewer-ready examples

**Prompts Used**:
1. "Create AnomalyDetector combining rule-based checks (from validation_rules.json) with Isolation Forest/LOF/AutoEncoder ensemble"
2. "Generate 20+ reviewer-ready examples with anomaly scores, rule violations, driver attributions, and plain-English narratives"

**Accepted Outputs**:
- `src/models/anomaly/anomaly.py` - RuleBasedChecker, MLAnomalyDetector, ExceptionClassifier, AnomalyDetector
- Ensemble weighting: 60% ML, 40% rules
- ReviewerExample dataclass with narratives and suggested actions
- Exception type classifier (supervised where labels exist)

**Rejected Outputs**:
- AutoEncoder required PyOD which had dependency issues - made optional with try/except
- Initial narratives were too generic - added specific driver references

**Human Review**: Verified 20+ curated examples with all required fields per PRD Module 4.

### Session 7: Scenario Simulation (2026-08-31)
**Objective**: Implement macro scenario stress testing

**Prompts Used**:
1. "Build ScenarioSimulator that applies macro_scenarios.csv assumptions (rate shifts, credit deterioration, HPI changes) to features and re-scores portfolio"
2. "Output segment-level projections and driver explanations for base/adverse/high-prepayment scenarios"

**Accepted Outputs**:
- `src/scenario/simulator.py` - ScenarioSimulator with feature perturbation and re-scoring
- Three scenarios: base, adverse_credit, high_prepayment
- Segment projections by vintage, credit band, state, servicer
- Driver explanations showing feature shifts and prediction deltas

**Rejected Outputs**:
- First version only perturbed interest_rate - expanded to credit status, DPD, modification flags, balance ratios
- Missing segment-level output - added groupby projections

**Human Review**: Confirmed all 3 scenarios produce aggregate + segment outputs with drivers.

### Session 8: Explainability (2026-08-31)
**Objective**: Add SHAP global/local explanations and error analysis

**Prompts Used**:
1. "Create ExplainabilityEngine with SHAP TreeExplainer for LightGBM, global summary plots, local waterfall plots, and false positive/negative case studies"
2. "Compute prediction uncertainty via tree variance for ensembles"

**Accepted Outputs**:
- `src/explainability/explainability.py` - ExplainabilityEngine with SHAP integration
- Global feature importance (top 20) per model
- Local explanations for any loan_id
- Error analysis with 5 FP/FN case studies per target
- Uncertainty estimates

**Rejected Outputs**:
- KernelExplainer too slow for 1000+ samples - switched to TreeExplainer for LightGBM
- Waterfall plots failed for multiclass - added class selection logic

**Human Review**: Verified SHAP values match model predictions, plots generate correctly.

### Session 9: LLM Copilot (2026-08-31)
**Objective**: Build governed LLM layer with grounding and logging

**Prompts Used**:
1. "Create ReviewerCopilot using OpenRouter free tier (Llama-3.1-8B) with RAG over data_dictionary.md and validation_rules.json"
2. "Implement 4 functions: reviewer notes, data Q&A, scenario summaries, rule explanations"
3. "Add governance: prompt logging, hallucination guard (numeric traceability), rejected examples curation"

**Accepted Outputs**:
- `src/copilot/copilot.py` - ReviewerCopilot, GroundingRetriever, HallucinationGuard, OpenRouterClient
- Grounding retrieval with keyword matching
- Hallucination guard checking output numbers against context
- JSONL interaction log with prompt, grounding refs, model, output, label="recommendation"
- 3+ curated rejected examples with corrections

**Rejected Outputs**:
- Initial RAG used FAISS but added complexity - simplified to keyword retrieval
- Hallucination guard was too strict on small integers - added threshold

**Human Review**: Confirmed all LLM outputs labeled as recommendations, logs complete, guard functional.

### Session 10: Pipeline Integration & CLI (2026-08-31)
**Objective**: Orchestrate all modules into single reproducible pipeline

**Prompts Used**:
1. "Create LoanIntelligencePipeline class that runs all 10 steps in order with proper data flow"
2. "Build Click CLI with commands: generate-data, run-all, profile, train, survival, anomaly, scenario, copilot, submit"
3. "Generate submission.csv matching template schema exactly"

**Accepted Outputs**:
- `src/pipeline.py` - LoanIntelligencePipeline with run_full_pipeline()
- `cli.py` - Full CLI with all subcommands
- Submission generator matching template columns exactly
- Copilot demo integrated into pipeline

**Rejected Outputs**:
- First pipeline run had memory issues with SHAP on full dataset - added sampling
- Submission column order mismatch - enforced template ordering

**Human Review**: End-to-end run completes in ~5 minutes on synthetic data, all deliverables produced.

## Summary Statistics
- **Total Python files created**: 15
- **Lines of code**: ~6,000
- **AI-generated code percentage**: ~85%
- **Human review checkpoints**: 10 (one per session)
- **Rejected/iterated outputs**: ~15 major revisions
- **Key lessons**: 
  - Time-aware splits are critical - random splits leak loan_id
  - Synthetic data must match schema exactly including all target columns
  - LLM governance requires explicit grounding retrieval, not just system prompts
  - Feature engineering must be groupby-aware for panel data
  - SHAP TreeExplainer is 100x faster than KernelExplainer for tree models

## Compliance with PRD
✅ All 8 modules implemented
✅ No LLM-based predictions (all from trained ML models)
✅ Time-aware splits with group awareness
✅ Synthetic data generator as fallback
✅ Full reproducibility with seeded RNGs
✅ Model cards, profiling reports, explainability, scenario outputs
✅ LLM copilot with grounding, logging, hallucination guard
✅ AI Development Log maintained
✅ Single-command pipeline execution