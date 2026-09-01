<div align="center">

# Loan Performance Intelligence Engine

**Intain Campus FinTech Challenge 2026 — AI Track**

*An ML-first system for loan-data profiling, performance prediction, anomaly detection,*  
*scenario simulation, explainability, and governed LLM-assisted review.*

---

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-26%20passed-16A34A?style=flat&logo=pytest&logoColor=white)](#running-tests)
[![LLM](https://img.shields.io/badge/LLM-DeepSeek-6366F1?style=flat)](https://platform.deepseek.com)
[![License](https://img.shields.io/badge/License-MIT-gray?style=flat)](LICENSE)

</div>

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Repository Structure](#2-repository-structure)
3. [Quickstart](#3-quickstart)
4. [Pipeline — 11 Stages](#4-pipeline--11-stages)
5. [Task 1 — Data Profiling](#5-task-1--data-profiling)
6. [Task 2 — Supervised Prediction](#6-task-2--supervised-prediction)
7. [Task 3 — Survival Modeling](#7-task-3--survival-modeling)
8. [Task 4 — Anomaly Detection](#8-task-4--anomaly-detection)
9. [Task 5 — Scenario Simulation](#9-task-5--scenario-simulation)
10. [Task 6 — Explainability](#10-task-6--explainability)
11. [Task 7 — LLM Reviewer Copilot](#11-task-7--llm-reviewer-copilot)
12. [Task 8 — Agentic Development Evidence](#12-task-8--agentic-development-evidence)
13. [Data](#13-data)
14. [Submission File](#14-submission-file)
15. [Running on GitHub Codespaces](#15-running-on-github-codespaces)
16. [Reproducing Results](#16-reproducing-results)
17. [Deliverables Checklist](#17-deliverables-checklist)

---

## 1. What This System Does

This submission addresses every required task in the challenge problem statement using a **strictly ML-first approach** — the LLM is used only to generate human-readable reviewer notes, never for classification or prediction.

| Challenge Task | Approach |
|---|---|
| **T1 — Data Profiling** | 14 validation rules, per-record quality flags, PSI drift, missingness, outlier detection |
| **T2 — Supervised Prediction** | 5 targets, LightGBM vs. LogisticRegression baseline, time-aware split, isotonic calibration |
| **T3 — Survival / Transition** | Kaplan-Meier, Cox PH (competing risks), discrete-time hazard model |
| **T4 — Anomaly Detection** | IsolationForest ensemble + rule-based flags, exception type classifier |
| **T5 — Scenario Simulation** | Base, adverse-credit, high-prepayment projections with segment-level breakdowns |
| **T6 — Explainability** | SHAP global importance, local per-loan explanations, FP/FN case studies, calibration curves |
| **T7 — LLM Copilot** | DeepSeek-powered reviewer notes, grounding retriever, hallucination guard, interaction log |
| **T8 — Agentic Evidence** | `AI_DEVELOPMENT_LOG.md` — prompts, accepted/rejected outputs, human review process |

---

## 2. Repository Structure

```
loan-performance-engine/
│
├── run_pipeline.py               ← One command runs all 11 stages end-to-end
├── demo_visuals.py               ← Generates 8 presentation charts from pipeline outputs
├── submission.csv                ← Final predictions (1,612 rows × 14 columns)
├── AI_DEVELOPMENT_LOG.md         ← Agentic coding evidence — Task 8
├── requirements.txt              ← Minimal, only actually-used packages
├── .env.example                  ← API key template (copy → .env, never commit)
│
├── src/
│   ├── pipeline/
│   │   ├── loader.py             ← Load, merge, reconcile servicer updates
│   │   ├── validation.py         ← Apply 14 deterministic rules → quality flags
│   │   └── synthetic_generator.py ← Schema-faithful synthetic data (replace with real data)
│   │
│   ├── profiling/
│   │   └── profile.py            ← Column stats, missingness, PSI drift, quality score
│   │
│   ├── features/
│   │   └── engineer.py           ← 157 leakage-safe features (lags, rolling windows, encodings)
│   │
│   ├── modeling/
│   │   ├── train_supervised.py   ← Baseline + LightGBM for 5 targets; calibration; model cards
│   │   ├── survival.py           ← Kaplan-Meier, Cox PH, discrete-time hazard
│   │   ├── anomaly.py            ← IsolationForest + rules; exception classifier
│   │   ├── scenario.py           ← 3 macro scenarios; segment projections
│   │   └── explain.py            ← SHAP global/local; FP/FN analysis
│   │
│   ├── llm_copilot/
│   │   └── reviewer.py           ← DeepSeek client; grounding retriever; hallucination guard
│   │
│   ├── evaluation/
│   │   ├── metrics.py            ← ROC-AUC, PR-AUC, F1, Brier score, recall @ precision
│   │   └── time_split.py         ← Time-aware split with leakage audit
│   │
│   ├── submission/
│   │   └── generate_submission.py ← Format predictions into required schema
│   │
│   └── utils/
│       ├── config.py             ← Settings singleton — get_settings()
│       ├── logging.py
│       └── reproducibility.py   ← Global seed = 42 everywhere
│
├── notebooks/
│   ├── 01_data_profiling.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_survival_analysis.ipynb
│   ├── 05_anomaly_detection.ipynb
│   ├── 06_scenario_simulation.ipynb
│   ├── 07_explainability.ipynb
│   └── 08_llm_copilot_demo.ipynb
│
├── config/
│   ├── settings.yaml             ← All tunable pipeline parameters
│   ├── validation_rules.json     ← 14 rule definitions (R001–R014)
│   ├── data_dictionary.md        ← Field definitions used for LLM grounding
│   ├── macro_scenarios.csv       ← Shock assumptions per scenario
│   └── submission_template.csv  ← Required 14-column output schema
│
├── models/
│   ├── classification/           ← 10 trained .pkl files + 10 model cards (.md)
│   ├── survival/                 ← 3 trained survival models
│   ├── anomaly/                  ← IsolationForest + exception classifier
│   └── feature_engineering/      ← Encoders, scalers, feature manifest CSV
│
├── reports/
│   ├── profiling/                ← train_profile.html  ←  open in browser
│   ├── modeling/                 ← detailed_results.json, model_comparison.csv
│   ├── explainability/           ← global_importance.json + 5 SHAP summary plots
│   ├── scenario/                 ← scenario_results.json, scenario_comparison.csv
│   ├── survival/                 ← survival_results.json
│   └── copilot/                  ← demo_outputs.json, rejected_examples.json
│
├── data/synthetic/               ← 4 CSV files (swap with organiser data when released)
├── tests/
│   └── test_pipeline.py          ← 26 unit and integration tests
└── logs/
    └── llm_interaction_log.jsonl ← Every LLM interaction logged
```

---

## 3. Quickstart

### Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Set API key *(optional — full pipeline runs without it)*

```bash
export DEEPSEEK_API_KEY=sk-your-key-here   # Mac / Linux / Codespaces
set  DEEPSEEK_API_KEY=sk-your-key-here     # Windows CMD
```

See [`.env.example`](.env.example) for all supported environment variables.

### Run tests

```bash
python -m pytest tests/ -q
# 26 passed
```

### Run the full pipeline

```bash
python run_pipeline.py --use-synthetic
# Completes in ~8–10 minutes. Generates all models, reports, and submission.csv.
```

### Generate demo charts

```bash
python demo_visuals.py
# Saves 8 charts to demo_charts/
```

---

## 4. Pipeline — 11 Stages

`run_pipeline.py` executes these stages in order:

```
Stage 1  │ Data Ingestion          Load CSVs, merge static attributes, reconcile servicer updates
Stage 2  │ Validation              Apply 14 deterministic rules → per-record quality flag (0/1/2)
Stage 3  │ Data Profiling          Distributions, missingness, outliers, PSI drift, quality score
Stage 4  │ Feature Engineering     157 leakage-safe features: lags, rolling windows, encodings
Stage 5  │ Supervised Models       10 models (2 per target), time-aware split, calibration
Stage 6  │ Survival Modeling       Kaplan-Meier, Cox PH (×2), discrete-time hazard
Stage 7  │ Anomaly Detection       IsolationForest + rules, exception classification
Stage 8  │ Scenario Simulation     Base / adverse-credit / high-prepayment projections
Stage 9  │ Explainability          SHAP global + local, FP/FN analysis, calibration curves
Stage 10 │ LLM Copilot             Reviewer notes, Q&A, scenario summary, rule explanations
Stage 11 │ Submission              Format into 14-column submission.csv
```

---

## 5. Task 1 — Data Profiling

**Report:** [`reports/profiling/train_profile.html`](reports/profiling/train_profile.html) *(open in browser)*

### What is profiled

- Column distributions (histograms, value counts, cardinality)
- Missing-value rates per column
- Outlier detection (IQR and Z-score methods)
- Cross-column relationship checks (e.g. `days_past_due` vs `current_status`)
- Train vs. test drift using Population Stability Index (PSI)
- Batch-level quality score and per-record quality flag

### Validation rules

14 deterministic rules are defined in [`config/validation_rules.json`](config/validation_rules.json) and applied before any modelling:

| Rule | Severity | Checks |
|---|---|---|
| R001 | Error | `current_balance` ≤ `original_balance × 1.05` |
| R002 | Error | `origination_month` < `reporting_month` |
| R003 | Error | `loan_age_months` ≥ 0 |
| R004 | Error | `remaining_term_months` ≥ 0 |
| R005 | Warning | `days_past_due` consistent with `current_status` |
| R006 | Warning | Prepaid/closed loans have no subsequent active records |
| R007 | Warning | `document_status` not null for active loans |
| R008 | Error | `interest_rate` within 0–30% bounds |
| R009 | Error | `credit_score_band` is one of the defined categories |
| R010 | Error | `ltv_band` is one of the defined categories |
| R011 | Error | `dti_band` is one of the defined categories |
| R012 | Warning | `document_status` is a valid enumerated value |
| R013 | Error | `prepayment_flag` and `default_flag` are mutually exclusive |
| R014 | Warning | `current_balance` ≥ 0 and ≤ `original_balance` |

Each record receives a quality flag: **0 = clean**, **1 = warning**, **2 = error**.

---

## 6. Task 2 — Supervised Prediction

### Targets

Five forward-looking targets are predicted simultaneously:

| Target | Description |
|---|---|
| `next_3m_delinquency_flag` | Will this loan be delinquent in the next 3 months? |
| `next_6m_delinquency_flag` | Will this loan be delinquent in the next 6 months? |
| `next_12m_default_flag` | Will this loan default within the next 12 months? |
| `next_12m_prepayment_flag` | Will this loan prepay within the next 12 months? |
| `next_state` | What is the loan's most likely state next month? (6 classes) |

### Results — validation set

| Target | Baseline ROC-AUC | **LightGBM ROC-AUC** | Baseline PR-AUC | **LightGBM PR-AUC** | Brier Score |
|---|---|---|---|---|---|
| 3-Month Delinquency | 0.582 | **0.729** | 0.164 | **0.421** | 0.085 |
| 6-Month Delinquency | 0.568 | **0.768** | 0.219 | **0.497** | 0.120 |
| **12-Month Default** | 0.640 | **0.887** | 0.177 | **0.736** | 0.065 |
| 12-Month Prepayment | 0.535 | **0.797** | 0.148 | **0.552** | 0.101 |
| Next State (7-class) | 0.669 | **0.830** | 0.231 | **0.496** | 0.136 |

### Time-aware split — no leakage

```
│◄── Month 1 ──────────────────── Month 36 ──►│
│                                              │
│   TRAIN (60%)   │  VAL (20%)  │  TEST (20%) │
│   Months 1–22   │  23–28      │  29–36      │
│                 ↑             ↑             │
│            Cutoff 1       Cutoff 2          │
```

The split is strictly chronological by `month_index`. No loan's future months ever appear in its own training window. Cross-set loan-level overlap is expected (same loan at different points in time) and is not leakage.

### Feature engineering

157 features are derived from 33 raw columns:

- **Lag features** — balance, DPD, status from 1, 2, 3, 6 months prior
- **Rolling statistics** — 3-month, 6-month, 12-month rolling mean, std, max
- **Trend features** — 3-month balance change, percentage change
- **Ratio features** — `current_balance / original_balance`, balance paid percentage
- **Delinquency streak** — consecutive months delinquent
- **Seasonality** — month of year, quarter, year-end flag
- **Categorical encodings** — credit band, LTV band, DTI band, state, status

All lag and rolling features use `.shift(1)` before aggregation — no same-month lookahead.

### Class imbalance and calibration

- `class_weight='balanced'` applied to all models
- Post-training isotonic regression calibration ensures predicted probabilities are well-calibrated (a predicted 0.7 means ~70% of such loans actually experience the event)

### Model cards

Individual model cards in `models/classification/MODEL_CARD_*.md` document objective, training data, top features, metrics, limitations, and known failure modes for every model.

---

## 7. Task 3 — Survival Modeling

**Report:** [`reports/survival/survival_results.json`](reports/survival/survival_results.json)

Four survival models are trained on a loan-level dataset (one row per loan, recording time-to-first-event):

| Model | Type | Purpose | C-index |
|---|---|---|---|
| **Kaplan-Meier** | Non-parametric | Baseline survival curve, no covariates | 0.994 |
| **Cox PH — Default** | Semi-parametric | Time-to-default with all loan covariates | 0.976 |
| **Cox PH — Prepayment** | Semi-parametric | Time-to-prepayment (competing risk) | 0.962 |
| **Discrete-Time Hazard** | Logistic regression on expanded panel | Month-by-month event probability | 0.977 |

**Censoring:** Loans that reach the end of observation without an event are right-censored at their last observed month. Cox models handle censored observations correctly in the partial likelihood.

**Competing risks:** Default and prepayment are modelled as competing events (event codes 1 and 2). The discrete-time model expands each loan into one row per month and predicts the binary probability of an event occurring at each step.

---

## 8. Task 4 — Anomaly Detection

**Examples:** [`reports/copilot/demo_outputs.json`](reports/copilot/demo_outputs.json) *(reviewer_note section)*

### How scores are computed

Two complementary methods are combined into a final anomaly score in \[0, 1\]:

1. **ML scoring** — `IsolationForest` trained on 9 numeric features. Records in sparse feature-space regions receive high isolation scores.
2. **Rule-based scoring** — Each of the 14 validation rules contributes a weight when violated. Error-level violations add more weight than warnings.

Records above the 95th-percentile threshold are flagged. A separate `RandomForestClassifier` then assigns each flagged record one of five **exception types**:

| Exception Type | Meaning |
|---|---|
| `data_quality` | Value outside expected range or format |
| `servicer_conflict` | Discrepancy between primary data and servicer update |
| `stale_record` | Last-updated timestamp suggests the record is outdated |
| `document_gap` | Missing document status for an active loan |
| `balance_anomaly` | Current balance inconsistent with origination balance |

### In the test set

- **81 records** flagged as anomalous (5% of 1,612 test rows)
- Anomaly score range: 0.08 – 0.68 (mean 0.18)
- `recommended_action = "Review - anomaly detected"` for all flagged records
- Top driver features recorded in the `top_drivers` column for every row

---

## 9. Task 5 — Scenario Simulation

**Report:** [`reports/scenario/scenario_results.json`](reports/scenario/scenario_results.json)

Three macro scenarios are applied by modifying input features and re-running the trained models:

| Scenario | Shock Applied | Projected Default Rate | Projected 3-Month Delinquency |
|---|---|---|---|
| **Base** | No change | 16.0% | 16.4% |
| **Adverse Credit** | Credit band degraded by one tier | **17.6%** | **23.8%** |
| **High Prepayment** | Prepayment probability scaled up | 16.0% | 16.4% |

Under the adverse-credit scenario, 3-month delinquency rises by **+7.4 percentage points** — a **+45% relative increase** from base. The impact is largest in the `620–659` credit score band.

### Segment projections

Results are broken down by:

- Credit score band (`<620`, `620–659`, `660–699`, `700–739`, `740–779`, `780+`)
- LTV band
- DTI band

Full segment tables are in `reports/scenario/scenario_results.json` and `reports/scenario/scenario_comparison.csv`.

---

## 10. Task 6 — Explainability

**Report:** [`reports/explainability/`](reports/explainability/)

### Global feature importance (SHAP)

SHAP `TreeExplainer` is run on all five LightGBM models. Top three drivers per target:

| Target | #1 Driver | #2 Driver | #3 Driver |
|---|---|---|---|
| 12-Month Default | `interest_rate` | `credit_score_band_encoded` | `is_delinquent` |
| 3-Month Delinquency | `is_delinquent` | `interest_rate` | `balance_paid_pct` |

SHAP summary plots for all five targets are saved in `reports/explainability/plots/`.

### Local explanations

For 500 sampled validation rows, per-row SHAP values explain exactly which features pushed each individual prediction up or down from the base rate.

### Error analysis

[`reports/explainability/error_analysis.json`](reports/explainability/error_analysis.json) documents false positive and false negative case studies — showing which features misled the model — used directly to populate the failure-mode sections of each model card.

### Calibration

Reliability diagrams are computed on the validation set. All models are post-calibrated with isotonic regression so that predicted probabilities correspond accurately to observed event rates.

---

## 11. Task 7 — LLM Reviewer Copilot

**Demo:** [`reports/copilot/demo_outputs.json`](reports/copilot/demo_outputs.json)  
**Interaction log:** [`logs/llm_interaction_log.jsonl`](logs/llm_interaction_log.jsonl)  
**Rejected examples:** [`reports/copilot/rejected_examples.json`](reports/copilot/rejected_examples.json)

### What the copilot does

The copilot uses **DeepSeek** (`deepseek-chat`) via its OpenAI-compatible REST API. It provides four governed functions:

| Function | What it does |
|---|---|
| **Reviewer Note** | Loan ID + model scores + SHAP drivers → plain-English review note for a human analyst |
| **Data Q&A** | Natural-language question → answer grounded strictly in `config/data_dictionary.md` |
| **Scenario Summary** | Scenario projections → plain-English portfolio risk narrative |
| **Rule Explanation** | Rule ID → what the rule checks, why it matters, what a violation means operationally |

### Governance controls

Every output:

- Is **labelled** `AI-generated recommendation — human review required`
- Is **logged** in `logs/llm_interaction_log.jsonl` with prompt, model, timestamp, and hallucination flag
- Passes a **hallucination guard** that verifies every number in the output appears in the grounding context
- Is treated as a **recommendation only** — the copilot never makes decisions

### Documented failures — rejected examples

Three real cases where LLM output was caught and corrected are documented in [`reports/copilot/rejected_examples.json`](reports/copilot/rejected_examples.json):

| # | Type | What went wrong |
|---|---|---|
| 1 | `reviewer_note` | Recommended foreclosure (a servicing decision). Cited market conditions not in the grounding context. |
| 2 | `data_qa` | Invented GSE regulatory thresholds (97%, 80% LTV) from training memory — not from the data dictionary. |
| 3 | `scenario_summary` | Fabricated a 35–40% default rate. Grounding context showed 17.6%. Caught by numeric traceability check. |

### API key setup

The copilot detects keys in this priority order:

```
1. DEEPSEEK_API_KEY    → api.deepseek.com  (deepseek-chat)
2. OPENROUTER_API_KEY  → openrouter.ai
3. No key              → smart mock responses (pipeline runs fully without a key)
```

Copy [`.env.example`](.env.example) to `.env` and fill in your key. Never commit `.env`.

---

## 12. Task 8 — Agentic Development Evidence

See [`AI_DEVELOPMENT_LOG.md`](AI_DEVELOPMENT_LOG.md) for the full agentic coding evidence, including:

- AI tools used and their roles
- Representative prompts from each development session
- Outputs that were accepted, rejected, or corrected by the human developer
- Human review and validation steps taken at each stage
- Approximate share of AI-generated code
- Lessons learned about governing AI-assisted ML development

---

## 13. Data

### Files

| File | Rows | Loans | Description |
|---|---|---|---|
| `loan_monthly_performance_train.csv` | 9,838 | 500 | Panel data — one row per loan per month — includes target labels |
| `loan_monthly_performance_test.csv` | 1,612 | 227 | Same schema, no target labels — predict these |
| `loan_static_attributes.csv` | 500 | 500 | Origination-level attributes per loan |
| `servicer_updates.csv` | ~1,500 | — | Second-source servicer reporting for conflict detection |

### Key columns

| Column | Type | Description |
|---|---|---|
| `loan_id` | String | Unique loan identifier |
| `month_index` | Integer | Months since origination (used for time-aware split) |
| `reporting_month` | Period[M] | Calendar month of the record |
| `current_balance` | Float | Outstanding principal balance |
| `days_past_due` | Integer | Calendar days payment is overdue |
| `current_status` | String | Current / 30-59 DPD / 60-89 DPD / 90+ DPD / Default / Prepaid |
| `credit_score_band` | String | Bucketed FICO band at origination |
| `next_12m_default_flag` | Binary | **Target:** 1 if loan defaults within 12 months |
| `next_state` | String | **Target:** Loan state at next reporting month |

Full field definitions are in [`config/data_dictionary.md`](config/data_dictionary.md).

### Replacing with real organiser data

```bash
# Drop the organiser's four files into data/synthetic/ using exactly these names:
data/synthetic/loan_monthly_performance_train.csv
data/synthetic/loan_monthly_performance_test.csv
data/synthetic/loan_static_attributes.csv
data/synthetic/servicer_updates.csv

# Re-run — no code changes needed:
python run_pipeline.py
```

---

## 14. Submission File

`submission.csv` — **1,612 rows × 14 columns**

| Column | Description |
|---|---|
| `loan_id` | Loan identifier |
| `reporting_month` | Reporting period |
| `month_index` | Integer month index |
| `next_3m_delinquency_prob` | P(delinquent in 3 months) |
| `next_6m_delinquency_prob` | P(delinquent in 6 months) |
| `next_12m_default_prob` | P(default in 12 months) |
| `next_12m_prepayment_prob` | P(prepayment in 12 months) |
| `next_state_pred` | Most likely next loan state |
| `next_state_prob` | Probability of predicted state |
| `exception_type` | Predicted exception category |
| `anomaly_score` | Record-level anomaly score \[0, 1\] |
| `top_drivers` | Top contributing features for the record |
| `recommended_action` | Review / Monitor / No action |
| `confidence` | Model confidence for the recommended action |

---

## 15. Running on GitHub Codespaces

This project is designed to run fully in a Codespace. No local installation required.

**Step 1 — Open a Codespace**

On this repository page: **Code → Codespaces → Create codespace on main**

**Step 2 — Add your DeepSeek API key as a secret** *(optional)*

GitHub → Profile → Settings → Codespaces → New secret:
- Name: `DEEPSEEK_API_KEY`
- Value: your key
- Repository: select this repo

Restart the Codespace after saving.

**Step 3 — Install and run**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q               # 26 passed
python run_pipeline.py --use-synthetic   # full pipeline
python demo_visuals.py                   # 8 charts → demo_charts/
```

---

## 16. Reproducing Results

Everything is deterministic. Random seed is fixed to `42` in `src/utils/reproducibility.py`.

```bash
python run_pipeline.py --use-synthetic
```

This command alone reproduces all models, reports, and `submission.csv` from scratch.

Pre-trained `.pkl` files and pre-generated reports are committed to the repository so the project works immediately after cloning — without needing to re-run the pipeline first.

---

## 17. Deliverables Checklist

| Deliverable | Location | Status |
|---|---|---|
| GitHub repository | This repo | ✅ |
| Reproducible pipeline | `run_pipeline.py` | ✅ |
| `submission.csv` | `submission.csv` | ✅ |
| Model cards (×10) | `models/classification/MODEL_CARD_*.md` | ✅ |
| Data intelligence report | `reports/profiling/train_profile.html` | ✅ |
| Explainability report | `reports/explainability/` | ✅ |
| Scenario report | `reports/scenario/scenario_results.json` | ✅ |
| LLM copilot demo | `reports/copilot/demo_outputs.json` | ✅ |
| Rejected LLM examples | `reports/copilot/rejected_examples.json` | ✅ |
| LLM interaction log | `logs/llm_interaction_log.jsonl` | ✅ |
| Notebooks (×8) | `notebooks/` | ✅ |
| AI Development Log | `AI_DEVELOPMENT_LOG.md` | ✅ |
| Demo video | *(to be added before submission)* | ⏳ |

---

<div align="center">

*Intain Campus FinTech Challenge 2026 — AI Track*  
*Seed: 42 · Python 3.9+ · LightGBM · SHAP · lifelines · DeepSeek*

</div>

