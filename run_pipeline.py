#!/usr/bin/env python3
"""
run_pipeline.py — One-command end-to-end pipeline runner
=========================================================
Usage:
    python run_pipeline.py --use-synthetic          # generate data + run all stages
    python run_pipeline.py                          # use data/synthetic/ as-is
    python run_pipeline.py --n-loans 2000           # smaller run for quick testing
"""

import argparse, sys, time, warnings
from pathlib import Path
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('run_pipeline')


def run(args):
    import pandas as pd, numpy as np, joblib, json
    from pathlib import Path
    from src.utils.config import get_settings
    from src.utils.reproducibility import set_global_seed

    config = get_settings()
    set_global_seed(42)
    T0 = time.time()

    # ── 1. Data ─────────────────────────────────────────────────────────────
    logger.info("[1/10] Data Ingestion")
    if args.use_synthetic:
        from src.pipeline.synthetic_generator import SyntheticDataGenerator, SyntheticConfig
        cfg = SyntheticConfig(n_loans=args.n_loans, n_months=36)
        gen = SyntheticDataGenerator(cfg)
        train_df, test_df, static_df, servicer_df = gen.generate_all()
        Path('data/synthetic').mkdir(parents=True, exist_ok=True)
        train_df.to_csv('data/synthetic/loan_monthly_performance_train.csv', index=False)
        test_df.to_csv('data/synthetic/loan_monthly_performance_test.csv', index=False)
        static_df.to_csv('data/synthetic/loan_static_attributes.csv', index=False)
        servicer_df.to_csv('data/synthetic/servicer_updates.csv', index=False)
        logger.info(f"  Generated: train={len(train_df):,}, test={len(test_df):,}")
    else:
        train_df = pd.read_csv('data/synthetic/loan_monthly_performance_train.csv')
        test_df  = pd.read_csv('data/synthetic/loan_monthly_performance_test.csv')
        servicer_df = pd.read_csv('data/synthetic/servicer_updates.csv')
        logger.info(f"  Loaded: train={len(train_df):,}, test={len(test_df):,}")

    for col in ['reporting_month', 'origination_month']:
        for df in [train_df, test_df]:
            if col in df.columns and not hasattr(df[col].dtype, 'freq'):
                df[col] = pd.to_datetime(df[col]).dt.to_period('M')

    # ── 2. Validation ────────────────────────────────────────────────────────
    logger.info("[2/10] Validation & Servicer Reconciliation")
    from src.pipeline.loader import reconcile_servicer_updates
    from src.pipeline.validation import load_rules, apply_rules, add_quality_flag
    train_df, _ = reconcile_servicer_updates(train_df, servicer_df)
    test_df, _  = reconcile_servicer_updates(test_df, servicer_df)
    rules = load_rules()
    violations_df, summary_df = apply_rules(train_df, rules)
    train_df = add_quality_flag(train_df, violations_df)
    logger.info(f"  Violations: {len(violations_df)}")

    # ── 3. Profiling ─────────────────────────────────────────────────────────
    logger.info("[3/10] Data Profiling")
    from src.profiling.profile import profile_data
    profile_data(train_df, test_df)

    # ── 4. Feature Engineering ───────────────────────────────────────────────
    logger.info("[4/10] Feature Engineering")
    from src.features.engineer import LeakageSafeFeatureEngineer
    fe = LeakageSafeFeatureEngineer(config)
    t0 = time.time()
    X_tr_full = fe.fit_transform(train_df, is_train=True)
    X_te = fe.transform(test_df)
    logger.info(f"  FE done in {time.time()-t0:.1f}s, features={X_tr_full.shape[1]}")

    # ── 5. Supervised Models ─────────────────────────────────────────────────
    logger.info("[5/10] Supervised Model Training")
    from src.evaluation.time_split import get_split_masks
    from src.modeling.train_supervised import train_classification_models
    target_cols = ['next_3m_delinquency_flag','next_6m_delinquency_flag',
                   'next_12m_default_flag','next_12m_prepayment_flag','next_state']
    y_full = train_df[target_cols]
    train_mask, val_mask, test_mask = get_split_masks(train_df)
    X_tr, X_val, X_ti = X_tr_full[train_mask], X_tr_full[val_mask], X_tr_full[test_mask]
    y_tr, y_val, y_ti = y_full[train_mask], y_full[val_mask], y_full[test_mask]
    if len(X_tr) > 40000:
        from sklearn.model_selection import train_test_split
        X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=40000,
                                             stratify=y_tr.iloc[:,0], random_state=42)
    logger.info(f"  Split: train={len(X_tr):,}, val={len(X_val):,}, test={len(X_ti):,}")
    t0 = time.time()
    clf_trainer = train_classification_models(X_tr, y_tr, X_val, y_val, X_ti, y_ti, config)
    logger.info(f"  Training done in {time.time()-t0:.1f}s")

    models = {t: joblib.load(f'models/classification/{t}_improved.pkl')
              for t in target_cols if Path(f'models/classification/{t}_improved.pkl').exists()}

    # ── 6. Survival Modeling ─────────────────────────────────────────────────
    logger.info("[6/10] Survival & Transition Modeling")
    from src.modeling.survival import SurvivalDataBuilder, train_survival_models
    builder = SurvivalDataBuilder()
    survival_df = builder.build_survival_dataset(train_df)
    train_loans = set(train_df[train_mask]['loan_id'].unique())
    val_loans   = set(train_df[val_mask]['loan_id'].unique())
    surv_train  = survival_df['loan_id'].isin(train_loans)
    surv_val    = survival_df['loan_id'].isin(val_loans)
    train_survival_models(survival_df, surv_train, surv_val, config=config)

    # ── 7. Anomaly Detection ─────────────────────────────────────────────────
    logger.info("[7/10] Anomaly & Exception Detection")
    from src.modeling.anomaly import AnomalyDetector
    detector = AnomalyDetector(config)
    detector.fit(train_df)
    train_anomaly = detector.detect(train_df)
    test_anomaly  = detector.detect(test_df)
    detector.save_artifacts()
    logger.info(f"  Train flagged: {train_anomaly.flags.sum()}, Test flagged: {test_anomaly.flags.sum()}")

    # ── 8. Scenario Simulation ───────────────────────────────────────────────
    logger.info("[8/10] Scenario Simulation")
    from src.modeling.scenario import run_scenario_simulation
    scenario_results = run_scenario_simulation(train_df, models, fe)

    # ── 9. Explainability ────────────────────────────────────────────────────
    logger.info("[9/10] Explainability")
    from src.modeling.explain import run_explainability
    y_val_dict = {t: train_df[t][val_mask] for t in target_cols}
    preds = {t: m.predict_proba(X_val) for t,m in models.items()}
    run_explainability(models, X_tr, X_val, y_val_dict, preds, config)

    # ── 10. LLM Copilot Demo ─────────────────────────────────────────────────
    logger.info("[10/11] LLM Copilot Demo")
    try:
        from src.llm_copilot.reviewer import ReviewerCopilot
        import json, os
        copilot = ReviewerCopilot(config)
        flagged_indices = np.where(train_anomaly.flags == 1)[0]
        if len(flagged_indices) > 0:
            fi = int(flagged_indices[0])
            flagged_row = train_df.iloc[fi]
            X_f = fe.transform(train_df.iloc[[fi]])
            model_outputs = {
                t: float(m.predict_proba(X_f)[0, 1]) for t, m in models.items()
                if 'next_state' not in t
            }
            shap_drivers = train_anomaly.drivers[fi] if train_anomaly.drivers[fi] else [
                {'feature': 'days_past_due', 'contribution': 0.45}
            ]
            note = copilot.generate_reviewer_note(
                str(flagged_row['loan_id']), model_outputs, shap_drivers, ['R014'], 'data_quality'
            )
        else:
            note = copilot.answer_data_question("What does days_past_due mean for loan monitoring?")

        qa = copilot.answer_data_question("What does days_past_due indicate?")

        class _SR:
            def __init__(self, a, s): self.aggregate_projections = a; self.segment_projections = s

        sc_json = json.load(open('reports/scenario/scenario_results.json')) if (
            Path('reports/scenario/scenario_results.json').exists()
        ) else {}
        scene_objs = {k: _SR(v.get('aggregate_projections', {}), v.get('segment_projections', {}))
                      for k, v in sc_json.items()}
        scen_summary = copilot.summarize_scenario(scene_objs) if scene_objs else "No scenario results."
        rule_exp = copilot.explain_validation_rule('R014')

        demo = {
            'reviewer_note': note,
            'data_qa': qa,
            'scenario_summary': scen_summary,
            'rule_explanation': rule_exp,
            'rejected_examples': copilot.get_rejected_examples(3),
        }
        Path('reports/copilot').mkdir(parents=True, exist_ok=True)
        json.dump(demo, open('reports/copilot/demo_outputs.json', 'w'), indent=2, default=str)
        logger.info(f"  Copilot: {len(copilot.interaction_log)} interactions logged")
    except Exception as e:
        logger.warning(f"  Copilot step skipped: {e}")

    # ── 11. Submission ───────────────────────────────────────────────────────
    logger.info("[11/11] Generating Submission")
    from src.submission.generate_submission import build_submission
    submission = build_submission(test_df, models, X_te, test_anomaly)

    elapsed = time.time() - T0
    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline completed in {elapsed:.1f}s")
    logger.info(f"Submission: submission.csv ({len(submission):,} rows)")
    logger.info(f"Reports:    reports/")
    logger.info(f"Models:     models/")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loan Performance Intelligence Engine')
    parser.add_argument('--use-synthetic', action='store_true', help='Generate synthetic data')
    parser.add_argument('--n-loans', type=int, default=5000, help='Number of loans (synthetic)')
    args = parser.parse_args()
    run(args)
