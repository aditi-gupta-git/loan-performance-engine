"""
Validation Layer
================
Apply deterministic validation rules from config/validation_rules.json.
Generates a per-record quality flag and rule-violation summary.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)


def load_rules(rules_path: str = "config/validation_rules.json") -> List[Dict]:
    """Load validation rules from JSON config."""
    p = Path(rules_path)
    if not p.exists():
        logger.warning(f"Rules file not found: {rules_path}")
        return []
    with open(p) as f:
        return json.load(f).get("rules", [])


def apply_rules(df: pd.DataFrame, rules: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply all rules to df.

    Returns
    -------
    (violations_df, summary_df)
        violations_df : rows × rule columns, one row per violation
        summary_df    : one row per rule, with violation count and rate
    """
    violations = []
    for rule in rules:
        try:
            mask = df.eval(rule["check"])
            bad = df[~mask].copy()
            if len(bad) > 0:
                bad["rule_id"] = rule["rule_id"]
                bad["rule_name"] = rule["name"]
                bad["severity"] = rule["severity"]
                bad["description"] = rule["description"]
                violations.append(bad)
        except Exception as e:
            logger.warning(f"Rule {rule['rule_id']} failed: {e}")

    if violations:
        violations_df = pd.concat(violations, ignore_index=True)
    else:
        violations_df = pd.DataFrame(columns=["loan_id", "rule_id", "rule_name", "severity"])

    # Summary
    if len(violations_df) > 0:
        summary = (
            violations_df.groupby(["rule_id", "rule_name", "severity"])
            .size()
            .reset_index(name="violation_count")
        )
        summary["violation_rate"] = summary["violation_count"] / len(df)
    else:
        summary = pd.DataFrame(columns=["rule_id", "rule_name", "severity", "violation_count", "violation_rate"])

    return violations_df, summary


def add_quality_flag(df: pd.DataFrame, violations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a per-record data_quality_flag column:
      0 = clean, 1 = warning, 2 = error
    """
    df = df.copy()
    df["data_quality_flag"] = 0

    if len(violations_df) == 0:
        return df

    # Error-level violations → flag = 2
    error_ids = set(violations_df[violations_df["severity"] == "error"]["loan_id"].tolist())
    warn_ids = set(violations_df[violations_df["severity"] == "warning"]["loan_id"].tolist())

    df.loc[df["loan_id"].isin(warn_ids), "data_quality_flag"] = 1
    df.loc[df["loan_id"].isin(error_ids), "data_quality_flag"] = 2

    logger.info(
        f"Quality flags: clean={( df['data_quality_flag']==0).sum()}, "
        f"warning={(df['data_quality_flag']==1).sum()}, "
        f"error={(df['data_quality_flag']==2).sum()}"
    )
    return df
