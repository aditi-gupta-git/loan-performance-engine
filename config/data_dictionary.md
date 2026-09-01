# Data Dictionary - Loan Performance Intelligence Engine

## Core Identifiers
| Field | Type | Description |
|-------|------|-------------|
| `loan_id` | string | Unique loan identifier |
| `month_index` | integer | Sequential month index (1 = first month of observation) |
| `reporting_month` | string (YYYY-MM) | Calendar month of the observation |
| `origination_month` | string (YYYY-MM) | Month when loan was originated |

## Loan Characteristics (Static)
| Field | Type | Description |
|-------|------|-------------|
| `original_balance` | float | Original principal balance at origination |
| `interest_rate` | float | Note rate at origination (decimal, e.g., 0.055 = 5.5%) |
| `credit_score_band` | categorical | Borrower credit score band at origination: `<620`, `620-659`, `660-699`, `700-739`, `740-779`, `780+` |
| `ltv_band` | categorical | Loan-to-value band at origination: `<60%`, `60-70%`, `70-80%`, `80-90%`, `90-100%`, `>100%` |
| `dti_band` | categorical | Debt-to-income band at origination: `<20%`, `20-30%`, `30-36%`, `36-43%`, `43-50%`, `>50%` |
| `state` | categorical | Property state (2-letter code) |
| `loan_purpose` | categorical | Purpose: `Purchase`, `Refinance`, `Cash-out Refinance` |
| `occupancy_type` | categorical | Occupancy: `Primary`, `Second Home`, `Investment` |
| `property_type` | categorical | Property type: `SFR`, `Condo`, `2-4 Unit`, `Manufactured` |
| `servicer_name` | categorical | Servicer name |

## Dynamic Monthly Features
| Field | Type | Description |
|-------|------|-------------|
| `loan_age_months` | integer | Months since origination |
| `remaining_term_months` | integer | Remaining months to maturity |
| `current_balance` | float | Current outstanding principal balance |
| `current_status` | categorical | Current delinquency status: `Current`, `30-59 DPD`, `60-89 DPD`, `90+ DPD`, `Prepaid`, `Closed`, `Defaulted` |
| `days_past_due` | integer | Days past due (0 if current) |
| `modification_flag` | binary | Whether loan was modified in this month (0/1) |
| `prepayment_flag` | binary | Whether loan prepaid in this month (0/1) |
| `default_flag` | binary | Whether loan defaulted in this month (0/1) |
| `loss_severity_band` | categorical | Loss severity if defaulted: `Low`, `Medium`, `High`, `NA` |
| `last_updated_at` | datetime | Timestamp of last update |
| `source_system` | categorical | Source system: `Primary`, `Servicer` |
| `document_status` | categorical | Document completeness: `Complete`, `Incomplete`, `Missing` |

## Target Variables (Derived)
| Field | Type | Description |
|-------|------|-------------|
| `next_3m_delinquency_flag` | binary | 1 if loan becomes 30+ DPD within next 3 months |
| `next_6m_delinquency_flag` | binary | 1 if loan becomes 30+ DPD within next 6 months |
| `next_12m_default_flag` | binary | 1 if loan defaults within next 12 months |
| `next_12m_prepayment_flag` | binary | 1 if loan prepays within next 12 months |
| `next_state` | categorical | Predicted state in next month: `Current`, `30-59 DPD`, `60-89 DPD`, `90+ DPD`, `Prepaid`, `Defaulted` |
| `exception_required` | binary | Whether record needs reviewer attention |
| `exception_type` | categorical | Type of exception: `data_quality`, `business_rule`, `pattern_anomaly`, `servicer_conflict` |

## Servicer Updates (Secondary Source)
| Field | Type | Description |
|-------|------|-------------|
| `loan_id` | string | Loan identifier |
| `reporting_month` | string (YYYY-MM) | Reporting month |
| `servicer_current_balance` | float | Balance per servicer |
| `servicer_current_status` | categorical | Status per servicer |
| `servicer_days_past_due` | integer | DPD per servicer |
| `servicer_last_updated` | datetime | Servicer update timestamp |

## Macro Scenarios
| Field | Type | Description |
|-------|------|-------------|
| `scenario_name` | string | Scenario identifier: `base`, `adverse_credit`, `high_prepayment` |
| `rate_shift` | float | Parallel shift in interest rates (decimal) |
| `unemployment_delta` | float | Change in unemployment rate (decimal) |
| `hpi_delta` | float | Change in home price index (decimal) |
| `credit_spread_widening` | float | Widening of credit spreads (decimal) |
| `prepayment_multiplier` | float | Multiplier for prepayment propensity |
| `default_multiplier` | float | Multiplier for default propensity |