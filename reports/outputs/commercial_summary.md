# Commercial Translation Summary

## Mapping
- Entity-month → HCP/customer-period
- Downstream risk event → response/adoption/engagement outcome
- Risk score → propensity/priority score
- Review capacity (top 5%) → field/channel capacity constraint

## NBA logic enabled
Score and rank customers, then select an operating point under constraints to drive action.

## Evidence prioritization works
- Base outcome rate (all): **0.0600**
- Top decile outcome rate: **0.2150**
- Top decile lift vs base: **3.58x**

## Top drivers (actionable)
- behavior_shift_30d
- anomaly_score
- policy_change_exposure
- prior_case_flag
- transactions_volume
- operational_metric_index

## Why this transfers
Same decision engine: predict → rank → apply constraints → explain → monitor. Only the objective changes.
