# Step-by-step (Compliance -> Commercial translation)

Each row = entity-month (analogous to HCP-month).

Signals:
- Intensity/exposure: transactions_volume, operational_metric_index
- Governance/policy: policy_deviation_rate, policy_change_exposure
- Behavioral change: behavior_shift_30d, anomaly_score
- Context: region, segment
- Data quality guardrail: data_quality_score
- History: prior_case_flag, prior_intervention_count_90d

Target:
- downstream_event (rare): escalation/finding (commercial analog: response/adoption)

Decisioning:
- lift-by-decile + capacity threshold (top-k queue)
- cost-based threshold simulation (FP vs FN tradeoff)
