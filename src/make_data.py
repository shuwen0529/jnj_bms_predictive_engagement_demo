import os
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1/(1+np.exp(-x))

def main(out_path="data/raw/synthetic_entity_month.csv", n_entities=2000, months=12, seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    regions = ["NE","SE","MW","SW","W"]
    segments = ["A_high","B_mid","C_low"]

    rows = []
    for e in range(1, n_entities+1):
        region = rng.choice(regions, p=[0.22,0.18,0.22,0.18,0.20])
        segment = rng.choice(segments, p=[0.20,0.50,0.30])

        baseline = rng.normal(0, 0.9) + (1 if segment=="A_high" else 0)*0.6
        prior_case_flag = rng.binomial(1, 0.08 + 0.05*sigmoid(baseline))

        prior_intervention_count_90d = 0

        for m in range(1, months+1):
            policy_change_exposure = rng.binomial(1, 0.08 + 0.18*(m in [4,8,12]))
            transactions_volume = rng.lognormal(mean=2.2 + 0.25*baseline, sigma=0.65)
            operational_metric_index = rng.normal(0.0 + 0.35*baseline, 1.0)

            policy_deviation_rate = np.clip(
                rng.beta(1.5, 18) + 0.03*policy_change_exposure + 0.02*sigmoid(baseline), 0, 1
            )
            anomaly_score = np.clip(
                rng.normal(0.5 + 0.45*sigmoid(baseline) + 0.8*policy_change_exposure, 0.7), 0, 5
            )
            behavior_shift_30d = rng.normal(0.0 + 0.55*baseline + 0.35*policy_change_exposure, 0.95)
            data_quality_score = np.clip(rng.normal(0.88 - 0.10*policy_change_exposure, 0.08), 0.45, 0.99)

            # intervention logs (synthetic): more likely when anomaly/deviation/prior case is high
            intervention_prob = sigmoid(-2.0 + 0.6*anomaly_score + 2.0*policy_deviation_rate + 0.9*prior_case_flag)
            intervention_this_month = rng.binomial(1, intervention_prob*0.35)
            prior_intervention_count_90d = min(6, prior_intervention_count_90d + intervention_this_month)

            # downstream event (rare); interventions reduce probability slightly (synthetic effect)
            logit = (
                -4.4
                + 0.30*np.log1p(transactions_volume)
                + 0.70*policy_change_exposure
                + 1.80*policy_deviation_rate
                + 0.55*anomaly_score
                + 0.55*behavior_shift_30d
                + 0.85*prior_case_flag
                + 0.25*operational_metric_index
                - 0.25*prior_intervention_count_90d
                - 1.10*(1-data_quality_score)
                + rng.normal(0, 0.35)
            )
            p = sigmoid(logit)
            downstream_event = rng.binomial(1, p)

            rows.append({
                "entity_id": e,
                "month": m,
                "region": region,
                "segment": segment,
                "transactions_volume": float(transactions_volume),
                "operational_metric_index": float(operational_metric_index),
                "policy_change_exposure": int(policy_change_exposure),
                "policy_deviation_rate": float(policy_deviation_rate),
                "anomaly_score": float(anomaly_score),
                "behavior_shift_30d": float(behavior_shift_30d),
                "prior_case_flag": int(prior_case_flag),
                "prior_intervention_count_90d": int(prior_intervention_count_90d),
                "data_quality_score": float(data_quality_score),
                "downstream_event": int(downstream_event),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} | rows={len(df):,} | event_rate={df['downstream_event'].mean():.4f}")

if __name__ == "__main__":
    main()
