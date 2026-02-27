# Predictive Engagement & Risk Prioritization Framework (Synthetic Demo)

End-to-end **predictive prioritization engine** framed as **J&J Compliance/Risk** work, with explicit mapping to
**Commercial Next Best Action (NBA) / Predictive Customer Engagement** (e.g., BMS).

## What it demonstrates
- Entity-period dataset + temporal feature engineering (leading indicators)
- Random Forest + Gradient Boosting baselines (XGBoost-ready pattern)
- Out-of-time validation (months 1–9 train, 10–12 test)
- Decision usefulness: PR, lift, calibration
- Capacity + cost-based threshold simulation (FP vs FN tradeoff)
- Explainability: permutation importance + partial dependence
- Drift monitoring: feature + score PSI
- Commercial translation artifact

## Quickstart
```bash
pip install -r requirements.txt
python src/make_data.py
python src/train.py --model gbdt
python src/evaluate.py --model_path models/gbdt_model.joblib --capacity_pct 0.05
python src/threshold_simulation.py --model_path models/gbdt_model.joblib
python src/explain.py --model_path models/gbdt_model.joblib
python src/drift_check.py --model_path models/gbdt_model.joblib
python src/commercial_translation.py --capacity_pct 0.05
```

## Outputs
- reports/outputs/decision_summary.md
- reports/outputs/threshold_simulation.csv
- reports/outputs/commercial_summary.md
- reports/figures/*.png
