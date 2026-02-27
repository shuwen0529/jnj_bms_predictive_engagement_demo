import argparse
import numpy as np
import pandas as pd
import joblib

TARGET = "downstream_event"
TIME_COL = "month"

def psi(a, b, bins=10):
    edges = np.quantile(a, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    a_hist, _ = np.histogram(a, bins=edges)
    b_hist, _ = np.histogram(b, bins=edges)
    a_p = (a_hist + 1e-6) / (a_hist.sum() + 1e-6)
    b_p = (b_hist + 1e-6) / (b_hist.sum() + 1e-6)
    return float(np.sum((b_p - a_p) * np.log(b_p / a_p)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/synthetic_entity_month.csv")
    ap.add_argument("--model_path", default="models/gbdt_model.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    base = df[df[TIME_COL] <= 9].copy()
    recent = df[df[TIME_COL] >= 10].copy()

    artifact = joblib.load(args.model_path)
    pipe = artifact["pipeline"]

    print("=== Feature drift (PSI) baseline months1-9 vs recent months10-12 ===")
    for c in ["transactions_volume", "policy_deviation_rate", "anomaly_score", "behavior_shift_30d", "data_quality_score"]:
        print(f"{c:24s} PSI={psi(base[c].values, recent[c].values):.4f}")

    base_scores = pipe.predict_proba(base.drop(columns=[TARGET]))[:, 1]
    recent_scores = pipe.predict_proba(recent.drop(columns=[TARGET]))[:, 1]

    print("\\n=== Score drift (PSI) ===")
    print(f"{'score':24s} PSI={psi(base_scores, recent_scores):.4f}")

if __name__ == "__main__":
    main()
