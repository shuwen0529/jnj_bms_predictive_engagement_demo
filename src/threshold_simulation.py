import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from src.utils import split_out_of_time

TARGET = "downstream_event"
TIME_COL = "month"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/synthetic_entity_month.csv")
    ap.add_argument("--model_path", default="models/gbdt_model.joblib")
    ap.add_argument("--fp_cost", type=float, default=1.0)
    ap.add_argument("--fn_cost", type=float, default=8.0)
    ap.add_argument("--grid", type=int, default=50)
    args = ap.parse_args()

    os.makedirs("reports/outputs", exist_ok=True)

    df = pd.read_csv(args.data)
    _, test_df = split_out_of_time(df, TIME_COL, 9, 10)

    artifact = joblib.load(args.model_path)
    pipe = artifact["pipeline"]

    X = test_df.drop(columns=[TARGET])
    y = test_df[TARGET].values
    scores = pipe.predict_proba(X)[:, 1]

    thresholds = np.quantile(scores, np.linspace(0.01, 0.99, args.grid))
    rows = []
    for t in thresholds:
        yhat = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        reviewed = tp + fp
        cost = args.fp_cost * fp + args.fn_cost * fn
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        rows.append({
            "threshold": float(t),
            "reviewed_count": int(reviewed),
            "review_rate": float(reviewed / len(y)),
            "precision": float(precision),
            "recall": float(recall),
            "expected_cost": float(cost),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })

    out = pd.DataFrame(rows).sort_values("expected_cost").reset_index(drop=True)
    out.to_csv("reports/outputs/threshold_simulation.csv", index=False)

    best = out.iloc[0]
    print("Best threshold by expected cost:")
    print(best[["threshold", "review_rate", "precision", "recall", "expected_cost"]].to_string())

if __name__ == "__main__":
    main()
