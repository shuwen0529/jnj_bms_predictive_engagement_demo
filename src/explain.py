import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from src.utils import split_out_of_time

TARGET = "downstream_event"
TIME_COL = "month"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/synthetic_entity_month.csv")
    ap.add_argument("--model_path", default="models/gbdt_model.joblib")
    ap.add_argument("--top_n", type=int, default=8)
    args = ap.parse_args()

    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/outputs", exist_ok=True)

    df = pd.read_csv(args.data)
    _, test_df = split_out_of_time(df, TIME_COL, 9, 10)

    artifact = joblib.load(args.model_path)
    pipe = artifact["pipeline"]

    X = test_df.drop(columns=[TARGET])
    y = test_df[TARGET].values

    r = permutation_importance(pipe, X, y, n_repeats=8, random_state=42, scoring="average_precision")
    imp = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean}).sort_values("importance", ascending=False)
    imp.to_csv("reports/outputs/permutation_importance.csv", index=False)

    top_feats = imp["feature"].head(args.top_n).tolist()

    plt.figure(figsize=(9, 4))
    plt.barh(imp["feature"].head(args.top_n)[::-1], imp["importance"].head(args.top_n)[::-1])
    plt.xlabel("Permutation importance (ΔAP)")
    plt.title("Top drivers (global)")
    plt.tight_layout()
    plt.savefig("reports/figures/permutation_importance.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(pipe, X, top_feats[:min(6, len(top_feats))], ax=ax)
    plt.tight_layout()
    plt.savefig("reports/figures/partial_dependence.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved permutation importance + PDP plots.")

if __name__ == "__main__":
    main()
