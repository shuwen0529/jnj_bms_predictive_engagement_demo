import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from src.utils import split_out_of_time

TARGET = "downstream_event"
TIME_COL = "month"

def lift_by_decile(y_true, scores, n_bins=10):
    df = pd.DataFrame({"y": y_true, "s": scores}).sort_values("s", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, n_bins, labels=False) + 1
    out = df.groupby("decile").agg(n=("y","size"), event_rate=("y","mean"), avg_score=("s","mean")).reset_index()
    base_rate = df["y"].mean()
    out["lift_vs_base"] = out["event_rate"] / (base_rate + 1e-12)
    return out, float(base_rate)

def capacity_threshold(scores, capacity_pct):
    k = max(1, int(np.floor(capacity_pct * len(scores))))
    cutoff = np.sort(scores)[-k]
    return float(cutoff), k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/synthetic_entity_month.csv")
    ap.add_argument("--model_path", default="models/gbdt_model.joblib")
    ap.add_argument("--capacity_pct", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/outputs", exist_ok=True)

    df = pd.read_csv(args.data)
    _, test_df = split_out_of_time(df, TIME_COL, 9, 10)

    artifact = joblib.load(args.model_path)
    pipe = artifact["pipeline"]

    X = test_df.drop(columns=[TARGET])
    y = test_df[TARGET].values
    scores = pipe.predict_proba(X)[:, 1]

    precision, recall, _ = precision_recall_curve(y, scores)
    ap_score = float(average_precision_score(y, scores))
    brier = float(brier_score_loss(y, scores))

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR (Out-of-time) | AP={ap_score:.3f}")
    plt.savefig("reports/figures/pr_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    lift_tbl, base_rate = lift_by_decile(y, scores, 10)
    lift_tbl.to_csv("reports/outputs/lift_by_decile.csv", index=False)

    plt.figure()
    plt.plot(lift_tbl["decile"], lift_tbl["lift_vs_base"], marker="o")
    plt.xlabel("Decile (1=highest score)")
    plt.ylabel("Lift vs base rate")
    plt.title("Lift by decile (Out-of-time)")
    plt.savefig("reports/figures/lift_by_decile.png", dpi=200, bbox_inches="tight")
    plt.close()

    frac_pos, mean_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration | Brier={brier:.3f}")
    plt.savefig("reports/figures/calibration.png", dpi=200, bbox_inches="tight")
    plt.close()

    cutoff, k = capacity_threshold(scores, args.capacity_pct)
    top_rate = float(np.mean(y[scores >= cutoff]))

    with open("reports/outputs/decision_summary.md", "w") as f:
        f.write("# Decision Summary (Out-of-time months 10–12)\\n\\n")
        f.write(f"- Base event rate: **{base_rate:.4f}**\\n")
        f.write(f"- Average Precision (PR AUC): **{ap_score:.4f}**\\n")
        f.write(f"- Brier score: **{brier:.4f}**\\n\\n")
        f.write("## Capacity-based prioritization\\n")
        f.write(f"- Capacity_pct: **{args.capacity_pct:.2%}** (top **{k}** entity-months)\\n")
        f.write(f"- Score cutoff: **{cutoff:.4f}**\\n")
        f.write(f"- Event rate in prioritized queue: **{top_rate:.4f}**\\n\\n")
        f.write("## Artifacts\\n")
        f.write("- reports/figures/pr_curve.png\\n")
        f.write("- reports/figures/lift_by_decile.png\\n")
        f.write("- reports/figures/calibration.png\\n")
        f.write("- reports/outputs/lift_by_decile.csv\\n")

    print("Wrote reports/outputs/decision_summary.md and figures.")

if __name__ == "__main__":
    main()
