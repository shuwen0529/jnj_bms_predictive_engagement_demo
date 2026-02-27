"""
Commercial Translation: Compliance -> Commercial Predictive Engagement / NBA

Run after:
- src/evaluate.py
- src/explain.py
"""
import os
import argparse
import pandas as pd

def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity_pct", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs("reports/outputs", exist_ok=True)

    lift = load_csv("reports/outputs/lift_by_decile.csv")
    imp = load_csv("reports/outputs/permutation_importance.csv")

    base_rate = top_dec_rate = top_dec_lift = None
    if lift is not None and len(lift):
        base_rate = float((lift["event_rate"] * lift["n"]).sum() / lift["n"].sum())
        top = lift.sort_values("decile").iloc[0]
        top_dec_rate = float(top["event_rate"])
        top_dec_lift = float(top["lift_vs_base"])

    top_drivers = []
    if imp is not None and len(imp):
        top = imp.sort_values("importance", ascending=False).head(6)
        top_drivers = [str(x) for x in top["feature"].tolist()]

    out_md = "reports/outputs/commercial_summary.md"
    with open(out_md, "w") as f:
        f.write("# Commercial Translation Summary\n\n")
        f.write("## Mapping\n")
        f.write("- Entity-month → HCP/customer-period\n")
        f.write("- Downstream risk event → response/adoption/engagement outcome\n")
        f.write("- Risk score → propensity/priority score\n")
        f.write(f"- Review capacity (top {args.capacity_pct:.0%}) → field/channel capacity constraint\n\n")

        f.write("## NBA logic enabled\n")
        f.write("Score and rank customers, then select an operating point under constraints to drive action.\n\n")

        if base_rate is not None:
            f.write("## Evidence prioritization works\n")
            f.write(f"- Base outcome rate (all): **{base_rate:.4f}**\n")
            f.write(f"- Top decile outcome rate: **{top_dec_rate:.4f}**\n")
            f.write(f"- Top decile lift vs base: **{top_dec_lift:.2f}x**\n\n")

        if top_drivers:
            f.write("## Top drivers (actionable)\n")
            for d in top_drivers:
                f.write(f"- {d}\n")
            f.write("\n")

        f.write("## Why this transfers\n")
        f.write("Same decision engine: predict → rank → apply constraints → explain → monitor. Only the objective changes.\n")

    print(f"Wrote {out_md}")

if __name__ == "__main__":
    main()
