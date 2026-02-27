import os
import argparse
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline

from src.utils import build_preprocess, split_out_of_time

CAT_COLS = ["region","segment"]
NUM_COLS = [
    "transactions_volume","operational_metric_index","policy_change_exposure",
    "policy_deviation_rate","anomaly_score","behavior_shift_30d",
    "prior_case_flag","prior_intervention_count_90d","data_quality_score","month"
]
TARGET = "downstream_event"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/synthetic_entity_month.csv")
    ap.add_argument("--model", choices=["rf","gbdt"], default="gbdt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    train_df, test_df = split_out_of_time(df, "month", 9, 10)

    X_train = train_df[CAT_COLS + NUM_COLS]
    y_train = train_df[TARGET].values
    X_test  = test_df[CAT_COLS + NUM_COLS]
    y_test  = test_df[TARGET].values

    preprocess = build_preprocess(CAT_COLS, NUM_COLS)

    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=20,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        out_path = "models/rf_model.joblib"
    else:
        clf = GradientBoostingClassifier(random_state=args.seed)
        out_path = "models/gbdt_model.joblib"

    pipe = Pipeline([("prep", preprocess), ("model", clf)])
    pipe.fit(X_train, y_train)

    scores = pipe.predict_proba(X_test)[:, 1]
    print("Out-of-time ROC-AUC:", round(roc_auc_score(y_test, scores), 4))
    print("Out-of-time Avg Precision:", round(average_precision_score(y_test, scores), 4))

    os.makedirs("models", exist_ok=True)
    joblib.dump({"pipeline": pipe}, out_path)
    print(f"Saved model: {out_path}")

if __name__ == "__main__":
    main()
