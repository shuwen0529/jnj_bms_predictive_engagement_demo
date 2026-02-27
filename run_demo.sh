#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw models reports/figures reports/outputs

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python  -m src.make_data
python  -m src.train --model gbdt
python  -m src.evaluate --model_path models/gbdt_model.joblib --capacity_pct 0.05
python  -m src.threshold_simulation --model_path models/gbdt_model.joblib
python  -m src.explain --model_path models/gbdt_model.joblib
python  -m src.drift_check --model_path models/gbdt_model.joblib
python  -m src.commercial_translation --capacity_pct 0.05

echo "Done. See reports/outputs/* and reports/figures/*"
