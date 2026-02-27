mkdir -Force data\raw, models, reports\figures, reports\outputs | Out-Null

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python src/make_data.py
python src/train.py --model gbdt
python src/evaluate.py --model_path models/gbdt_model.joblib --capacity_pct 0.05
python src/threshold_simulation.py --model_path models/gbdt_model.joblib
python src/explain.py --model_path models/gbdt_model.joblib
python src/drift_check.py --model_path models/gbdt_model.joblib
python src/commercial_translation.py --capacity_pct 0.05

Write-Host "Done. See reports/outputs/* and reports/figures/*"
