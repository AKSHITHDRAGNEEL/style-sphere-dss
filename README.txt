STYLE SPHERE DSS — FINAL ANALYTICALLY VERIFIED PACKAGE

Run on Windows: double-click run_windows.bat.
Or from this folder: streamlit run app.py

Inputs: Style_Sphere_Model.xlsx
Model: model.py
Dashboard: app.py
Verification: TEST_REPORT.txt

The model-backed dashboard supports:
- Online and store demand changes
- Pooling limit: 30% / 40% / 50%
- Online fill-rate target
- Enablement-cost multiplier
- Store stockout penalty
- Current promotion / warehouse-disruption scenario switches
- Model-recommended or manual store selection
- Monte Carlo simulation with promotion/disruption probabilities
- Store-level stockout risk

The Python model is the single source of truth for the dashboard.

Analytical methods implemented:
- Fixed-charge transportation / store enablement
- Weighted Goal Programming with deviation variables
- MINIMAX cost-vs-service pooling sensitivity
- Monte Carlo risk simulation
