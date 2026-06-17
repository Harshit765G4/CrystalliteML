#!/usr/bin/env python3
# ============================================================
#  CrystalliteML — Main Pipeline
#  Models: Random Forest + XGBoost
#  Run  : python main.py
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader   import load_data
from src.model         import train_models, plot_metrics_table
from src.shap_analysis import run_shap
from src.pdp_analysis  import run_pdp
from src.report        import save_text_report


def main():
    print("\n" + "╔" + "═" * 55 + "╗")
    print("║       CrystalliteML — RF + XGBoost Pipeline        ║")
    print("╚" + "═" * 55 + "╝\n")

    # ── Step 1: Load Data ──────────────────────────────────────
    print("━" * 57)
    print("  STEP 1 / 5  —  Loading Data")
    print("━" * 57)
    X, y, df = load_data(verbose=True)

    # ── Step 2: Train Models ───────────────────────────────────
    print("\n" + "━" * 57)
    print("  STEP 2 / 5  —  Training RF + XGBoost")
    print("━" * 57)
    rf, xgb, rf_m, xgb_m = train_models(X, y)

    # ── Step 3: SHAP Analysis ──────────────────────────────────
    print("\n" + "━" * 57)
    print("  STEP 3 / 5  —  SHAP Analysis (RF + XGBoost)")
    print("━" * 57)
    rf_shap, xgb_shap = run_shap(rf, xgb, X)

    # ── Step 4: Partial Dependence ─────────────────────────────
    print("\n" + "━" * 57)
    print("  STEP 4 / 5  —  Partial Dependence Plots")
    print("━" * 57)
    run_pdp(rf, xgb, X)

    # ── Step 5: Reports & Metrics Table ───────────────────────
    print("\n" + "━" * 57)
    print("  STEP 5 / 5  —  Metrics Table + Text Report")
    print("━" * 57)
    plot_metrics_table(rf_m, xgb_m, rf_shap, xgb_shap)
    save_text_report(rf_m, xgb_m, rf_shap, xgb_shap)

    # ── Final Summary ──────────────────────────────────────────
    print("\n" + "╔" + "═" * 55 + "╗")
    print("║              ✅  Pipeline Complete                  ║")
    print("╚" + "═" * 55 + "╝")
    print("\n  Outputs saved to ./outputs/")
    print("  ├── plots/   → 7 PNG files")
    print("  │     0_actual_vs_predicted.png")
    print("  │     1_shap_importance_bar.png")
    print("  │     2_shap_summary.png")
    print("  │     3_shap_dep_temperature.png")
    print("  │     4_shap_dep_time.png")
    print("  │     5_pdp_temperature.png")
    print("  │     6_pdp_time.png")
    print("  │     8_metrics_table.png")
    print("  ├── models/  → random_forest.pkl + xgboost.pkl")
    print("  └── reports/ → results_report.txt\n")


if __name__ == "__main__":
    main()
