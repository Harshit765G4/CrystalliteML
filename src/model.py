# ============================================================
#  CrystalliteML — Model Training & Evaluation
#  Models: Random Forest + XGBoost
# ============================================================
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from src.config import (RF_PARAMS, XGB_PARAMS, CV_FOLDS, RANDOM_SEED,
                        MODEL_DIR, PLOT_DIR, C1, C2, C3, ACC, BG, GRID, DPI,
                        FEATURE_LABELS)


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "axes.edgecolor": "#AEB8CC", "axes.labelcolor": "#1A1E2E",
        "xtick.color": "#1A1E2E", "ytick.color": "#1A1E2E",
        "text.color": "#1A1E2E", "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linestyle": "--", "grid.alpha": 0.6,
    })


def _compute_metrics(model, X, y, cv):
    """Compute train + CV metrics for any sklearn-compatible model."""
    y_pred    = model.predict(X)
    train_r2  = r2_score(y, y_pred)
    train_rmse= np.sqrt(mean_squared_error(y, y_pred))
    train_mae = mean_absolute_error(y, y_pred)

    cv_r2  = cross_val_score(model, X, y, cv=cv, scoring="r2")
    cv_mse = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    cv_mae = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")

    return {
        "train_r2":    round(train_r2, 4),
        "cv_r2_mean":  round(cv_r2.mean(), 4),
        "cv_r2_std":   round(cv_r2.std(), 4),
        "train_rmse":  round(train_rmse, 2),
        "cv_rmse":     round(np.sqrt(-cv_mse.mean()), 2),
        "train_mae":   round(train_mae, 2),
        "cv_mae":      round(-cv_mae.mean(), 2),
        "y_pred":      y_pred,
    }


def _save_plot(fig, name: str):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✅  Plot saved  → {path}")


def _plot_actual_vs_predicted(y, y_pred_rf, y_pred_xgb):
    """Side-by-side Actual vs Predicted for RF and XGBoost."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, y_pred, color, title in zip(
        axes,
        [y_pred_rf, y_pred_xgb],
        [C1, C2],
        ["Random Forest", "XGBoost"],
    ):
        r2 = r2_score(y, y_pred)
        ax.scatter(y, y_pred, color=color, edgecolors="white",
                   s=55, alpha=0.82, zorder=3)
        lims = [min(y.min(), y_pred.min()) - 2,
                max(y.max(), y_pred.max()) + 2]
        ax.plot(lims, lims, "--", color="#888", lw=1.5, label="Perfect fit")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual Crystallite Size (nm)", fontsize=11)
        ax.set_ylabel("Predicted Crystallite Size (nm)", fontsize=11)
        ax.set_title(f"Actual vs Predicted — {title}",
                     fontsize=12, fontweight="bold", color=color, pad=10)
        ax.text(0.05, 0.92, f"Train R² = {r2:.4f}",
                transform=ax.transAxes, fontsize=10,
                color=color, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True)

    fig.tight_layout()
    _save_plot(fig, "0_actual_vs_predicted.png")


# ════════════════════════════════════════════════════════════
#  PUBLIC: train both models
# ════════════════════════════════════════════════════════════

def train_models(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Train Random Forest + XGBoost, compute metrics, save models.

    Returns
    -------
    rf, xgb, rf_metrics, xgb_metrics
    """
    _apply_style()
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # ── Random Forest ─────────────────────────────────────────
    print("\n  [1/2] Training Random Forest …")
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X, y)
    rf_m = _compute_metrics(rf, X, y, cv)
    rf_m.update({
        "model_name": "Random Forest",
        "n_estimators": RF_PARAMS["n_estimators"],
        "max_depth":    str(RF_PARAMS["max_depth"]),
        "min_samples_split": RF_PARAMS["min_samples_split"],
        "min_samples_leaf":  RF_PARAMS["min_samples_leaf"],
        "max_features":      RF_PARAMS["max_features"],
        "cv_strategy":       f"{CV_FOLDS}-Fold KFold (shuffle=True)",
    })

    # ── XGBoost ───────────────────────────────────────────────
    print("  [2/2] Training XGBoost …")
    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(X, y)
    xgb_m = _compute_metrics(xgb, X, y, cv)
    xgb_m.update({
        "model_name":       "XGBoost",
        "n_estimators":     XGB_PARAMS["n_estimators"],
        "max_depth":        str(XGB_PARAMS["max_depth"]),
        "learning_rate":    XGB_PARAMS["learning_rate"],
        "subsample":        XGB_PARAMS["subsample"],
        "colsample_bytree": XGB_PARAMS["colsample_bytree"],
        "cv_strategy":      f"{CV_FOLDS}-Fold KFold (shuffle=True)",
    })

    # ── Print comparison ──────────────────────────────────────
    print("\n" + "=" * 57)
    print(f"  {'Metric':<25} {'RF':>12} {'XGBoost':>12}")
    print("=" * 57)
    for k, label in [
        ("train_r2",   "Train R²"),
        ("cv_r2_mean", "CV R² Mean"),
        ("cv_r2_std",  "CV R² Std"),
        ("train_rmse", "Train RMSE (nm)"),
        ("cv_rmse",    "CV RMSE (nm)"),
        ("train_mae",  "Train MAE (nm)"),
        ("cv_mae",     "CV MAE (nm)"),
    ]:
        print(f"  {label:<25} {str(rf_m[k]):>12} {str(xgb_m[k]):>12}")
    print("=" * 57)

    # ── Save models ───────────────────────────────────────────
    for name, mdl in [("random_forest.pkl", rf), ("xgboost.pkl", xgb)]:
        path = os.path.join(MODEL_DIR, name)
        with open(path, "wb") as f:
            pickle.dump(mdl, f)
        print(f"  ✅  Model saved → {path}")

    # ── Actual vs Predicted (side by side) ───────────────────
    _plot_actual_vs_predicted(y, rf_m["y_pred"], xgb_m["y_pred"])

    return rf, xgb, rf_m, xgb_m


# ════════════════════════════════════════════════════════════
#  PUBLIC: combined metrics table
# ════════════════════════════════════════════════════════════

def plot_metrics_table(rf_m: dict, xgb_m: dict,
                       rf_shap: dict, xgb_shap: dict):
    """Render side-by-side RF vs XGBoost metrics table PNG."""
    _apply_style()

    rows = [
        ["Train R²",
         str(rf_m["train_r2"]), str(xgb_m["train_r2"])],
        ["CV R² (mean ± std)",
         f"{rf_m['cv_r2_mean']} ± {rf_m['cv_r2_std']}",
         f"{xgb_m['cv_r2_mean']} ± {xgb_m['cv_r2_std']}"],
        ["Train RMSE (nm)",
         str(rf_m["train_rmse"]), str(xgb_m["train_rmse"])],
        ["CV RMSE (nm)",
         str(rf_m["cv_rmse"]), str(xgb_m["cv_rmse"])],
        ["Train MAE (nm)",
         str(rf_m["train_mae"]), str(xgb_m["train_mae"])],
        ["CV MAE (nm)",
         str(rf_m["cv_mae"]), str(xgb_m["cv_mae"])],
        ["SHAP — Temperature",
         f"{rf_shap.get('Temperature (K)', 0):.3f}",
         f"{xgb_shap.get('Temperature (K)', 0):.3f}"],
        ["SHAP — Time",
         f"{rf_shap.get('Time (h)', 0):.3f}",
         f"{xgb_shap.get('Time (h)', 0):.3f}"],
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Random Forest", "XGBoost"],
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor(C1)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF1F9")
        else:
            cell.set_facecolor(BG)

    ax.set_title("Model Comparison — RF vs XGBoost",
                 fontsize=13, fontweight="bold", color=C1, pad=12)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "8_metrics_table.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✅  Metrics table saved → {path}")
