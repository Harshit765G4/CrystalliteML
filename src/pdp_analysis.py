# ============================================================
#  CrystalliteML — Partial Dependence Plots (RF + XGBoost)
#  FIX: X must be float64 before calling partial_dependence
# ============================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence

from src.config import (PLOT_DIR, FEATURE_LABELS, C1, C2, BG, GRID, DPI)


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "axes.edgecolor": "#AEB8CC", "axes.labelcolor": "#1A1E2E",
        "xtick.color": "#1A1E2E", "ytick.color": "#1A1E2E",
        "text.color": "#1A1E2E", "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linestyle": "--", "grid.alpha": 0.6,
    })


def _save(fig, name: str):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✅  Plot saved → {path}")


def run_pdp(rf, xgb, X: np.ndarray):
    """
    Generate side-by-side PDP plots for Temperature and Time
    for both RF and XGBoost.

    X must be float64 — guaranteed by data_loader.
    """
    _apply_style()
    print("\n  [PDP] Computing partial dependence …")

    # Extra safety cast (prevents the integer dtype ValueError)
    X = X.astype(np.float64)

    pd_rf_temp  = partial_dependence(rf,  X, features=[0],
                                     grid_resolution=100, kind="average")
    pd_rf_time  = partial_dependence(rf,  X, features=[1],
                                     grid_resolution=100, kind="average")
    pd_xgb_temp = partial_dependence(xgb, X, features=[0],
                                     grid_resolution=100, kind="average")
    pd_xgb_time = partial_dependence(xgb, X, features=[1],
                                     grid_resolution=100, kind="average")

    _plot_pdp_temp(pd_rf_temp, pd_xgb_temp, X)
    _plot_pdp_time(pd_rf_time, pd_xgb_time, X)


# ── Private helpers ──────────────────────────────────────────

def _annotate_steepest(ax, grid, avg, color):
    grad   = np.gradient(avg, grid)
    peak_i = np.argmax(np.abs(grad))
    ax.annotate(f"Steepest rise\n≈ {grid[peak_i]:.0f} K",
                xy=(grid[peak_i], avg[peak_i]),
                xytext=(grid[peak_i] + 80, avg[peak_i] - 4),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                fontsize=8.5, color=color, fontweight="bold")


def _annotate_saturation(ax, grid, avg, color):
    grad  = np.gradient(avg, grid)
    max_g = grad.max()
    sat_i = np.argmax(grad < max_g * 0.20)
    if sat_i > 0:
        ax.annotate(f"Saturation onset\n≈ {grid[sat_i]:.1f} h",
                    xy=(grid[sat_i], avg[sat_i]),
                    xytext=(grid[sat_i] + 0.4, avg[sat_i] + 1.5),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                    fontsize=8.5, color=color, fontweight="bold")


def _plot_pdp_temp(pd_rf, pd_xgb, X):
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, pd_res, color, title in zip(
        axes, [pd_rf, pd_xgb], [C1, C2],
        ["Random Forest", "XGBoost"]
    ):
        grid = pd_res["grid_values"][0]
        avg  = pd_res["average"][0]
        ax.plot(grid, avg, color=color, lw=2.5, zorder=3)
        ax.fill_between(grid, avg, avg.min(),
                        alpha=0.12, color=color)
        rug_y = avg.min() - (avg.max() - avg.min()) * 0.05
        ax.scatter(X[:, 0], np.full(len(X), rug_y),
                   s=14, color=color, alpha=0.35,
                   marker="|", linewidths=0.9)
        _annotate_steepest(ax, grid, avg, color)
        ax.set_xlabel("Temperature (K)", fontsize=11)
        ax.set_ylabel("Predicted Crystallite Size (nm)", fontsize=11)
        ax.set_title(f"PDP: Temperature — {title}",
                     fontsize=12, fontweight="bold", color=color, pad=8)
        ax.grid(True)

    fig.set_facecolor(BG)
    fig.tight_layout()
    _save(fig, "5_pdp_temperature.png")


def _plot_pdp_time(pd_rf, pd_xgb, X):
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, pd_res, color, title in zip(
        axes, [pd_rf, pd_xgb], [C1, C2],
        ["Random Forest", "XGBoost"]
    ):
        grid = pd_res["grid_values"][0]
        avg  = pd_res["average"][0]
        ax.plot(grid, avg, color=color, lw=2.5, zorder=3)
        ax.fill_between(grid, avg, avg.min(),
                        alpha=0.12, color=color)
        rug_y = avg.min() - (avg.max() - avg.min()) * 0.05
        ax.scatter(X[:, 1], np.full(len(X), rug_y),
                   s=14, color=color, alpha=0.35,
                   marker="|", linewidths=0.9)
        _annotate_saturation(ax, grid, avg, color)
        ax.set_xlabel("Time (h)", fontsize=11)
        ax.set_ylabel("Predicted Crystallite Size (nm)", fontsize=11)
        ax.set_title(f"PDP: Time — {title}",
                     fontsize=12, fontweight="bold", color=color, pad=8)
        ax.grid(True)

    fig.set_facecolor(BG)
    fig.tight_layout()
    _save(fig, "6_pdp_time.png")
