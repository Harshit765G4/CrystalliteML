# ============================================================
#  CrystalliteML — SHAP Analysis (RF + XGBoost)
# ============================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import shap

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


def run_shap(rf, xgb, X: np.ndarray) -> tuple:
    """
    Compute SHAP for both RF and XGBoost.
    Generates 6 plots (3 per model) + 1 side-by-side importance bar.

    Returns
    -------
    rf_importance  : dict {feature: mean_abs_shap}
    xgb_importance : dict {feature: mean_abs_shap}
    """
    _apply_style()
    print("\n  [SHAP] Computing RF SHAP values …")
    rf_explainer  = shap.TreeExplainer(rf)
    rf_shap_vals  = rf_explainer.shap_values(X)

    print("  [SHAP] Computing XGBoost SHAP values …")
    xgb_explainer = shap.TreeExplainer(xgb)
    xgb_shap_vals = xgb_explainer.shap_values(X)

    rf_mean_abs  = np.abs(rf_shap_vals).mean(axis=0)
    xgb_mean_abs = np.abs(xgb_shap_vals).mean(axis=0)

    rf_importance  = {FEATURE_LABELS[i]: float(rf_mean_abs[i])
                      for i in range(len(FEATURE_LABELS))}
    xgb_importance = {FEATURE_LABELS[i]: float(xgb_mean_abs[i])
                      for i in range(len(FEATURE_LABELS))}

    print(f"\n  RF  — Temperature SHAP: {rf_mean_abs[0]:.4f} | "
          f"Time SHAP: {rf_mean_abs[1]:.4f}")
    print(f"  XGB — Temperature SHAP: {xgb_mean_abs[0]:.4f} | "
          f"Time SHAP: {xgb_mean_abs[1]:.4f}")

    # ── Plot 1: Side-by-side Importance Bar ───────────────────
    _plot_importance_bar_comparison(rf_mean_abs, xgb_mean_abs)

    # ── Plots 2–3: Summary (Beeswarm) ────────────────────────
    _plot_summary_sidebyside(rf_shap_vals, xgb_shap_vals, X)

    # ── Plots 4–5: Dependence — Temperature ───────────────────
    _plot_dependence_temp_sidebyside(rf_shap_vals, xgb_shap_vals, X)

    # ── Plots 6–7: Dependence — Time ──────────────────────────
    _plot_dependence_time_sidebyside(rf_shap_vals, xgb_shap_vals, X)

    return rf_importance, xgb_importance


# ── Private plot helpers ──────────────────────────────────────

def _plot_importance_bar_comparison(rf_abs, xgb_abs):
    _apply_style()
    x     = np.arange(len(FEATURE_LABELS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars1 = ax.bar(x - width/2, rf_abs,  width, label="Random Forest",
                   color=C1, edgecolor="white", linewidth=0.6)
    bars2 = ax.bar(x + width/2, xgb_abs, width, label="XGBoost",
                   color=C2, edgecolor="white", linewidth=0.6)

    for bars, vals in [(bars1, rf_abs), (bars2, xgb_abs)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_LABELS, fontsize=11)
    ax.set_ylabel("Mean |SHAP value|  (nm)", fontsize=11)
    ax.set_title("SHAP Global Feature Importance — RF vs XGBoost",
                 fontsize=13, fontweight="bold", color=C1, pad=10)
    ax.legend(fontsize=10); ax.grid(axis="y")
    fig.tight_layout()
    _save(fig, "1_shap_importance_bar.png")


def _plot_summary_sidebyside(rf_sv, xgb_sv, X):
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sv, title, color in zip(
        axes, [rf_sv, xgb_sv],
        ["Random Forest", "XGBoost"], [C1, C2]
    ):
        plt.sca(ax)
        shap.summary_plot(sv, X, feature_names=FEATURE_LABELS,
                          plot_type="dot", show=False, color_bar=True,
                          max_display=2, alpha=0.78)
        ax.set_title(f"SHAP Summary — {title}",
                     fontsize=12, fontweight="bold", color=color, pad=8)
        ax.set_xlabel("SHAP value (nm)", fontsize=10)
        ax.set_facecolor(BG)

    fig.set_facecolor(BG)
    fig.tight_layout()
    _save(fig, "2_shap_summary.png")


def _plot_dependence_temp_sidebyside(rf_sv, xgb_sv, X):
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, sv, title, color in zip(
        axes, [rf_sv, xgb_sv],
        ["Random Forest", "XGBoost"], [C1, C2]
    ):
        sc = ax.scatter(X[:, 0], sv[:, 0], c=X[:, 1], cmap="plasma",
                        s=60, alpha=0.82, edgecolors="white",
                        linewidths=0.4, zorder=3)
        fig.colorbar(sc, ax=ax, pad=0.02, label="Time (h)")

        z  = np.polyfit(X[:, 0], sv[:, 0], 3)
        xr = np.linspace(X[:, 0].min(), X[:, 0].max(), 300)
        ax.plot(xr, np.poly1d(z)(xr), color="white", lw=3, zorder=4)
        ax.plot(xr, np.poly1d(z)(xr), color=color, lw=2,
                ls="--", label="Cubic trend", zorder=5)
        ax.axhline(0, color="#aaa", lw=0.9, ls=":")

        ax.set_xlabel("Temperature (K)", fontsize=11)
        ax.set_ylabel("SHAP value (nm)", fontsize=11)
        ax.set_title(f"SHAP Dependence: Temperature — {title}",
                     fontsize=12, fontweight="bold", color=color, pad=8)
        ax.legend(fontsize=9); ax.grid(True)

    fig.set_facecolor(BG)
    fig.tight_layout()
    _save(fig, "3_shap_dep_temperature.png")


def _plot_dependence_time_sidebyside(rf_sv, xgb_sv, X):
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, sv, title, color in zip(
        axes, [rf_sv, xgb_sv],
        ["Random Forest", "XGBoost"], [C1, C2]
    ):
        sc = ax.scatter(X[:, 1], sv[:, 1], c=X[:, 0], cmap="viridis",
                        s=60, alpha=0.82, edgecolors="white",
                        linewidths=0.4, zorder=3)
        fig.colorbar(sc, ax=ax, pad=0.02, label="Temperature (K)")

        z  = np.polyfit(X[:, 1], sv[:, 1], 2)
        xr = np.linspace(X[:, 1].min(), X[:, 1].max(), 300)
        ax.plot(xr, np.poly1d(z)(xr), color="white", lw=3, zorder=4)
        ax.plot(xr, np.poly1d(z)(xr), color=color, lw=2,
                ls="--", label="Quadratic trend", zorder=5)
        ax.axhline(0, color="#aaa", lw=0.9, ls=":")

        ax.set_xlabel("Time (h)", fontsize=11)
        ax.set_ylabel("SHAP value (nm)", fontsize=11)
        ax.set_title(f"SHAP Dependence: Time — {title}",
                     fontsize=12, fontweight="bold", color=color, pad=8)
        ax.legend(fontsize=9); ax.grid(True)

    fig.set_facecolor(BG)
    fig.tight_layout()
    _save(fig, "4_shap_dep_time.png")
