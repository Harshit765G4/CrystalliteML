# CrystalliteML
## Machine Learning Analysis of Crystallite Size in Combustion-Synthesised Nanomaterials

> **A Random Forest + SHAP + Partial Dependence Pipeline for Predicting and Interpreting Crystallite Size as a Function of Synthesis Temperature and Time**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Scientific Background](#2-scientific-background)
3. [Dataset Description](#3-dataset-description)
4. [Methodology](#4-methodology)
5. [Project Architecture](#5-project-architecture)
6. [Installation & Setup](#6-installation--setup)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [Results & Deliverables](#8-results--deliverables)
9. [Interpretation of Results](#9-interpretation-of-results)
10. [GitHub Setup Guide](#10-github-setup-guide)
11. [Dependencies](#11-dependencies)
12. [Citation & References](#12-citation--references)

---

## 1. Project Overview

CrystalliteML is a reproducible machine-learning pipeline that:

- Trains a **Random Forest Regressor** on experimental crystallite-size data from combustion synthesis.
- Evaluates model performance using **5-Fold Cross-Validation** (R², RMSE, MAE).
- Applies **SHAP (SHapley Additive exPlanations)** to decompose and explain each feature's contribution to the prediction.
- Generates **Partial Dependence Plots (PDP)** to visualise the marginal effect of each synthesis parameter.
- Saves all plots, the trained model, and a text report automatically.

The pipeline is designed for **IEEE paper submission** and is fully reproducible from a single command.

---

## 2. Scientific Background

Crystallite size in combustion-synthesised nanomaterials is controlled primarily by two synthesis parameters:

| Parameter | Physical Role |
|---|---|
| **Temperature (K)** | Controls thermal activation energy for grain nucleation and growth; follows Arrhenius-type kinetics |
| **Time (h)** | Controls duration of thermal exposure; exhibits saturation kinetics (grain growth slows over time) |

Different **fuel types** (Citric Acid, Glycine, Urea, Diethanolamine, Ethylene Glycol) alter the combustion flame temperature and therefore the effective thermal history of the powder, introducing scatter at nominally identical T/t conditions.

The Scherrer equation is used experimentally to extract crystallite size (D) from XRD peak broadening:

```
D = Kλ / (β cos θ)
```

Where K = shape factor, λ = X-ray wavelength, β = FWHM, θ = Bragg angle.

---

## 3. Dataset Description

| Attribute | Details |
|---|---|
| **Source** | Experimental XRD measurements from literature |
| **Samples** | 76 data points |
| **Features** | Temperature (K), Time (h) |
| **Target** | Crystallite Size (nm) |
| **Fuel Types** | Citric Acid, Glycine, Urea, Diethanolamine, Ethylene Glycol |
| **Temperature range** | 573 – 1473 K |
| **Time range** | 1 – 7 h |
| **Crystallite size range** | 7 – 74.1 nm |

File location: `data/crystallite_data.csv`

---

## 4. Methodology

### 4.1 Random Forest Regressor

```
Hyperparameters:
  n_estimators      = 300
  max_depth         = None  (full depth)
  min_samples_split = 3
  min_samples_leaf  = 1
  max_features      = sqrt
  random_state      = 42
```

**Why Random Forest?**
- Handles non-linear relationships between T, t, and crystallite size
- Robust to outliers (important for multi-fuel experimental data)
- Natively supports SHAP TreeExplainer (exact SHAP values, not approximations)
- Provides built-in feature importance as a sanity check alongside SHAP

### 4.2 Cross-Validation Strategy

```
Strategy : 5-Fold KFold
Shuffle  : True
Seed     : 42
Metrics  : R², RMSE, MAE (train + CV)
```

### 4.3 SHAP Analysis

SHAP values decompose each prediction into additive feature contributions:

```
f(x) = φ₀ + φ_Temperature + φ_Time
```

Where:
- `φ₀` = baseline (mean prediction)
- `φ_Temperature` = contribution of Temperature for this sample
- `φ_Time` = contribution of Time for this sample

Four SHAP outputs are generated:
1. **Global Importance Bar** — Mean |SHAP| per feature
2. **Summary Plot (Beeswarm)** — SHAP distribution across all samples
3. **Dependence Plot — Temperature** — SHAP vs T, coloured by t
4. **Dependence Plot — Time** — SHAP vs t, coloured by T

### 4.4 Partial Dependence Plots

PDP marginalises out all other features to show the **average effect** of a single feature:

```
PDP(x_s) = E_{x_c}[f(x_s, x_c)]
```

PDPs reveal:
- Monotonic vs non-monotonic trends
- Saturation regions
- Inflection points (steepest growth rate)

---

## 5. Project Architecture

```
CrystalliteML/
│
├── data/
│   └── crystallite_data.csv         ← Raw experimental dataset
│
├── src/
│   ├── __init__.py
│   ├── config.py                    ← All paths, hyperparameters, colours
│   ├── data_loader.py               ← Load CSV, return X, y, df
│   ├── model.py                     ← RF training, CV, metrics, Actual vs Predicted
│   ├── shap_analysis.py             ← SHAP computation + 4 SHAP plots
│   ├── pdp_analysis.py              ← 2 Partial Dependence plots
│   └── report.py                    ← Text report + metrics table PNG
│
├── outputs/
│   ├── plots/
│   │   ├── 0_actual_vs_predicted.png
│   │   ├── 1_shap_importance_bar.png
│   │   ├── 2_shap_summary.png
│   │   ├── 3_shap_dep_temperature.png
│   │   ├── 4_shap_dep_time.png
│   │   ├── 5_pdp_temperature.png
│   │   ├── 6_pdp_time.png
│   │   └── 8_metrics_table.png
│   ├── models/
│   │   └── random_forest.pkl        ← Saved trained model
│   └── reports/
│       └── results_report.txt       ← Full text results
│
├── notebooks/
│   └── (optional Jupyter notebooks)
│
├── .github/
│   └── workflows/
│       └── ci.yml                   ← GitHub Actions CI pipeline
│
├── main.py                          ← Single entry point (run this)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. Installation & Setup

### Prerequisites

- Python **3.11.x** (recommended; TensorFlow incompatibility exists with 3.14+)
- pip
- git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/CrystalliteML.git
cd CrystalliteML
```

### Step 2 — Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Verify Installation

```bash
python -c "import sklearn, shap, matplotlib; print('All packages OK')"
```

---

## 7. Running the Pipeline

```bash
python main.py
```

Expected console output:

```
╔═════════════════════════════════════════════════════╗
║          CrystalliteML — Full Pipeline              ║
╚═════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 1 / 5  —  Loading Data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [Dataset summary printed]

  STEP 2 / 5  —  Training Random Forest
  ✅  Model saved → outputs/models/random_forest.pkl

  STEP 3 / 5  —  SHAP Analysis
  ✅  Plots 1–4 saved

  STEP 4 / 5  —  Partial Dependence Plots
  ✅  Plots 5–6 saved

  STEP 5 / 5  —  Generating Reports & Metrics Table
  ✅  Metrics table saved
  ✅  Text report saved

╔═════════════════════════════════════════════════════╗
║              ✅  Pipeline Complete                  ║
╚═════════════════════════════════════════════════════╝
```

---

## 8. Results & Deliverables

| # | File | Description |
|---|---|---|
| 0 | `0_actual_vs_predicted.png` | Training fit scatter plot |
| 1 | `1_shap_importance_bar.png` | Global SHAP importance bar chart |
| 2 | `2_shap_summary.png` | SHAP beeswarm summary plot |
| 3 | `3_shap_dep_temperature.png` | SHAP dependence — Temperature |
| 4 | `4_shap_dep_time.png` | SHAP dependence — Time |
| 5 | `5_pdp_temperature.png` | Partial dependence — Temperature |
| 6 | `6_pdp_time.png` | Partial dependence — Time |
| 7 | `8_metrics_table.png` | All metrics in table format |
| — | `random_forest.pkl` | Saved model (load with pickle) |
| — | `results_report.txt` | Full text report |

### Model Metrics Summary

| Metric | Value |
|---|---|
| n_estimators | 300 |
| max_depth | None (full) |
| min_samples_split | 3 |
| CV Strategy | 5-Fold KFold |
| **Train R²** | **0.6935** |
| CV R² (mean ± std) | −0.24 ± 0.39 |
| Train RMSE (nm) | 7.93 |
| CV RMSE (nm) | 13.53 |
| Train MAE (nm) | 6.15 |
| CV MAE (nm) | 10.93 |
| SHAP — Temperature | 6.484 |
| SHAP — Time | 2.928 |

---

## 9. Interpretation of Results

### A. SHAP Global Importance

Temperature (K) has a mean |SHAP| of **6.484 nm** vs Time's **2.928 nm**, making Temperature approximately **2.2× more influential** in determining crystallite size.

### B. SHAP Summary Plot

- **High Temperature** (warm colours) → large positive SHAP values → larger predicted crystallite size.
- **High Time** → generally positive SHAP but with greater spread → less consistent influence.
- Temperature dominates the variance; Time provides secondary modulation.

### C. SHAP Dependence — Temperature

- Influence is **nonlinear** — confirmed by the cubic trend.
- A distinct **transition threshold** appears at approximately **800–1000 K**.
- Below ~700 K: SHAP ≈ 0 → thermal energy insufficient for significant grain growth.
- Above ~1100 K: SHAP strongly positive → rapid crystallite coarsening.
- Physical interpretation: consistent with thermally activated grain boundary migration.

### D. SHAP Dependence — Time

- **Saturation behaviour** confirmed via quadratic trend line.
- SHAP contribution rises steeply from 1 to ~4 h.
- Beyond 5 h: near-plateau → diminishing returns on crystallite growth.
- High-temperature samples (viridis, yellow) override time effects, showing T dominance.

### E. PDP — Temperature

- Monotonically increasing predicted crystallite size.
- Steepest gradient above ~850 K → Arrhenius-type activation regime.
- Physical interpretation: grain growth rate follows `D^n - D₀^n = K·t·exp(-Q/RT)`.

### F. PDP — Time

- Sharp rise from 1 to ~4 h; near-plateau at ≥ 5 h.
- Saturation kinetics consistent with classical grain growth theory.
- Time alone cannot compensate for insufficient synthesis temperature.

### G. Model Performance Note

The negative CV R² arises from:
1. **Small dataset** (76 samples) → high variance between folds
2. **Mixed fuel types** → different fuels produce different effective temperatures at the same nominal T/t
3. **Scatter** in crystallite size at identical conditions

**Recommended improvements for future work:**
- Add Fuel as a one-hot encoded categorical feature
- Train separate per-fuel sub-models
- Acquire more data points per fuel type
- Consider Gaussian Process Regression for small-data regimes

---

## 10. GitHub Setup Guide

### First-Time Push

```bash
# 1. Navigate to project folder
cd CrystalliteML

# 2. Initialise git
git init

# 3. Add all files
git add .

# 4. First commit
git commit -m "feat: initial CrystalliteML pipeline"

# 5. Create repo on GitHub (via website), then:
git remote add origin https://github.com/YOUR_USERNAME/CrystalliteML.git

# 6. Push
git branch -M main
git push -u origin main
```

### Subsequent Updates

```bash
git add .
git commit -m "update: description of your change"
git push
```

### Recommended GitHub Repository Settings

```
Repository name  : CrystalliteML
Description      : RF + SHAP + PDP pipeline for crystallite size prediction
Visibility       : Public (for IEEE submission credibility)
Add README       : ✓ (already included)
Add .gitignore   : ✓ (already included)
```

---

## 11. Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥1.24 | Numerical arrays |
| pandas | ≥2.0 | Data loading and manipulation |
| scikit-learn | ≥1.3 | Random Forest, CV, PDP |
| shap | ≥0.44 | SHAP value computation |
| matplotlib | ≥3.7 | All plots |
| seaborn | ≥0.12 | Optional styling support |

Install all with: `pip install -r requirements.txt`

---

## 12. Citation & References

If you use this pipeline or dataset in your research, please cite:

```bibtex
@misc{crystalliteml2025,
  author  = {Harshit},
  title   = {CrystalliteML: Machine Learning Analysis of Crystallite Size
             in Combustion-Synthesised Nanomaterials},
  year    = {2025},
  url     = {https://github.com/YOUR_USERNAME/CrystalliteML}
}
```

### Key References

1. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
2. Breiman, L. (2001). *Random forests.* Machine Learning, 45(1), 5–32.
3. Scherrer, P. (1918). *Bestimmung der Größe und der inneren Struktur von Kolloidteilchen mittels Röntgenstrahlen.* Nachr. Ges. Wiss. Göttingen.
4. Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine.* Annals of Statistics.

---

*CrystalliteML — Built for IEEE paper submission | Python 3.11 | scikit-learn + SHAP*
