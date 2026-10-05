# 🔬 CrystalliteML

### Machine Learning Analysis of Crystallite Size in Combustion-Synthesised Nanomaterials

CrystalliteML is a reproducible Python pipeline for modeling **crystallite size** from synthesis **temperature** and **time**, then interpreting the learned relationships with **SHAP** and **Partial Dependence Plots (PDP)**.

The current implementation compares two tree-based regression models:

- **Random Forest Regressor**
- **XGBoost Regressor**

The pipeline automatically trains both models, evaluates them with 5-fold cross-validation, generates explainability plots, saves trained models, and produces a text report.

## 🎯 Project Overview

The pipeline answers two practical questions:

1. How well can machine-learning models predict crystallite size from synthesis conditions?
2. Which synthesis parameter—temperature or time—has the strongest influence on the predictions?

Current model inputs:

```text
Temperature_K
Time_h
```

Target:

```text
Crystallite_Size
```

The dataset also records **Fuel**, but the current training pipeline does **not** use Fuel as a model feature.

## 🧪 Scientific Context

Crystallite growth in combustion-synthesised materials is influenced by thermal exposure.

- **Temperature** affects thermal activation and grain-growth kinetics.
- **Time** controls how long the material remains under the synthesis conditions.
- Different fuels can introduce additional variation even when nominal temperature and time are similar.

The crystallite sizes in the dataset are experimentally derived values, while the machine-learning stage learns an empirical relationship between the recorded synthesis conditions and measured crystallite size.

## 📊 Dataset

Dataset file:

```text
data/crystallite_data.csv
```

The current repository contains **76 observations** with the following fields:

| Column | Meaning |
|---|---|
| `Temperature_K` | Synthesis temperature in Kelvin |
| `Time_h` | Synthesis duration in hours |
| `Fuel` | Fuel used during combustion synthesis |
| `Crystallite_Size` | Measured crystallite size |

The training code uses only `Temperature_K` and `Time_h` as predictors.

Observed ranges in the repository data are approximately:

- Temperature: **573–1473 K**
- Time: **1–7 h**
- Crystallite size: **7–74.1 nm**

## 🧠 Models

### Random Forest Regressor

Configuration in `src/config.py`:

```text
n_estimators      = 300
max_depth         = None
min_samples_split = 3
min_samples_leaf  = 1
max_features      = "sqrt"
random_state      = 42
n_jobs            = -1
```

### XGBoost Regressor

```text
n_estimators      = 300
max_depth         = 4
learning_rate     = 0.05
subsample         = 0.8
colsample_bytree  = 0.8
reg_alpha         = 0.1
reg_lambda        = 1.0
random_state      = 42
n_jobs            = -1
```

### Cross-validation

Both models are evaluated using:

```text
5-Fold KFold
shuffle = True
random_state = 42
```

Reported metrics include:

- Train R²
- Cross-validation R² mean and standard deviation
- Train RMSE
- Cross-validation RMSE
- Train MAE
- Cross-validation MAE

## 🔍 Explainability

CrystalliteML uses **SHAP TreeExplainer** for both tree-based models.

The pipeline calculates mean absolute SHAP values for:

- Temperature
- Time

It also generates:

### SHAP global importance

Compares feature contribution magnitude between Random Forest and XGBoost.

### SHAP summary plots

Shows the distribution of feature-level SHAP effects across the dataset.

### SHAP dependence — Temperature

Plots temperature against its SHAP contribution and colours points by synthesis time. The implementation overlays a cubic trend.

### SHAP dependence — Time

Plots time against its SHAP contribution and colours points by temperature. The implementation overlays a quadratic trend.

## 📈 Partial Dependence Analysis

The PDP module uses scikit-learn's `partial_dependence` to estimate the average model response as one feature changes.

Generated analyses:

- Temperature → predicted crystallite size
- Time → predicted crystallite size

The plots include annotations for:

- the steepest temperature rise
- the detected saturation onset in time

The input matrix is explicitly converted to `float64` before PDP calculation to avoid dtype-related errors.

## 🔄 Full Pipeline

Run everything through the single entry point:

```bash
python main.py
```

The five stages are:

```text
1. Load Data
      ↓
2. Train Random Forest + XGBoost
      ↓
3. SHAP Analysis
      ↓
4. Partial Dependence Plots
      ↓
5. Reports + Metrics Table
```

The main pipeline is implemented as:

```text
main.py
  ├── src.data_loader.load_data()
  ├── src.model.train_models()
  ├── src.shap_analysis.run_shap()
  ├── src.pdp_analysis.run_pdp()
  └── src.report.save_text_report()
```

## 📁 Project Structure

```text
CrystalliteML/
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── crystallite_data.csv
│
├── outputs/
│   ├── models/
│   │   ├── random_forest.pkl
│   │   └── xgboost.pkl
│   ├── plots/
│   │   ├── 0_actual_vs_predicted.png
│   │   ├── 1_shap_importance_bar.png
│   │   ├── 2_shap_summary.png
│   │   ├── 3_shap_dep_temperature.png
│   │   ├── 4_shap_dep_time.png
│   │   ├── 5_pdp_temperature.png
│   │   ├── 6_pdp_time.png
│   │   └── 8_metrics_table.png
│   └── reports/
│       └── results_report.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── model.py
│   ├── pdp_analysis.py
│   ├── report.py
│   └── shap_analysis.py
│
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

### Key modules

| Module | Responsibility |
|---|---|
| `src/config.py` | Paths, feature names, model hyperparameters, CV settings |
| `src/data_loader.py` | CSV loading and numeric conversion |
| `src/model.py` | Model training, CV metrics, model serialization, actual-vs-predicted plot |
| `src/shap_analysis.py` | SHAP values and SHAP plots |
| `src/pdp_analysis.py` | Partial dependence analysis and plots |
| `src/report.py` | Text report generation |

## ⚙️ Installation

Recommended environment: **Python 3.11**.

Create a virtual environment:

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The repository currently requires:

```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
shap>=0.44.0
matplotlib>=3.7.0
seaborn>=0.12.0
xgboost>=2.0.0
```

## ▶️ Running the Project

From the repository root:

```bash
python main.py
```

The pipeline creates or refreshes the `outputs/` directory contents.

## 📦 Generated Outputs

### Models

```text
outputs/models/random_forest.pkl
outputs/models/xgboost.pkl
```

Both are serialized with Python pickle and can be loaded with the corresponding Python/scikit-learn/XGBoost environment.

### Plots

```text
0_actual_vs_predicted.png
1_shap_importance_bar.png
2_shap_summary.png
3_shap_dep_temperature.png
4_shap_dep_time.png
5_pdp_temperature.png
6_pdp_time.png
8_metrics_table.png
```

### Report

```text
outputs/reports/results_report.txt
```

The report records:

- model parameters
- train/CV metrics
- SHAP importance
- interpretation notes
- a timestamp for the run

## 📈 Stored Results

The repository currently contains a generated report from a previous run.

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Train R² | **0.6904** | 0.5338 |
| CV R² mean | **-0.2679** | -0.2862 |
| CV R² std | 0.4119 | 0.4019 |
| Train RMSE (nm) | 7.97 | 9.78 |
| CV RMSE (nm) | **13.65** | 13.77 |
| Train MAE (nm) | **6.22** | 8.07 |
| CV MAE (nm) | **11.04** | 11.24 |
| SHAP Temperature | 6.4739 | **6.8949** |
| SHAP Time | 2.8456 | 2.0869 |

These are **stored results from the committed `results_report.txt`**, not a benchmark rerun during README generation.

### What the results suggest

Random Forest has the slightly lower stored CV RMSE and MAE, while both models have negative mean CV R² on this small dataset. This indicates limited out-of-sample predictive reliability despite substantially better training fit.

The SHAP magnitudes show that **Temperature contributes more strongly than Time** to the predictions in both stored runs.

## 🔬 Interpretation Notes

The committed report describes the following patterns in the generated analyses:

- Temperature is roughly **2.2×** as important as Time by the reported RF SHAP values.
- Temperature effects are nonlinear, with a stronger change in SHAP contribution in the higher-temperature region.
- Time shows a saturation-like trend, with diminishing influence at longer durations.
- The PDPs show an increasing predicted crystallite size as temperature rises and a weaker/diminishing time effect.

These should be interpreted as **model-derived patterns**, not as proof of causal physical relationships.

## ⚠️ Important Modeling Limitations

The current dataset is very small for a machine-learning regression problem: only 76 observations are available, and the training pipeline uses just two predictors.

More importantly, the `Fuel` column is present in the dataset but is excluded from training. Since different fuels can produce different combustion behavior, leaving Fuel out can contribute to unexplained variance.

The stored negative CV R² values indicate that the current models do not generalize strongly across the selected K-fold splits.

The current evaluation also uses standard shuffled KFold. For scientific reporting, additional validation strategies should be considered, especially when multiple observations may share similar experimental conditions.

## 🚀 Recommended Extensions

- Encode **Fuel** as a categorical feature.
- Add physically meaningful descriptors beyond temperature and time.
- Increase the number of observations and replicate measurements.
- Compare grouped or stratified validation strategies appropriate for experimental batches.
- Add uncertainty intervals around predictions.
- Evaluate Gaussian Process Regression for small-data behavior.
- Tune hyperparameters using nested or repeated cross-validation.
- Add an explicit held-out test set.
- Compare feature-only and fuel-aware models with an ablation study.
- Track experiment provenance and data source metadata.

## 🤖 Continuous Integration

GitHub Actions is configured in:

```text
.github/workflows/ci.yml
```

The workflow runs on pushes and pull requests targeting `main` or `master` and:

1. checks out the repository
2. installs Python 3.11
3. installs `requirements.txt`
4. executes `python main.py`
5. uploads the generated `outputs/` directory as a workflow artifact

This provides an automated reproducibility check for the complete pipeline.

## 📚 References

The project draws on established methods including:

1. Breiman, L. — **Random Forests**, Machine Learning (2001).
2. Chen, T. & Guestrin, C. — **XGBoost: A Scalable Tree Boosting System** (2016).
3. Lundberg, S. M. & Lee, S.-I. — **A Unified Approach to Interpreting Model Predictions** (2017).
4. Scherrer, P. — **Determination of the Size and Internal Structure of Colloid Particles by X-Rays** (1918).

## 👤 Author

**Harshit Garg**

GitHub: [Harshit765G4](https://github.com/Harshit765G4)

---

**CrystalliteML** — machine-learning prediction and explainability for crystallite growth experiments.
