# ============================================================
#  CrystalliteML — Configuration
# ============================================================
import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "crystallite_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

for _d in [PLOT_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Features / Target ────────────────────────────────────────
FEATURES       = ["Temperature_K", "Time_h"]
TARGET         = "Crystallite_Size"
FEATURE_LABELS = ["Temperature (K)", "Time (h)"]

# ── Random Forest hyperparameters ───────────────────────────
RF_PARAMS = dict(
    n_estimators      = 300,
    max_depth         = None,
    min_samples_split = 3,
    min_samples_leaf  = 1,
    max_features      = "sqrt",
    random_state      = 42,
    n_jobs            = -1,
)

# ── XGBoost hyperparameters ──────────────────────────────────
XGB_PARAMS = dict(
    n_estimators      = 300,
    max_depth         = 4,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    verbosity         = 0,
)

# ── Cross-validation ─────────────────────────────────────────
CV_FOLDS    = 5
RANDOM_SEED = 42

# ── Plot colours ─────────────────────────────────────────────
C1   = "#2E3192"   # deep indigo   (RF)
C2   = "#E8192C"   # vivid red     (XGB / Time)
C3   = "#00AEEF"   # sky blue
ACC  = "#F7941D"   # amber
BG   = "#F7F9FC"
GRID = "#DDE3ED"

DPI  = 180
