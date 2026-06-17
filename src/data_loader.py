# ============================================================
#  CrystalliteML — Data Loader
# ============================================================
import pandas as pd
import numpy as np
from src.config import DATA_PATH, FEATURES, TARGET


def load_data(verbose: bool = True) -> tuple:
    """
    Load crystallite dataset.

    Returns
    -------
    X   : ndarray float64, shape (n_samples, 2)
    y   : ndarray float64, shape (n_samples,)
    df  : original DataFrame
    """
    df = pd.read_csv(DATA_PATH)

    # ── CRITICAL: cast to float64 (fixes PDP integer error) ──
    X = df[FEATURES].values.astype(np.float64)
    y = df[TARGET].values.astype(np.float64)

    if verbose:
        print("=" * 55)
        print("  Dataset Summary")
        print("=" * 55)
        print(df[FEATURES + [TARGET]].describe().round(2).to_string())
        print(f"\n  Fuel types : {df['Fuel'].unique().tolist()}")
        print(f"  Total rows : {len(df)}")
        print(f"  X dtype    : {X.dtype}  ← float64 confirmed")
        print("=" * 55)

    return X, y, df
