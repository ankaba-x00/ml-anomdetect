import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .common import apply_custom_theme


def plot_log_candidates(
    feature_name: str, 
    feature_values: pd.Series, 
    folder: Path, 
    fname: str = "plot_log_candidates.png", 
    show: bool = False,
) -> None:
    """Histograms of value distribution for raw signals and log-transformed signals."""
    apply_custom_theme()

    try:
        # suppresses divide-by-zero and invalid log warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            log_vals = np.log1p(feature_values)
        if not np.all(np.isfinite(log_vals)):
            raise ValueError("[ERROR] log1p produced non-finite values; skipping log-scale plot.")
        
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.hist(feature_values, bins=200)
        plt.title(f"Raw Scale: {feature_name}")
        plt.yscale("log")
        plt.subplot(1,2,2)
        plt.hist(np.log1p(feature_values), bins=200)
        plt.title(f"Log1p Scale: {feature_name}")
        plt.savefig(folder / fname, dpi=160)
        if show: plt.show()
        plt.close()

    except (Exception, BaseException):
        print(f"[INFO] Log scaling not possible for feature '{feature_name}'.")
