"""
LaLonde National Supported Work (NSW) dataset loader.

The Dehejia-Wahba sample from the NSW randomized experiment.
Treatment: job training program
Outcome (Y): RE78 (real earnings in 1978, post-experiment)
Surrogate (S): RE75 (real earnings in 1975, pre-experiment but correlated)
Covariates: age, education, black, hispanic, married, nodegree, re74
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def load_lalonde() -> Dict[str, np.ndarray]:
    """
    Load the Dehejia-Wahba NSW sample.

    Returns
    -------
    dict with keys:
        T  : (n,) int array, treatment {0, 1}
        S  : (n,) float array, surrogate (RE75)
        Y  : (n,) float array, primary outcome (RE78)
        X  : (n, p) float array, covariates
        df : pandas DataFrame with raw data
    """
    url = "https://users.nber.org/~rdehejia/data/nsw_dw.dta"
    df = pd.read_stata(url)

    T = df["treat"].values.astype(np.int32)
    S = df["re75"].values.astype(np.float64)  # surrogate
    Y = df["re78"].values.astype(np.float64)  # outcome

    # Covariates
    covariate_cols = ["age", "education", "black", "hispanic", "married", "nodegree", "re74"]
    X = df[covariate_cols].values.astype(np.float64)

    return dict(T=T, S=S, Y=Y, X=X, df=df)


def compute_ground_truth_ate(T: np.ndarray, Y: np.ndarray) -> float:
    """Compute full-sample ground truth ATE via difference-in-means on Y."""
    t1 = T == 1
    t0 = T == 0
    return float(Y[t1].mean() - Y[t0].mean())
