"""
Hillstrom email marketing dataset loader and preprocessor.

The Hillstrom dataset (Kevin Hillstrom's MineThatData E-Mail Analytics Challenge)
contains data from an email marketing campaign. Users were randomized into:
    - Control: no email
    - Men's email: received men's merchandise email
    - Women's email: received women's merchandise email

Surrogate: visit (binary: did the user visit the website within 2 weeks)
Primary outcome: conversion (binary: did the user make a purchase)
Covariates: recency, history_segment, mens, womens, zip_code, newbie, channel
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_hillstrom() -> Dict[str, np.ndarray]:
    """
    Load and preprocess the Hillstrom email marketing dataset.

    Returns
    -------
    dict with keys:
        T          : (n,) int array, treatment {0, 1} (email vs no-email)
        S          : (n,) float array, surrogate (visit)
        Y          : (n,) float array, primary outcome (conversion)
        X          : (n, p) float array, covariates (dummy-encoded)
        df         : pandas DataFrame with raw data
    """
    from sklift.datasets import fetch_hillstrom

    # Load visit (surrogate) and conversion (primary outcome) separately
    bunch_visit = fetch_hillstrom(target_col="visit")
    bunch_conv = fetch_hillstrom(target_col="conversion")

    df = bunch_visit.data.copy()
    df["treatment"] = bunch_visit.treatment
    df["visit"] = bunch_visit.target
    df["conversion"] = bunch_conv.target

    # Treatment: email (men's or women's) vs no-email control
    df["T"] = (df["treatment"] != "No E-Mail").astype(int)

    # Surrogate: visit
    S = df["visit"].values.astype(np.float64)

    # Primary outcome: conversion
    Y = df["conversion"].values.astype(np.float64)

    # Treatment indicator
    T = df["T"].values.astype(np.int32)

    # Covariates: encode categoricals as dummies
    covariate_cols = ["recency", "mens", "womens", "newbie"]
    categorical_cols = ["history_segment", "zip_code", "channel"]

    X_parts = [df[covariate_cols].values.astype(np.float64)]

    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        X_parts.append(dummies.values.astype(np.float64))

    X = np.hstack(X_parts)

    return dict(T=T, S=S, Y=Y, X=X, df=df)


def compute_ground_truth_ate(T: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute full-sample ground truth ATE via difference-in-means on Y.
    """
    t1 = T == 1
    t0 = T == 0
    return float(Y[t1].mean() - Y[t0].mean())
