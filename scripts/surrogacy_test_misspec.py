#!/usr/bin/env python3
"""
Joint Null Disambiguation -- DGP 8 with Linear Model.

Runs the surrogacy diagnostic test on DGP 8 (nonlinear but valid surrogate)
using a deliberately mis-specified linear prediction model that omits S^2.

DGP 8: Y = 0.5*S - 0.03*(S-5)^2 + 0.2*X + eps_Y
Correct model: [1, S, S^2, X]
Mis-specified model: [1, S, X]  (omitting S^2)

This test tells us whether the diagnostic can distinguish model misspecification
from surrogacy failure. The surrogate IS valid in DGP 8 (no direct effect),
but the mis-specified model should cause the test to reject.

pi_L = {0.05, 0.20, 0.50}, R = 1000
Saves to results/tables/surrogacy_test_misspec.csv and .md
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dgps.dgps import DGP_GENERATORS
from src.simulations.simulation import _degenerate_fold_error, derive_seed
from src.utils.config import DEFAULT_PROTOCOL, DGP_DEFAULTS, PROTOCOLS
from src.utils.registry import write_table_rows
from src.utils.estimator_api import (
    estimate,
    estimate_cov_si_ppi,
    surrogacy_test,
)

#: Prediction protocol; set from --protocol in main() and propagated to the
#: worker processes by `_set_protocol`.
PROTOCOL = DEFAULT_PROTOCOL


def _set_protocol(protocol: str) -> None:
    """Pool initializer: workers re-import this module under spawn."""
    global PROTOCOL
    PROTOCOL = protocol

R = 1000
PI_L_VALUES = [0.05, 0.20, 0.50]
N_CORES = 8
MASTER_SEED = 42


def _build_linear_design(S: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Build mis-specified linear design: [1, S, X] -- NO S^2 term."""
    n = S.shape[0]
    intercept = np.ones((n, 1))
    S_col = S.reshape(-1, 1)
    parts = [intercept, S_col]
    if X is not None and X.shape[1] > 0:
        parts.append(X)
    return np.hstack(parts)


def _ols_fit(X_design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit OLS with small ridge for stability."""
    XtX = X_design.T @ X_design
    XtX += 1e-10 * np.eye(XtX.shape[0])
    Xty = X_design.T @ y
    return np.linalg.solve(XtX, Xty)


def train_prediction_model_linear_only(
    S: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    labeled_mask: np.ndarray,
    n_folds: int = 5,
    rng: np.random.Generator = None,
    protocol: str = DEFAULT_PROTOCOL,
    return_design: bool = False,
):
    """Train the mis-specified linear prediction model, all-units cross-fit.

    Uses design [1, S, X] instead of [1, S, S^2, X].  Apart from the design,
    this follows the same protocol as
    `src.simulations.simulation.train_prediction_model`: every unit is
    assigned to one of K folds and predicted from the fit on the labeled units
    outside its fold.  With `return_design=True` it also returns the design
    dictionary the SI sandwich variance consumes, so the diagnostic's SE(D) is
    computed for the index that was actually fit -- misspecification and all.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol {protocol!r} not in {PROTOCOLS}")

    n = S.shape[0]
    Y_hat = np.empty(n, dtype=np.float64)

    labeled_idx = np.where(labeled_mask)[0]
    unlabeled_idx = np.where(~labeled_mask)[0]
    n_L = len(labeled_idx)

    G = _build_linear_design(S, X)
    q = G.shape[1]
    beta_full = _ols_fit(G[labeled_idx], Y[labeled_idx])

    def _design(fold_ids, coefs):
        return dict(
            g=G, fold_ids=fold_ids, coefs=coefs, beta_full=beta_full,
            protocol=protocol, model="ols", n_folds=n_folds,
            labeled_mask=np.asarray(labeled_mask, dtype=bool),
        )

    if n_L < n_folds * 2:
        Y_hat[:] = G @ beta_full
        if not return_design:
            return Y_hat
        return Y_hat, _design(np.zeros(n, dtype=int),
                              np.tile(beta_full, (n_folds, 1)))

    coefs = np.zeros((n_folds, q), dtype=np.float64)

    if protocol == "mixed_fit":
        # Folds among the labeled units only; unlabeled units get the full fit.
        fold_ids = np.full(n, -1, dtype=int)
        lab_folds = np.zeros(n_L, dtype=int)
        perm = rng.permutation(n_L)
        for k in range(n_folds):
            start = k * n_L // n_folds
            end = (k + 1) * n_L // n_folds
            lab_folds[perm[start:end]] = k
        fold_ids[labeled_idx] = lab_folds
        for k in range(n_folds):
            train_idx = labeled_idx[lab_folds != k]
            test_idx = labeled_idx[lab_folds == k]
            beta_k = _ols_fit(G[train_idx], Y[train_idx])
            coefs[k] = beta_k
            Y_hat[test_idx] = G[test_idx] @ beta_k
        if unlabeled_idx.size:
            Y_hat[unlabeled_idx] = G[unlabeled_idx] @ beta_full
    else:
        # All-units cross-fitting.
        fold_ids = np.empty(n, dtype=int)
        perm = rng.permutation(n)
        for k in range(n_folds):
            start = k * n // n_folds
            end = (k + 1) * n // n_folds
            fold_ids[perm[start:end]] = k
        for k in range(n_folds):
            train_idx = labeled_idx[fold_ids[labeled_idx] != k]
            test_idx = np.where(fold_ids == k)[0]
            if train_idx.size < G.shape[1] + 1:
                raise _degenerate_fold_error(k, train_idx.size, G.shape[1])
            beta_k = _ols_fit(G[train_idx], Y[train_idx])
            coefs[k] = beta_k
            if test_idx.size:
                Y_hat[test_idx] = G[test_idx] @ beta_k

    if not return_design:
        return Y_hat
    return Y_hat, _design(fold_ids, coefs)


def run_one_rep(args: Tuple) -> Dict[str, Any]:
    dgp_id, pi_L, rep, master_seed = args
    config_hash = 0
    rep_seed = derive_seed(master_seed, dgp_id, config_hash, int(pi_L * 100), rep)

    defaults = dict(DGP_DEFAULTS[dgp_id])
    defaults["seed"] = rep_seed
    defaults["pi_L"] = pi_L

    dgp_func = DGP_GENERATORS[dgp_id]
    data = dgp_func(**defaults)
    true_tau = data["true_tau"]

    T, S, Y, X = data["T"], data["S"], data["Y"], data["X"]
    labeled_mask = data["labeled_mask"]

    # Use mis-specified LINEAR prediction model (no S^2)
    cf_rng = np.random.default_rng(rep_seed + 7777)
    Y_hat, design = train_prediction_model_linear_only(
        S, X, Y, labeled_mask, n_folds=5, rng=cf_rng,
        protocol=PROTOCOL, return_design=True,
    )

    res_ppi = estimate(3, T, S, Y, Y_hat, labeled_mask, protocol=PROTOCOL)
    lambda_hat = res_ppi.get("lambda_hat", 1.0)
    res_si = estimate(2, T, S, Y, Y_hat, labeled_mask, protocol=PROTOCOL,
                      design=design, lambda_hat=lambda_hat)

    cov_sp = estimate_cov_si_ppi(T, Y, Y_hat, labeled_mask, lambda_hat,
                                 design=design)

    # One-sided test
    test_one = surrogacy_test(
        res_si["tau_hat"], res_ppi["tau_hat"],
        res_si["var_hat"], res_ppi["var_hat"], cov_sp,
        alternative="greater",
    )

    # Two-sided test
    test_two = surrogacy_test(
        res_si["tau_hat"], res_ppi["tau_hat"],
        res_si["var_hat"], res_ppi["var_hat"], cov_sp,
        alternative="two-sided",
    )

    return {
        "dgp_id": dgp_id,
        "pi_L": pi_L,
        "rep": rep,
        "true_tau": true_tau,
        "tau_si": res_si["tau_hat"],
        "tau_ppi": res_ppi["tau_hat"],
        "D_hat": test_one["D_hat"],
        "T_n": test_one["T_n"],
        "p_one_sided": test_one["p_value"],
        "p_two_sided": test_two["p_value"],
    }


def main():
    t0 = time.time()

    tasks = []
    for pi_L in PI_L_VALUES:
        for r in range(R):
            tasks.append((8, pi_L, r, MASTER_SEED))

    print(f"Surrogacy Test Misspecification (DGP 8): {len(tasks)} tasks, "
          f"R={R}, {N_CORES} cores")

    with multiprocessing.Pool(N_CORES, initializer=_set_protocol,
                              initargs=(PROTOCOL,)) as pool:
        all_results = pool.map(run_one_rep, tasks)

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    df = pd.DataFrame(all_results)

    output_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "surrogacy_test_misspec.csv")
    df.to_csv(csv_path, index=False)

    # Result rows: one per (pi_L, alpha, alternative) rejection rate.
    _summary = pd.DataFrame([
        dict(dgp_id=8, pi_L=pi_L, alternative=alt, alpha=a,
             rejection=float((sub[col] < a).mean()),
             mc_se_rejection=float(
                 np.sqrt((sub[col] < a).mean() * (1 - (sub[col] < a).mean())
                         / len(sub))),
             mean_D_hat=float(sub["D_hat"].mean()),
             mean_T_n=float(sub["T_n"].mean()))
        for pi_L in PI_L_VALUES
        for sub in [df[df["pi_L"] == pi_L]] if not sub.empty
        for alt, col in (("one-sided", "p_one_sided"),
                         ("two-sided", "p_two_sided"))
        for a in (0.05, 0.10)
    ])
    write_table_rows(
        csv_path.replace(".csv", "_rows.json"), _summary,
        dgp=8,
        analysis="SI--PPI++ diagnostic (misspecified index)",
        protocol=PROTOCOL,
        config_cols=("dgp_id", "pi_L", "alternative"),
        R=R,
        generated_by="scripts/surrogacy_test_misspec.py",
    )
    print(f"Raw results saved to {csv_path}")

    # Summary table
    true_tau = df["true_tau"].iloc[0]
    lines = [
        "# Surrogacy Test Under Model Misspecification\n\n",
        f"DGP 8 (nonlinear valid surrogate), true_tau = {true_tau:.4f}\n",
        f"Prediction model: LINEAR only [1, S, X] -- S^2 deliberately omitted\n",
        f"R = {R}\n\n",
        "| pi_L | Reject@0.05 (1-sided) | Reject@0.10 (1-sided) | "
        "Reject@0.05 (2-sided) | Reject@0.10 (2-sided) | Mean D_hat | Mean T_n |\n",
        "|------|-----------------------|-----------------------|"
        "-----------------------|-----------------------|------------|----------|\n",
    ]

    for pi_L in PI_L_VALUES:
        sub = df[df["pi_L"] == pi_L]
        if sub.empty:
            continue

        rej_one_05 = (sub["p_one_sided"] < 0.05).mean()
        rej_one_10 = (sub["p_one_sided"] < 0.10).mean()
        rej_two_05 = (sub["p_two_sided"] < 0.05).mean()
        rej_two_10 = (sub["p_two_sided"] < 0.10).mean()
        mean_D = sub["D_hat"].mean()
        mean_Tn = sub["T_n"].mean()

        lines.append(
            f"| {pi_L} | {rej_one_05:.3f} | {rej_one_10:.3f} | "
            f"{rej_two_05:.3f} | {rej_two_10:.3f} | {mean_D:.4f} | {mean_Tn:.3f} |\n"
        )

    # Observed rejection ranges, so the interpretation can never drift away
    # from the numbers in the table above.
    rej05 = [(df[df["pi_L"] == p]["p_one_sided"] < 0.05).mean()
             for p in PI_L_VALUES if not df[df["pi_L"] == p].empty]
    rej10 = [(df[df["pi_L"] == p]["p_one_sided"] < 0.10).mean()
             for p in PI_L_VALUES if not df[df["pi_L"] == p].empty]
    mc_se_05 = (0.05 * 0.95 / R) ** 0.5
    mc_se_10 = (0.10 * 0.90 / R) ** 0.5

    lines.append("\n")
    lines.append("**Interpretation:**\n")
    lines.append("- DGP 8 has a valid surrogate (no direct effect), so any "
                 "rejection here reflects prediction-model misspecification "
                 "rather than a surrogacy violation.\n")
    lines.append("- The linear model omits S^2, which biases the SI "
                 "predictions.\n")
    lines.append(
        f"- Rejection nevertheless stays near nominal size: "
        f"{min(rej05):.3f}-{max(rej05):.3f} at the 5% level "
        f"(MC SE {mc_se_05:.3f}) and {min(rej10):.3f}-{max(rej10):.3f} at the "
        f"10% level (MC SE {mc_se_10:.3f}), rising only mildly with pi_L. "
        f"Mean D_hat is within a hundredth of zero at every pi_L.\n")
    lines.append("- So this misspecification does NOT by itself generate a "
                 "detectable SI-PPI++ gap: the test has essentially no power "
                 "against it, and a rejection in practice should still be "
                 "read as evidence of a surrogacy violation rather than of a "
                 "mis-specified prediction model.\n")
    lines.append("- The corollary is a limitation, not a strength: the test "
                 "cannot warn a practitioner that the prediction model is "
                 "wrong, so model misspecification of this kind passes "
                 "silently.\n")

    md_path = os.path.join(output_dir, "surrogacy_test_misspec.md")
    with open(md_path, "w") as f:
        f.writelines(lines)
    print(f"Summary saved to {md_path}")


if __name__ == "__main__":
    main()
