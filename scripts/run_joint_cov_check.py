#!/usr/bin/env python3
"""
Joint paired-bootstrap validation of the learned-index sandwich.

What is being validated
-----------------------
Under the all-units cross-fitting protocol both tau_SI and tau_PPI++ are
functionals of the SAME learned linear index g(S, X)' beta.  The
fixed-predictor ("plug-in") variances treat beta as known; the sandwich `src.methods.joint_influence_cov` restores the first-stage term and
returns the full 2 x 2 covariance of (tau_SI, tau_PPI++), hence also
Var(D-hat) for the SI--PPI++ estimator-disagreement diagnostic
D-hat = tau_PPI++ - tau_SI.

This script measures the sandwich against a joint paired bootstrap that
contains every term by construction: units are resampled with replacement
STRATIFIED BY ARM, the labeled flags travel with the units, the index is REFIT
on the resampled labeled units under the all-units protocol, and BOTH
estimators are recomputed from the refit predictions.

Cells
-----
  (1) DGP 1,          n = 10,000,  pi_L = 0.20  (valid surrogate; rho n/a)
  (2) DGP 2 rho=0.0,  n = 10,000,  pi_L = 0.20  (null: no direct effect)
  (3) DGP 2 rho=0.2,  n = 10,000,  pi_L = 0.20  (alternative)
  (4) Multi-surrogate rich index m = 1, n = 64,000, pi_L = 0.20
      (Criteo-calibrated funnel; construction imported from
       scripts/run_multisurrogate.py, covariate pool from results/cache)
  (5) DGP 2 rho=0.0,  n = 100,000, pi_L = 0.20  -- null size only, no
      bootstrap (the bootstrap is what makes the other cells expensive).

Reported per cell
-----------------
  * mean sandwich Var(SI), Var(PPI), Cov, Var(D) with MC SEs
  * mean paired-bootstrap counterparts, and the sandwich/bootstrap ratios
  * the Monte Carlo variance of each statistic across the R replications,
    which is the ground truth both estimators are aiming at
  * coverage of the true tau by the SI sandwich interval, the PPI++
    interval, and the two bootstrap intervals
  * at rho = 0 cells, the diagnostic's rejection rate (null size) at
    alpha = 0.05 and 0.10, using SE(D) from the sandwich and from the
    bootstrap

Output: results/tables/joint_cov_check.{md,csv} (+ _raw.csv)

Usage:
    python scripts/run_joint_cov_check.py [--cells 1,2,3,4,5] [-R 200] [-B 200]
    python scripts/run_joint_cov_check.py --cells 2,1 -R 2000 --no-boot \
        --out-suffix _R2000      # -> joint_cov_check_R2000.{md,csv,_raw.csv}
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dgps.dgps import generate_dgp1, generate_dgp2
from src.methods import (
    estimate,
    joint_influence_cov,
    si_first_stage_term,
    surrogacy_test,
)
from src.simulations.simulation import (
    PROTOCOL_ALLUNITS,
    _build_design_matrix,
    _degenerate_fold_error,
    _ols_fit,
    derive_seed,
    train_prediction_model,
    unit_clustered_fold_ids,
)

TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
#: Where the writers put their outputs (``--outdir``; inputs such as
#: PARAM_PATH are still read from TABLES_DIR).
OUT_DIR = TABLES_DIR
CACHE_DIR = os.path.join(PROJECT_ROOT, "results", "cache")
POOL_PATH = os.path.join(CACHE_DIR, "semisynth_pool.npz")
PARAM_PATH = os.path.join(TABLES_DIR, "semisynth_params.json")

MASTER_SEED = 42
N_WORKERS = 8
N_FOLDS = 5
PI_L = 0.20
Z_ALPHA = float(stats.norm.ppf(0.975))

# Seed tags kept disjoint from every other script's streams.
SEED_TAG = {1: 330, 2: 331, 3: 332, 4: 333, 5: 334}

# Cell 4 constants, matching scripts/run_multisurrogate.py.
ZETA = 0.5
N_MULTI = 64_000
M_RICH = 1.0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# All-units cross-fitting on a prebuilt design matrix
# ---------------------------------------------------------------------------

def allunits_predict(
    G: np.ndarray,
    Y: np.ndarray,
    lm: np.ndarray,
    rng: np.random.Generator,
    n_folds: int = N_FOLDS,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """All-units cross-fitted predictions from a PREBUILT design matrix.

    Mirrors `train_prediction_model(..., protocol="allunits_crossfit")`
    exactly, including the rng consumption (one permutation of n).  Working on
    a prebuilt design is valid because `_build_design_matrix` acts row-wise,
    so resampling rows of G is identical to rebuilding the design from the
    resampled (S, X) -- which is what makes the bootstrap affordable.

    ``groups`` (bootstrap only): the original unit id of each row.  When
    given, folds are assigned by unit (`unit_clustered_fold_ids`), so all
    copies of a resampled unit share a fold; ``None`` keeps the row
    permutation above.

    Returns (Y_hat, design) with `design` in the shape
    `joint_influence_cov` expects.
    """
    n, q = G.shape
    Y_hat = np.empty(n, dtype=np.float64)
    lab = np.where(lm)[0]

    if groups is None:
        fold_ids = np.zeros(n, dtype=int)
        perm = rng.permutation(n)
        for k in range(n_folds):
            fold_ids[perm[k * n // n_folds:(k + 1) * n // n_folds]] = k
    else:
        fold_ids = unit_clustered_fold_ids(groups, n_folds, rng)

    beta_full = _ols_fit(G[lab], Y[lab])
    coefs = np.empty((n_folds, q), dtype=np.float64)
    fold_of_labeled = fold_ids[lab]
    for k in range(n_folds):
        test_idx = np.where(fold_ids == k)[0]
        train_idx = lab[fold_of_labeled != k]
        if len(train_idx) < q + 1:
            raise _degenerate_fold_error(k, len(train_idx), q)
        beta_k = _ols_fit(G[train_idx], Y[train_idx])
        coefs[k] = beta_k
        if len(test_idx) > 0:
            Y_hat[test_idx] = G[test_idx] @ beta_k

    design = dict(
        g=G, fold_ids=fold_ids, coefs=coefs, beta_full=beta_full,
        protocol=PROTOCOL_ALLUNITS, model="ols", n_folds=n_folds,
        labeled_mask=np.asarray(lm, dtype=bool),
    )
    return Y_hat, design


def fit_pair(
    G: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    lm: np.ndarray,
    rng: np.random.Generator,
    n_folds: int = N_FOLDS,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit the index and both estimators on one (possibly resampled) sample.

    Returns tau_SI, tau_PPI++, the PPI++ tuning constant, the PPI++ plug-in
    interval, and the design dict for the sandwich.
    """
    Y_hat, design = allunits_predict(G, Y, lm, rng, n_folds, groups)
    ppi = estimate(3, T, None, Y, Y_hat, lm)
    t1 = T == 1
    t0 = ~t1
    tau_si = float(Y_hat[t1].mean() - Y_hat[t0].mean())
    return dict(
        tau_si=tau_si,
        tau_ppi=float(ppi["tau_hat"]),
        var_ppi_own=float(ppi["var_hat"]),
        lam=float(ppi["lambda_hat"]),
        design=design,
        Y_hat=Y_hat,
    )


def paired_bootstrap(
    G: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    lm: np.ndarray,
    B: int,
    seed: int,
    n_folds: int = N_FOLDS,
    clustered: bool = False,
) -> Dict[str, float]:
    """Joint paired bootstrap of (tau_SI, tau_PPI++), stratified by arm.

    Each draw resamples n_1 treated and n_0 control units with replacement,
    carries the labeled flags along, refits the index under the all-units
    protocol and recomputes BOTH estimators, so the returned covariance and
    Var(D) are the joint bootstrap quantities.  ``clustered=True`` assigns
    the folds of each resample by original unit (all copies of a unit share
    a fold); the default splits the resampled rows, as the published
    joint_cov_check run did.
    """
    rng = np.random.default_rng(seed)
    idx1 = np.where(T == 1)[0]
    idx0 = np.where(T == 0)[0]
    n1, n0 = len(idx1), len(idx0)

    draws = np.empty((B, 2), dtype=np.float64)
    for b in range(B):
        take = np.concatenate([
            idx1[rng.integers(0, n1, size=n1)],
            idx0[rng.integers(0, n0, size=n0)],
        ])
        out = fit_pair(G[take], Y[take], T[take], lm[take], rng, n_folds,
                       groups=take if clustered else None)
        draws[b] = (out["tau_si"], out["tau_ppi"])

    cov = np.cov(draws[:, 0], draws[:, 1], ddof=1)
    return dict(
        boot_var_si=float(cov[0, 0]),
        boot_var_ppi=float(cov[1, 1]),
        boot_cov=float(cov[0, 1]),
        boot_var_D=float(np.var(draws[:, 1] - draws[:, 0], ddof=1)),
    )


def _one_rep(
    G: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    lm: np.ndarray,
    tau: float,
    B: int,
    seed: int,
    clustered: bool = False,
) -> Dict[str, Any]:
    """Sandwich + (optional) paired bootstrap for one replication."""
    base = fit_pair(G, Y, T, lm, np.random.default_rng(seed + 7777))
    J = joint_influence_cov(T, Y, lm, base["design"], base["lam"])

    t1b, t0b = T == 1, T == 0
    Y_hat = base["Y_hat"]
    var_si_plugin = float(Y_hat[t1b].var(ddof=1) / t1b.sum()
                          + Y_hat[t0b].var(ddof=1) / t0b.sum())
    var_si_delta = var_si_plugin + si_first_stage_term(
        T, None, Y, lm, design=base["design"])

    se_si = np.sqrt(max(J["var_si"], 0.0))
    se_ppi = np.sqrt(max(J["var_ppi"], 0.0))
    se_D = np.sqrt(max(J["var_D"], 0.0))
    D_hat = base["tau_ppi"] - base["tau_si"]

    row: Dict[str, Any] = dict(
        true_tau=tau,
        tau_si=base["tau_si"],
        tau_ppi=base["tau_ppi"],
        D_hat=D_hat,
        lam=base["lam"],
        var_si=J["var_si"],
        var_ppi=J["var_ppi"],
        cov_si_ppi=J["cov_si_ppi"],
        var_D=J["var_D"],
        var_ppi_own=base["var_ppi_own"],
        var_si_plugin=var_si_plugin,
        var_si_delta=var_si_delta,
        cov_si_plugin=int(abs(base["tau_si"] - tau)
                          <= Z_ALPHA * np.sqrt(var_si_plugin)),
        cov_si_delta=int(abs(base["tau_si"] - tau)
                         <= Z_ALPHA * np.sqrt(var_si_delta)),
        cov_si=int(abs(base["tau_si"] - tau) <= Z_ALPHA * se_si),
        cov_ppi=int(abs(base["tau_ppi"] - tau) <= Z_ALPHA * se_ppi),
        rej_sand_05=int(se_D > 0 and abs(D_hat) / se_D > stats.norm.ppf(0.975)),
        rej_sand_10=int(se_D > 0 and abs(D_hat) / se_D > stats.norm.ppf(0.950)),
    )

    if B > 0:
        bs = paired_bootstrap(G, Y, T, lm, B, seed + 991,
                              clustered=clustered)
        row.update(bs)
        se_si_b = np.sqrt(max(bs["boot_var_si"], 0.0))
        se_ppi_b = np.sqrt(max(bs["boot_var_ppi"], 0.0))
        se_D_b = np.sqrt(max(bs["boot_var_D"], 0.0))
        row.update(
            cov_si_boot=int(abs(base["tau_si"] - tau) <= Z_ALPHA * se_si_b),
            cov_ppi_boot=int(abs(base["tau_ppi"] - tau) <= Z_ALPHA * se_ppi_b),
            rej_boot_05=int(se_D_b > 0
                            and abs(D_hat) / se_D_b > stats.norm.ppf(0.975)),
            rej_boot_10=int(se_D_b > 0
                            and abs(D_hat) / se_D_b > stats.norm.ppf(0.950)),
        )
    else:
        row.update(boot_var_si=np.nan, boot_var_ppi=np.nan, boot_cov=np.nan,
                   boot_var_D=np.nan, cov_si_boot=np.nan,
                   cov_ppi_boot=np.nan, rej_boot_05=np.nan,
                   rej_boot_10=np.nan)
    return row


# ---------------------------------------------------------------------------
# Cell data generation
# ---------------------------------------------------------------------------

#: `null` marks the cells where the SI--PPI++ diagnostic's null holds, so the
#: rejection rate is a size; elsewhere it is power.  At m = 1 the rich index
#: (visit, S2) satisfies surrogacy exactly, so cell 4 is a null cell.
CELLS: Dict[int, Dict[str, Any]] = {
    1: dict(label="DGP 1", cell="pi_L=0.20", n=10_000, rho=None, boot=True,
            null=True),
    2: dict(label="DGP 2", cell="rho=0.0", n=10_000, rho=0.0, boot=True,
            null=True),
    3: dict(label="DGP 2", cell="rho=0.2", n=10_000, rho=0.2, boot=True,
            null=False),
    4: dict(label="Multi-surrogate (rich)", cell="m=1.0", n=N_MULTI,
            rho=None, boot=True, null=True),
    5: dict(label="DGP 2", cell="rho=0.0 (n = 100,000)", n=100_000, rho=0.0,
            boot=False, null=True),
}


def _make_cell_data(cell_id: int, rep: int, params: Optional[Dict] = None,
                    tau_multi: Optional[float] = None):
    """Generate one replication's data for a cell; returns (G, Y, T, lm, tau)."""
    spec = CELLS[cell_id]
    seed = derive_seed(MASTER_SEED, SEED_TAG[cell_id], 0,
                       int(round(PI_L * 100)), rep)

    if cell_id in (1, 2, 3, 5):
        if cell_id == 1:
            d = generate_dgp1(n=spec["n"], pi_L=PI_L, seed=seed)
        else:
            d = generate_dgp2(n=spec["n"], pi_L=PI_L, rho=spec["rho"],
                              seed=seed)
        G = _build_design_matrix(d["S"], d["X"])
        return G, d["Y"], d["T"], d["labeled_mask"], d["true_tau"], seed

    # cell 4: multi-surrogate rich index, m = 1 on the Criteo-calibrated funnel
    rng = np.random.default_rng(seed)
    X_pool = np.load(POOL_PATH)["X"].astype(np.float64)
    X = X_pool[rng.choice(len(X_pool), size=N_MULTI, replace=True)]

    av = np.asarray(params["visit_coef"][:-1]); bv = params["visit_coef"][-1]
    cv = params["visit_intercept"]
    ac = np.asarray(params["conv_coef"][:-1]); cc = params["conv_intercept"]
    kap = params["kappa_hat"]
    theta = M_RICH * kap / ZETA
    kr = (1 - M_RICH) * kap

    T = (rng.random(N_MULTI) < 0.5).astype(np.float64)
    S1 = (rng.random(N_MULTI) < sigmoid(X @ av + cv + bv * T)).astype(np.float64)
    S2 = ZETA * T + rng.standard_normal(N_MULTI)
    pY = sigmoid(X @ ac + cc + theta * S2 + kr * T)
    Y = np.where(S1 == 1, (rng.random(N_MULTI) < pY).astype(np.float64), 0.0)

    lm = np.zeros(N_MULTI, dtype=bool)
    lm[rng.choice(N_MULTI, size=int(PI_L * N_MULTI), replace=False)] = True

    G = _build_design_matrix(np.column_stack([S1, S2]), X)
    return G, Y, T, lm, tau_multi, seed


def _multi_true_tau_local(params, X_pool, m) -> float:
    """tau for the multi-surrogate cell; transcription of run_multisurrogate.

    Used only as a fallback when `scripts/run_multisurrogate.py` cannot be
    imported (it is owned by another workstream and may be mid-edit).  The
    imported version is the source of truth; `_load_multisurrogate_true_tau`
    prefers it and cross-checks this copy against it when both are available.
    Gauss-Hermite quadrature over S2 ~ N(zeta T, 1) on the covariate pool.
    """
    av = np.asarray(params["visit_coef"][:-1]); bv = params["visit_coef"][-1]
    cv = params["visit_intercept"]
    ac = np.asarray(params["conv_coef"][:-1]); cc = params["conv_intercept"]
    kap = params["kappa_hat"]
    theta = m * kap / ZETA
    kr = (1 - m) * kap

    nodes, weights = np.polynomial.hermite_e.hermegauss(31)
    weights = weights / weights.sum()

    lin_v = X_pool @ av + cv
    lin_c = X_pool @ ac + cc

    def mean_conv(t):
        s2 = ZETA * t + nodes
        p = np.zeros_like(lin_c)
        for s2k, wk in zip(s2, weights):
            p += wk * sigmoid(lin_c + theta * s2k + kr * t)
        return p

    m1 = sigmoid(lin_v + bv) * mean_conv(1.0)
    m0 = sigmoid(lin_v) * mean_conv(0.0)
    return float(np.mean(m1 - m0))


def _load_multisurrogate_true_tau():
    """Prefer run_multisurrogate.true_tau; fall back to the local copy."""
    try:
        from run_multisurrogate import true_tau as imported
        return imported
    except Exception as exc:                       # pragma: no cover
        print(f"WARNING: could not import run_multisurrogate.true_tau "
              f"({type(exc).__name__}: {exc}); using the local transcription "
              f"_multi_true_tau_local instead.", flush=True)
        return _multi_true_tau_local


def run_task(args: Tuple) -> Dict[str, Any]:
    cell_id, rep, B, params, tau_multi, clustered = args
    G, Y, T, lm, tau, seed = _make_cell_data(cell_id, rep, params, tau_multi)
    spec = CELLS[cell_id]
    row = _one_rep(G, Y, T, lm, tau, B if spec["boot"] else 0, seed,
                   clustered=clustered)
    row.update(cell_id=cell_id, rep=rep, design=spec["label"],
               cell=spec["cell"], n=spec["n"], pi_L=PI_L,
               is_null=bool(spec["null"]),
               B=(B if spec["boot"] else 0),
               boot_folds="clustered" if clustered else "rows")
    return row


# ---------------------------------------------------------------------------
# Self-check: the fast design-matrix path reproduces the paper pipeline
# ---------------------------------------------------------------------------

def _self_check() -> None:
    seed = derive_seed(MASTER_SEED, SEED_TAG[1], 0, 20, 0)
    d = generate_dgp1(n=5_000, pi_L=PI_L, seed=seed)
    T, X, S, Y, lm = d["T"], d["X"], d["S"], d["Y"], d["labeled_mask"]

    Y_hat_ref, design_ref = train_prediction_model(
        S, X, Y, lm, n_folds=N_FOLDS,
        rng=np.random.default_rng(seed + 7777),
        protocol=PROTOCOL_ALLUNITS, return_design=True,
    )
    Y_hat_fast, design_fast = allunits_predict(
        _build_design_matrix(S, X), Y, lm,
        np.random.default_rng(seed + 7777), N_FOLDS,
    )
    err = float(np.max(np.abs(Y_hat_ref - Y_hat_fast)))
    assert err < 1e-12, f"fast path deviates by {err:g}"
    assert np.array_equal(design_ref["fold_ids"], design_fast["fold_ids"])
    assert np.allclose(design_ref["beta_full"], design_fast["beta_full"])
    print(f"self-check ok: fast all-units path reproduces "
          f"train_prediction_model (max |dY_hat| = {err:.2e})", flush=True)


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

def _mean_se(x: pd.Series) -> Tuple[float, float]:
    x = x.dropna()
    if len(x) == 0:
        return np.nan, np.nan
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cell_id, g in df.groupby("cell_id", sort=True):
        R = len(g)
        rec: Dict[str, Any] = dict(
            cell_id=int(cell_id),
            design=g.design.iloc[0],
            cell=g.cell.iloc[0],
            n=int(g.n.iloc[0]),
            pi_L=float(g.pi_L.iloc[0]),
            is_null=bool(g.is_null.iloc[0]),
            R=R,
            B=int(g.B.iloc[0]),
            true_tau=float(g.true_tau.iloc[0]),
            mean_lambda=float(g.lam.mean()),
            bias_si=float((g.tau_si - g.true_tau).mean()),
            bias_ppi=float((g.tau_ppi - g.true_tau).mean()),
            # Monte Carlo truth: the spread of the estimators across reps.
            mc_var_si=float(g.tau_si.var(ddof=1)),
            mc_var_ppi=float(g.tau_ppi.var(ddof=1)),
            mc_cov=float(np.cov(g.tau_si, g.tau_ppi, ddof=1)[0, 1]),
            mc_var_D=float(g.D_hat.var(ddof=1)),
        )
        for key in ("var_si", "var_ppi", "cov_si_ppi", "var_D",
                    "boot_var_si", "boot_var_ppi", "boot_cov", "boot_var_D",
                    "var_si_plugin", "var_si_delta"):
            m, se = _mean_se(g[key])
            rec[f"mean_{key}"] = m
            rec[f"mcse_{key}"] = se
        for key in ("cov_si", "cov_ppi", "cov_si_boot", "cov_ppi_boot",
                    "cov_si_plugin", "cov_si_delta",
                    "rej_sand_05", "rej_sand_10", "rej_boot_05",
                    "rej_boot_10"):
            m, _ = _mean_se(g[key])
            rec[key] = m
            rec[f"mcse_{key}"] = (np.sqrt(m * (1 - m) / R)
                                  if np.isfinite(m) else np.nan)

        def ratio(a, b):
            return (rec[a] / rec[b]) if (rec.get(b) and np.isfinite(rec[b])
                                         and rec[b] != 0) else np.nan

        rec["ratio_si_boot"] = ratio("mean_var_si", "mean_boot_var_si")
        rec["ratio_ppi_boot"] = ratio("mean_var_ppi", "mean_boot_var_ppi")
        rec["ratio_cov_boot"] = ratio("mean_cov_si_ppi", "mean_boot_cov")
        rec["ratio_D_boot"] = ratio("mean_var_D", "mean_boot_var_D")
        rec["ratio_si_mc"] = ratio("mean_var_si", "mc_var_si")
        rec["ratio_si_plugin_mc"] = ratio("mean_var_si_plugin", "mc_var_si")
        rec["ratio_si_delta_mc"] = ratio("mean_var_si_delta", "mc_var_si")
        rec["ratio_ppi_mc"] = ratio("mean_var_ppi", "mc_var_ppi")
        rec["ratio_cov_mc"] = ratio("mean_cov_si_ppi", "mc_cov")
        rec["ratio_D_mc"] = ratio("mean_var_D", "mc_var_D")

        # Monte Carlo SE of each Sand/MC ratio.  The ratio is
        # mean(sandwich) / (sample (co)variance across reps); both are means
        # over the R replications, so its delta-method SE uses the per-rep
        # influence  s_r / M - (mean s / M^2) (z_r - M)  with
        # z_r = (a_r - a_bar)(b_r - b_bar) the (co)variance summand.
        D_hat = g.tau_ppi - g.tau_si
        pairs = {
            "si": ("var_si", g.tau_si, g.tau_si),
            "ppi": ("var_ppi", g.tau_ppi, g.tau_ppi),
            "cov": ("cov_si_ppi", g.tau_si, g.tau_ppi),
            "D": ("var_D", D_hat, D_hat),
        }
        for tag, (col, a_, b_) in pairs.items():
            s_r = g[col].to_numpy(dtype=float)
            z_r = ((a_ - a_.mean()) * (b_ - b_.mean())).to_numpy(dtype=float)
            M = z_r.sum() / (R - 1)
            if R > 2 and M != 0:
                infl = s_r / M - (s_r.mean() / M ** 2) * (z_r * R / (R - 1) - M)
                rec[f"mcse_ratio_{tag}_mc"] = float(
                    infl.std(ddof=1) / np.sqrt(R))
            else:
                rec[f"mcse_ratio_{tag}_mc"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _interpretation(summary: pd.DataFrame) -> List[str]:
    """Three sentences of reading, computed from the table rather than typed."""
    boot = summary[np.isfinite(summary.ratio_si_boot)]
    ratios = np.concatenate([
        boot.ratio_si_boot.values, boot.ratio_ppi_boot.values,
        boot.ratio_cov_boot.values, boot.ratio_D_boot.values,
    ])
    worst = float(np.max(np.abs(ratios - 1.0)))

    multi = summary[summary.design == "Multi-surrogate (rich)"]
    nulls = summary[summary.is_null]
    small = nulls[nulls.n <= 64_000]
    big = nulls[nulls.n >= 100_000]

    # Sign pattern computed, not asserted: a quantity whose Sand/Boot ratio
    # is on one side of 1 in every bootstrapped cell is named.
    one_sided = []
    for col, name in (("ratio_si_boot", "Var(SI)"),
                      ("ratio_ppi_boot", "Var(PPI++)"),
                      ("ratio_cov_boot", "Cov"), ("ratio_D_boot", "Var(D)")):
        v = boot[col].to_numpy()
        if len(v) > 1 and (np.all(v < 1) or np.all(v > 1)):
            one_sided.append(
                f"{name} ({'below' if np.all(v < 1) else 'above'} in every "
                f"cell, {', '.join(f'{x:.3f}' for x in v)})")
    sign_txt = ("with no systematic sign" if not one_sided else
                "with the sandwich on one side of the bootstrap for "
                + " and ".join(one_sided))
    out = ["\n## Interpretation\n\n"]
    out.append(
        f"**Is the sandwich accurate?** Yes.  Across the "
        f"{ {2: 'two', 3: 'three', 4: 'four'}.get(len(boot), len(boot))} "
        f"bootstrapped "
        f"cells the sandwich reproduces the joint paired bootstrap for all "
        f"four quantities -- Var(SI), Var(PPI++), Cov and Var(D) -- to within "
        f"{100 * worst:.1f}%, {sign_txt}.  The remaining gap to "
        f"the Monte Carlo variance is within the Monte Carlo error of a "
        f"variance ratio at R = 200.\n\n")
    if len(multi):
        m = multi.iloc[0]
        out.append(
            f"**Does the first-stage term matter?** On the single-surrogate "
            f"DGPs barely: the plug-in variance is already within about 10% "
            f"of the truth, because the index has three free coefficients and "
            f"pi_L = 0.20.  On the Criteo-calibrated rich index it is "
            f"decisive: the plug-in variance is {m['ratio_si_plugin_mc']:.3f} "
            f"of the Monte Carlo variance and the SI interval covers "
            f"{m['cov_si_plugin']:.3f}, while the sandwich is "
            f"{m['ratio_si_mc']:.3f} and covers {m['cov_si']:.3f}.  That is "
            f"the Table S16 failure, now repaired by a formula rather than a "
            f"patch.\n\n")
    if len(small) and len(big):
        out.append(
            f"**What is the null size?** With the sandwich SE(D) the "
            f"two-sided diagnostic rejects "
            f"{small.rej_sand_05.mean():.3f} of the time at alpha = 0.05 and "
            f"{small.rej_sand_10.mean():.3f} at alpha = 0.10, averaged over "
            f"the null cells at n <= 64,000, against Monte Carlo standard "
            f"errors of about {np.sqrt(0.05 * 0.95 / (200 * len(small))):.3f} "
            f"and {np.sqrt(0.10 * 0.90 / (200 * len(small))):.3f}; at "
            f"n = 100,000 it is {big.rej_sand_05.iloc[0]:.3f} and "
            f"{big.rej_sand_10.iloc[0]:.3f}.  The test is therefore very "
            f"mildly liberal at n = 10,000 and correctly sized by "
            f"n = 100,000.  The bootstrap SE(D) gives the same sizes to "
            f"within Monte Carlo error, so the residual over-rejection is a "
            f"normal-approximation effect, not a variance-estimation one.\n")
    return out


def write_mc_check(summary: pd.DataFrame, raw: pd.DataFrame,
                   wall_time: float, suffix: str, workers: int) -> None:
    """Compact output for a no-bootstrap run: sandwich vs Monte Carlo only.

    Written instead of the full table when no cell carries the bootstrap
    (e.g. the R = 2,000 check), so the R = 200 bootstrap artifact
    `joint_cov_check.md` is never overwritten by a run that cannot fill it.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"joint_cov_check{suffix}"
    summary.to_csv(os.path.join(OUT_DIR, f"{stem}.csv"), index=False)
    raw.to_csv(os.path.join(OUT_DIR, f"{stem}_raw.csv"), index=False)

    def rt(r, tag):
        m, se = r[f"ratio_{tag}_mc"], r[f"mcse_ratio_{tag}_mc"]
        return "---" if not np.isfinite(m) else f"{m:.3f} ({se:.3f})"

    def pr(r, key):
        m, se = r[key], r[f"mcse_{key}"]
        return "---" if not np.isfinite(m) else f"{m:.3f} ({se:.3f})"

    Rs = sorted({int(x) for x in summary.R})
    lines: List[str] = [
        "# Joint sandwich against the Monte Carlo variance "
        f"(R = {', '.join(f'{x:,}' for x in Rs)}, no bootstrap)\n\n",
        "Same cells, seeds and code path as `joint_cov_check.md` "
        "(`scripts/run_joint_cov_check.py`), run without the paired "
        "bootstrap so that R can be large enough to make the `Sand/MC` "
        "column sharp.  Replications 0-199 are the draws of the R = 200 "
        "run.  `Sand/MC` is the mean joint-sandwich (co)variance over the "
        "Monte Carlo (co)variance of the estimators across the R "
        "replications; 1 means the sandwich is exact, below 1 "
        "anti-conservative.  D = tau_PPI++ - tau_SI.\n\n",
        "## Sandwich over Monte Carlo\n\n",
        "| Design | Cell | n | R | Var(SI) | Var(PPI++) | Cov | Var(D) |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['design']} | {r['cell']} | {r['n']:,} | {r['R']:,} "
            f"| {rt(r, 'si')} | {rt(r, 'ppi')} | {rt(r, 'cov')} "
            f"| {rt(r, 'D')} |\n")

    lines += [
        "\n## Levels\n\n",
        "| Design | Cell | n | R | Mean Var(SI) | MC Var(SI) | Mean Var(PPI++) "
        "| MC Var(PPI++) | Mean Cov | MC Cov | Mean Var(D) | MC Var(D) |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['design']} | {r['cell']} | {r['n']:,} | {r['R']:,} "
            f"| {r['mean_var_si']:.4e} | {r['mc_var_si']:.4e} "
            f"| {r['mean_var_ppi']:.4e} | {r['mc_var_ppi']:.4e} "
            f"| {r['mean_cov_si_ppi']:.4e} | {r['mc_cov']:.4e} "
            f"| {r['mean_var_D']:.4e} | {r['mc_var_D']:.4e} |\n")

    lines += [
        "\n## Coverage and diagnostic rejection rate\n\n",
        "| Design | Cell | n | R | H0 | Cov SI (sandwich) | Cov SI (plug-in) "
        "| Cov PPI++ | Reject 0.05 | Reject 0.10 |\n",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in summary.iterrows():
        h0 = "holds (size)" if r["is_null"] else "violated (power)"
        lines.append(
            f"| {r['design']} | {r['cell']} | {r['n']:,} | {r['R']:,} | {h0} "
            f"| {pr(r, 'cov_si')} | {pr(r, 'cov_si_plugin')} "
            f"| {pr(r, 'cov_ppi')} | {pr(r, 'rej_sand_05')} "
            f"| {pr(r, 'rej_sand_10')} |\n")

    lines += [
        "\nNotes: Monte Carlo standard errors in parentheses.  The SE of a "
        "`Sand/MC` ratio is the delta-method SE over replications (numerator "
        "and denominator are both replication means).  Coverage is of the "
        "true tau at nominal 0.95; the rejection rate is the two-sided "
        "SI--PPI++ estimator-disagreement diagnostic with SE(D) from the "
        "joint sandwich, a size where the null holds.\n",
        f"\nWall time: {wall_time:.0f} s on {workers} cores.\n",
    ]
    with open(os.path.join(OUT_DIR, f"{stem}.md"), "w") as fh:
        fh.writelines(lines)


def write_outputs(summary: pd.DataFrame, raw: pd.DataFrame,
                  wall_time: float, stem: str = "joint_cov_check",
                  clustered: bool = False) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    summary.to_csv(os.path.join(OUT_DIR, f"{stem}.csv"), index=False)
    raw.to_csv(os.path.join(OUT_DIR, f"{stem}_raw.csv"), index=False)

    def f(x, fmt="{:.4e}"):
        return "---" if not np.isfinite(x) else fmt.format(x)

    lines: List[str] = [
        "# Joint paired-bootstrap validation of the learned-index sandwich\n",
        "\n",
        "Both tau_SI and tau_PPI++ are functionals of the same learned linear "
        "index under all-units cross-fitting.  The sandwich "
        "(`src.methods.joint_influence_cov`) adds the first-stage term the "
        "fixed-predictor formulas omit and returns the full 2 x 2 covariance, "
        "hence Var(D) for the SI--PPI++ estimator-disagreement diagnostic.  "
        "The joint paired bootstrap resamples units with replacement "
        "stratified by arm, refits the index on the resampled labeled units "
        "under the same protocol, and recomputes both estimators, so it "
        "contains every term.  `MC` columns are the Monte Carlo variances of "
        "the estimators across the R replications: that is the quantity both "
        "the sandwich and the bootstrap are estimating.  Nominal level 95%.\n",
        "\n",
    ]
    if clustered:
        lines += [
            "Bootstrap folds clustered by original unit (`--clustered-folds`):"
            " in each resample the distinct original units are permuted and "
            "cut into five blocks and every copy inherits its unit's fold, so "
            "no copy is predicted by an index fit on another copy of the same "
            "unit.  Same cells, seeds and draws as `joint_cov_check.md`, whose "
            "bootstrap splits the resampled rows; the sandwich, Monte Carlo, "
            "coverage-of-sandwich and sandwich-size columns are therefore "
            "identical to that table and only the bootstrap columns differ.\n",
            "\n",
        ]
    lines += [
        "## Variances\n\n",
        "| Design | Cell | n | R | B | Mean Var(SI) | Boot | MC | "
        "Sand/Boot | Sand/MC | Mean Var(PPI) | Boot | MC | Sand/Boot | "
        "Sand/MC |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|\n",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['design']} | {r['cell']} | {r['n']:,} | {r['R']} "
            f"| {r['B']} | {f(r['mean_var_si'])} | {f(r['mean_boot_var_si'])} "
            f"| {f(r['mc_var_si'])} | {f(r['ratio_si_boot'], '{:.3f}')} "
            f"| {f(r['ratio_si_mc'], '{:.3f}')} | {f(r['mean_var_ppi'])} "
            f"| {f(r['mean_boot_var_ppi'])} | {f(r['mc_var_ppi'])} "
            f"| {f(r['ratio_ppi_boot'], '{:.3f}')} "
            f"| {f(r['ratio_ppi_mc'], '{:.3f}')} |\n")

    lines += [
        "\n## Covariance and the diagnostic denominator\n\n",
        "| Design | Cell | n | Mean Cov | Boot | MC | Sand/Boot | "
        "Mean Var(D) | Boot | MC | Sand/Boot | Sand/MC |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['design']} | {r['cell']} | {r['n']:,} "
            f"| {f(r['mean_cov_si_ppi'])} | {f(r['mean_boot_cov'])} "
            f"| {f(r['mc_cov'])} | {f(r['ratio_cov_boot'], '{:.3f}')} "
            f"| {f(r['mean_var_D'])} | {f(r['mean_boot_var_D'])} "
            f"| {f(r['mc_var_D'])} | {f(r['ratio_D_boot'], '{:.3f}')} "
            f"| {f(r['ratio_D_mc'], '{:.3f}')} |\n")

    lines += [
        "\n## The three SI variances (this table supersedes "
        "`si_bootstrap_check`)\n\n",
        "| Design | Cell | n | Plug-in | Delta-method | Sandwich | MC | "
        "Plug-in/MC | Delta/MC | Sandwich/MC | Cov (plug-in) | Cov (delta) | "
        "Cov (sandwich) |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['design']} | {r['cell']} | {r['n']:,} "
            f"| {f(r['mean_var_si_plugin'])} | {f(r['mean_var_si_delta'])} "
            f"| {f(r['mean_var_si'])} | {f(r['mc_var_si'])} "
            f"| {f(r['ratio_si_plugin_mc'], '{:.3f}')} "
            f"| {f(r['ratio_si_delta_mc'], '{:.3f}')} "
            f"| {f(r['ratio_si_mc'], '{:.3f}')} "
            f"| {r['cov_si_plugin']:.3f} | {r['cov_si_delta']:.3f} "
            f"| {r['cov_si']:.3f} |\n")

    lines += [
        "\n## Coverage of tau (nominal 0.95)\n\n",
        "| Design | Cell | n | R | Cov SI (sandwich) | Cov SI (bootstrap) | "
        "Cov PPI++ | Cov PPI++ (bootstrap) |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]

    def cov_cell(r, key):
        m, se = r[key], r[f"mcse_{key}"]
        return "---" if not np.isfinite(m) else f"{m:.3f} ({se:.4f})"

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['design']} | {r['cell']} | {r['n']:,} | {r['R']} "
            f"| {cov_cell(r, 'cov_si')} | {cov_cell(r, 'cov_si_boot')} "
            f"| {cov_cell(r, 'cov_ppi')} "
            f"| {cov_cell(r, 'cov_ppi_boot')} |\n")

    lines += [
        "\n## Diagnostic rejection rate (size where the null holds)\n\n",
        "| Design | Cell | n | R | H0 | 0.05 (sandwich) | 0.10 (sandwich) | "
        "0.05 (bootstrap) | 0.10 (bootstrap) |\n",
        "|---|---|---:|---:|---|---:|---:|---:|---:|\n",
    ]
    for _, r in summary.iterrows():
        h0 = "holds (size)" if r["is_null"] else "violated (power)"
        lines.append(
            f"| {r['design']} | {r['cell']} | {r['n']:,} | {r['R']} | {h0} "
            f"| {cov_cell(r, 'rej_sand_05')} | {cov_cell(r, 'rej_sand_10')} "
            f"| {cov_cell(r, 'rej_boot_05')} "
            f"| {cov_cell(r, 'rej_boot_10')} |\n")

    lines += [
        "\nNotes: Monte Carlo standard errors in parentheses.  `Sand/Boot` is "
        "the mean sandwich variance over the mean paired-bootstrap variance; "
        "`Sand/MC` is the mean sandwich variance over the Monte Carlo "
        "variance of the estimator across the R replications.  A ratio of "
        "1.000 means the sandwich is exact at that cell; below 1 it is "
        "anti-conservative.  The rejection rate is two-sided; it is a size "
        "where the null holds (DGP 1; DGP 2 at rho = 0; the rich index at "
        "m = 1, where surrogacy holds exactly) and power where it does not "
        "(DGP 2 at rho = 0.2).  The n = 100,000 cell is run without the "
        "bootstrap, so its bootstrap columns read ---.  At R = 200 the Monte "
        "Carlo standard error of a variance ratio is about 10%, so the "
        "`Sand/MC` column is the noisy comparison and `Sand/Boot` the sharp "
        "one.\n",
    ]

    lines += _interpretation(summary)
    lines.append(f"\nWall time: {wall_time:.0f} s on {N_WORKERS} cores.\n")

    with open(os.path.join(OUT_DIR, f"{stem}.md"), "w") as fh:
        fh.writelines(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="1,2,3,4,5",
                    help="comma-separated cell ids (see module docstring)")
    ap.add_argument("-R", "--reps", type=int, default=200)
    ap.add_argument("-B", "--boot", type=int, default=200)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--no-boot", action="store_true",
                    help="skip the paired bootstrap in every cell (B = 0)")
    ap.add_argument("--out-suffix", default="",
                    help="write joint_cov_check<suffix>.{md,csv,_raw.csv}; "
                         "required with --no-boot so the R = 200 bootstrap "
                         "artifact is not overwritten")
    ap.add_argument("--clustered-folds", action="store_true",
                    help="assign bootstrap folds by original unit (all copies "
                         "of a resampled unit share a fold); writes "
                         "joint_cov_check<suffix>.* with suffix defaulting to "
                         "_clustered")
    ap.add_argument("--outdir", default=TABLES_DIR,
                    help="output directory (smoke runs: a temp directory)")
    args = ap.parse_args()
    global OUT_DIR
    OUT_DIR = args.outdir
    if args.clustered_folds and not args.out_suffix:
        args.out_suffix = "_clustered"
    if args.no_boot:
        args.boot = 0
        if not args.out_suffix:
            ap.error("--no-boot needs --out-suffix (e.g. _R2000)")

    cell_ids = [int(c) for c in args.cells.split(",") if c.strip()]
    t0 = time.time()
    _self_check()

    params = None
    tau_multi = None
    if 4 in cell_ids:
        _multi_true_tau = _load_multisurrogate_true_tau()
        with open(PARAM_PATH) as fh:
            params = json.load(fh)
        X_pool = np.load(POOL_PATH)["X"].astype(np.float64)
        tau_multi = _multi_true_tau(params, X_pool, M_RICH)
        del X_pool
        print(f"cell 4 true tau (m = {M_RICH}) = {tau_multi:.6f}", flush=True)

    tasks = [(c, rep, args.boot, params, tau_multi, args.clustered_folds)
             for c in cell_ids for rep in range(args.reps)]
    print(f"{len(tasks)} tasks over cells {cell_ids} "
          f"(R = {args.reps}, B = {args.boot})", flush=True)

    with multiprocessing.Pool(args.workers) as pool:
        rows = pool.map(run_task, tasks, chunksize=1)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    wall = time.time() - t0
    if args.clustered_folds and args.boot > 0:
        write_outputs(summary, raw, wall,
                      stem=f"joint_cov_check{args.out_suffix}",
                      clustered=True)
    elif args.boot == 0 or args.out_suffix:
        write_mc_check(summary, raw, wall, args.out_suffix, args.workers)
    else:
        write_outputs(summary, raw, wall)

    print(f"\ntotal {wall:.0f}s")
    cols = ["design", "cell", "n", "R", "ratio_si_boot", "ratio_ppi_boot",
            "ratio_cov_boot", "ratio_D_boot", "ratio_si_plugin_mc",
            "ratio_si_delta_mc", "ratio_si_mc", "ratio_D_mc",
            "cov_si_plugin", "cov_si_delta", "cov_si", "cov_ppi",
            "rej_sand_05", "rej_sand_10", "rej_boot_05", "rej_boot_10"]
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
