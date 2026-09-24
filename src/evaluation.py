"""
Evaluation metrics for surrogate metrics simulation.

Computes Bias, RMSE, Coverage, Correct Decision Rate, Cumulative Regret,
and Relative Efficiency from simulation result DataFrames.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-method metric computation
# ---------------------------------------------------------------------------

def compute_bias(
    tau_hats: np.ndarray, true_tau: float
) -> Dict[str, float]:
    """
    Compute bias and its Monte Carlo standard error.

    Parameters
    ----------
    tau_hats : (R,) array of point estimates across replications
    true_tau : true ATE

    Returns
    -------
    dict with keys: bias, rel_bias (%), mc_se_bias
    """
    valid = ~np.isnan(tau_hats)
    if valid.sum() < 2:
        return dict(bias=np.nan, rel_bias=np.nan, mc_se_bias=np.nan)

    tau_hats = tau_hats[valid]
    R = len(tau_hats)

    errors = tau_hats - true_tau
    bias = errors.mean()
    s = tau_hats.std(ddof=1)
    mc_se = s / np.sqrt(R)

    rel_bias = (bias / true_tau * 100.0) if abs(true_tau) > 1e-12 else np.nan

    return dict(bias=bias, rel_bias=rel_bias, mc_se_bias=mc_se)


def compute_rmse(
    tau_hats: np.ndarray, true_tau: float
) -> Dict[str, float]:
    """
    Compute RMSE and its decomposition into bias^2 + variance.

    Returns
    -------
    dict with keys: rmse, mse, bias_sq, variance
    """
    valid = ~np.isnan(tau_hats)
    if valid.sum() < 2:
        return dict(rmse=np.nan, mse=np.nan, bias_sq=np.nan, variance=np.nan)

    tau_hats = tau_hats[valid]
    R = len(tau_hats)

    errors = tau_hats - true_tau
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    bias = errors.mean()
    bias_sq = bias ** 2
    variance = tau_hats.var(ddof=1) * (R - 1) / R  # population variance

    return dict(rmse=rmse, mse=mse, bias_sq=bias_sq, variance=variance)


def compute_coverage(
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    true_tau: float,
) -> Dict[str, float]:
    """
    Compute empirical coverage and its Monte Carlo standard error.

    Returns
    -------
    dict with keys: coverage, mc_se_coverage, under_coverage (bool),
                    over_coverage (bool)
    """
    valid = ~(np.isnan(ci_lower) | np.isnan(ci_upper))
    if valid.sum() < 2:
        return dict(
            coverage=np.nan, mc_se_coverage=np.nan,
            under_coverage=False, over_coverage=False,
        )

    ci_lower = ci_lower[valid]
    ci_upper = ci_upper[valid]
    R = len(ci_lower)

    covers = (ci_lower <= true_tau) & (true_tau <= ci_upper)
    cov = covers.mean()
    mc_se = np.sqrt(cov * (1.0 - cov) / R)

    # MC band for R = 2000: approximately [0.940, 0.960]
    mc_z = 1.96
    nominal = 0.95
    mc_band_half = mc_z * np.sqrt(nominal * (1 - nominal) / R)
    under = cov < (nominal - mc_band_half)
    over = cov > (nominal + mc_band_half)

    return dict(
        coverage=cov, mc_se_coverage=mc_se,
        under_coverage=bool(under), over_coverage=bool(over),
    )


def compute_cdr(
    tau_hats: np.ndarray,
    V_hats: np.ndarray,
    true_tau: float,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Compute Correct Decision Rate (CDR).

    Two-sided test: reject H_0: tau = 0 if |tau_hat / se| > z_{1-alpha/2}.

    Returns
    -------
    dict with keys: cdr, mc_se_cdr, rejection_rate
    """
    from scipy import stats as sp_stats

    valid = ~(np.isnan(tau_hats) | np.isnan(V_hats))
    if valid.sum() < 2:
        return dict(cdr=np.nan, mc_se_cdr=np.nan, rejection_rate=np.nan)

    tau_hats = tau_hats[valid]
    V_hats = V_hats[valid]
    R = len(tau_hats)

    z_crit = sp_stats.norm.ppf(1.0 - alpha / 2.0)

    se = np.sqrt(np.maximum(V_hats, 0.0))
    # Avoid division by zero
    z_stats = np.where(se > 1e-15, np.abs(tau_hats / se), 0.0)
    rejects = z_stats > z_crit

    rejection_rate = rejects.mean()

    # Correct decision depends on truth
    if abs(true_tau) < 1e-12:
        # Null is true -> correct = not reject
        correct = ~rejects
    else:
        # Null is false -> correct = reject AND correct sign
        correct_sign = np.sign(tau_hats) == np.sign(true_tau)
        correct = rejects & correct_sign

    cdr = correct.astype(float).mean()
    mc_se = np.sqrt(cdr * (1.0 - cdr) / R)

    return dict(cdr=cdr, mc_se_cdr=mc_se, rejection_rate=rejection_rate)


def _compute_cdr_varying(
    tau_hats: np.ndarray,
    V_hats: np.ndarray,
    true_taus: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Compute CDR when true_tau varies per replication.

    Uses per-replication true_tau for the null/sign checks instead of a
    scalar average, so that replications where true_tau crosses zero are
    handled correctly.
    """
    from scipy import stats as sp_stats

    valid = ~(np.isnan(tau_hats) | np.isnan(V_hats) | np.isnan(true_taus))
    if valid.sum() < 2:
        return dict(cdr=np.nan, mc_se_cdr=np.nan, rejection_rate=np.nan)

    tau_hats = tau_hats[valid]
    V_hats = V_hats[valid]
    true_taus = true_taus[valid]
    R = len(tau_hats)

    z_crit = sp_stats.norm.ppf(1.0 - alpha / 2.0)

    se = np.sqrt(np.maximum(V_hats, 0.0))
    z_stats = np.where(se > 1e-15, np.abs(tau_hats / se), 0.0)
    rejects = z_stats > z_crit

    rejection_rate = rejects.mean()

    # Per-replication correct decision
    null_true = np.abs(true_taus) < 1e-12
    correct_sign = np.sign(tau_hats) == np.sign(true_taus)
    correct = np.where(null_true, ~rejects, rejects & correct_sign)

    cdr = correct.astype(float).mean()
    mc_se = np.sqrt(cdr * (1.0 - cdr) / R)

    return dict(cdr=cdr, mc_se_cdr=mc_se, rejection_rate=rejection_rate)


def summarize_portfolio_regret(
    regrets: np.ndarray,
    oracle_gains: np.ndarray,
) -> Dict[str, float]:
    """Summarize the DGP 6 portfolio regret over replications.

    Each replication draws its OWN portfolio of K experiments, so the oracle
    gain ``V*_r = sum_k max(tau_{r,k}, 0)`` is a random variable and not a
    constant of the design.  Both inputs are therefore length-R vectors.

    Relative regret is reported two ways:

    * ``rel_regret``  -- the RATIO OF MEANS, ``100 * mean_r(regret_r) /
      mean_r(V*_r)``.  This is the quantity the paper's tables print, and it
      is the same definition as ``scripts/run_portfolio_sensitivity.py``.
    * ``rel_regret_mean_of_ratios`` -- ``100 * mean_r(regret_r / V*_r)``,
      reported alongside as a robustness read.  The two differ by the
      covariance between regret and oracle gain and agree to within their
      Monte Carlo error here.

    ``mc_se_rel_regret`` is the delta-method (paired influence-function)
    Monte Carlo SE of the ratio of means:
    ``sd_r(regret_r - rho * V*_r) / (sqrt(R) * mean(V*))`` with
    ``rho = mean(regret) / mean(V*)``.

    Returns
    -------
    dict with keys: mean_cumul_regret, mc_se_regret, oracle_gain,
    mc_se_oracle_gain, rel_regret, mc_se_rel_regret,
    rel_regret_mean_of_ratios, mc_se_rel_regret_mean_of_ratios
    """
    reg = np.asarray(regrets, dtype=float)
    og = np.asarray(oracle_gains, dtype=float)
    if og.ndim == 0 or og.size == 1:
        og = np.full(reg.shape, float(og))

    valid = ~(np.isnan(reg) | np.isnan(og))
    reg = reg[valid]
    og = og[valid]
    R = len(reg)

    out: Dict[str, float] = dict(
        mean_cumul_regret=np.nan, mc_se_regret=np.nan,
        oracle_gain=np.nan, mc_se_oracle_gain=np.nan,
        rel_regret=np.nan, mc_se_rel_regret=np.nan,
        rel_regret_mean_of_ratios=np.nan,
        mc_se_rel_regret_mean_of_ratios=np.nan,
    )
    if R == 0:
        return out

    mean_regret = float(reg.mean())
    mean_og = float(og.mean())
    out["mean_cumul_regret"] = mean_regret
    out["oracle_gain"] = mean_og
    if R > 1:
        out["mc_se_regret"] = float(reg.std(ddof=1) / np.sqrt(R))
        out["mc_se_oracle_gain"] = float(og.std(ddof=1) / np.sqrt(R))

    if mean_og > 1e-12:
        rho = mean_regret / mean_og
        out["rel_regret"] = 100.0 * rho
        if R > 1:
            infl = reg - rho * og
            out["mc_se_rel_regret"] = float(
                100.0 * infl.std(ddof=1) / (np.sqrt(R) * mean_og)
            )

    pos = og > 1e-12
    if pos.any():
        ratios = reg[pos] / og[pos]
        out["rel_regret_mean_of_ratios"] = float(100.0 * ratios.mean())
        if pos.sum() > 1:
            out["mc_se_rel_regret_mean_of_ratios"] = float(
                100.0 * ratios.std(ddof=1) / np.sqrt(pos.sum())
            )
    return out


def compute_cumulative_regret(
    decisions_list: Sequence[np.ndarray],
    true_taus: np.ndarray,
) -> Dict[str, float]:
    """
    Compute cumulative regret for DGP 6 (portfolio).

    Parameters
    ----------
    decisions_list : list of (K,) bool arrays, one per replication
    true_taus : (K,) array of true ATEs shared by every replication, or a
        length-R sequence of (K,) arrays when each replication draws its own
        portfolio (the DGP 6 case).

    Returns
    -------
    dict from :func:`summarize_portfolio_regret`
    """
    R = len(decisions_list)

    taus_arr = np.asarray(true_taus, dtype=float)
    per_rep = taus_arr.ndim == 2
    if per_rep and taus_arr.shape[0] != R:
        raise ValueError(
            "true_taus has {} rows but there are {} replications".format(
                taus_arr.shape[0], R
            )
        )

    regrets = np.zeros(R)
    oracle_gains = np.zeros(R)
    for r, decisions in enumerate(decisions_list):
        taus_r = taus_arr[r] if per_rep else taus_arr
        d_r = np.asarray(decisions, dtype=bool)
        oracle_gains[r] = float(np.sum(np.maximum(taus_r, 0.0)))
        regrets[r] = float(
            np.sum(np.where(taus_r > 0, np.where(d_r, 0.0, taus_r), 0.0))
            + np.sum(np.where(taus_r < 0, np.where(d_r, -taus_r, 0.0), 0.0))
        )

    return summarize_portfolio_regret(regrets, oracle_gains)


def compute_relative_efficiency(
    rmse_baseline: float, rmse_method: float
) -> float:
    """
    Relative efficiency: RE = RMSE_baseline / RMSE_method.

    Values > 1 mean the method is more efficient than baseline (Method 0).
    """
    if np.isnan(rmse_baseline) or np.isnan(rmse_method) or rmse_method < 1e-15:
        return np.nan
    return rmse_baseline / rmse_method


# ---------------------------------------------------------------------------
# Aggregate evaluation from simulation DataFrame
# ---------------------------------------------------------------------------

def evaluate_simulation_results(
    df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute all evaluation metrics from a simulation results DataFrame.

    Parameters
    ----------
    df : DataFrame with columns:
         replication, method_id, method_name, tau_hat, V_hat,
         ci_lower, ci_upper, true_tau
         Optionally: cumulative_regret, oracle_gain, decisions, true_taus

    Returns
    -------
    DataFrame with one row per method and columns for all metrics.
    """
    method_ids = sorted(df["method_id"].unique())

    # Check if true_tau varies across replications (e.g., DGP 3)
    # If it does, use per-replication true_tau for evaluation
    true_tau_series = df.groupby("replication")["true_tau"].first()
    true_tau_varies = true_tau_series.nunique() > 1
    true_tau_scalar = float(true_tau_series.mean())

    rows = []
    rmse_by_method: Dict[int, float] = {}

    for m_id in method_ids:
        mdf = df[df["method_id"] == m_id].sort_values("replication")
        method_name = mdf["method_name"].iloc[0]

        tau_hats = mdf["tau_hat"].values
        V_hats = mdf["V_hat"].values
        ci_lower = mdf["ci_lower"].values
        ci_upper = mdf["ci_upper"].values

        if true_tau_varies:
            # Use per-replication true_tau
            true_taus = mdf["true_tau"].values
            # Bias: mean(tau_hat - true_tau_r) across replications
            errors = tau_hats - true_taus
            bias_val = float(np.nanmean(errors))
            mc_se = float(np.nanstd(errors) / np.sqrt(np.sum(~np.isnan(errors))))
            rel_bias = float(bias_val / true_tau_scalar * 100) if abs(true_tau_scalar) > 1e-12 else np.nan
            bias_res = {"bias": bias_val, "rel_bias": rel_bias, "mc_se_bias": mc_se}
            # RMSE: sqrt(mean((tau_hat - true_tau_r)^2))
            rmse_val = float(np.sqrt(np.nanmean(errors ** 2)))
            rmse_res = {"rmse": rmse_val, "mse": rmse_val**2,
                        "bias_sq": bias_val**2, "variance": rmse_val**2 - bias_val**2}
            # Coverage: per-replication CI covers per-replication true_tau
            covers = (ci_lower <= true_taus) & (true_taus <= ci_upper)
            cov_val = float(np.nanmean(covers))
            mc_se_cov = float(np.sqrt(cov_val * (1 - cov_val) / len(covers)))
            cov_res = {"coverage": cov_val, "mc_se_coverage": mc_se_cov,
                       "under_coverage": cov_val < 0.95 - 1.96 * mc_se_cov,
                       "over_coverage": cov_val > 0.95 + 1.96 * mc_se_cov}
            # CDR: use per-replication true_tau for correct sign/null checks
            cdr_res = _compute_cdr_varying(tau_hats, V_hats, true_taus, alpha=alpha)
        else:
            true_tau = true_tau_scalar
            bias_res = compute_bias(tau_hats, true_tau)
            rmse_res = compute_rmse(tau_hats, true_tau)
            cov_res = compute_coverage(ci_lower, ci_upper, true_tau)
            cdr_res = compute_cdr(tau_hats, V_hats, true_tau, alpha=alpha)

        rmse_by_method[m_id] = rmse_res["rmse"]

        row = dict(
            method_id=m_id,
            method_name=method_name,
            true_tau=true_tau_scalar,
            n_valid=int((~np.isnan(tau_hats)).sum()),
            **bias_res,
            **rmse_res,
            **cov_res,
            **cdr_res,
        )

        # DGP 6 portfolio metrics.  The oracle gain is a per-replication
        # random variable (each replication draws its own portfolio), so it
        # is AVERAGED over replications; relative regret is the ratio of
        # means, matching scripts/run_portfolio_sensitivity.py.
        if "cumulative_regret" in mdf.columns:
            reg_vals = mdf["cumulative_regret"].values
            og_vals = (
                mdf["oracle_gain"].values if "oracle_gain" in mdf.columns
                else np.full(len(reg_vals), np.nan)
            )
            row.update(summarize_portfolio_regret(reg_vals, og_vals))

        rows.append(row)

    # Compute relative efficiency (vs. Method 0)
    rmse_baseline = rmse_by_method.get(0, np.nan)
    for row in rows:
        m_id = row["method_id"]
        row["relative_efficiency"] = compute_relative_efficiency(
            rmse_baseline, rmse_by_method.get(m_id, np.nan)
        )

    result_df = pd.DataFrame(rows)
    return result_df


def summarize_across_configs(
    eval_dfs: Sequence[pd.DataFrame],
    config_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Concatenate evaluation results across multiple configurations.

    Parameters
    ----------
    eval_dfs : list of evaluation DataFrames (from evaluate_simulation_results)
    config_labels : list of string labels for each config

    Returns
    -------
    Combined DataFrame with a 'config' column.
    """
    combined = []
    for i, edf in enumerate(eval_dfs):
        edf = edf.copy()
        edf["config"] = config_labels[i] if config_labels else f"config_{i}"
        combined.append(edf)
    return pd.concat(combined, ignore_index=True)
