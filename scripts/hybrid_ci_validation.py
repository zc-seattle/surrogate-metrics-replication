#!/usr/bin/env python3
"""
Hybrid CI Validation — lighter configuration.

Changes from v1:
  - R=500 (not 2000)
  - B=200 (not 500)
  - Only 2 configs: DGP 1 pi_L=0.20, DGP 2 rho=0.2 pi_L=0.20
  - Pool(4) instead of Pool(8)
  - Saves standardized estimates for Q-Q plotting

Reports: coverage of sim-calibrated CI vs bootstrap CI, CI widths,
         raw standardized estimates for Q-Q plots.
Outputs (in --outdir, default results/tables):
  hybrid_ci_validation.csv        one row per replication (raw data, including
                                  z_standardized for Q-Q plots)
  hybrid_ci_validation_rows.json  summary result rows, one per (cell, CI
                                  method)
  hybrid_ci_validation.md         the table and summary statistics
`--from-raw` rebuilds the rows and the table from an existing CSV.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dgps.dgps import DGP_GENERATORS
from src.simulations.simulation import derive_seed
from src.utils.config import DEFAULT_PROTOCOL, DGP_DEFAULTS, PROTOCOLS
from src.utils.config import make_result_row
from src.utils.registry import metrics_from_errors, write_rows
from src.utils.estimator_api import (
    estimate,
    hybrid_estimator,
    hybrid_sigma,
    surrogacy_test,
    train_prediction_model,
)

# --- Lighter configuration ---
R = 500
B_BOOT = 200

#: Prediction protocol; overridden by --protocol in main().
PROTOCOL = DEFAULT_PROTOCOL

def _set_protocol(protocol: str) -> None:
    """Pool initializer: propagate --protocol into the worker processes.

    Workers re-import this module under the spawn start method, so a global
    set in main() would not reach them.
    """
    global PROTOCOL
    PROTOCOL = protocol
M_SIM = 10_000
HYBRID_C = 1.5
ALPHA = 0.05
N_CORES = 4
MASTER_SEED = 42

# Only 2 configurations
DGP_CONFIGS = [
    (1, {}, [0.20]),
    (2, {"rho": 0.2}, [0.20]),
]


def _compute_hybrid_from_arrays(T, S, Y, Y_hat, labeled_mask, rng_seed,
                                design):
    """Compute SI, PPI++, and hybrid from pre-computed arrays.

    `design` is the dictionary from
    train_prediction_model(..., return_design=True): the SI variance and the
    hybrid's Sigma_hat are the joint sandwich for the learned index, so they
    cannot be computed from the predictions alone.
    """
    res_ppi = estimate(3, T, S, Y, Y_hat, labeled_mask, protocol=PROTOCOL)
    lambda_hat = res_ppi.get("lambda_hat", 1.0)
    res_si = estimate(2, T, S, Y, Y_hat, labeled_mask, protocol=PROTOCOL,
                      design=design, lambda_hat=lambda_hat)

    # Sigma_hat: the joint sandwich covariance for the learned index.
    sigma = hybrid_sigma(T, Y, labeled_mask, design, lambda_hat)
    var_si_j, cov_sp, var_ppi_j = (
        float(sigma[0, 0]), float(sigma[0, 1]), float(sigma[1, 1])
    )

    res_hyb = hybrid_estimator(
        res_si["tau_hat"], res_ppi["tau_hat"],
        var_si_j, var_ppi_j,
        cov_sp, c=HYBRID_C, alpha=ALPHA, M=M_SIM, rng_seed=rng_seed,
    )

    return res_si, res_ppi, res_hyb, lambda_hat, cov_sp


def run_one_rep(args: Tuple) -> Dict[str, Any]:
    # `B` travels with the task: a global set in main() would not reach the
    # spawned workers, so --B used to be ignored.
    dgp_id, overrides, pi_L, rep, master_seed = args[:5]
    B = int(args[5]) if len(args) > 5 else B_BOOT
    config_hash = int(hashlib.sha256(str(overrides).encode()).hexdigest()[:8], 16) % 1000
    rep_seed = derive_seed(master_seed, dgp_id, config_hash, int(pi_L * 100), rep)

    defaults = dict(DGP_DEFAULTS[dgp_id])
    defaults.update(overrides)
    defaults["seed"] = rep_seed
    defaults["pi_L"] = pi_L

    dgp_func = DGP_GENERATORS[dgp_id]
    data = dgp_func(**defaults)
    true_tau = data["true_tau"]

    T, S, Y, X = data["T"], data["S"], data["Y"], data["X"]
    labeled_mask = data["labeled_mask"]

    cf_rng = np.random.default_rng(rep_seed + 7777)
    Y_hat, design = train_prediction_model(
        S, X, Y, labeled_mask, n_folds=5, rng=cf_rng,
        protocol=PROTOCOL, return_design=True,
    )

    # --- Main estimates ---
    res_si, res_ppi, res_hyb, lambda_hat, cov_sp = _compute_hybrid_from_arrays(
        T, S, Y, Y_hat, labeled_mask, rng_seed=rep_seed + 9999, design=design
    )

    tau_hybrid = res_hyb["tau_hybrid"]
    sim_ci_lower = res_hyb["ci_lower"]
    sim_ci_upper = res_hyb["ci_upper"]

    # --- Bootstrap CI (B resamples, default 200, stratified by T) ---
    boot_rng = np.random.default_rng(rep_seed + 5555)
    idx_t1 = np.where(T == 1)[0]
    idx_t0 = np.where(T == 0)[0]

    boot_tau_hybrids = np.empty(B)
    for b in range(B):
        # Resample units stratified by treatment
        boot_idx_t1 = boot_rng.choice(idx_t1, size=len(idx_t1), replace=True)
        boot_idx_t0 = boot_rng.choice(idx_t0, size=len(idx_t0), replace=True)
        boot_idx = np.concatenate([boot_idx_t1, boot_idx_t0])

        T_b = T[boot_idx]
        S_b = S[boot_idx]
        Y_b = Y[boot_idx]
        X_b = X[boot_idx]
        lm_b = labeled_mask[boot_idx]

        # Retrain prediction model on bootstrap sample
        b_rng = np.random.default_rng(rep_seed + 10000 + b)
        Y_hat_b, design_b = train_prediction_model(
            S_b, X_b, Y_b, lm_b, n_folds=5, rng=b_rng,
            protocol=PROTOCOL, return_design=True,
        )

        # Recompute SI, PPI++, test stat, weight, hybrid.  The bootstrap
        # refits the index inside each resample, so its covariance is the
        # joint sandwich too.
        res_ppi_b = estimate(3, T_b, S_b, Y_b, Y_hat_b, lm_b,
                             protocol=PROTOCOL)
        lam_b = res_ppi_b.get("lambda_hat", 1.0)
        res_si_b = estimate(2, T_b, S_b, Y_b, Y_hat_b, lm_b,
                            protocol=PROTOCOL, design=design_b,
                            lambda_hat=lam_b)
        sigma_b = hybrid_sigma(T_b, Y_b, lm_b, design_b, lam_b)
        var_si_b, cov_b, var_ppi_b = (
            float(sigma_b[0, 0]), float(sigma_b[0, 1]), float(sigma_b[1, 1])
        )

        # Only T_n is used for the bootstrap weight, so `alternative` does
        # not affect the result; passed explicitly to pin the convention.
        test_b = surrogacy_test(
            res_si_b["tau_hat"], res_ppi_b["tau_hat"],
            var_si_b, var_ppi_b, cov_b,
            alternative="two-sided",
        )
        T_n_b = test_b["T_n"]
        w_b = HYBRID_C / (HYBRID_C + T_n_b ** 2)
        boot_tau_hybrids[b] = w_b * res_si_b["tau_hat"] + (1.0 - w_b) * res_ppi_b["tau_hat"]

    boot_ci_lower = float(np.percentile(boot_tau_hybrids, 100 * ALPHA / 2))
    boot_ci_upper = float(np.percentile(boot_tau_hybrids, 100 * (1.0 - ALPHA / 2)))

    # --- Standardized estimate for Q-Q ---
    sim_ci_width = sim_ci_upper - sim_ci_lower
    se_sim = sim_ci_width / (2 * 1.96)
    se_sim = max(se_sim, 1e-12)
    z_standardized = (tau_hybrid - true_tau) / se_sim

    return {
        "dgp_id": dgp_id,
        "overrides": str(overrides),
        "pi_L": pi_L,
        "rep": rep,
        "true_tau": true_tau,
        "tau_hybrid": tau_hybrid,
        "tau_si": res_si["tau_hat"],
        "tau_ppi": res_ppi["tau_hat"],
        "w": res_hyb["w"],
        "T_n": res_hyb["T_n"],
        "lambda_hat": lambda_hat,
        # Sim-calibrated CI
        "sim_ci_lower": sim_ci_lower,
        "sim_ci_upper": sim_ci_upper,
        "sim_ci_width": sim_ci_width,
        "sim_covers": float(sim_ci_lower <= true_tau <= sim_ci_upper),
        # Bootstrap CI
        "boot_ci_lower": boot_ci_lower,
        "boot_ci_upper": boot_ci_upper,
        "boot_ci_width": float(boot_ci_upper - boot_ci_lower),
        "boot_covers": float(boot_ci_lower <= true_tau <= boot_ci_upper),
        # Q-Q data
        "z_standardized": z_standardized,
    }


#: (method label, md label, CI-column prefix)
CI_METHODS = [
    ("Hybrid (sim-calibrated CI)", "Sim-calibrated", "sim"),
    ("Hybrid (bootstrap CI)", "Bootstrap", "boot"),
]


def summary_rows(df: pd.DataFrame, protocol: str, B: int,
                 seed: int = MASTER_SEED) -> List[Dict[str, Any]]:
    """One result row per (cell, CI method).

    Both rows share the hybrid point estimate (bias, RMSE); they differ in
    coverage and width.  The median width and the cell's summary statistics
    (weight, T_n, lambda_hat, standardized-z moments and quantiles) are
    carried as extra fields, so every number in the md traces to a row.
    """
    rows: List[Dict[str, Any]] = []
    for dgp_id, overrides, pi_Ls in DGP_CONFIGS:
        for pi_L in pi_Ls:
            g = df[(df["dgp_id"] == dgp_id)
                   & (df["overrides"] == str(overrides))
                   & (df["pi_L"] == pi_L)]
            if g.empty:
                continue
            tt = g["true_tau"].to_numpy(float)
            err = g["tau_hybrid"].to_numpy(float) - tt
            z = g["z_standardized"]
            q = np.percentile(z, [5, 25, 50, 75, 95])
            stats = {
                "mean_w": float(g["w"].mean()), "sd_w": float(g["w"].std()),
                "mean_T_n": float(g["T_n"].mean()),
                "sd_T_n": float(g["T_n"].std()),
                "mean_lambda_hat": float(g["lambda_hat"].mean()),
                "z_mean": float(z.mean()), "z_sd": float(z.std()),
                "z_q05": float(q[0]), "z_q25": float(q[1]),
                "z_q50": float(q[2]), "z_q75": float(q[3]),
                "z_q95": float(q[4]),
            }
            name = f"DGP{dgp_id}" + "".join(
                f"_{k}{v}" for k, v in overrides.items()) + f"_piL{pi_L:.2f}"
            for label, md_label, pre in CI_METHODS:
                width = g[f"{pre}_ci_width"]
                rows.append(make_result_row(
                    dgp=dgp_id, config_name=name,
                    params={"dgp_id": dgp_id, **overrides}, pi_L=pi_L,
                    n=DGP_DEFAULTS[dgp_id].get("n"), R=g["rep"].nunique(),
                    protocol=protocol, method_id=-1, method_label=label,
                    alpha=ALPHA, target="ATE", seed=seed,
                    metrics=metrics_from_errors(
                        err, true_tau=tt,
                        covers=g[f"{pre}_covers"].to_numpy(float),
                        ci_width=width.to_numpy(float)),
                    extra={"analysis": "hybrid_ci_validation",
                           "table_label": md_label, "ci_method": pre,
                           "median_ci_width": float(width.median()),
                           "c": HYBRID_C, "B": B, "M": M_SIM,
                           **dict(overrides), **stats},
                ))
    return rows


def build_markdown(df: pd.DataFrame, R: int, B: int) -> List[str]:
    lines = [
        "# Hybrid CI Validation\n\n",
        f"R = {R}, B = {B}, M = {M_SIM}, c = {HYBRID_C}\n\n",
    ]

    for dgp_id, overrides, pi_Ls in DGP_CONFIGS:
        label = f"DGP {dgp_id}"
        if overrides:
            label += f" ({', '.join(f'{k}={v}' for k, v in overrides.items())})"

        sub = df[(df["dgp_id"] == dgp_id) & (df["overrides"] == str(overrides))]
        if sub.empty:
            continue

        true_tau = sub["true_tau"].iloc[0]
        lines.append(f"## {label} (true_tau = {true_tau:.4f})\n\n")
        lines.append("| pi_L | CI Method | Coverage | Mean Width | Median Width |\n")
        lines.append("|------|-----------|----------|------------|-------------|\n")

        for pi_L in pi_Ls:
            pl_sub = sub[sub["pi_L"] == pi_L]
            if pl_sub.empty:
                continue

            # Sim-calibrated
            sim_cov = pl_sub["sim_covers"].mean()
            sim_width_mean = pl_sub["sim_ci_width"].mean()
            sim_width_median = pl_sub["sim_ci_width"].median()
            lines.append(
                f"| {pi_L} | Sim-calibrated | {sim_cov:.3f} | "
                f"{sim_width_mean:.4f} | {sim_width_median:.4f} |\n"
            )

            # Bootstrap
            boot_cov = pl_sub["boot_covers"].mean()
            boot_width_mean = pl_sub["boot_ci_width"].mean()
            boot_width_median = pl_sub["boot_ci_width"].median()
            lines.append(
                f"| {pi_L} | Bootstrap | {boot_cov:.3f} | "
                f"{boot_width_mean:.4f} | {boot_width_median:.4f} |\n"
            )

        # Summary stats
        lines.append("\n### Summary Statistics\n\n")
        for pi_L in pi_Ls:
            pl_sub = sub[sub["pi_L"] == pi_L]
            if pl_sub.empty:
                continue
            lines.append(f"**pi_L = {pi_L}:**\n")
            lines.append(f"- Mean weight w: {pl_sub['w'].mean():.3f} "
                         f"(SD: {pl_sub['w'].std():.3f})\n")
            lines.append(f"- Mean T_n: {pl_sub['T_n'].mean():.3f} "
                         f"(SD: {pl_sub['T_n'].std():.3f})\n")
            lines.append(f"- Mean lambda_hat: {pl_sub['lambda_hat'].mean():.3f}\n")
            lines.append(f"- Standardized z: mean={pl_sub['z_standardized'].mean():.3f}, "
                         f"SD={pl_sub['z_standardized'].std():.3f}\n")
            lines.append(f"- Q-Q quantiles (5th, 25th, 50th, 75th, 95th): "
                         f"{np.percentile(pl_sub['z_standardized'], [5, 25, 50, 75, 95]).round(3).tolist()}\n")
            lines.append("\n")
    return lines


def write_outputs(df: pd.DataFrame, outdir: str, R: int, B: int,
                  protocol: str, write_raw: bool = True) -> None:
    """Raw CSV (per replication), summary result rows, and the md table."""
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "hybrid_ci_validation.csv")
    if write_raw:
        df.to_csv(csv_path, index=False)
        print(f"Raw results saved to {csv_path}")
    write_rows(
        os.path.join(outdir, "hybrid_ci_validation_rows.json"),
        summary_rows(df, protocol, B),
        generated_by="scripts/hybrid_ci_validation.py", protocol=protocol,
    )
    md_path = os.path.join(outdir, "hybrid_ci_validation.md")
    with open(md_path, "w") as f:
        f.writelines(build_markdown(df, R, B))
    print(f"Summary saved to {md_path}")


def main():
    import argparse

    global R, PROTOCOL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--R", type=int, default=R)
    ap.add_argument("--B", type=int, default=B_BOOT)
    ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    ap.add_argument("--outdir",
                    default=os.path.join(PROJECT_ROOT, "results", "tables"))
    ap.add_argument("--from-raw", metavar="CSV", default=None,
                    help="rebuild rows + md from an existing per-replication "
                         "CSV; no simulation (pass the --B the run used)")
    args = ap.parse_args()
    PROTOCOL = args.protocol

    if args.from_raw:
        df = pd.read_csv(args.from_raw, float_precision="round_trip")
        R_raw = int(df.groupby(["dgp_id", "overrides", "pi_L"])["rep"]
                    .nunique().max())
        write_outputs(df, args.outdir, R_raw, args.B, PROTOCOL,
                      write_raw=False)
        return

    R = args.R

    t0 = time.time()

    tasks = []
    for dgp_id, overrides, pi_Ls in DGP_CONFIGS:
        for pi_L in pi_Ls:
            for r in range(R):
                tasks.append((dgp_id, overrides, pi_L, r, MASTER_SEED,
                              args.B))

    print(f"Hybrid CI Validation v2: {len(tasks)} tasks, R={R}, B={args.B}, "
          f"{N_CORES} cores")

    with multiprocessing.Pool(N_CORES, initializer=_set_protocol,
                              initargs=(PROTOCOL,)) as pool:
        all_results = pool.map(run_one_rep, tasks)

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    df = pd.DataFrame(all_results)
    write_outputs(df, args.outdir, R, args.B, PROTOCOL)


if __name__ == "__main__":
    main()
