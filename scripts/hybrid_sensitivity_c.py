#!/usr/bin/env python3
"""Hybrid estimator sensitivity analysis for the tuning parameter c.

Sweeps c ∈ {0.5, 1.0, 1.5, 2.0, 3.0} across DGPs 1, 2 (ρ=0, 0.2, 0.4), and 9.
For each configuration, runs R=1000 replications with π_L ∈ {0.05, 0.20, 0.50}.
Reports: Bias, RMSE, Coverage, Mean w, RE vs LO.

Uses 8-core multiprocessing.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.dgps.dgps import generate_dgp1, generate_dgp2, generate_dgp9
from src.simulations.simulation import derive_seed
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.registry import write_table_rows
from src.utils.estimator_api import (
    estimate,
    hybrid_estimator,
    hybrid_sigma,
    train_prediction_model,
)

# ── Configuration ──────────────────────────────────────────────────────────

C_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]
PI_L_VALUES = [0.05, 0.20, 0.50]
R = 1000
N = 10_000

#: Prediction protocol; overridden by --protocol in main().
PROTOCOL = DEFAULT_PROTOCOL
MASTER_SEED = 42
N_WORKERS = 8

# DGP configurations: (dgp_id, label, extra_kwargs)
DGP_CONFIGS = [
    (1, "DGP1", {}),
    (2, "DGP2_rho0.0", {"rho": 0.0}),
    (2, "DGP2_rho0.2", {"rho": 0.2}),
    (2, "DGP2_rho0.4", {"rho": 0.4}),
    (9, "DGP9", {}),
]

DGP_FUNCS = {1: generate_dgp1, 2: generate_dgp2, 9: generate_dgp9}


# ── Single-replication worker ─────────────────────────────────────────────

def run_one_rep(args: Tuple) -> List[Dict[str, Any]]:
    """Run one replication for a given (dgp_config, pi_L, rep_id).

    Returns a list of result dicts, one per c value.
    """
    dgp_id, dgp_label, dgp_extra, pi_L, rep_id, protocol = args

    config_id = int(hashlib.sha256(dgp_label.encode()).hexdigest()[:8], 16) % 10000
    pi_L_id = int(pi_L * 100)
    seed = derive_seed(MASTER_SEED, dgp_id, config_id, pi_L_id, rep_id)

    # Generate data
    dgp_func = DGP_FUNCS[dgp_id]
    dgp_kwargs = {"n": N, "pi_L": pi_L, "seed": seed}
    dgp_kwargs.update(dgp_extra)
    data = dgp_func(**dgp_kwargs)

    T = data["T"]
    S = data["S"]
    Y = data["Y"]
    X = data["X"]
    labeled_mask = data["labeled_mask"]
    true_tau = data["true_tau"]

    # Train prediction model
    cf_rng = np.random.default_rng(seed + 7777)
    Y_hat, design = train_prediction_model(
        S, X, Y, labeled_mask, n_folds=5, rng=cf_rng,
        protocol=protocol, return_design=True,
    )

    # Run SI and PPI++ in their primary configurations
    res_ppi = estimate(3, T, S, Y, Y_hat, labeled_mask, protocol=protocol)
    res_si = estimate(2, T, S, Y, Y_hat, labeled_mask, protocol=protocol,
                      design=design,
                      lambda_hat=res_ppi.get("lambda_hat", 0.0))
    res_lo = estimate(0, T, S, Y, Y_hat, labeled_mask, protocol=protocol)

    tau_si = res_si["tau_hat"]
    var_si = res_si["var_hat"]
    tau_ppi = res_ppi["tau_hat"]
    var_ppi = res_ppi["var_hat"]
    lambda_hat = res_ppi.get("lambda_hat", 1.0)
    var_lo = res_lo["var_hat"]

    # Covariance
    # Sigma_hat: the joint sandwich covariance for the learned index.
    _sigma = hybrid_sigma(T, Y, labeled_mask, design, lambda_hat)
    var_si = float(_sigma[0, 0])
    var_ppi = float(_sigma[1, 1])
    cov_sp = float(_sigma[0, 1])

    results = []
    for c_val in C_VALUES:
        res_hyb = hybrid_estimator(
            tau_si, tau_ppi, var_si, var_ppi, cov_sp,
            c=c_val, alpha=0.05, M=10_000, rng_seed=seed + int(c_val * 1000),
        )

        tau_hyb = res_hyb["tau_hybrid"]
        ci_lo = res_hyb["ci_lower"]
        ci_hi = res_hyb["ci_upper"]
        w = res_hyb["w"]

        covers = 1.0 if ci_lo <= true_tau <= ci_hi else 0.0

        results.append({
            "dgp_label": dgp_label,
            "pi_L": pi_L,
            "c": c_val,
            "rep": rep_id,
            "true_tau": true_tau,
            "tau_hyb": tau_hyb,
            "covers": covers,
            "w": w,
            "var_lo": var_lo,
        })

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse

    global R, PROTOCOL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--R", type=int, default=R)
    ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    args = ap.parse_args()
    R = args.R
    PROTOCOL = args.protocol

    print("Hybrid Sensitivity for c")
    print(f"  C values: {C_VALUES}")
    print(f"  DGP configs: {[c[1] for c in DGP_CONFIGS]}")
    print(f"  pi_L values: {PI_L_VALUES}")
    print(f"  R={R}, n={N}")
    print()

    # Build task list
    tasks = []
    for dgp_id, dgp_label, dgp_extra in DGP_CONFIGS:
        for pi_L in PI_L_VALUES:
            for r in range(R):
                tasks.append((dgp_id, dgp_label, dgp_extra, pi_L, r,
                              PROTOCOL))

    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks}")

    t0 = time.time()

    with Pool(N_WORKERS) as pool:
        nested_results = pool.map(run_one_rep, tasks, chunksize=10)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Flatten results
    all_rows = []
    for batch in nested_results:
        all_rows.extend(batch)

    df = pd.DataFrame(all_rows)

    # Aggregate
    summary_rows = []
    for (dgp_label, pi_L, c_val), grp in df.groupby(["dgp_label", "pi_L", "c"]):
        true_tau = grp["true_tau"].iloc[0]
        bias = grp["tau_hyb"].mean() - true_tau
        rmse = np.sqrt(((grp["tau_hyb"] - true_tau) ** 2).mean())
        coverage = grp["covers"].mean()
        mean_w = grp["w"].mean()

        # RE vs LO: Var(LO) / MSE(hybrid)
        mean_var_lo = grp["var_lo"].mean()
        mse_hyb = ((grp["tau_hyb"] - true_tau) ** 2).mean()
        re_vs_lo = np.sqrt(mean_var_lo / mse_hyb) if mse_hyb > 0 else np.nan

        summary_rows.append({
            "DGP": dgp_label,
            "pi_L": pi_L,
            "c": c_val,
            "true_tau": round(true_tau, 4),
            "Bias": round(bias, 5),
            "RMSE": round(rmse, 5),
            "Coverage": round(coverage, 3),
            "Mean_w": round(mean_w, 3),
            "RE_vs_LO": round(re_vs_lo, 3),
        })

    summary = pd.DataFrame(summary_rows)

    # Save
    out_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "hybrid_sensitivity_c.csv")
    md_path = os.path.join(out_dir, "hybrid_sensitivity_c.md")

    summary.to_csv(csv_path, index=False)
    write_table_rows(
        csv_path.replace(".csv", "_rows.json"), summary,
        dgp="hybrid",
        analysis="hybrid_sensitivity_c",
        protocol=PROTOCOL,
        config_cols=("DGP", "c"),
        R=R,
        generated_by="scripts/hybrid_sensitivity_c.py",
    )

    # Build markdown table
    lines = [
        "# Hybrid Estimator Sensitivity to c",
        "",
        f"n={N}, R={R}",
        "",
        "| DGP | pi_L | c | true_tau | Bias | RMSE | Coverage | Mean_w | RE_vs_LO |",
        "|-----|------|---|----------|------|------|----------|--------|----------|",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['DGP']} | {r['pi_L']:.2f} | {r['c']:.1f} "
            f"| {r['true_tau']:.4f} | {r['Bias']:.5f} | {r['RMSE']:.5f} "
            f"| {r['Coverage']:.3f} | {r['Mean_w']:.3f} | {r['RE_vs_LO']:.3f} |"
        )

    md_text = "\n".join(lines) + "\n"
    with open(md_path, "w") as f:
        f.write(md_text)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved MD:  {md_path}")
    print()
    print(md_text)


if __name__ == "__main__":
    main()
