#!/usr/bin/env python3
"""Minimax objective computation for hybrid estimator tuning parameter c.

Computes max_ρ R_excess(c, ρ) as a function of c for two DGP 1 parameterizations:
  1. Default (R² ≈ 0.56, σ_ε_Y = 1.0)
  2. Low-R² (R² ≈ 0.25, σ_ε_Y = 2.5)

For each parameterization:
  - Sweep c ∈ {0.1, 0.2, ..., 5.0} (50 values)
  - Sweep ρ ∈ {0, 0.05, 0.10, ..., 0.80} (17 values)
  - At each (c, ρ): R_excess = MSE(hybrid) - min(MSE(SI), MSE(PPI++))
  - Record max_ρ R_excess(c, ρ) for each c
  - Find c* = argmin_c max_ρ R_excess(c, ρ)

Uses 8-core multiprocessing. R=500 replications per cell, n=10000, π_L=0.20.
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

from src.dgps.dgps import generate_dgp2
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

C_VALUES = [round(0.1 * i, 1) for i in range(1, 51)]  # 0.1, 0.2, ..., 5.0
RHO_VALUES = [round(0.05 * i, 2) for i in range(17)]   # 0.0, 0.05, ..., 0.80
R = 500
N = 10_000

#: Prediction protocol; overridden by --protocol in main().
PROTOCOL = DEFAULT_PROTOCOL
PI_L = 0.20
MASTER_SEED = 42
N_WORKERS = 8

# DGP parameterizations:
# Default DGP 1: sigma_Y=1.0 → R² ≈ 0.56
# Low-R²:        sigma_Y=2.5 → R² ≈ 0.25
DGP_VARIANTS = [
    ("default_R2_0.56", {"sigma_Y": 1.0}),
    ("low_R2_0.25",     {"sigma_Y": 2.5}),
]


# ── Single-cell worker ────────────────────────────────────────────────────

def run_one_rep(args: Tuple) -> Dict[str, Any]:
    """Run one replication for a given (variant, rho, rep_id).

    Returns MSE components for all c values.
    """
    variant_label, variant_kwargs, rho, rep_id, protocol = args

    config_id = int(hashlib.sha256(variant_label.encode()).hexdigest()[:8], 16) % 10000
    rho_id = int(rho * 100)
    seed = derive_seed(MASTER_SEED, 2, config_id, rho_id, rep_id)

    # Generate data from DGP 2 (partial mediation with given rho)
    dgp_kwargs = {
        "n": N,
        "pi_L": PI_L,
        "rho": rho,
        "seed": seed,
    }
    dgp_kwargs.update(variant_kwargs)
    data = generate_dgp2(**dgp_kwargs)

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

    tau_si = res_si["tau_hat"]
    var_si = res_si["var_hat"]
    tau_ppi = res_ppi["tau_hat"]
    var_ppi = res_ppi["var_hat"]
    lambda_hat = res_ppi.get("lambda_hat", 1.0)

    # Covariance
    # Sigma_hat: the joint sandwich covariance for the learned index.
    _sigma = hybrid_sigma(T, Y, labeled_mask, design, lambda_hat)
    var_si = float(_sigma[0, 0])
    var_ppi = float(_sigma[1, 1])
    cov_sp = float(_sigma[0, 1])

    # Squared errors for SI and PPI++
    se_si = (tau_si - true_tau) ** 2
    se_ppi = (tau_ppi - true_tau) ** 2

    # Compute hybrid for each c value
    result = {
        "variant": variant_label,
        "rho": rho,
        "rep": rep_id,
        "true_tau": true_tau,
        "se_si": se_si,
        "se_ppi": se_ppi,
    }

    for c_val in C_VALUES:
        res_hyb = hybrid_estimator(
            tau_si, tau_ppi, var_si, var_ppi, cov_sp,
            c=c_val, alpha=0.05, M=10_000, rng_seed=seed + int(c_val * 1000),
        )
        tau_hyb = res_hyb["tau_hybrid"]
        se_hyb = (tau_hyb - true_tau) ** 2
        result[f"se_hyb_c{c_val}"] = se_hyb

    return result


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

    print("Minimax Objective Computation")
    print(f"  c values: {len(C_VALUES)} ({C_VALUES[0]} to {C_VALUES[-1]})")
    print(f"  rho values: {len(RHO_VALUES)} ({RHO_VALUES[0]} to {RHO_VALUES[-1]})")
    print(f"  R={R}, n={N}, pi_L={PI_L}")
    print(f"  DGP variants: {[v[0] for v in DGP_VARIANTS]}")
    total_cells = len(DGP_VARIANTS) * len(RHO_VALUES) * R
    print(f"  Total replications: {total_cells}")
    print()

    # Build task list
    tasks = []
    for variant_label, variant_kwargs in DGP_VARIANTS:
        for rho in RHO_VALUES:
            for r in range(R):
                tasks.append((variant_label, variant_kwargs, rho, r,
                              PROTOCOL))

    t0 = time.time()

    with Pool(N_WORKERS) as pool:
        all_results = pool.map(run_one_rep, tasks, chunksize=20)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    df = pd.DataFrame(all_results)

    # ── Compute MSE and R_excess for each (variant, rho, c) ──────────────

    full_grid_rows = []

    for variant_label, _ in DGP_VARIANTS:
        for rho in RHO_VALUES:
            mask = (df["variant"] == variant_label) & (df["rho"] == rho)
            sub = df[mask]

            mse_si = sub["se_si"].mean()
            mse_ppi = sub["se_ppi"].mean()
            min_mse_oracle = min(mse_si, mse_ppi)

            for c_val in C_VALUES:
                col = f"se_hyb_c{c_val}"
                mse_hyb = sub[col].mean()
                r_excess = mse_hyb - min_mse_oracle

                full_grid_rows.append({
                    "variant": variant_label,
                    "rho": rho,
                    "c": c_val,
                    "mse_si": mse_si,
                    "mse_ppi": mse_ppi,
                    "min_mse_oracle": min_mse_oracle,
                    "mse_hybrid": mse_hyb,
                    "R_excess": r_excess,
                })

    grid_df = pd.DataFrame(full_grid_rows)

    # ── Compute minimax objective: max_ρ R_excess(c, ρ) for each (variant, c) ──

    summary_rows = []
    for variant_label, _ in DGP_VARIANTS:
        for c_val in C_VALUES:
            mask = (grid_df["variant"] == variant_label) & (grid_df["c"] == c_val)
            sub = grid_df[mask]
            max_R_excess = sub["R_excess"].max()
            worst_rho = sub.loc[sub["R_excess"].idxmax(), "rho"]
            summary_rows.append({
                "variant": variant_label,
                "c": c_val,
                "max_R_excess": max_R_excess,
                "worst_rho": worst_rho,
            })

    summary_df = pd.DataFrame(summary_rows)

    # Find c* for each variant
    print("\n" + "=" * 60)
    print("MINIMAX RESULTS")
    print("=" * 60)

    md_lines = [
        "# Minimax Objective for Hybrid Estimator Tuning Parameter c",
        "",
        f"n={N}, pi_L={PI_L}, R={R}",
        "",
    ]

    for variant_label, _ in DGP_VARIANTS:
        mask = summary_df["variant"] == variant_label
        sub = summary_df[mask].copy()
        idx_star = sub["max_R_excess"].idxmin()
        c_star = sub.loc[idx_star, "c"]
        max_R_excess_star = sub.loc[idx_star, "max_R_excess"]
        worst_rho_star = sub.loc[idx_star, "worst_rho"]

        print(f"\n{variant_label}:")
        print(f"  c* = {c_star:.1f}")
        print(f"  max_rho R_excess(c*, rho) = {max_R_excess_star:.6f}")
        print(f"  worst-case rho at c* = {worst_rho_star:.2f}")

        md_lines.append(f"## {variant_label}")
        md_lines.append("")
        md_lines.append(f"- **c* = {c_star:.1f}**")
        md_lines.append(f"- max_rho R_excess(c*, rho) = {max_R_excess_star:.6f}")
        md_lines.append(f"- Worst-case rho at c* = {worst_rho_star:.2f}")
        md_lines.append("")

        # Show top-5 c values near optimum
        sub_sorted = sub.sort_values("max_R_excess").head(5)
        md_lines.append("| c | max_rho R_excess | worst rho |")
        md_lines.append("|---|------------------|-----------|")
        for _, row in sub_sorted.iterrows():
            md_lines.append(
                f"| {row['c']:.1f} | {row['max_R_excess']:.6f} "
                f"| {row['worst_rho']:.2f} |"
            )
        md_lines.append("")

    # ── Save ──────────────────────────────────────────────────────────────

    out_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "minimax_objective.csv")
    md_path = os.path.join(out_dir, "minimax_objective.md")

    grid_df.to_csv(csv_path, index=False)
    write_table_rows(
        csv_path.replace(".csv", "_rows.json"), grid_df,
        dgp=2,
        analysis="minimax_objective",
        protocol=PROTOCOL,
        config_cols=("variant", "rho", "c"),
        R=R,
        generated_by="scripts/minimax_objective.py",
    )

    md_text = "\n".join(md_lines) + "\n"
    with open(md_path, "w") as f:
        f.write(md_text)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved MD:  {md_path}")


if __name__ == "__main__":
    main()
