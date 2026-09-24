#!/usr/bin/env python3
"""Best-Case Composite Proxy Regime.

Shows that the Composite Proxy (CP) method achieves near-SI efficiency
with valid coverage when historical calibration experiments are drawn
from the same DGP 7 family (multi-surrogate, J=3).

Design:
  - K_hist=50 historical experiments from DGP 7
  - Test experiments also from DGP 7
  - Methods: 0 (LO), 2 (SI), 3 (PPI++), 5 (CP)
  - pi_L in {0.05, 0.20, 0.50}
  - R=500 replications
  - 8-core multiprocessing
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.dgps.dgps import generate_dgp7
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.registry import write_table_rows
from src.simulations.simulation import (
    train_prediction_model,
    derive_seed,
    estimate_composite_weight_from_historical_multi,
    run_single_replication,
    METHOD_DISPLAY_NAMES,
)

# ── Configuration ─────────────────────────────────────────────────────────
#: Prediction protocol; overridden by --protocol in main().
PROTOCOL = DEFAULT_PROTOCOL
PI_L_VALUES = [0.05, 0.20, 0.50]
METHOD_IDS = [0, 2, 3, 5]
R = 500
N = 10_000
K_HIST = 50
MASTER_SEED = 42
N_WORKERS = 8


def _generate_historical(seed: int = 999) -> List[Dict[str, Any]]:
    """Generate K_hist=50 historical experiments from DGP 7 family."""
    return estimate_composite_weight_from_historical_multi(
        K_hist=K_HIST,
        J=3,
        gammas=[0.3, 0.2, 0.1],
        betas=[0.3, 0.4, 0.2],
        sigma_tau=0.10,
        seed=seed,
    )


# Pre-generate historical experiments once (module level for pickling)
_HIST_EXPERIMENTS = _generate_historical(seed=999)


def run_one_rep(args: Tuple[float, int]) -> List[Dict[str, Any]]:
    """Run one replication for a given pi_L."""
    pi_L, rep_idx = args

    pi_L_id = int(round(pi_L * 100))
    seed = derive_seed(MASTER_SEED, dgp_id=7, config_id=99, pi_L_id=pi_L_id, rep=rep_idx)

    dgp_kwargs = dict(n=N, pi_L=pi_L, seed=seed)

    results = run_single_replication(
        dgp_id=7,
        dgp_kwargs=dgp_kwargs,
        method_ids=METHOD_IDS,
        seed=seed,
        historical_experiments=_HIST_EXPERIMENTS,
        prediction_model="ols",
    )

    for res in results:
        res["pi_L"] = pi_L
        res["replication"] = rep_idx

    return results


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary metrics for each (pi_L, method) cell."""
    rows = []
    for (pi_L, m_id), grp in df.groupby(["pi_L", "method_id"]):
        true_tau = grp["true_tau"].iloc[0]
        tau_hats = grp["tau_hat"].values
        ci_lowers = grp["ci_lower"].values
        ci_uppers = grp["ci_upper"].values

        n_valid = len(tau_hats[~np.isnan(tau_hats)])
        if n_valid == 0:
            continue

        valid = ~np.isnan(tau_hats)
        th = tau_hats[valid]
        cl = ci_lowers[valid]
        cu = ci_uppers[valid]

        bias = np.mean(th) - true_tau
        variance = np.var(th, ddof=1)
        rmse = np.sqrt(np.mean((th - true_tau) ** 2))
        coverage = np.mean((cl <= true_tau) & (true_tau <= cu))
        rejection_rate = np.mean(
            (np.sign(cl) == np.sign(cu))  # CI excludes zero
        )

        # Monte Carlo SEs
        mc_se_bias = np.std(th, ddof=1) / np.sqrt(n_valid)
        mc_se_coverage = np.sqrt(coverage * (1 - coverage) / n_valid)
        mc_se_rmse = np.std((th - true_tau) ** 2, ddof=1) / (2 * rmse * np.sqrt(n_valid)) if rmse > 0 else np.nan

        # Relative efficiency vs Labeled-Only
        lo_th = df[(df["pi_L"] == pi_L) & (df["method_id"] == 0)]["tau_hat"]
        lo_th = lo_th.values[~np.isnan(lo_th.values)]
        lo_rmse = np.sqrt(np.mean((lo_th - true_tau) ** 2)) if len(lo_th) else np.nan
        # Paper convention: RE = RMSE_LO / RMSE_method (an RMSE ratio).
        rel_eff = lo_rmse / rmse if rmse > 0 and not np.isnan(lo_rmse) else np.nan
        ess_mult = rel_eff ** 2 if not np.isnan(rel_eff) else np.nan

        rows.append({
            "pi_L": pi_L,
            "method_id": int(m_id),
            "method_name": METHOD_DISPLAY_NAMES.get(int(m_id), f"Method {int(m_id)}"),
            "true_tau": true_tau,
            "n_valid": n_valid,
            "bias": bias,
            "mc_se_bias": mc_se_bias,
            "rmse": rmse,
            "mc_se_rmse": mc_se_rmse,
            "variance": variance,
            "coverage": coverage,
            "mc_se_coverage": mc_se_coverage,
            "rejection_rate": rejection_rate,
            "relative_efficiency": rel_eff,
            "ess_multiplier": ess_mult,
        })

    return pd.DataFrame(rows)


def main():
    import argparse

    global R, PROTOCOL
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--R", type=int, default=R,
                     help="Monte Carlo replications (smoke runs use a few)")
    _ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    _args = _ap.parse_args()
    R = _args.R
    PROTOCOL = _args.protocol

    print("Best-Case Composite Proxy Regime")
    print(f"  DGP 7 (multi-surrogate, J=3), n={N}")
    print(f"  K_hist={K_HIST} historical experiments (same DGP family)")
    print(f"  Methods: {METHOD_IDS}")
    print(f"  pi_L values: {PI_L_VALUES}")
    print(f"  R={R}, Workers={N_WORKERS}")
    print()

    # Build task list
    tasks: List[Tuple[float, int]] = []
    for pi_L in PI_L_VALUES:
        for r in range(R):
            tasks.append((pi_L, r))

    total = len(tasks)
    print(f"Total tasks: {total}")

    t0 = time.time()

    with Pool(N_WORKERS) as pool:
        all_results_nested = pool.map(run_one_rep, tasks, chunksize=10)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    # Flatten
    all_results = []
    for batch in all_results_nested:
        all_results.extend(batch)

    df_raw = pd.DataFrame(all_results)
    df_summary = compute_metrics(df_raw)

    # Print
    print("\n" + "=" * 80)
    print("Best-Case Composite Proxy: Summary")
    print("=" * 80)
    for pi_L in PI_L_VALUES:
        sub = df_summary[df_summary["pi_L"] == pi_L]
        print(f"\npi_L = {pi_L}:")
        print(sub[["method_name", "bias", "rmse", "coverage", "relative_efficiency"]].to_string(index=False))

    # Save CSV
    csv_path = os.path.join(PROJECT_ROOT, "results", "tables", "cp_best_case.csv")
    df_summary.to_csv(csv_path, index=False)
    write_table_rows(
        csv_path.replace(".csv", "_rows.json"), df_summary,
        dgp=7,
        analysis="cp_best_case",
        protocol=PROTOCOL,
        label_col="method_name",
        config_cols=("pi_L",),
        R_col="n_valid",
        generated_by="scripts/cp_best_case.py",
    )
    print(f"\nSaved CSV: {csv_path}")

    # Save markdown
    md_path = os.path.join(PROJECT_ROOT, "results", "tables", "cp_best_case.md")
    with open(md_path, "w") as f:
        f.write("# Best-Case Composite Proxy Regime\n\n")
        f.write(f"**DGP 7** (multi-surrogate, J=3), n={N}, K_hist={K_HIST}, R={R}\n\n")
        f.write("Historical experiments drawn from the **same DGP 7 family**, "
                "so CP weights are well-calibrated.\n\n")

        for pi_L in PI_L_VALUES:
            sub = df_summary[df_summary["pi_L"] == pi_L]
            f.write(f"## pi_L = {pi_L}\n\n")
            f.write("| Method | Bias | MC SE(Bias) | RMSE | Coverage | MC SE(Cov) | Rel. Eff. | ESS multiplier |\n")
            f.write("|--------|-----:|------------:|-----:|---------:|-----------:|----------:|---------------:|\n")
            for _, row in sub.iterrows():
                f.write(f"| {row['method_name']} "
                        f"| {row['bias']:.6f} "
                        f"| {row['mc_se_bias']:.6f} "
                        f"| {row['rmse']:.6f} "
                        f"| {row['coverage']:.3f} "
                        f"| {row['mc_se_coverage']:.4f} "
                        f"| {row['relative_efficiency']:.3f} "
                        f"| {row['ess_multiplier']:.3f} |\n")
            f.write("\n")

    print(f"Saved MD:  {md_path}")


if __name__ == "__main__":
    main()
