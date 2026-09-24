#!/usr/bin/env python3
"""
Run AIPW through DGPs 1-6 at the standard pi_L sweep.

R=500 per cell, with multiprocessing.
Saves results to results/tables/aipw_full_eval.md.

Usage:
    python run_aipw_full.py [--R 500] [--cores 8]
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.simulations.simulation import run_single_replication, derive_seed
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS

#: Prediction protocol; overridden by --protocol in main().
PROTOCOL = DEFAULT_PROTOCOL
from src.utils.registry import rows_from_replications, write_rows
from src.utils.config import DGP_DEFAULTS, PI_L_SWEEP_STANDARD


# Methods to evaluate: LO (baseline) and AIPW
METHOD_IDS = [0, 6]  # 0 = Labeled-Only, 6 = AIPW
METHOD_NAMES = {0: "LO", 6: "AIPW"}

# DGPs to evaluate
DGP_IDS = [1, 2, 3, 4, 5, 6]

# pi_L sweep
PI_L_VALUES = [0.05, 0.10, 0.20, 0.50, 0.80, 1.00]

# DGP 5 has its own pi_L values (determined by q)
DGP5_PI_L_VALUES = [0.05, 0.10, 0.15, 0.20, 0.30]

# DGP 6 has its own pi_L values
DGP6_PI_L_VALUES = [0.10, 0.20, 0.50, 1.00]


def _get_pi_L_values(dgp_id: int) -> List[float]:
    if dgp_id == 5:
        return DGP5_PI_L_VALUES
    if dgp_id == 6:
        return DGP6_PI_L_VALUES
    return PI_L_VALUES


def run_one_cell(args: Tuple) -> List[Dict[str, Any]]:
    """Run all replications for one (dgp, pi_L) cell."""
    dgp_id, pi_L, R, master_seed = args

    defaults = dict(DGP_DEFAULTS[dgp_id])

    results = []
    for r in range(R):
        rep_seed = derive_seed(master_seed, dgp_id, 0, int(pi_L * 100), r)

        dgp_kwargs = dict(defaults)
        dgp_kwargs["seed"] = rep_seed
        if dgp_id == 5:
            dgp_kwargs["q"] = pi_L
        else:
            dgp_kwargs["pi_L"] = pi_L

        try:
            rep_results = run_single_replication(
                dgp_id=dgp_id,
                dgp_kwargs=dgp_kwargs,
                method_ids=METHOD_IDS,
                seed=rep_seed,
            )
        except Exception as e:
            print(f"Error in DGP {dgp_id}, pi_L={pi_L}, rep={r}: {e}")
            continue

        for res in rep_results:
            res["dgp_id"] = dgp_id
            res["pi_L"] = pi_L
            res["replication"] = r
            results.append(res)

    return results


def main():
    parser = argparse.ArgumentParser(description="AIPW full evaluation")
    parser.add_argument("--protocol", choices=PROTOCOLS,
                        default=DEFAULT_PROTOCOL)
    parser.add_argument("--R", type=int, default=500, help="Replications per cell")
    parser.add_argument("--cores", type=int, default=None, help="Number of cores")
    parser.add_argument("--seed", type=int, default=42, help="Master seed")
    args = parser.parse_args()
    global PROTOCOL
    PROTOCOL = args.protocol

    R = args.R
    cores = args.cores or min(8, os.cpu_count() or 1)
    master_seed = args.seed

    # Build task list
    tasks = []
    for dgp_id in DGP_IDS:
        for pi_L in _get_pi_L_values(dgp_id):
            tasks.append((dgp_id, pi_L, R, master_seed))

    print(f"AIPW Full Evaluation: {len(tasks)} cells, R={R}, {cores} cores")

    t0 = time.time()
    with multiprocessing.Pool(cores) as pool:
        all_results_nested = pool.map(run_one_cell, tasks)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    all_results = [r for batch in all_results_nested for r in batch]
    df = pd.DataFrame(all_results)

    # Compute summary and write markdown table
    output_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    lines = ["# AIPW Full Evaluation (DGPs 1-6)\n\n"]
    lines.append(f"R = {R} replications per cell\n\n")
    lines.append("RE = RMSE(labeled-only) / RMSE(method); >1 means more efficient.\n")
    lines.append("ESS multiplier = RE^2 (effective-sample-size scale).\n\n")

    for dgp_id in DGP_IDS:
        dgp_sub = df[df["dgp_id"] == dgp_id]
        if dgp_sub.empty:
            continue

        lines.append(f"## DGP {dgp_id}\n\n")
        lines.append("| pi_L | Method | Bias | RMSE | Coverage | RE | ESS multiplier |\n")
        lines.append("|------|--------|------|------|----------|----|----------------|\n")

        for pi_L in _get_pi_L_values(dgp_id):
            pl_sub = dgp_sub[dgp_sub["pi_L"] == pi_L]
            if pl_sub.empty:
                continue

            # LO MSE for RE calculation
            lo_sub = pl_sub[pl_sub["method_id"] == 0]
            if lo_sub.empty or dgp_id == 6:
                lo_mse = np.nan
            else:
                lo_mse = ((lo_sub["tau_hat"] - lo_sub["true_tau"]) ** 2).mean()
            lo_rmse = np.sqrt(lo_mse) if not np.isnan(lo_mse) else np.nan

            for m_id in METHOD_IDS:
                m_sub = pl_sub[pl_sub["method_id"] == m_id]
                if m_sub.empty:
                    continue

                if dgp_id == 6:
                    # Portfolio: report cumulative regret
                    mean_regret = m_sub["cumulative_regret"].mean()
                    lines.append(
                        f"| {pi_L} | {METHOD_NAMES[m_id]} | "
                        f"regret={mean_regret:.4f} | - | - | - | - |\n"
                    )
                else:
                    bias = (m_sub["tau_hat"] - m_sub["true_tau"]).mean()
                    mse = ((m_sub["tau_hat"] - m_sub["true_tau"]) ** 2).mean()
                    rmse = np.sqrt(mse)
                    covers = (
                        (m_sub["ci_lower"] <= m_sub["true_tau"])
                        & (m_sub["true_tau"] <= m_sub["ci_upper"])
                    ).mean()
                    # Paper convention: RE = RMSE_LO / RMSE_method.
                    re = lo_rmse / rmse if rmse > 0 and not np.isnan(lo_rmse) else np.nan
                    ess = re ** 2 if not np.isnan(re) else np.nan
                    re_str = f"{re:.2f}" if not np.isnan(re) else "-"
                    ess_str = f"{ess:.2f}" if not np.isnan(ess) else "-"

                    lines.append(
                        f"| {pi_L} | {METHOD_NAMES[m_id]} | {bias:.4f} | "
                        f"{rmse:.4f} | {covers:.3f} | {re_str} | {ess_str} |\n"
                    )

        lines.append("\n")

    output_path = os.path.join(output_dir, "aipw_full_eval.md")
    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"Results saved to {output_path}")

    # Save raw CSV
    csv_path = os.path.join(output_dir, "aipw_full_eval.csv")
    # Drop non-serializable columns for DGP 6
    save_cols = [c for c in df.columns if c not in ("decisions", "true_taus")]
    df[save_cols].to_csv(csv_path, index=False)
    _rows = []
    for (dgp_id, pi_L), g in df.groupby(["dgp_id", "pi_L"]):
        _rows += rows_from_replications(
            g, dgp=int(dgp_id),
            config_name=f"DGP{int(dgp_id)}_piL{pi_L:.2f}", params={},
            pi_L=float(pi_L), R=int(g["replication"].nunique()),
            protocol=PROTOCOL, seed=42,
        )
    write_rows(csv_path.replace(".csv", "_rows.json"), _rows,
               generated_by="run_aipw_full.py", protocol=PROTOCOL)
    print(f"Result rows saved ({len(_rows)} rows)")
    print(f"Raw results saved to {csv_path}")


if __name__ == "__main__":
    main()
