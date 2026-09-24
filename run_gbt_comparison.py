#!/usr/bin/env python3
"""
Simulation 1: GBT vs OLS comparison across DGPs 1, 2, 4, 8.

Compares OLS and GBT prediction models across multiple DGPs and pi_L values.
Methods: LO, SI, PPI++, CP, AIPW (skip NS, GREG to save time).
R=500, saves to results/tables/gbt_comparison.md
"""

from __future__ import annotations

import argparse
import ast
import hashlib
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

from src.simulations.simulation import (
    run_single_replication, derive_seed,
    estimate_composite_weight_from_historical,
)
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS

#: Prediction protocol; overridden by --protocol in main().
PROTOCOL = DEFAULT_PROTOCOL
from src.utils.registry import rows_from_replications, write_rows
from src.utils.config import DGP_DEFAULTS

# Methods: LO=0, SI=2, PPI++=3, CP=5, AIPW=6
METHOD_IDS = [0, 2, 3, 5, 6]
METHOD_NAMES = {0: "LO", 2: "SI", 3: "PPI++", 5: "CP", 6: "AIPW"}

PI_L_VALUES = [0.05, 0.20, 0.50]

#: What the SI interval in each row is; printed under the md header.  The
#: result rows carry the same information in `si_variance`.
SI_VARIANCE_NOTE = ('Interval note: the OLS surrogate-index rows use the joint sandwich (si_variance = "sandwich"), except DGP 4, whose SI index is the external calibration fit (n_cal = 50,000) with the plug-in variance. Every other GBT surrogate-index row (DGP 4 uses the same external OLS fit under both models) uses si_variance = "plugin": a tree index has no linear design, so the Neyman variance of the imputed outcomes is used with the trees treated as fixed. This is a heuristic interval that ignores first-stage uncertainty; `gbt_variance_check.md` measures it against a paired bootstrap that refits the trees. PPI++ uses its exact variance under both models.\n')
PREDICTION_MODELS = ["ols", "gbt"]

# DGP configurations
DGP_CONFIGS = [
    (1, {}),
    (2, {"rho": 0.0}),
    (2, {"rho": 0.2}),
    (2, {"rho": 0.4}),
    (4, {"delta_beta": 0.0}),
    (4, {"delta_beta": 0.2}),
    (8, {}),
]


def run_one_cell(args: Tuple) -> List[Dict[str, Any]]:
    dgp_id, overrides, pi_L, pred_model, R, master_seed, hist_exps = args

    defaults = dict(DGP_DEFAULTS[dgp_id])
    defaults.update(overrides)

    results = []
    for r in range(R):
        rep_seed = derive_seed(master_seed, dgp_id, int(hashlib.sha256(str(overrides).encode()).hexdigest()[:8], 16) % 1000, int(pi_L * 100), r)

        dgp_kwargs = dict(defaults)
        dgp_kwargs["seed"] = rep_seed
        dgp_kwargs["pi_L"] = pi_L

        try:
            rep_results = run_single_replication(
                dgp_id=dgp_id,
                dgp_kwargs=dgp_kwargs,
                method_ids=METHOD_IDS,
                seed=rep_seed,
                historical_experiments=hist_exps,
                prediction_model=pred_model,
            )
        except Exception as e:
            print(f"Error: DGP {dgp_id}, {overrides}, pi_L={pi_L}, {pred_model}, rep={r}: {e}")
            continue

        for res in rep_results:
            res["dgp_id"] = dgp_id
            res["overrides"] = str(overrides)
            res["pi_L"] = pi_L
            res["prediction_model"] = pred_model
            res["replication"] = r
            results.append(res)

    return results


def build_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """One result row per (DGP, overrides, pi_L, prediction model, method).

    The grouping must include the DGP overrides: DGP 2 runs at three rho
    values and DGP 4 at two delta_beta values, and grouping on the DGP id
    alone pooled their replications into one row.
    """
    rows: List[Dict[str, Any]] = []
    for (dgp_id, ov, pi_L, pm), g in df.groupby(
            ["dgp_id", "overrides", "pi_L", "prediction_model"], sort=True):
        overrides = ast.literal_eval(ov) if isinstance(ov, str) else dict(ov)
        tag = "".join(f"_{k}{v}" for k, v in overrides.items())
        rows += rows_from_replications(
            g, dgp=int(dgp_id),
            config_name=f"DGP{int(dgp_id)}{tag}_piL{pi_L:.2f}_{pm}",
            params={"prediction_model": pm, **overrides}, pi_L=float(pi_L),
            R=int(g["replication"].nunique()), protocol=PROTOCOL, seed=42,
        )
    return rows


def rebuild_rows_from_csv() -> None:
    """Rewrite gbt_comparison_rows.json from the stored per-replication CSV."""
    csv_path = os.path.join(PROJECT_ROOT, "results", "tables",
                            "gbt_comparison.csv")
    df = pd.read_csv(csv_path, keep_default_na=True)
    rows = build_rows(df)
    write_rows(csv_path.replace(".csv", "_rows.json"), rows,
               generated_by="run_gbt_comparison.py (rows rebuilt from "
                            "gbt_comparison.csv by --rows-from-csv)",
               protocol=str(df["protocol"].iloc[0]))
    print(f"Result rows rebuilt from {csv_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="GBT vs OLS comparison")
    parser.add_argument("--protocol", choices=PROTOCOLS,
                        default=DEFAULT_PROTOCOL)
    parser.add_argument("--R", type=int, default=500)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows-from-csv", action="store_true",
                        help="rebuild gbt_comparison_rows.json from the "
                             "stored per-replication CSV and exit")
    args = parser.parse_args()
    global PROTOCOL
    PROTOCOL = args.protocol
    if args.rows_from_csv:
        rebuild_rows_from_csv()
        return

    R = args.R
    cores = args.cores or min(8, os.cpu_count() or 1)
    master_seed = args.seed

    # Generate historical experiments for composite proxy
    _, hist_exps = estimate_composite_weight_from_historical(K_hist=30, seed=999)

    tasks = []
    for dgp_id, overrides in DGP_CONFIGS:
        for pi_L in PI_L_VALUES:
            for pred_model in PREDICTION_MODELS:
                tasks.append((dgp_id, overrides, pi_L, pred_model, R, master_seed, hist_exps))

    print(f"GBT Comparison: {len(tasks)} cells, R={R}, {cores} cores")

    t0 = time.time()
    with multiprocessing.Pool(cores) as pool:
        all_results_nested = pool.map(run_one_cell, tasks)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    all_results = [r for batch in all_results_nested for r in batch]
    df = pd.DataFrame(all_results)

    output_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    lines = ["# GBT vs OLS Comparison\n\n"]
    lines.append(f"R = {R} replications per cell\n\n")
    lines.append("RE = RMSE(labeled-only) / RMSE(method) within the same cell "
                 "and prediction model; >1 means more efficient.\n")
    lines.append("ESS multiplier = RE^2 (effective-sample-size scale).\n\n")
    lines.append(SI_VARIANCE_NOTE + "\n")

    for dgp_id, overrides in DGP_CONFIGS:
        dgp_sub = df[(df["dgp_id"] == dgp_id) & (df["overrides"] == str(overrides))]
        if dgp_sub.empty:
            continue

        label = f"DGP {dgp_id}"
        if overrides:
            label += f" ({', '.join(f'{k}={v}' for k, v in overrides.items())})"
        lines.append(f"## {label}\n\n")
        lines.append("| pi_L | Model | Method | Bias | RMSE | Coverage | RE | ESS multiplier |\n")
        lines.append("|------|-------|--------|------|------|----------|----|----------------|\n")

        for pi_L in PI_L_VALUES:
            for pred_model in PREDICTION_MODELS:
                sub = dgp_sub[(dgp_sub["pi_L"] == pi_L) & (dgp_sub["prediction_model"] == pred_model)]
                if sub.empty:
                    continue

                lo_sub = sub[sub["method_id"] == 0]
                lo_mse = ((lo_sub["tau_hat"] - lo_sub["true_tau"]) ** 2).mean() if not lo_sub.empty else np.nan
                lo_rmse = np.sqrt(lo_mse) if not np.isnan(lo_mse) else np.nan

                for m_id in METHOD_IDS:
                    m_sub = sub[sub["method_id"] == m_id]
                    if m_sub.empty:
                        continue
                    bias = (m_sub["tau_hat"] - m_sub["true_tau"]).mean()
                    mse = ((m_sub["tau_hat"] - m_sub["true_tau"]) ** 2).mean()
                    rmse = np.sqrt(mse)
                    covers = ((m_sub["ci_lower"] <= m_sub["true_tau"]) & (m_sub["true_tau"] <= m_sub["ci_upper"])).mean()
                    # Paper convention: RE = RMSE_LO / RMSE_method.
                    re = lo_rmse / rmse if rmse > 0 and not np.isnan(lo_rmse) else np.nan
                    ess = re ** 2 if not np.isnan(re) else np.nan
                    re_str = f"{re:.2f}" if not np.isnan(re) else "-"
                    ess_str = f"{ess:.2f}" if not np.isnan(ess) else "-"

                    lines.append(f"| {pi_L} | {pred_model.upper()} | {METHOD_NAMES[m_id]} | {bias:.4f} | {rmse:.4f} | {covers:.3f} | {re_str} | {ess_str} |\n")

        lines.append("\n")

    output_path = os.path.join(output_dir, "gbt_comparison.md")
    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"Results saved to {output_path}")

    csv_path = os.path.join(output_dir, "gbt_comparison.csv")
    df.to_csv(csv_path, index=False)
    _rows = build_rows(df)
    write_rows(csv_path.replace(".csv", "_rows.json"), _rows,
               generated_by="run_gbt_comparison.py", protocol=PROTOCOL)
    print(f"Result rows saved ({len(_rows)} rows)")
    print(f"Raw results saved to {csv_path}")


if __name__ == "__main__":
    main()
