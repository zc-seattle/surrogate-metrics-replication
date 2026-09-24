#!/usr/bin/env python3
"""
Simulation 3: DGP 10 (Nonlinear + Partial Mediation).

R=500, rho = {0, 0.2, 0.4}, pi_L = {0.05, 0.20, 0.50}.
All methods. Saves to results/tables/dgp10_results.md
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

from src.simulations.simulation import (
    run_single_replication, derive_seed,
    estimate_composite_weight_from_historical,
)
from src.utils.config import DGP_DEFAULTS
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.registry import rows_from_replications, write_rows

#: Prediction protocol; overridden by --protocol in main().
PROTOCOL = DEFAULT_PROTOCOL

METHOD_IDS = [0, 1, 2, 3, 4, 5, 6]
METHOD_NAMES = {0: "LO", 1: "NS", 2: "SI", 3: "PPI++", 4: "GREG", 5: "CP", 6: "AIPW"}

PI_L_VALUES = [0.05, 0.20, 0.50]
RHO_VALUES = [0.0, 0.2, 0.4]


#: Short labels for the markdown convenience table.  The table is built from
#: the result rows (one row per method CONFIGURATION), never by grouping the
#: replication frame on method_id, which would pool e.g. PPI++ with
#: PPI++ (clipped) since both carry id 3.
SHORT_LABEL = {"Labeled-Only": "LO", "Naive Surrogate": "NS", "SI": "SI",
               "PPI++": "PPI++", "GREG": "GREG", "Composite Proxy": "CP",
               "AIPW": "AIPW", "PPI++ (clipped)": "PPI++ (clipped)"}
LABEL_ORDER = list(SHORT_LABEL)


def _md_estimation_lines(rows, pi_L):
    """Markdown lines for one pi_L cell, read from the result rows."""
    out = []
    cell = [r for r in rows if abs(float(r["pi_L"]) - pi_L) < 1e-12]
    cell.sort(key=lambda r: (LABEL_ORDER.index(r["method_label"])
                             if r["method_label"] in LABEL_ORDER else 99,
                             r["method_label"]))
    for r in cell:
        re_ = r.get("relative_efficiency")
        re_str = "-" if re_ is None or not np.isfinite(re_) else f"{re_:.2f}"
        out.append(
            f"| {pi_L} | {SHORT_LABEL.get(r['method_label'], r['method_label'])} "
            f"| {r['bias']:.4f} | {r['rmse']:.4f} | {r['coverage']:.3f} "
            f"| {re_str} |\n")
    return out


def run_one_cell(args: Tuple) -> List[Dict[str, Any]]:
    rho, pi_L, R, master_seed, hist_exps = args

    defaults = dict(DGP_DEFAULTS[10])
    defaults["rho"] = rho

    results = []
    for r in range(R):
        rep_seed = derive_seed(master_seed, 10, int(rho * 100), int(pi_L * 100), r)

        dgp_kwargs = dict(defaults)
        dgp_kwargs["seed"] = rep_seed
        dgp_kwargs["pi_L"] = pi_L

        try:
            rep_results = run_single_replication(
                dgp_id=10,
                dgp_kwargs=dgp_kwargs,
                method_ids=METHOD_IDS,
                seed=rep_seed,
                historical_experiments=hist_exps,
            )
        except Exception as e:
            print(f"Error: DGP 10, rho={rho}, pi_L={pi_L}, rep={r}: {e}")
            continue

        for res in rep_results:
            res["rho"] = rho
            res["pi_L"] = pi_L
            res["replication"] = r
            results.append(res)

    return results


def main():
    parser = argparse.ArgumentParser(description="DGP 10 evaluation")
    parser.add_argument("--protocol", choices=PROTOCOLS,
                        default=DEFAULT_PROTOCOL)
    parser.add_argument("--R", type=int, default=500)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    R = args.R
    cores = args.cores or min(8, os.cpu_count() or 1)
    global PROTOCOL
    PROTOCOL = args.protocol
    master_seed = args.seed

    _, hist_exps = estimate_composite_weight_from_historical(K_hist=30, seed=999)

    tasks = []
    for rho in RHO_VALUES:
        for pi_L in PI_L_VALUES:
            tasks.append((rho, pi_L, R, master_seed, hist_exps))

    print(f"DGP 10 Evaluation: {len(tasks)} cells, R={R}, {cores} cores")

    t0 = time.time()
    with multiprocessing.Pool(cores) as pool:
        all_results_nested = pool.map(run_one_cell, tasks)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    all_results = [r for batch in all_results_nested for r in batch]
    df = pd.DataFrame(all_results)

    output_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    _rows = []
    for (rho, pi_L), g in df.groupby(["rho", "pi_L"]):
        _rows += rows_from_replications(
            g,
            dgp=10, config_name=f"DGP10_rho{rho:.1f}_piL{pi_L:.2f}",
            params={"rho": rho}, pi_L=pi_L, n=DGP_DEFAULTS[10]["n"],
            R=int(g["replication"].nunique()),
            protocol=PROTOCOL, seed=master_seed,
        )

    lines = ["# DGP 10: Nonlinear + Partial Mediation Results\n\n"]
    lines.append(f"R = {R}\n\n")

    for rho in RHO_VALUES:
        rho_sub = df[df["rho"] == rho]
        if rho_sub.empty:
            continue

        true_tau = rho_sub["true_tau"].iloc[0]
        lines.append(f"## rho = {rho} (true_tau = {true_tau:.4f})\n\n")
        lines.append("| pi_L | Method | Bias | RMSE | Coverage | RE |\n")
        lines.append("|------|--------|------|------|----------|----|\n")

        for pi_L in PI_L_VALUES:
            sub = rho_sub[rho_sub["pi_L"] == pi_L]
            if sub.empty:
                continue

            lines += _md_estimation_lines(
                [r for r in _rows if abs(r["params"].get("rho", -1) - rho) < 1e-12],
                pi_L)

        lines.append("\n")

    output_path = os.path.join(output_dir, "dgp10_results.md")
    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"Results saved to {output_path}")

    csv_path = os.path.join(output_dir, "dgp10_results.csv")
    df.to_csv(csv_path, index=False)
    write_rows(os.path.join(os.path.dirname(csv_path), "dgp10_rows.json"),
               _rows, generated_by="run_dgp10.py", protocol=PROTOCOL)
    print(f"Result rows saved: dgp10_rows.json ({len(_rows)} rows)")

    print(f"Raw results saved to {csv_path}")


if __name__ == "__main__":
    main()
