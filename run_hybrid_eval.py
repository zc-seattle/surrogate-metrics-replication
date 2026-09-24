#!/usr/bin/env python3
"""
Simulation 6: Hybrid Estimator Evaluation.

DGPs 1, 2 (rho=0, 0.2, 0.4, 0.8), 9
pi_L = {0.05, 0.20, 0.50}, R=500

Reports: tau_hybrid, w (average weight), bias, RMSE, coverage, RE.
Compares to SI and PPI++ standalone.

Outputs (in --outdir, default results/tables):
  hybrid_eval.csv        one row per replication (the raw data)
  hybrid_eval_rows.json  summary result rows, one per (cell, estimator)
  hybrid_eval.md         the paper table
`--from-raw` rebuilds the summary rows and the table from an existing
hybrid_eval.csv without re-simulating.

"""

from __future__ import annotations

import argparse
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

from src.dgps.dgps import DGP_GENERATORS
from src.simulations.simulation import derive_seed
from src.utils.config import DEFAULT_PROTOCOL, DGP_DEFAULTS, PROTOCOLS
from src.utils.registry import (
    replications_long,
    rows_from_replications,
    write_rows,
)
from src.utils.estimator_api import (
    estimate,
    hybrid_estimator,
    hybrid_sigma,
    surrogacy_test,
    train_prediction_model,
)

PI_L_VALUES = [0.05, 0.20, 0.50]

DGP_CONFIGS = [
    (1, {}),
    (2, {"rho": 0.0}),
    (2, {"rho": 0.2}),
    (2, {"rho": 0.4}),
    (2, {"rho": 0.8}),
    (9, {}),
]


def run_one_rep(args: Tuple) -> Dict[str, Any]:
    dgp_id, overrides, pi_L, rep, master_seed, protocol = args
    config_hash = int(hashlib.sha256(str(overrides).encode()).hexdigest()[:8], 16) % 1000
    rep_seed = derive_seed(master_seed, dgp_id, config_hash, int(pi_L * 100), rep)

    defaults = dict(DGP_DEFAULTS[dgp_id])
    defaults.update(overrides)
    defaults["seed"] = rep_seed
    defaults["pi_L"] = pi_L

    dgp_func = DGP_GENERATORS[dgp_id]
    data = dgp_func(**defaults)
    true_tau = data["true_tau"]

    cf_rng = np.random.default_rng(rep_seed + 7777)
    Y_hat, design = train_prediction_model(
        data["S"], data["X"], data["Y"],
        data["labeled_mask"], n_folds=5, rng=cf_rng,
        protocol=protocol, return_design=True,
    )

    T, S, Y, lm = data["T"], data["S"], data["Y"], data["labeled_mask"]

    # LO baseline, SI, and PPI++, all in their primary configurations.
    lo_res = estimate(0, T, S, Y, Y_hat, lm, protocol=protocol)
    ppi_res = estimate(3, T, S, Y, Y_hat, lm, protocol=protocol)
    lambda_hat = ppi_res.get("lambda_hat", 0.0)
    si_res = estimate(2, T, S, Y, Y_hat, lm, protocol=protocol,
                      design=design, lambda_hat=lambda_hat)

    # Sigma_hat: the 2 x 2 joint sandwich covariance of (tau_SI, tau_PPI) for
    # the learned index.  The hybrid's weight and interval are built from it.
    sigma = hybrid_sigma(T, Y, lm, design, lambda_hat)
    var_si_j, cov_sp, var_ppi_j = (
        float(sigma[0, 0]), float(sigma[0, 1]), float(sigma[1, 1])
    )

    # Hybrid
    hyb_res = hybrid_estimator(
        si_res["tau_hat"], ppi_res["tau_hat"],
        var_si_j, var_ppi_j,
        cov_sp, rng_seed=rep_seed + 9999,
    )

    result = {
        "dgp_id": dgp_id,
        "overrides": str(overrides),
        "pi_L": pi_L,
        "rep": rep,
        "true_tau": true_tau,
        # SI
        "tau_lo": lo_res["tau_hat"],
        "tau_si": si_res["tau_hat"],
        "var_si_sandwich": var_si_j,
        "var_ppi_sandwich": var_ppi_j,
        "cov_si_ppi_sandwich": cov_sp,
        "protocol": protocol,
        "ci_lower_si": si_res["ci_lower"],
        "ci_upper_si": si_res["ci_upper"],
        # PPI++
        "tau_ppi": ppi_res["tau_hat"],
        "ci_lower_ppi": ppi_res["ci_lower"],
        "ci_upper_ppi": ppi_res["ci_upper"],
        # Hybrid
        "tau_hybrid": hyb_res["tau_hybrid"],
        "w": hyb_res["w"],
        "ci_lower_hybrid": hyb_res["ci_lower"],
        "ci_upper_hybrid": hyb_res["ci_upper"],
        "T_n": hyb_res["T_n"],
    }
    return result


#: (method-spec key or (key, label), tau column, CI lower, CI upper, md label)
ESTIMATORS = [
    ("lo", "tau_lo", None, None, "LO"),
    ("si", "tau_si", "ci_lower_si", "ci_upper_si", "SI"),
    ("ppi", "tau_ppi", "ci_lower_ppi", "ci_upper_ppi", "PPI++"),
    (("hybrid", "Hybrid"), "tau_hybrid", "ci_lower_hybrid",
     "ci_upper_hybrid", "Hybrid"),
]


def _cell_name(dgp_id: int, overrides: Dict[str, Any], pi_L: float) -> str:
    name = f"DGP{dgp_id}"
    for k, v in overrides.items():
        name += f"_{k}{v}"
    return f"{name}_piL{pi_L:.2f}"


def summary_rows(df: pd.DataFrame, protocol: str,
                 seed: int = 42) -> List[Dict[str, Any]]:
    """One result row per (DGP cell, pi_L, estimator), from the raw frame.

    Bias, RMSE, coverage and RE = RMSE(LO) / RMSE(method) are the md table's
    columns; MC SEs come from `rows_from_replications`.  The hybrid row also
    carries the mean weight (`mean_w`) and its constant c.
    """
    rows: List[Dict[str, Any]] = []
    for dgp_id, overrides in DGP_CONFIGS:
        for pi_L in PI_L_VALUES:
            g = df[(df["dgp_id"] == dgp_id)
                   & (df["overrides"] == str(overrides))
                   & (df["pi_L"] == pi_L)]
            if g.empty:
                continue
            long = replications_long(
                g, [e[:4] for e in ESTIMATORS], protocol=protocol)
            cell = rows_from_replications(
                long, dgp=dgp_id,
                config_name=_cell_name(dgp_id, overrides, pi_L),
                params={"dgp_id": dgp_id, **overrides},
                pi_L=pi_L, n=DGP_DEFAULTS[dgp_id].get("n"),
                protocol=protocol, seed=seed,
            )
            md_label = {(e[0] if isinstance(e[0], str) else e[0][0]): e[4]
                        for e in ESTIMATORS}
            order = [e[0] if isinstance(e[0], str) else e[0][0]
                     for e in ESTIMATORS]
            cell.sort(key=lambda r: order.index(r["method_key"]))
            for r in cell:
                r["analysis"] = "hybrid_eval"
                r["table_label"] = md_label[r["method_key"]]
                for k, v in overrides.items():
                    r[k] = v
                if r["method_key"] == "hybrid":
                    r["mean_w"] = float(g["w"].mean())
                    r["c"] = 1.5
            rows.extend(cell)
    return rows


def build_markdown(df: pd.DataFrame, R: int) -> List[str]:
    lines = ["# Hybrid Estimator Evaluation\n\n"]
    lines.append(f"R = {R}\n\n")

    for dgp_id, overrides in DGP_CONFIGS:
        label = f"DGP {dgp_id}"
        if overrides:
            label += f" ({', '.join(f'{k}={v}' for k, v in overrides.items())})"

        sub = df[(df["dgp_id"] == dgp_id) & (df["overrides"] == str(overrides))]
        if sub.empty:
            continue

        true_tau = sub["true_tau"].iloc[0]
        lines.append(f"## {label} (true_tau = {true_tau:.4f})\n\n")
        lines.append("| pi_L | Estimator | Bias | RMSE | Coverage | Mean w | RE vs LO |\n")
        lines.append("|------|-----------|------|------|----------|--------|----------|\n")

        for pi_L in PI_L_VALUES:
            pl_sub = sub[sub["pi_L"] == pi_L]
            if pl_sub.empty:
                continue

            tt = pl_sub["true_tau"]

            lo_mse = ((pl_sub["tau_lo"] - tt) ** 2).mean()
            for est_name, tau_col, ci_lo_col, ci_hi_col in [
                ("SI", "tau_si", "ci_lower_si", "ci_upper_si"),
                ("PPI++", "tau_ppi", "ci_lower_ppi", "ci_upper_ppi"),
                ("Hybrid", "tau_hybrid", "ci_lower_hybrid", "ci_upper_hybrid"),
            ]:
                tau_vals = pl_sub[tau_col]
                bias = (tau_vals - tt).mean()
                mse = ((tau_vals - tt) ** 2).mean()
                rmse = np.sqrt(mse)
                covers = ((pl_sub[ci_lo_col] <= tt) & (tt <= pl_sub[ci_hi_col])).mean()
                mean_w = pl_sub["w"].mean() if est_name == "Hybrid" else np.nan
                w_str = f"{mean_w:.3f}" if not np.isnan(mean_w) else "-"

                # RE vs LO, RMSE ratio (paper convention, Section 3.2)
                re = np.sqrt(lo_mse / mse) if mse > 0 else np.nan
                re_str = f"{re:.2f}" if not np.isnan(re) else "-"

                lines.append(f"| {pi_L} | {est_name} | {bias:.4f} | {rmse:.4f} | {covers:.3f} | {w_str} | {re_str} |\n")

        lines.append("\n")
    return lines


def write_outputs(df: pd.DataFrame, outdir: str, R: int, protocol: str,
                  seed: int = 42, write_raw: bool = True) -> None:
    """Raw CSV (per replication), summary result rows, and the md table."""
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "hybrid_eval.csv")
    if write_raw:
        df.to_csv(csv_path, index=False)
        print(f"Raw results saved to {csv_path}")
    write_rows(
        os.path.join(outdir, "hybrid_eval_rows.json"),
        summary_rows(df, protocol, seed),
        generated_by="run_hybrid_eval.py", protocol=protocol,
    )
    output_path = os.path.join(outdir, "hybrid_eval.md")
    with open(output_path, "w") as f:
        f.writelines(build_markdown(df, R))
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid estimator evaluation")
    parser.add_argument("--R", type=int, default=500)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", choices=PROTOCOLS,
                        default=DEFAULT_PROTOCOL)
    parser.add_argument("--outdir",
                        default=os.path.join(PROJECT_ROOT, "results", "tables"))
    parser.add_argument("--from-raw", metavar="CSV", default=None,
                        help="rebuild rows + md from an existing per-"
                             "replication hybrid_eval.csv; no simulation")
    args = parser.parse_args()

    if args.from_raw:
        df = pd.read_csv(args.from_raw, float_precision="round_trip")
        R = int(df.groupby(["dgp_id", "overrides", "pi_L"])["rep"]
                .nunique().max())
        protocol = (str(df["protocol"].iloc[0]) if "protocol" in df.columns
                    else args.protocol)
        write_outputs(df, args.outdir, R, protocol, args.seed,
                      write_raw=False)
        return

    R = args.R
    cores = args.cores or min(8, os.cpu_count() or 1)
    master_seed = args.seed

    tasks = []
    for dgp_id, overrides in DGP_CONFIGS:
        for pi_L in PI_L_VALUES:
            for r in range(R):
                tasks.append((dgp_id, overrides, pi_L, r, master_seed,
                              args.protocol))

    print(f"Hybrid Estimator Eval: {len(tasks)} tasks, R={R}, {cores} cores, "
          f"protocol={args.protocol}")

    t0 = time.time()
    with multiprocessing.Pool(cores) as pool:
        all_results = pool.map(run_one_rep, tasks)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    df = pd.DataFrame(all_results)
    write_outputs(df, args.outdir, R, args.protocol, master_seed)


if __name__ == "__main__":
    main()
