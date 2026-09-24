#!/usr/bin/env python3
"""
Binary Switch vs Hybrid Estimator Comparison.

Binary switch: surrogacy test at alpha=0.10. If reject -> PPI++. If fail -> SI.
Hybrid: hybrid estimator with c=1.5.

DGP configs:
  DGP 1 (pi_L={0.05, 0.20})
  DGP 2 (rho={0, 0.2, 0.4}, pi_L=0.20)
  DGP 9 (pi_L=0.20)

R=2000 per cell.

Outputs (in --outdir, default results/tables):
  switch_vs_hybrid.csv        one row per replication (the raw data)
  switch_vs_hybrid_rows.json  summary result rows, one per (cell, estimator)
                              plus the diagnostic's rejection rate
  switch_vs_hybrid.md         the paper table
`--from-raw` rebuilds the summary rows and the table from an existing
switch_vs_hybrid.csv without re-simulating.
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
from src.methods import ppi_plus_bootstrap
from src.simulations.simulation import derive_seed
from src.utils.config import DEFAULT_PROTOCOL, DGP_DEFAULTS, PROTOCOLS
from src.utils.config import make_result_row
from src.utils.registry import (
    metrics_from_errors,
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


# (dgp_id, overrides, pi_L)
DGP_CONFIGS = [
    (1, {}, 0.05),
    (1, {}, 0.20),
    (2, {"rho": 0.0}, 0.20),
    (2, {"rho": 0.2}, 0.20),
    (2, {"rho": 0.4}, 0.20),
    (9, {}, 0.20),
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

    # SI and PPI++
    ppi_res = estimate(3, T, S, Y, Y_hat, lm, protocol=protocol)
    lambda_hat = ppi_res.get("lambda_hat", 0.0)
    si_res = estimate(2, T, S, Y, Y_hat, lm, protocol=protocol,
                      design=design, lambda_hat=lambda_hat)

    # Sigma_hat: the joint sandwich covariance for the learned index.
    sigma = hybrid_sigma(T, Y, lm, design, lambda_hat)
    var_si_j, cov_sp, var_ppi_j = (
        float(sigma[0, 0]), float(sigma[0, 1]), float(sigma[1, 1])
    )

    # Bootstrap SE for PPI++
    boot_res = ppi_plus_bootstrap(T, S, Y, Y_hat, lm, B=200, boot_seed=rep_seed + 2222)

    # Surrogacy test (two-sided, using bootstrap variance)
    test_res = surrogacy_test(
        si_res["tau_hat"], ppi_res["tau_hat"],
        var_si_j, boot_res["var_hat"],
        cov_sp, alternative="two-sided",
    )

    # Binary switch at alpha=0.10
    switch_reject = test_res["p_value"] < 0.10
    if switch_reject:
        switch_tau = ppi_res["tau_hat"]
        switch_ci_lo = ppi_res["ci_lower"]
        switch_ci_hi = ppi_res["ci_upper"]
        switch_choice = "PPI++"
    else:
        switch_tau = si_res["tau_hat"]
        switch_ci_lo = si_res["ci_lower"]
        switch_ci_hi = si_res["ci_upper"]
        switch_choice = "SI"

    # Hybrid (c=1.5)
    hyb_res = hybrid_estimator(
        si_res["tau_hat"], ppi_res["tau_hat"],
        var_si_j, var_ppi_j,
        cov_sp, c=1.5, rng_seed=rep_seed + 9999,
    )

    return {
        "dgp_id": dgp_id,
        "overrides": str(overrides),
        "pi_L": pi_L,
        "rep": rep,
        "true_tau": true_tau,
        # SI
        "tau_si": si_res["tau_hat"],
        "ci_lower_si": si_res["ci_lower"],
        "ci_upper_si": si_res["ci_upper"],
        # PPI++
        "tau_ppi": ppi_res["tau_hat"],
        "ci_lower_ppi": ppi_res["ci_lower"],
        "ci_upper_ppi": ppi_res["ci_upper"],
        # Surrogacy test
        "test_p_value": test_res["p_value"],
        "test_reject_010": int(switch_reject),
        # Binary switch
        "tau_switch": switch_tau,
        "ci_lower_switch": switch_ci_lo,
        "ci_upper_switch": switch_ci_hi,
        "switch_choice": switch_choice,
        # Hybrid
        "tau_hybrid": hyb_res["tau_hybrid"],
        "w_hybrid": hyb_res["w"],
        "ci_lower_hybrid": hyb_res["ci_lower"],
        "ci_upper_hybrid": hyb_res["ci_upper"],
    }


#: (method-spec key or (key, label), tau column, CI lower, CI upper, md label)
ESTIMATORS = [
    ("si", "tau_si", "ci_lower_si", "ci_upper_si", "SI"),
    ("ppi", "tau_ppi", "ci_lower_ppi", "ci_upper_ppi", "PPI++"),
    (("switch", "Switch"), "tau_switch", "ci_lower_switch",
     "ci_upper_switch", "Switch"),
    (("hybrid", "Hybrid"), "tau_hybrid", "ci_lower_hybrid",
     "ci_upper_hybrid", "Hybrid"),
]

DIAG_LABEL = "SI--PPI++ diagnostic (two-sided)"


def _key(method) -> str:
    return method if isinstance(method, str) else method[0]


def summary_rows(df: pd.DataFrame, protocol: str,
                 seed: int = 42) -> List[Dict[str, Any]]:
    """One result row per (cell, estimator) plus one diagnostic row.

    The Switch row carries the fraction of replications that chose PPI++
    (`switch_ppi_fraction`); the Hybrid row the mean weight.  The diagnostic
    row is the two-sided test at alpha = 0.10 (bootstrap PPI++ variance) that
    drives the switch.
    """
    rows: List[Dict[str, Any]] = []
    md_label = {_key(e[0]): e[4] for e in ESTIMATORS}
    order = [_key(e[0]) for e in ESTIMATORS]
    for dgp_id, overrides, pi_L in DGP_CONFIGS:
        g = df[(df["dgp_id"] == dgp_id)
               & (df["overrides"] == str(overrides))
               & (df["pi_L"] == pi_L)]
        if g.empty:
            continue
        name = f"DGP{dgp_id}" + "".join(
            f"_{k}{v}" for k, v in overrides.items()) + f"_piL{pi_L:.2f}"
        common = dict(dgp=dgp_id, config_name=name,
                      params={"dgp_id": dgp_id, **overrides}, pi_L=pi_L,
                      n=DGP_DEFAULTS[dgp_id].get("n"), protocol=protocol,
                      seed=seed)
        cell = rows_from_replications(
            replications_long(g, [e[:4] for e in ESTIMATORS],
                              protocol=protocol),
            **common)
        cell.sort(key=lambda r: order.index(r["method_key"]))
        for r in cell:
            r["table_label"] = md_label[r["method_key"]]
            if r["method_key"] == "switch":
                r["switch_ppi_fraction"] = float(
                    (g["switch_choice"] == "PPI++").mean())
                r["switch_alpha"] = 0.10
            if r["method_key"] == "hybrid":
                r["mean_w"] = float(g["w_hybrid"].mean())
                r["c"] = 1.5
        diag = make_result_row(
            **{**common, "R": g["rep"].nunique()},
            method_id=-1, method_label=DIAG_LABEL, alpha=0.10,
            metrics=metrics_from_errors(
                None, true_tau=g["true_tau"].to_numpy(float),
                reject=g["test_reject_010"].to_numpy(float)),
            extra={"method_key": "diagnostic",
                   "ppi_variance": "bootstrap (B=200)"},
        )
        cell.append(diag)
        for r in cell:
            r["analysis"] = "switch_vs_hybrid"
            for k, v in overrides.items():
                r[k] = v
        rows.extend(cell)
    return rows


def build_markdown(df: pd.DataFrame, R: int) -> List[str]:
    lines = ["# Binary Switch vs Hybrid Estimator Comparison\n\n"]
    lines.append(f"R = {R}\n")
    lines.append("Binary switch: surrogacy test (alpha=0.10, two-sided). Reject -> PPI++, fail -> SI.\n")
    lines.append("Hybrid: Cauchy-kernel interpolation with c=1.5.\n\n")

    for dgp_id, overrides, pi_L in DGP_CONFIGS:
        label = f"DGP {dgp_id}"
        if overrides:
            label += f" ({', '.join(f'{k}={v}' for k, v in overrides.items())})"
        label += f", pi_L={pi_L}"

        sub = df[(df["dgp_id"] == dgp_id) & (df["overrides"] == str(overrides)) & (df["pi_L"] == pi_L)]
        if sub.empty:
            continue

        true_tau = sub["true_tau"].iloc[0]
        lines.append(f"## {label} (true_tau = {true_tau:.4f})\n\n")

        # Rejection rate of surrogacy test
        rr = sub["test_reject_010"].mean()
        ppi_frac = (sub["switch_choice"] == "PPI++").mean()
        lines.append(f"Surrogacy test rejection rate (alpha=0.10): {rr:.3f}\n")
        lines.append(f"Switch chooses PPI++: {ppi_frac:.3f}\n\n")

        lines.append("| Estimator | Bias | RMSE | Coverage | Mean w |\n")
        lines.append("|-----------|------|------|----------|--------|\n")

        tt = sub["true_tau"]

        for est_name, tau_col, ci_lo_col, ci_hi_col in [
            ("SI", "tau_si", "ci_lower_si", "ci_upper_si"),
            ("PPI++", "tau_ppi", "ci_lower_ppi", "ci_upper_ppi"),
            ("Switch", "tau_switch", "ci_lower_switch", "ci_upper_switch"),
            ("Hybrid", "tau_hybrid", "ci_lower_hybrid", "ci_upper_hybrid"),
        ]:
            tau_vals = sub[tau_col]
            bias = (tau_vals - tt).mean()
            mse = ((tau_vals - tt) ** 2).mean()
            rmse = np.sqrt(mse)
            covers = ((sub[ci_lo_col] <= tt) & (tt <= sub[ci_hi_col])).mean()
            mean_w = sub["w_hybrid"].mean() if est_name == "Hybrid" else np.nan
            w_str = f"{mean_w:.3f}" if not np.isnan(mean_w) else "-"
            lines.append(f"| {est_name} | {bias:.4f} | {rmse:.4f} | {covers:.3f} | {w_str} |\n")

        lines.append("\n")
    return lines


def write_outputs(df: pd.DataFrame, outdir: str, R: int, protocol: str,
                  seed: int = 42, write_raw: bool = True) -> None:
    """Raw CSV (per replication), summary result rows, and the md table."""
    os.makedirs(outdir, exist_ok=True)
    if write_raw:
        df.to_csv(os.path.join(outdir, "switch_vs_hybrid.csv"), index=False)
    write_rows(
        os.path.join(outdir, "switch_vs_hybrid_rows.json"),
        summary_rows(df, protocol, seed),
        generated_by="run_switch_vs_hybrid.py", protocol=protocol,
    )
    output_path = os.path.join(outdir, "switch_vs_hybrid.md")
    with open(output_path, "w") as f:
        f.writelines(build_markdown(df, R))
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Switch vs Hybrid comparison")
    parser.add_argument("--R", type=int, default=2000)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", choices=PROTOCOLS,
                        default=DEFAULT_PROTOCOL)
    parser.add_argument("--outdir",
                        default=os.path.join(PROJECT_ROOT, "results", "tables"))
    parser.add_argument("--from-raw", metavar="CSV", default=None,
                        help="rebuild rows + md from an existing per-"
                             "replication switch_vs_hybrid.csv; no simulation")
    args = parser.parse_args()

    if args.from_raw:
        df = pd.read_csv(args.from_raw, float_precision="round_trip")
        R = int(df.groupby(["dgp_id", "overrides", "pi_L"])["rep"]
                .nunique().max())
        write_outputs(df, args.outdir, R, args.protocol, args.seed,
                      write_raw=False)
        return

    R = args.R
    cores = args.cores
    master_seed = args.seed

    tasks = []
    for dgp_id, overrides, pi_L in DGP_CONFIGS:
        for r in range(R):
            tasks.append((dgp_id, overrides, pi_L, r, master_seed,
                          args.protocol))

    print(f"Switch vs Hybrid: {len(DGP_CONFIGS)} cells x R={R} = {len(tasks)} tasks, {cores} cores")

    t0 = time.time()
    with multiprocessing.Pool(cores) as pool:
        all_results = pool.map(run_one_rep, tasks)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    df = pd.DataFrame(all_results)
    write_outputs(df, args.outdir, R, args.protocol, master_seed)


if __name__ == "__main__":
    main()
