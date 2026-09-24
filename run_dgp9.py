#!/usr/bin/env python3
"""
Simulation 2: Antagonistic Surrogate (DGP 9).

DGP 9 has S positively responding to T but Y negatively related to S,
plus a positive direct effect. This differentiates GREG from PPI++.

Also runs surrogacy test and hybrid estimator.
R=500, saves to results/tables/dgp9_results.md
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
from src.dgps.dgps import generate_dgp9
from src.methods import (
    estimate, surrogacy_test, estimate_cov_si_ppi, hybrid_estimator,
)
from src.utils.config import (
    DEFAULT_PROTOCOL,
    DGP_DEFAULTS,
    PROTOCOLS,
    resolve_specs,
    spec_by_key,
)
from src.utils.registry import rows_from_replications, write_rows
from src.utils.estimator_api import (
    estimate,
    estimate_cov_si_ppi,
    hybrid_estimator,
    surrogacy_test,
    train_prediction_model,
)


#: Prediction protocol; set from --protocol in main() and propagated to the
#: worker processes by `_set_protocol`.
PROTOCOL = DEFAULT_PROTOCOL


def _set_protocol(protocol: str) -> None:
    """Pool initializer: workers re-import this module under spawn."""
    global PROTOCOL
    PROTOCOL = protocol

# All methods including GREG
METHOD_IDS = [0, 1, 2, 3, 4, 5, 6]

# DGP 9 is the antagonistic surrogate: it is where the [0, 1] clip on the
# PPI++ coefficient binds, so this DGP's table prints the primary PPI++ beside
# the clipped ablation.  Every other configuration is the primary one.
METHOD_SPECS = resolve_specs(METHOD_IDS) + [spec_by_key("ppi_clipped")]
METHOD_NAMES = {0: "LO", 1: "NS", 2: "SI", 3: "PPI++", 4: "GREG", 5: "CP", 6: "AIPW"}

PI_L_VALUES = [0.05, 0.20, 0.50]


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


def run_one_cell(args: Tuple) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    pi_L, R, master_seed, hist_exps = args
    defaults = dict(DGP_DEFAULTS[9])

    est_results = []
    test_results = []
    hybrid_results = []

    for r in range(R):
        rep_seed = derive_seed(master_seed, 9, 0, int(pi_L * 100), r)

        dgp_kwargs = dict(defaults)
        dgp_kwargs["seed"] = rep_seed
        dgp_kwargs["pi_L"] = pi_L

        try:
            rep_results = run_single_replication(
                dgp_id=9,
                dgp_kwargs=dgp_kwargs,
                method_specs=METHOD_SPECS,
                seed=rep_seed,
                historical_experiments=hist_exps,
                protocol=PROTOCOL,
            )
        except Exception as e:
            print(f"Error: DGP 9, pi_L={pi_L}, rep={r}: {e}")
            continue

        for res in rep_results:
            res["pi_L"] = pi_L
            res["replication"] = r
            est_results.append(res)

        # Run surrogacy test and hybrid estimator
        try:
            data = generate_dgp9(**dgp_kwargs)
            cf_rng = np.random.default_rng(rep_seed + 7777)
            Y_hat, design = train_prediction_model(
                data["S"], data["X"], data["Y"],
                data["labeled_mask"], n_folds=5, rng=cf_rng,
                protocol=PROTOCOL, return_design=True,
            )

            si_res = estimate(2, data["T"], data["S"], data["Y"], Y_hat, data["labeled_mask"], protocol=PROTOCOL, design=design)
            ppi_res = estimate(3, data["T"], data["S"], data["Y"], Y_hat, data["labeled_mask"], protocol=PROTOCOL)

            lambda_hat = ppi_res.get("lambda_hat", 0.0)
            cov_sp = estimate_cov_si_ppi(
                data["T"], data["Y"], Y_hat, data["labeled_mask"], lambda_hat, design=design)

            test_res = surrogacy_test(
                si_res["tau_hat"], ppi_res["tau_hat"],
                si_res["var_hat"], ppi_res["var_hat"],
                cov_sp, alternative="two-sided",
            )
            test_res["pi_L"] = pi_L
            test_res["replication"] = r
            test_res["true_tau"] = data["true_tau"]
            test_results.append(test_res)

            hyb_res = hybrid_estimator(
                si_res["tau_hat"], ppi_res["tau_hat"],
                si_res["var_hat"], ppi_res["var_hat"],
                cov_sp, rng_seed=rep_seed + 9999,
            )
            hyb_res["pi_L"] = pi_L
            hyb_res["replication"] = r
            hyb_res["true_tau"] = data["true_tau"]
            hybrid_results.append(hyb_res)

        except Exception as e:
            print(f"Test/hybrid error: pi_L={pi_L}, rep={r}: {e}")

    return est_results, test_results, hybrid_results


def main():
    parser = argparse.ArgumentParser(description="DGP 9 (antagonistic surrogate)")
    parser.add_argument("--R", type=int, default=500)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    R = args.R
    cores = args.cores or min(8, os.cpu_count() or 1)
    master_seed = args.seed

    _, hist_exps = estimate_composite_weight_from_historical(K_hist=30, seed=999)

    tasks = [(pi_L, R, master_seed, hist_exps) for pi_L in PI_L_VALUES]

    print(f"DGP 9 Evaluation: {len(tasks)} cells, R={R}, {cores} cores")

    t0 = time.time()
    with multiprocessing.Pool(cores) as pool:
        results_nested = pool.map(run_one_cell, tasks)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    all_est = [r for batch in results_nested for r in batch[0]]
    all_test = [r for batch in results_nested for r in batch[1]]
    all_hybrid = [r for batch in results_nested for r in batch[2]]

    df_est = pd.DataFrame(all_est)
    df_test = pd.DataFrame(all_test)
    df_hybrid = pd.DataFrame(all_hybrid)

    output_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    lines = ["# DGP 9: Antagonistic Surrogate Results\n\n"]
    lines.append(f"True ATE = 0.41 (beta_YS*gamma_S + delta = -0.09 + 0.5)\n")
    lines.append(f"R = {R}\n\n")

    # Estimation results
    lines.append("## Estimation Results\n\n")
    lines.append("| pi_L | Method | Bias | RMSE | Coverage | RE |\n")
    lines.append("|------|--------|------|------|----------|----|\n")

    _rows = []
    for pi_L, g in df_est.groupby("pi_L"):
        _rows += rows_from_replications(
            g,
            dgp=9, config_name=f"DGP9_piL{pi_L:.2f}", params={}, pi_L=pi_L,
            n=DGP_DEFAULTS[9]["n"], R=int(g["replication"].nunique()),
            protocol=PROTOCOL, seed=42,
        )
    for pi_L in PI_L_VALUES:
        lines += _md_estimation_lines(_rows, pi_L)

    # Surrogacy test results
    lines.append("\n## Surrogacy Test Results\n\n")
    lines.append("| pi_L | Mean D_hat | Mean SE_D | Rejection Rate (alpha=0.05) |\n")
    lines.append("|------|-----------|----------|----------------------------|\n")

    for pi_L in PI_L_VALUES:
        sub = df_test[df_test["pi_L"] == pi_L]
        if sub.empty:
            continue
        mean_D = sub["D_hat"].mean()
        mean_SE = sub["SE_D"].mean()
        reject_rate = (sub["p_value"] < 0.05).mean()
        lines.append(f"| {pi_L} | {mean_D:.4f} | {mean_SE:.4f} | {reject_rate:.3f} |\n")

    # Hybrid estimator results
    lines.append("\n## Hybrid Estimator Results\n\n")
    lines.append("| pi_L | Mean tau_hybrid | Mean w | Bias | RMSE | Coverage |\n")
    lines.append("|------|---------------|--------|------|------|----------|\n")

    for pi_L in PI_L_VALUES:
        sub = df_hybrid[df_hybrid["pi_L"] == pi_L]
        if sub.empty:
            continue
        mean_tau = sub["tau_hybrid"].mean()
        mean_w = sub["w"].mean()
        true_tau = sub["true_tau"].iloc[0]
        bias = mean_tau - true_tau
        mse = ((sub["tau_hybrid"] - sub["true_tau"]) ** 2).mean()
        rmse = np.sqrt(mse)
        covers = ((sub["ci_lower"] <= sub["true_tau"]) & (sub["true_tau"] <= sub["ci_upper"])).mean()
        lines.append(f"| {pi_L} | {mean_tau:.4f} | {mean_w:.3f} | {bias:.4f} | {rmse:.4f} | {covers:.3f} |\n")

    output_path = os.path.join(output_dir, "dgp9_results.md")
    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"Results saved to {output_path}")

    csv_path = os.path.join(output_dir, "dgp9_results.csv")
    df_est.to_csv(csv_path, index=False)
    write_rows(os.path.join(os.path.dirname(csv_path), "dgp9_rows.json"),
               _rows, generated_by="run_dgp9.py", protocol=PROTOCOL)
    print(f"Result rows saved: dgp9_rows.json ({len(_rows)} rows)")

    print(f"Raw results saved to {csv_path}")


if __name__ == "__main__":
    main()
