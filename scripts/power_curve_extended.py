#!/usr/bin/env python3
"""Extended power curves for the surrogacy diagnostic test.

Sweeps n ∈ {1000, 10000, 100000}, π_L ∈ {0.05, 0.20, 0.50},
ρ ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8}.
Runs BOTH one-sided and two-sided tests.
R=500 (R=500 for n=100K speed).

Uses 8-core multiprocessing.
"""

from __future__ import annotations

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
from src.methods import ppi_plus
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.registry import write_table_rows
from src.utils.estimator_api import (
    estimate,
    estimate_cov_si_ppi,
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

# ── Configuration ──────────────────────────────────────────────────────────

N_VALUES = [1_000, 10_000, 100_000]
PI_L_VALUES = [0.05, 0.20, 0.50]
RHO_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
R = 500
ALPHA = 0.05
MASTER_SEED = 42
N_WORKERS = 8


# ── Single-replication worker ─────────────────────────────────────────────

def run_one_rep(args: Tuple) -> Dict[str, Any]:
    """Run one replication for a given (n, pi_L, rho, rep_id).

    Returns a dict with rejection flags for both one-sided and two-sided tests.
    """
    n, pi_L, rho, rep_id = args

    config_id = int(n / 1000)
    pi_L_id = int(pi_L * 100)
    rho_id = int(rho * 100)
    seed = derive_seed(MASTER_SEED, dgp_id=2, config_id=config_id * 100 + rho_id,
                       pi_L_id=pi_L_id, rep=rep_id)

    # Generate data
    data = generate_dgp2(n=n, pi_L=pi_L, rho=rho, seed=seed)

    T = data["T"]
    S = data["S"]
    Y = data["Y"]
    X = data["X"]
    labeled_mask = data["labeled_mask"]
    true_tau = data["true_tau"]

    # Train prediction model
    cf_rng = np.random.default_rng(seed + 7777)
    Y_hat, design = train_prediction_model(S, X, Y, labeled_mask, n_folds=5, rng=cf_rng, protocol=PROTOCOL, return_design=True)

    # Run SI and PPI++
    res_si = estimate(2, T, S, Y, Y_hat, labeled_mask, protocol=PROTOCOL,
                      design=design)
    res_ppi = ppi_plus(T, S, Y, Y_hat, labeled_mask)

    tau_si = res_si["tau_hat"]
    var_si = res_si["var_hat"]
    tau_ppi = res_ppi["tau_hat"]
    var_ppi = res_ppi["var_hat"]
    lambda_hat = res_ppi.get("lambda_hat", 0.0)

    # Covariance
    cov_sp = estimate_cov_si_ppi(T, Y, Y_hat, labeled_mask, lambda_hat, design=design)

    # One-sided test (H1: delta > 0)
    test_1s = surrogacy_test(
        tau_si=tau_si, tau_ppi=tau_ppi,
        var_si=var_si, var_ppi=var_ppi,
        cov_si_ppi=cov_sp, alternative="greater",
    )
    p_1s = test_1s["p_value"]

    # Two-sided test
    test_2s = surrogacy_test(
        tau_si=tau_si, tau_ppi=tau_ppi,
        var_si=var_si, var_ppi=var_ppi,
        cov_si_ppi=cov_sp, alternative="two-sided",
    )
    p_2s = test_2s["p_value"]

    valid = 1 if (np.isfinite(p_1s) and np.isfinite(p_2s)) else 0
    reject_1s = 1 if (valid and p_1s < ALPHA) else 0
    reject_2s = 1 if (valid and p_2s < ALPHA) else 0

    return {
        "n": n,
        "pi_L": pi_L,
        "rho": rho,
        "rep": rep_id,
        "reject_1s": reject_1s,
        "reject_2s": reject_2s,
        "valid": valid,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse

    global R, PROTOCOL
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--R", type=int, default=R,
                     help="Monte Carlo replications (smoke runs use a few)")
    _ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    _args = _ap.parse_args()
    PROTOCOL = _args.protocol
    R = _args.R

    print("Extended Power Curves")
    print(f"  n: {N_VALUES}")
    print(f"  pi_L: {PI_L_VALUES}")
    print(f"  rho: {RHO_VALUES}")
    print(f"  R={R}, alpha={ALPHA}")
    print()

    # Build task list
    tasks = []
    for n in N_VALUES:
        for pi_L in PI_L_VALUES:
            for rho in RHO_VALUES:
                for r in range(R):
                    tasks.append((n, pi_L, rho, r))

    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks}")

    t0 = time.time()

    with Pool(N_WORKERS) as pool:
        results = pool.map(run_one_rep, tasks, chunksize=20)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    df = pd.DataFrame(results)

    # Aggregate
    summary_rows = []
    for (n, pi_L, rho), grp in df.groupby(["n", "pi_L", "rho"]):
        n_valid = grp["valid"].sum()
        n_reject_1s = grp["reject_1s"].sum()
        n_reject_2s = grp["reject_2s"].sum()

        rate_1s = n_reject_1s / n_valid if n_valid > 0 else np.nan
        rate_2s = n_reject_2s / n_valid if n_valid > 0 else np.nan

        # True delta
        mediated = 0.5 * 0.3  # beta_YS * gamma_S
        delta = (rho / (1.0 - rho)) * mediated if rho < 1.0 else np.inf
        true_tau = mediated + delta

        summary_rows.append({
            "n": int(n),
            "pi_L": pi_L,
            "rho": rho,
            "delta": round(delta, 4),
            "true_tau": round(true_tau, 4),
            "n_valid": int(n_valid),
            "reject_rate_1s": round(rate_1s, 4),
            "reject_rate_2s": round(rate_2s, 4),
        })

    summary = pd.DataFrame(summary_rows)

    # Save
    out_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "power_curve_extended.csv")
    md_path = os.path.join(out_dir, "power_curve_extended.md")

    summary.to_csv(csv_path, index=False)
    write_table_rows(
        csv_path.replace(".csv", "_rows.json"), summary,
        dgp=2,
        analysis="SI--PPI++ diagnostic",
        protocol=PROTOCOL,
        config_cols=("n", "rho"),
        n_col="n",
        R_col="n_valid",
        generated_by="scripts/power_curve_extended.py",
    )

    # Markdown
    lines = [
        "# Extended Power Curves: Surrogacy Diagnostic Test",
        "",
        f"DGP 2, alpha={ALPHA}, R={R}",
        "",
        "| n | pi_L | rho | delta | true_tau | n_valid | reject_1s | reject_2s |",
        "|---|------|-----|-------|----------|---------|-----------|-----------|",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['n']):,} | {r['pi_L']:.2f} | {r['rho']:.1f} "
            f"| {r['delta']:.4f} | {r['true_tau']:.4f} "
            f"| {int(r['n_valid'])} | {r['reject_rate_1s']:.4f} "
            f"| {r['reject_rate_2s']:.4f} |"
        )

    md_text = "\n".join(lines) + "\n"
    with open(md_path, "w") as f:
        f.write(md_text)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved MD:  {md_path}")
    print()

    # Print compact summary
    for n in N_VALUES:
        for pi_L in PI_L_VALUES:
            sub = summary[(summary["n"] == n) & (summary["pi_L"] == pi_L)]
            size_1s = sub[sub["rho"] == 0.0]["reject_rate_1s"].values[0]
            print(f"n={n:>7,}, pi_L={pi_L:.2f}: size(1s)={size_1s:.4f}")


if __name__ == "__main__":
    main()
