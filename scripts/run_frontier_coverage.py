#!/usr/bin/env python3
"""
SI and PPI++ coverage on a dense rho grid for the detection-damage
frontier figure. Matches the DGP 2 protocol of the main simulations and
the rho grid of results/tables/power_curve_extended.csv.

Grid: rho in {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6}
      n in {10000, 100000}, pi_L = 0.20, R = 500

Saves results/tables/frontier_coverage.csv
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dgps.dgps import generate_dgp2
from src.simulations.simulation import derive_seed
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.registry import write_table_rows
from src.utils.estimator_api import (
    estimate,
    train_prediction_model,
)


#: Prediction protocol; set from --protocol in main() and propagated to the
#: worker processes by `_set_protocol`.
PROTOCOL = DEFAULT_PROTOCOL


def _set_protocol(protocol: str) -> None:
    """Pool initializer: workers re-import this module under spawn."""
    global PROTOCOL
    PROTOCOL = protocol


def run_one(args: Tuple) -> Dict[str, Any]:
    rep, rho, n, pi_L, master_seed = args
    rep_seed = derive_seed(master_seed, 2, int(rho * 1000),
                           int(pi_L * 100) * 100 + n // 1000, rep)
    data = generate_dgp2(n=n, pi_L=pi_L, rho=rho, seed=rep_seed)
    T, S, Y, lm = data["T"], data["S"], data["Y"], data["labeled_mask"]
    cf_rng = np.random.default_rng(rep_seed + 7777)
    Y_hat, design = train_prediction_model(S, data["X"], Y, lm, n_folds=5, rng=cf_rng, protocol=PROTOCOL, return_design=True)
    tau = data["true_tau"]

    si = estimate(2, T, S, Y, Y_hat, lm, protocol=PROTOCOL, design=design)
    ppi = estimate(3, T, S, Y, Y_hat, lm, protocol=PROTOCOL)
    return dict(
        rep=rep, rho=rho, n=n, pi_L=pi_L, true_tau=tau,
        si_covers=int(si["ci_lower"] <= tau <= si["ci_upper"]),
        ppi_covers=int(ppi["ci_lower"] <= tau <= ppi["ci_upper"]),
        si_bias=si["tau_hat"] - tau,
        ppi_bias=ppi["tau_hat"] - tau,
    )


def main():
    import argparse

    global PROTOCOL
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--R", type=int, default=500,
                     help="Monte Carlo replications (smoke runs use a few)")
    _ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    _args = _ap.parse_args()
    PROTOCOL = _args.protocol

    R = _args.R
    master_seed = 42
    rho_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
    n_values = [10_000, 100_000]
    pi_L = 0.20

    tasks = [(r, rho, n, pi_L, master_seed)
             for rho in rho_values for n in n_values for r in range(R)]
    print(f"{len(tasks)} tasks", flush=True)
    t0 = time.time()
    with multiprocessing.Pool(8) as pool:
        results = pool.map(run_one, tasks)
    print(f"done in {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(results)
    agg = df.groupby(["n", "pi_L", "rho"]).agg(
        si_coverage=("si_covers", "mean"),
        ppi_coverage=("ppi_covers", "mean"),
        si_bias=("si_bias", "mean"),
        R=("rep", "count"),
    ).reset_index()
    out = os.path.join(PROJECT_ROOT, "results", "tables",
                       "frontier_coverage.csv")
    agg.to_csv(out, index=False)
    write_table_rows(
        str(out).replace(".csv", "_rows.json"), agg,
        dgp=2,
        analysis="frontier_coverage",
        protocol=PROTOCOL,
        config_cols=("n", "rho"),
        n_col="n",
        R_col="R",
        generated_by="scripts/run_frontier_coverage.py",
    )
    print(agg.to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
