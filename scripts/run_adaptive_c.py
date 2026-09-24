#!/usr/bin/env python3
"""
Covariance-adaptive choice of the hybrid tuning constant c.

The limit-minimax c* of the Prop.-9 analysis depends on the nuisance
covariance (sig_S, sig_P, sig_SP), so the "c = 1.5 is near-minimax" claim
is specific to the calibrated cell. This script makes the choice adaptive:
on each replication, estimate the covariance from the data
    sig_S^2 = n var_SI,  sig_P^2 = n var_PPI,  sig_SP = n cov_SI,PPI,
compute c*(sig_hat) = argmin_c max_h [ r(h; c) - oracle(h) ] by the same
1-D integral as run_limit_experiment.py, and run the hybrid with c = c*.

Cells: DGP 1 (pi_L = 0.05, 0.20, 0.50), DGP 2 (rho = 0.2, 0.4 at
pi_L = 0.20), DGP 9 (pi_L = 0.20). For each cell: fixed c = 1.5 vs
adaptive c, R = 500. Also reports the cell-level c* at the mean estimated
covariance.

Output (in --outdir, default results/tables):
  adaptive_c_raw.csv    one row per (replication, estimator)
  adaptive_c_rows.json  summary result rows, one per (cell, estimator)
  adaptive_c.md         the table
`--from-raw` rebuilds the rows and the table from an existing raw CSV.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dgps.dgps import generate_dgp1, generate_dgp2, generate_dgp9
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.config import make_result_row, spec_by_key
from src.utils.registry import metrics_from_errors, write_rows
from src.utils.estimator_api import (
    estimate,
    hybrid_estimator,
    hybrid_sigma,
    train_prediction_model,
)

#: cell -> (dgp id, pi_L, DGP overrides), for the result rows
CELL_META = {
    "DGP1 piL=0.05": (1, 0.05, {}),
    "DGP1 piL=0.20": (1, 0.20, {}),
    "DGP1 piL=0.50": (1, 0.50, {}),
    "DGP2 rho=0.2": (2, 0.20, {"rho": 0.2}),
    "DGP2 rho=0.4": (2, 0.20, {"rho": 0.4}),
    "DGP9": (9, 0.20, {}),
}

CELLS = [
    ("DGP1 piL=0.05", lambda s: generate_dgp1(n=10_000, pi_L=0.05, seed=s)),
    ("DGP1 piL=0.20", lambda s: generate_dgp1(n=10_000, pi_L=0.20, seed=s)),
    ("DGP1 piL=0.50", lambda s: generate_dgp1(n=10_000, pi_L=0.50, seed=s)),
    ("DGP2 rho=0.2",  lambda s: generate_dgp2(n=10_000, pi_L=0.20, rho=0.2, seed=s)),
    ("DGP2 rho=0.4",  lambda s: generate_dgp2(n=10_000, pi_L=0.20, rho=0.4, seed=s)),
    ("DGP9",          lambda s: generate_dgp9(n=10_000, pi_L=0.20, seed=s)),
]

C_GRID = np.arange(0.3, 4.01, 0.05)


def minimax_c(sig_S, sig_P, sig_SP):
    """Limit-minimax c over the excess risk, vectorized over (c, h, V)."""
    sig_D2 = sig_P**2 + sig_S**2 - 2 * sig_SP
    if sig_D2 <= 1e-12:
        return 1.5  # degenerate; keep default
    sig_D = np.sqrt(sig_D2)
    gamma = (sig_P**2 - sig_SP) / sig_D2

    h = np.linspace(0, 8 * sig_D, 41)                        # (H,)
    z = np.linspace(-10, 10, 1601)                           # std grid (Z,)
    dens = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    V = h[:, None] + sig_D * z[None, :]                      # (H, Z)
    oracle = np.minimum(sig_P**2, sig_S**2 + h**2)           # (H,)

    excess = np.empty(len(C_GRID))
    for i, c in enumerate(C_GRID):
        w = c / (c + (V / sig_D) ** 2)
        integrand = (gamma * (V - h[:, None]) - w * V) ** 2  # (H, Z)
        Ev = np.trapz(integrand * dens[None, :], z, axis=1)
        r = sig_P**2 - gamma**2 * sig_D2 + Ev
        excess[i] = np.max(r - oracle)
    return float(C_GRID[int(np.argmin(excess))])


def run_one(args):
    rep, cell_id, protocol = args
    name, gen = CELLS[cell_id]
    d = gen(880_000 + rep)
    n = len(d["T"])
    Yh, design = train_prediction_model(
        d["S"], d["X"], d["Y"], d["labeled_mask"], n_folds=5,
        rng=np.random.default_rng(rep + 7777),
        protocol=protocol, return_design=True,
    )
    pp = estimate(3, d["T"], d["S"], d["Y"], Yh, d["labeled_mask"],
                  protocol=protocol)
    lam = pp.get("lambda_hat", 0.0)
    si = estimate(2, d["T"], d["S"], d["Y"], Yh, d["labeled_mask"],
                  protocol=protocol, design=design, lambda_hat=lam)
    lo = estimate(0, d["T"], d["S"], d["Y"], Yh, d["labeled_mask"],
                  protocol=protocol)
    # Sigma_hat: the joint sandwich covariance for the learned index.
    sigma = hybrid_sigma(d["T"], d["Y"], d["labeled_mask"], design, lam)
    var_si_j, cov_sp, var_ppi_j = (
        float(sigma[0, 0]), float(sigma[0, 1]), float(sigma[1, 1])
    )
    tau = d["true_tau"]

    sig_S = np.sqrt(n * var_si_j)
    sig_P = np.sqrt(n * var_ppi_j)
    sig_SP = n * cov_sp
    c_star = minimax_c(sig_S, sig_P, sig_SP)

    out = [dict(rep=rep, cell=name, method="LO", c=np.nan, true_tau=tau,
                bias=lo["tau_hat"] - tau,
                covers=int(lo["ci_lower"] <= tau <= lo["ci_upper"]))]
    for label, c in [("hybrid c=1.5", 1.5), ("hybrid adaptive", c_star)]:
        hyb = hybrid_estimator(si["tau_hat"], pp["tau_hat"], var_si_j,
                               var_ppi_j, cov_sp, c=c, rng_seed=rep)
        out.append(dict(rep=rep, cell=name, method=label, c=c, true_tau=tau,
                        bias=hyb["tau_hybrid"] - tau,
                        covers=int(hyb["ci_lower"] <= tau <= hyb["ci_upper"])))
    return out


#: md estimator label -> (method-spec key or (key, label))
METHODS = {
    "LO": "lo",
    "hybrid c=1.5": ("hybrid_c1.5", "Hybrid (c=1.5)"),
    "hybrid adaptive": ("hybrid_adaptive", "Hybrid (adaptive c)"),
}


def _true_tau(df: pd.DataFrame, name: str) -> float:
    """The cell's true tau: from the raw file when recorded, else from the
    DGP (it does not depend on the replication seed)."""
    sub = df[df.cell == name]
    if "true_tau" in sub.columns and sub["true_tau"].notna().any():
        return float(sub["true_tau"].dropna().iloc[0])
    gen = dict(CELLS)[name]
    return float(gen(880_000)["true_tau"])


def summary_rows(df: pd.DataFrame, protocol: str, R: int,
                 seed: int = 42) -> list:
    """One result row per (cell, estimator): bias, RMSE, coverage, RE vs
    LO (RMSE ratio) and MC SEs; hybrid rows carry the mean c (`mean_c`)."""
    rows = []
    for name, _ in CELLS:
        sub = df[df.cell == name]
        if sub.empty:
            continue
        dgp_id, pi_L, overrides = CELL_META[name]
        tau = _true_tau(df, name)
        lo_rmse = float(np.sqrt((sub[sub.method == "LO"].bias ** 2).mean()))
        for md_label, method in METHODS.items():
            ms = sub[sub.method == md_label]
            if ms.empty:
                continue
            metrics = metrics_from_errors(
                ms.bias.to_numpy(float), true_tau=tau,
                covers=ms.covers.to_numpy(float), rmse_baseline=lo_rmse)
            extra = {"analysis": "adaptive_c", "table_label": md_label,
                     "cell": name, **overrides}
            common = dict(dgp=dgp_id, config_name=name,
                          params={"cell": name, "dgp_id": dgp_id,
                                  **overrides},
                          pi_L=pi_L, n=10_000, R=ms.rep.nunique(),
                          protocol=protocol, seed=seed, metrics=metrics)
            if isinstance(method, str):
                row = make_result_row(spec=spec_by_key(method), **common,
                                      extra={"method_key": method, **extra})
            else:
                extra["mean_c"] = float(ms.c.mean())
                row = make_result_row(method_id=-1, method_label=method[1],
                                      **common,
                                      extra={"method_key": method[0],
                                             **extra})
            rows.append(row)
    return rows


def build_markdown(df: pd.DataFrame, R: int) -> list:
    lines = ["# Covariance-adaptive c vs fixed c = 1.5\n",
             f"n = 10,000, R = {R}. Adaptive c = limit-minimax c* at the "
             "replication's estimated (sig_S, sig_P, sig_SP).\n\n",
             "| Cell | Estimator | Mean c | Bias | RMSE | Coverage "
             "| RE vs LO (RMSE ratio) |\n|---|---|---|---|---|---|---|\n"]
    for name, _ in CELLS:
        sub = df[df.cell == name]
        lo_rmse = np.sqrt((sub[sub.method == "LO"].bias ** 2).mean())
        for meth in ("hybrid c=1.5", "hybrid adaptive"):
            ms = sub[sub.method == meth]
            rmse = np.sqrt((ms.bias ** 2).mean())
            lines.append(f"| {name} | {meth} | {ms.c.mean():.2f} "
                         f"| {ms.bias.mean():+.4f} | {rmse:.4f} "
                         f"| {ms.covers.mean():.3f} | {lo_rmse / rmse:.2f} |\n")
    return lines


def write_outputs(df: pd.DataFrame, outdir: str, R: int, protocol: str,
                  write_raw: bool = True) -> None:
    os.makedirs(outdir, exist_ok=True)
    if write_raw:
        df.to_csv(os.path.join(outdir, "adaptive_c_raw.csv"), index=False)
    write_rows(
        os.path.join(outdir, "adaptive_c_rows.json"),
        summary_rows(df, protocol, R),
        generated_by="scripts/run_adaptive_c.py", protocol=protocol,
    )
    with open(os.path.join(outdir, "adaptive_c.md"), "w") as f:
        f.writelines(build_markdown(df, R))


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    ap.add_argument("--outdir",
                    default=os.path.join(PROJECT_ROOT, "results", "tables"))
    ap.add_argument("--from-raw", metavar="CSV", default=None,
                    help="rebuild rows + md from an existing raw CSV; "
                         "no simulation")
    args = ap.parse_args()

    if args.from_raw:
        df = pd.read_csv(args.from_raw, float_precision="round_trip")
        R = int(df.groupby("cell")["rep"].nunique().max())
        write_outputs(df, args.outdir, R, args.protocol, write_raw=False)
        return

    t0 = time.time()
    R = args.R
    tasks = [(rep, c, args.protocol)
             for c in range(len(CELLS)) for rep in range(R)]
    print(f"{len(tasks)} tasks", flush=True)
    with multiprocessing.Pool(8) as pool:
        nested = pool.map(run_one, tasks)
    df = pd.DataFrame([r for b in nested for r in b])
    write_outputs(df, args.outdir, R, args.protocol)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
