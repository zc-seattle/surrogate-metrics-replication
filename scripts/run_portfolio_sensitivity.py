#!/usr/bin/env python3
"""
Portfolio loss-function sensitivity.

This script re-runs the DGP 6 portfolio with two additions:

  1. A per-launch cost c_launch (in tau units), subtracted from the realized
     payoff of every launched experiment.  Realized payoff is
         sum_k d_k * (tau_k - c_launch),
     the oracle launches exactly when tau_k > c_launch, and
         regret = sum_k [ max(tau_k - c, 0) - d_k * (tau_k - c) ].
     At c = 0 this is algebraically the paper's current regret.
     Reference scale: the oracle gain over K = 200 experiments is about 4.3,
     i.e. ~0.0215 per experiment, so c = 0.01 per launch is material.

  2. A fraction f_ant of ANTAGONISTIC experiments (DGP 9 style): the
     surrogate still responds positively to treatment, but Y depends
     negatively on S, so the mediated component has the opposite sign to the
     true effect and a direct effect restores it:
         beta_YS = -0.3,  mediated_k = -tau_k,  delta_k = 2 * tau_k,
         gamma_S_k = -tau_k / beta_YS = tau_k / 0.3.
     A surrogate-index reading of such an experiment gets the SIGN wrong.
     Non-antagonistic experiments are the standard valid-surrogate DGP 6
     construction (beta_YS = 0.5, delta_k = 0).

Two decision rules are reported from the same estimates:
  * `tau=0`   -- the paper's rule: launch iff the two-sided test of
                 H0: tau_k = 0 rejects at alpha and tau_hat_k > 0.
  * `tau=c`   -- the cost-aware rule: launch iff the two-sided test of
                 H0: tau_k = c_launch rejects and tau_hat_k > c_launch.

Grid: K = 200, pi_L = 0.20, c_launch in {0, 0.01, 0.02} x f_ant in {0, 0.10},
R = 200.

Output: results/tables/portfolio_sensitivity.md / .csv

Usage:
    python scripts/run_portfolio_sensitivity.py [--R 200] [--K 200]
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
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")

MASTER_SEED = 42
SEED_TAG = 270          # distinct from every core-grid DGP id
N_WORKERS = 8

# DGP 6 defaults (src/utils/config.py DGP6_DEFAULTS)
N_MIN, N_MAX = 500, 5_000
PI_0 = 0.5              # fraction of true nulls
SIGMA_TAU = 0.10
ALPHA_S, BETA_SX, SIGMA_S = 5.0, 1.0, 2.0
ALPHA_Y, BETA_YS, BETA_YX, SIGMA_Y = 0.0, 0.5, 0.2, 1.0
BETA_YS_ANT = -0.3      # DGP 9's antagonistic outcome loading
ALPHA_DECISION = 0.05

PI_L = 0.20
C_LAUNCH_GRID = [0.0, 0.01, 0.02]
F_ANT_GRID = [0.0, 0.10]

METHODS: List[Tuple[int, str, Dict[str, Any]]] = [
    (0, "Labeled-Only", {}),
    (1, "Naive Surrogate", {}),
    (2, "Surrogate Index", {}),
    (3, "PPI++ (implemented rule)", {"corrected_variance": True}),
    (4, "GREG", {}),
    (9, "PPI++ (exact rule)", {}),
]
METHOD_NAMES = [name for _, name, _ in METHODS]
RULES = ["tau=0", "tau=c"]

Z_CRIT = stats.norm.ppf(1.0 - ALPHA_DECISION / 2.0)


def build_portfolio(K: int, f_ant: float, seed: int):
    """Generate one portfolio of K experiments; returns (experiments, taus)."""
    rng = np.random.default_rng(seed)

    n_ks = rng.integers(N_MIN, N_MAX + 1, size=K)
    is_null = rng.random(K) < PI_0
    tau_ks = np.where(is_null, 0.0, rng.normal(0, SIGMA_TAU, size=K))
    antagonistic = rng.random(K) < f_ant

    experiments = []
    for k in range(K):
        n_k = int(n_ks[k])
        tau_k = float(tau_ks[k])

        if antagonistic[k]:
            # Sign-reversing: mediated component = -tau_k, direct = 2*tau_k.
            b_ys = BETA_YS_ANT
            gamma_k = -tau_k / b_ys
            delta_k = 2.0 * tau_k
        else:
            b_ys = BETA_YS
            gamma_k = tau_k / b_ys
            delta_k = 0.0

        exp_rng = np.random.default_rng(seed * 100_000 + k + 1)
        X_k = exp_rng.standard_normal((n_k, 1))
        T_k = exp_rng.binomial(1, 0.5, size=n_k).astype(np.int32)
        S_k = (ALPHA_S + BETA_SX * X_k[:, 0] + gamma_k * T_k
               + exp_rng.normal(0, SIGMA_S, size=n_k))
        Y_k = (ALPHA_Y + b_ys * S_k + BETA_YX * X_k[:, 0] + delta_k * T_k
               + exp_rng.normal(0, SIGMA_Y, size=n_k))
        n_L = int(np.floor(PI_L * n_k))
        lm_k = np.zeros(n_k, dtype=bool)
        lm_k[exp_rng.permutation(n_k)[:n_L]] = True

        experiments.append((T_k, X_k, S_k, Y_k, lm_k))
    return experiments, tau_ks, antagonistic


def run_one(args: Tuple) -> List[Dict[str, Any]]:
    rep, K, f_ant = args
    seed = derive_seed(MASTER_SEED, SEED_TAG, int(round(f_ant * 100)), K, rep)
    experiments, tau_ks, antagonistic = build_portfolio(K, f_ant, seed)

    # (method, rule, c) -> launch decisions
    tau_hats = {name: np.zeros(K) for name in METHOD_NAMES}
    ses = {name: np.full(K, np.inf) for name in METHOD_NAMES}

    for k, (T_k, X_k, S_k, Y_k, lm_k) in enumerate(experiments):
        Y_hat_k, design = train_prediction_model(
            S_k, X_k, Y_k, lm_k, n_folds=5,
            rng=np.random.default_rng(seed * 10_000 + k + 8888),
            protocol=PROTOCOL, return_design=True,
        )
        for m_id, name, kw in METHODS:
            mkw = dict(kw)
            if m_id in (2, 5):
                mkw["design"] = design
            r = estimate(m_id, T_k, S_k, Y_k, Y_hat_k, lm_k,
                         protocol=PROTOCOL, **mkw)
            th, vh = r["tau_hat"], r["var_hat"]
            if np.isfinite(th) and np.isfinite(vh) and vh > 0:
                tau_hats[name][k] = th
                ses[name][k] = np.sqrt(vh)

    rows = []
    for c in C_LAUNCH_GRID:
        net = tau_ks - c
        oracle_gain = float(np.sum(np.maximum(net, 0.0)))
        for name in METHOD_NAMES:
            th, se = tau_hats[name], ses[name]
            for rule in RULES:
                h0 = 0.0 if rule == "tau=0" else c
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = np.where(np.isfinite(se) & (se > 1e-15),
                                 (th - h0) / se, 0.0)
                launch = (np.abs(z) > Z_CRIT) & (th > h0)
                realized = float(np.sum(net[launch]))
                regret = oracle_gain - realized
                rows.append(dict(
                    rep=rep, K=K, f_ant=f_ant, c_launch=c, rule=rule,
                    method=name, regret=regret, realized=realized,
                    oracle_gain=oracle_gain,
                    n_launched=int(launch.sum()),
                    n_launched_harmful=int((launch & (net < 0)).sum()),
                    n_missed_good=int((~launch & (net > 0)).sum()),
                    n_antagonistic=int(antagonistic.sum()),
                ))
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (K, f_ant, c, rule, method), g in raw.groupby(
            ["K", "f_ant", "c_launch", "rule", "method"]):
        R = len(g)
        og = float(g.oracle_gain.mean())
        mr = float(g.regret.mean())
        rows.append(dict(
            K=int(K), f_ant=float(f_ant), c_launch=float(c), rule=rule,
            method=method, R=R, oracle_gain=og,
            mean_regret=mr,
            mc_se_regret=float(g.regret.std(ddof=1) / np.sqrt(R)),
            rel_regret_pct=(100.0 * mr / og) if og > 1e-12 else np.nan,
            mean_launched=float(g.n_launched.mean()),
            mean_launched_harmful=float(g.n_launched_harmful.mean()),
            mean_missed_good=float(g.n_missed_good.mean()),
        ))
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHOD_NAMES)}
    out["_o"] = out.method.map(order)
    return (out.sort_values(["rule", "f_ant", "c_launch", "_o"])
              .drop(columns="_o").reset_index(drop=True))


def write_markdown(summ: pd.DataFrame, path: str) -> None:
    R = int(summ.R.max())
    K = int(summ.K.max())
    lines = [
        "# Portfolio loss-function sensitivity (DGP 6 with launch costs and "
        "antagonistic experiments)\n\n",
        f"K = {K} experiments per portfolio, pi_L = {PI_L}, R = {R} "
        f"portfolios, launch test at alpha = {ALPHA_DECISION}.\n\n"
        "Payoff of a launched experiment is tau_k - c_launch, so the oracle "
        "launches exactly when tau_k > c_launch and\n\n"
        "    regret = sum_k [ max(tau_k - c, 0) - d_k (tau_k - c) ].\n\n"
        "At c = 0 this is the paper's current regret. `f_ant` is the fraction "
        "of antagonistic (sign-reversing, DGP 9-style) experiments in the "
        "portfolio: the surrogate responds positively to treatment but Y "
        "loads negatively on S, so the mediated component has the wrong "
        "sign. Decision rules: `tau=0` is the paper's cost-blind rule "
        "(reject H0: tau_k = 0, launch if the estimate is positive); "
        "`tau=c` is the cost-aware rule (reject H0: tau_k = c_launch). "
        "Parenthesised figures are Monte Carlo standard errors.\n\n",
    ]
    for rule in RULES:
        lines.append(f"## Decision rule: {rule}\n\n")
        for f_ant in sorted(summ.f_ant.unique()):
            lines.append(f"### Antagonistic fraction f_ant = {f_ant:.2f}\n\n")
            lines.append(
                "| c_launch | Oracle gain | Method | Mean regret | MC SE | "
                "Rel. regret (%) | Mean launched | Harmful launched | "
                "Good missed |\n"
                "|---:|---:|---|---:|---:|---:|---:|---:|---:|\n"
            )
            sub = summ[(summ.rule == rule) & (summ.f_ant == f_ant)]
            for _, r in sub.iterrows():
                lines.append(
                    f"| {r.c_launch:.2f} | {r.oracle_gain:.3f} | {r.method} "
                    f"| {r.mean_regret:.4f} | {r.mc_se_regret:.4f} "
                    f"| {r.rel_regret_pct:.1f} | {r.mean_launched:.1f} "
                    f"| {r.mean_launched_harmful:.2f} "
                    f"| {r.mean_missed_good:.1f} |\n"
                )
            lines.append("\n")
    with open(path, "w") as f:
        f.writelines(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--K", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    tasks = [(rep, args.K, f) for f in F_ANT_GRID for rep in range(args.R)]
    print(f"{len(tasks)} portfolios (K={args.K}, R={args.R}, "
          f"f_ant in {F_ANT_GRID})", flush=True)
    with multiprocessing.Pool(N_WORKERS) as pool:
        nested = pool.map(run_one, tasks, chunksize=1)
    raw = pd.DataFrame([r for b in nested for r in b])
    summ = summarize(raw)

    os.makedirs(TABLES_DIR, exist_ok=True)
    raw.to_csv(os.path.join(TABLES_DIR, "portfolio_sensitivity_raw.csv"),
               index=False)
    summ.to_csv(os.path.join(TABLES_DIR, "portfolio_sensitivity.csv"),
                index=False)
    write_table_rows(
        os.path.join(TABLES_DIR, "portfolio_sensitivity_rows.json"), summ,
        dgp=6,
        analysis="portfolio_sensitivity",
        protocol=PROTOCOL,
        label_col="method",
        config_cols=("K", "f_ant", "c_launch", "rule"),
        R_col="R",
        generated_by="scripts/run_portfolio_sensitivity.py",
    )
    write_markdown(summ, os.path.join(TABLES_DIR,
                                      "portfolio_sensitivity.md"))

    cols = ["rule", "f_ant", "c_launch", "method", "mean_regret",
            "mc_se_regret", "rel_regret_pct", "mean_launched",
            "mean_launched_harmful"]
    print("\n" + summ[cols].to_string(index=False))
    print(f"\nWrote results/tables/portfolio_sensitivity.{{md,csv}} "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
