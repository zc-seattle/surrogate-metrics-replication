#!/usr/bin/env python3
"""
Calendar-time review-date experiment.

This script runs the worked example explicitly.

Worked example
--------------
  * Enrollment is uniform over a 4-week window; eventual sample n = 10,000,
    so e_i ~ U(0, 4) weeks.
  * The surrogate resolves immediately; the primary outcome needs a 2-week
    maturation window.
  * DGP 1 outcome model (valid surrogacy, no drift): with the package
    defaults tau = beta_YS * gamma_S = 0.5 * 0.3 = 0.15.

At review date t (weeks):
  * enrolled  = {i : e_i <= t},  so n(t) = min(t, 4)/4 * 10,000
  * labeled   = {i : e_i + 2 <= t}  (matured outcomes)
  * pi_L(t)   = (t - 2)/t      for t <= 4
              = (t - 2)/4      for 4 <= t <= 6

The same population draw is reused across review dates within a replication,
so the curves are paired.

Estimators
----------
  LO               method 0
  SI               method 2
  PPI++ (impl.)    method 3 with corrected_variance=True -- the plug-in
                   tuning rule the paper implements, with the corrected
                   variance
  PPI++ (exact)    method 9 -- exact-variance tuning rule + corrected
                   variance

Reported per (t, estimator): mean 95% CI width, coverage of the
eventual-sample ATE tau = 0.15, and power to reject H0: tau = 0 at 5%.
Also: the earliest review date at which each estimator's mean CI width
matches labeled-only's width at t = 6 (full maturation), by linear
interpolation on the width curve.

Output: results/tables/calendar_review.md / .csv

Usage:
    python scripts/run_calendar_review.py [--R 1000]
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
SEED_TAG = 260          # distinct from every core-grid DGP id
N_WORKERS = 8

N_EVENTUAL = 10_000
ENROLL_WEEKS = 4.0
OUTCOME_WINDOW = 2.0
REVIEW_DATES = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]

# DGP 1 defaults (src/utils/config.py DGP1_DEFAULTS)
ALPHA_S, BETA_SX, GAMMA_S, SIGMA_S = 5.0, 1.0, 0.3, 2.0
ALPHA_Y, BETA_YS, BETA_YX, SIGMA_Y = 0.0, 0.5, 0.2, 1.0
TRUE_TAU = BETA_YS * GAMMA_S        # 0.15

Z95 = stats.norm.ppf(0.975)

# Method id 9 is an alias of id 3 in its default configuration, so the two
# PPI++ rows are the primary (exact rule) and the plug-in-rule ablation.
ESTIMATORS = [
    (0, "LO", {}),
    (2, "SI", {}),
    (3, "PPI++", {}),
    (3, "PPI++ (plug-in lambda)", {"lambda_rule": "plugin"}),
]
ESTIMATOR_NAMES = [name for _, name, _ in ESTIMATORS]


def pi_L_of_t(t: float) -> float:
    """Labeled fraction among the ENROLLED units at review date t."""
    if t <= OUTCOME_WINDOW:
        return 0.0
    if t <= ENROLL_WEEKS:
        return (t - OUTCOME_WINDOW) / t
    return (t - OUTCOME_WINDOW) / ENROLL_WEEKS


def run_one(rep: int) -> List[Dict[str, Any]]:
    """One replication: one population, evaluated at every review date."""
    seed = derive_seed(MASTER_SEED, SEED_TAG, 0, 0, rep)
    rng = np.random.default_rng(seed)

    # Eventual population (DGP 1 outcome model, no drift)
    e = rng.uniform(0.0, ENROLL_WEEKS, size=N_EVENTUAL)   # enrollment time
    X = rng.standard_normal((N_EVENTUAL, 1))
    T = rng.binomial(1, 0.5, size=N_EVENTUAL).astype(np.int32)
    S = (ALPHA_S + BETA_SX * X[:, 0] + GAMMA_S * T
         + rng.normal(0, SIGMA_S, size=N_EVENTUAL))
    Y = (ALPHA_Y + BETA_YS * S + BETA_YX * X[:, 0]
         + rng.normal(0, SIGMA_Y, size=N_EVENTUAL))

    rows: List[Dict[str, Any]] = []
    for t in REVIEW_DATES:
        enrolled = e <= t
        matured = (e + OUTCOME_WINDOW) <= t
        n_t = int(enrolled.sum())
        n_L = int(matured.sum())
        if n_L < 20 or n_t - n_L < 0:
            continue

        T_t, X_t, S_t, Y_t = T[enrolled], X[enrolled], S[enrolled], Y[enrolled]
        lm_t = matured[enrolled]

        Y_hat, design = train_prediction_model(
            S_t, X_t, Y_t, lm_t, n_folds=5,
            rng=np.random.default_rng(seed + 7777),
            protocol=PROTOCOL, return_design=True,
        )

        for m_id, name, kw in ESTIMATORS:
            mkw = dict(kw)
            if m_id in (2, 5):
                mkw["design"] = design
            r = estimate(m_id, T_t, S_t, Y_t, Y_hat, lm_t,
                         protocol=PROTOCOL, **mkw)
            se = np.sqrt(max(r["var_hat"], 0.0))
            rows.append(dict(
                rep=rep, t=t, n_t=n_t, n_L=n_L,
                pi_L_realized=n_L / n_t, method=name,
                tau_hat=r["tau_hat"],
                width=r["ci_upper"] - r["ci_lower"],
                covers=int(r["ci_lower"] <= TRUE_TAU <= r["ci_upper"]),
                reject0=int(se > 0 and abs(r["tau_hat"] / se) > Z95),
            ))
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (t, method), g in raw.groupby(["t", "method"]):
        R = len(g)
        cov = float(g.covers.mean())
        pw = float(g.reject0.mean())
        rows.append(dict(
            t=float(t), method=method, R=R,
            n_t=int(g.n_t.mean().round()), n_L=int(g.n_L.mean().round()),
            pi_L_formula=pi_L_of_t(float(t)),
            pi_L_realized=float(g.pi_L_realized.mean()),
            mean_width=float(g.width.mean()),
            mc_se_width=float(g.width.std(ddof=1) / np.sqrt(R)),
            coverage=cov, mc_se_coverage=float(np.sqrt(cov * (1 - cov) / R)),
            power=pw, mc_se_power=float(np.sqrt(pw * (1 - pw) / R)),
        ))
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(ESTIMATOR_NAMES)}
    out["_o"] = out.method.map(order)
    return out.sort_values(["t", "_o"]).drop(columns="_o").reset_index(drop=True)


def earliest_t_matching(summ: pd.DataFrame, target_width: float
                        ) -> Dict[str, Optional[float]]:
    """Earliest review date at which mean CI width <= target_width.

    Linear interpolation on the (t, mean width) curve. Returns None if the
    estimator never reaches the target on the review-date grid.
    """
    out: Dict[str, Optional[float]] = {}
    for method in ESTIMATOR_NAMES:
        sub = summ[summ.method == method].sort_values("t")
        ts = sub.t.to_numpy(dtype=float)
        ws = sub.mean_width.to_numpy(dtype=float)
        hit = None
        if len(ws) and ws[0] <= target_width:
            hit = float(ts[0])
        else:
            for i in range(len(ts) - 1):
                if ws[i] > target_width >= ws[i + 1]:
                    frac = (ws[i] - target_width) / (ws[i] - ws[i + 1])
                    hit = float(ts[i] + frac * (ts[i + 1] - ts[i]))
                    break
        out[method] = hit
    return out


def write_markdown(summ: pd.DataFrame, matching: Dict[str, Optional[float]],
                   target_width: float, path: str) -> None:
    R = int(summ.R.max())
    lines = [
        "# Calendar-time review-date experiment\n\n",
        f"Uniform enrollment over a {ENROLL_WEEKS:.0f}-week window, eventual "
        f"n = {N_EVENTUAL:,}; {OUTCOME_WINDOW:.0f}-week outcome maturation "
        "window; surrogate resolves immediately; DGP 1 outcome model (valid "
        f"surrogacy, no drift), true tau = {TRUE_TAU:.2f}. R = {R}.\n\n"
        "At review date t: enrolled = {i : e_i <= t}; labeled = matured = "
        "{i : e_i + 2 <= t}; pi_L(t) = (t-2)/t for t <= 4 and (t-2)/4 for "
        "4 <= t <= 6. Coverage is of the eventual-sample ATE tau = 0.15; "
        "power is the rejection rate of H0: tau = 0 at the 5% level. "
        "Widths are 95% CI widths. `PPI++` is the primary configuration "
        "(exact-variance tuning rule, common unclipped coefficient, exact "
        "variance); `PPI++ (plug-in lambda)` is the plug-in tuning rule "
        "on the same replications. SI uses the joint sandwich variance for "
        "the learned index.\n\n",
        "| t (weeks) | n(t) | n_L(t) | pi_L(t) | Method | Mean CI width | "
        "MC SE | Coverage | MC SE | Power vs tau=0 | MC SE |\n"
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in summ.iterrows():
        lines.append(
            f"| {r.t:.1f} | {r.n_t:,} | {r.n_L:,} | {r.pi_L_formula:.3f} "
            f"| {r.method} | {r.mean_width:.5f} | {r.mc_se_width:.5f} "
            f"| {r.coverage:.3f} | {r.mc_se_coverage:.4f} "
            f"| {r.power:.3f} | {r.mc_se_power:.4f} |\n"
        )
    lines.append(
        f"\n## Weeks saved: earliest review date matching LO at t = 6\n\n"
        f"Labeled-only mean CI width at t = 6 (full maturation, "
        f"pi_L = 1.00) is {target_width:.5f}. The table gives the earliest "
        "review date at which each estimator's mean CI width is at least as "
        "narrow, by linear interpolation on the width curve.\n\n"
        "| Method | Earliest t (weeks) | Weeks saved vs t = 6 |\n"
        "|---|---:|---:|\n"
    )
    for method in ESTIMATOR_NAMES:
        hit = matching[method]
        if hit is None:
            lines.append(f"| {method} | never on grid | --- |\n")
        else:
            lines.append(f"| {method} | {hit:.2f} | {6.0 - hit:.2f} |\n")
    with open(path, "w") as f:
        f.writelines(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=1000)
    args = ap.parse_args()

    t0 = time.time()
    print(f"{args.R} replications x {len(REVIEW_DATES)} review dates",
          flush=True)
    with multiprocessing.Pool(N_WORKERS) as pool:
        nested = pool.map(run_one, range(args.R), chunksize=5)
    raw = pd.DataFrame([r for b in nested for r in b])

    summ = summarize(raw)

    lo6 = summ[(summ.method == "LO") & (np.isclose(summ.t, 6.0))]
    target_width = float(lo6.mean_width.iloc[0])
    matching = earliest_t_matching(summ, target_width)

    os.makedirs(TABLES_DIR, exist_ok=True)
    raw.to_csv(os.path.join(TABLES_DIR, "calendar_review_raw.csv"), index=False)
    summ.to_csv(os.path.join(TABLES_DIR, "calendar_review.csv"), index=False)
    write_table_rows(
        os.path.join(TABLES_DIR, "calendar_review_rows.json"), summ,
        dgp=11,
        analysis="calendar_review",
        protocol=PROTOCOL,
        label_col="method",
        config_cols=("t",),
        pi_L_col="pi_L_realized",
        n_col="n_t",
        R_col="R",
        generated_by="scripts/run_calendar_review.py",
    )
    write_markdown(summ, matching, target_width,
                   os.path.join(TABLES_DIR, "calendar_review.md"))

    print("\n" + summ[["t", "n_t", "n_L", "pi_L_formula", "method",
                       "mean_width", "coverage", "power"]].to_string(index=False))
    print(f"\nLO mean CI width at t=6: {target_width:.5f}")
    for method in ESTIMATOR_NAMES:
        hit = matching[method]
        s = "never on grid" if hit is None else f"{hit:.2f} weeks"
        print(f"  earliest t matching that width -- {method}: {s}")
    print(f"\nWrote results/tables/calendar_review.{{md,csv}} "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
