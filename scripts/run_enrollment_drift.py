#!/usr/bin/env python3
"""
DGP 11: enrollment-time labeling under drift.

The maturation-curve framing labels the earliest-enrolled cohort, not a
random subset. This script tests that labeling mechanism directly:

  e_i ~ U(0, 1)                       enrollment time
  labeled  <=>  e_i <= pi_L quantile  (earliest-enrolled cohort)

Two drift channels, dialed separately:
  (a) mix drift: X_i = d * (e_i - 0.5) + N(0,1); the covariate mix of
      early enrollees differs from the final population. The conditional
      outcome model Y | S, X is stable.
  (b) effect drift: a direct treatment effect on Y of size g_e * e grows
      over the enrollment window (a channel bypassing S), so the labeled
      (early) cohort's effect differs from the full-sample ATE and no
      conditional-on-(S, X) model can recover the difference. The same
      DGP under a random (MCAR) split is included as the contrast: there
      the drift hurts only SI, not LO or PPI++.

Outcome model:
  S = 5 + X + gamma_i * T + N(0, 2),   gamma_i = 0.5 + g_x * X
  Y = 0.3 * S + 0.2 * X + g_e * e * T + N(0, 1)
  true tau = 0.3 * mean(gamma_i) + g_e * mean(e)   (realized-sample ATE)

Surrogacy holds iff g_e = 0; mix drift alone (d > 0) preserves it.
Estimators (primary): LO (0), SI (2, joint sandwich variance), PPI++ (3,
exact tuning rule, common unclipped coefficient, exact variance), AIPW (6,
logistic propensity in S).

Ablation, run on the same draws and the same fitted prediction model in every
configuration: PPI++ under the plug-in tuning rule with the exact variance,
and the whole plug-in-rule estimator (plug-in rule, clipped, plug-in variance).  Mix
drift is where the tuning rule matters, so the comparison is paired cell by
cell rather than run separately.

Output: results/tables/enrollment_drift.md / _raw.csv
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

from src.utils.config import (
    DEFAULT_PROTOCOL,
    PROTOCOLS,
    make_result_row,
    spec_by_label,
)
from src.utils.registry import write_rows
from src.utils.estimator_api import (
    estimate,
    train_prediction_model,
)


#: Prediction protocol; set from --protocol in main() and propagated to the
#: worker processes by `_set_protocol`.
PROTOCOL = DEFAULT_PROTOCOL

#: Base seed recorded in the result rows for this analysis.
SEED_BASE = 42


def _set_protocol(protocol: str) -> None:
    """Pool initializer: workers re-import this module under spawn."""
    global PROTOCOL
    PROTOCOL = protocol



N = 10_000
PI_L = 0.20

# (name, mix drift d, effect-in-X g_x, effect-in-time g_e, labeling)
CONFIGS = [
    ("MCAR baseline",              0.0, 0.3, 0.0, "mcar"),
    ("Enrollment, no drift",       0.0, 0.3, 0.0, "early"),
    ("Mix drift d=1",              1.0, 0.3, 0.0, "early"),
    ("Mix drift d=2",              2.0, 0.3, 0.0, "early"),
    ("Effect drift g_e=0.2, MCAR", 0.0, 0.3, 0.2, "mcar"),
    ("Effect drift g_e=0.2",       0.0, 0.3, 0.2, "early"),
    ("Effect drift g_e=0.4",       0.0, 0.3, 0.4, "early"),
    ("Mix + effect drift",         1.0, 0.3, 0.2, "early"),
]

#: (table name, method label, method id, estimator kwargs), in table order.
#: The first four are the primary configurations; the last two are the paired
#: PPI++ tuning-rule ablation.
METHODS = [
    ("LO", "Labeled-Only", 0, {}),
    ("SI", "SI", 2, {"design": True}),
    ("PPI++", "PPI++", 3, {}),
    ("AIPW", "AIPW", 6, {"propensity_model": "logistic"}),
    ("PPI++ (plug-in lambda)", "PPI++ (plug-in lambda)", 3,
     {"lambda_rule": "plugin", "clip": False, "per_arm": False,
      "variance": "exact"}),
    ("PPI++ (plug-in lambda, plug-in variance)",
     "PPI++ (plug-in lambda, plug-in variance)", 3,
     {"lambda_rule": "plugin", "clip": True, "per_arm": False,
      "variance": "plugin"}),
]

#: The names the markdown Method column prints, in order.
METHOD_NAMES = [m[0] for m in METHODS]


def run_one(args):
    rep, cfg_id = args
    name, d, g_x, g_e, labeling = CONFIGS[cfg_id]
    rng = np.random.default_rng(770_000 + rep * 10 + cfg_id)

    e = rng.uniform(0, 1, N)
    X = (d * (e - 0.5) + rng.standard_normal(N)).reshape(-1, 1)
    T = (rng.random(N) < 0.5).astype(np.float64)
    gamma = 0.5 + g_x * X[:, 0]
    S = 5.0 + X[:, 0] + gamma * T + rng.normal(0, 2.0, N)
    Y = (0.3 * S + 0.2 * X[:, 0] + g_e * e * T
         + rng.normal(0, 1.0, N))
    tau = 0.3 * float(gamma.mean()) + g_e * float(e.mean())

    n_L = int(PI_L * N)
    lm = np.zeros(N, dtype=bool)
    if labeling == "early":
        lm[np.argsort(e)[:n_L]] = True
    else:
        lm[rng.choice(N, size=n_L, replace=False)] = True

    Yh, design = train_prediction_model(S, X, Y, lm, n_folds=5,
                                rng=np.random.default_rng(rep + 7777),
        protocol=PROTOCOL, return_design=True,
    )
    out = []
    # Method 2 needs the design dict, or the SI interval silently falls back
    # to the plug-in variance.  Method id 9 is an alias of 3, so it is no
    # longer run as a separate estimator.  Every configuration, primary and
    # ablation, runs on this one draw and this one fitted prediction model.
    for mname, _label, m_id, kw in METHODS:
        kw = dict(kw)
        if kw.pop("design", False):
            kw["design"] = design
        r = estimate(m_id, T, S, Y, Yh, lm, protocol=PROTOCOL, **kw)
        out.append(dict(rep=rep, config=name, method=mname, tau=tau,
                        bias=r["tau_hat"] - tau,
                        covers=int(r["ci_lower"] <= tau <= r["ci_upper"])))
    return out


def main():
    import argparse

    global PROTOCOL
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--R", type=int, default=1000,
                     help="Monte Carlo replications (smoke runs use a few)")
    _ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    _args = _ap.parse_args()
    PROTOCOL = _args.protocol

    t0 = time.time()
    R = _args.R
    tasks = [(rep, c) for c in range(len(CONFIGS)) for rep in range(R)]
    print(f"{len(tasks)} tasks", flush=True)
    with multiprocessing.Pool(8, initializer=_set_protocol,
                              initargs=(PROTOCOL,)) as pool:
        nested = pool.map(run_one, tasks)
    df = pd.DataFrame([r for b in nested for r in b])
    outdir = os.path.join(PROJECT_ROOT, "results", "tables")
    df.to_csv(os.path.join(outdir, "enrollment_drift_raw.csv"), index=False)

    # Result rows, one per (config, method configuration).
    reg_rows = []
    for name, *_ in CONFIGS:
        sub = df[df.config == name]
        tau_mean = float(sub.tau.mean())
        lo_rmse = float(np.sqrt((sub[sub.method == "LO"].bias ** 2).mean()))
        for m, label, _m_id, _kw in METHODS:
            ms = sub[sub.method == m]
            if not len(ms):
                continue
            rmse = float(np.sqrt((ms.bias ** 2).mean()))
            bias = float(ms.bias.mean())
            cov = float(ms.covers.mean())
            n_valid = int(len(ms))
            reg_rows.append(make_result_row(
                dgp=11, config_name=name, params={"config": name},
                pi_L=PI_L, n=N, R=R, protocol=PROTOCOL,
                spec=spec_by_label(label),
                alpha=0.05, target="ATE", seed=SEED_BASE,
                metrics={
                    "true_tau": tau_mean, "n_valid": n_valid, "bias": bias,
                    "rel_bias": (100 * bias / tau_mean) if tau_mean else float("nan"),
                    "rmse": rmse, "coverage": cov,
                    "mc_se_coverage": float(np.sqrt(cov * (1 - cov) / n_valid)),
                    "mc_se_bias": float(ms.bias.std(ddof=1) / np.sqrt(n_valid)),
                    "relative_efficiency": (lo_rmse / rmse) if rmse else float("nan"),
                },
                extra={"analysis": "enrollment_drift"},
            ))
    write_rows(os.path.join(outdir, "enrollment_drift_rows.json"), reg_rows,
               generated_by="scripts/run_enrollment_drift.py",
               protocol=PROTOCOL)

    band = {2000: "[0.940, 0.960]", 1000: "[0.936, 0.964]",
            500: "[0.931, 0.969]", 200: "[0.920, 0.980]"}.get(R, "")
    lines = ["# DGP 11: enrollment-time labeling under drift\n",
             f"n = {N}, pi_L = {PI_L}, R = {R}. Labeled set = earliest-"
             "enrolled cohort except in MCAR configs.\n\n",
             "| Config | tau | Method | Bias | RelBias% | RMSE | Coverage "
             "| RE |\n|---|---|---|---|---|---|---|---|\n"]
    for name, *_ in CONFIGS:
        sub = df[df.config == name]
        tau_mean = sub.tau.mean()
        lo_rmse = np.sqrt((sub[sub.method == "LO"].bias ** 2).mean())
        for m in METHOD_NAMES:
            ms = sub[sub.method == m]
            if not len(ms):
                continue
            rmse = np.sqrt((ms.bias ** 2).mean())
            lines.append(
                f"| {name} | {tau_mean:.4f} | {m} | {ms.bias.mean():+.4f} "
                f"| {100 * ms.bias.mean() / tau_mean:+.1f} | {rmse:.4f} "
                f"| {ms.covers.mean():.3f} | {lo_rmse / rmse:.2f} |\n")
    lines.append(
        "\nNotes: the first four rows of each configuration are the primary "
        "configurations; `PPI++ (plug-in lambda)` (plug-in tuning rule, "
        "unclipped, exact variance) and `PPI++ (plug-in lambda, plug-in "
        "variance)` (the whole plug-in-rule estimator: plug-in rule, clipped, "
        "plug-in variance) are the paired PPI++ tuning-rule ablation, run on "
        "the same draws and the same fitted prediction model. RE is "
        "RMSE(Labeled-Only) / RMSE(method) in the same configuration."
        + (f" The Monte Carlo band for coverage at R = {R} is {band}.\n"
           if band else "\n"))
    with open(os.path.join(outdir, "enrollment_drift.md"), "w") as f:
        f.writelines(lines)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
