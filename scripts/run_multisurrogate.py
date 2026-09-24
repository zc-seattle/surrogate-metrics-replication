#!/usr/bin/env python3
"""
Multi-dimensional surrogate on the Criteo-calibrated semi-synthetic funnel.

Question: how much of a surrogacy violation does a richer
surrogate index absorb? The single-binary-visit testbed makes the violation
maximally hidden; real platforms track additional engagement signals that
may mediate part of the direct effect.

Construction: start from the fitted Criteo funnel (run_semisynthetic.py
params). Add a post-treatment engagement proxy
    S2 = zeta * T + N(0, 1),        zeta = 0.5,
and split the observed conversion-stage violation kappa_hat into a part
mediated by S2 and a residual direct effect:
    conversion index = X @ a_c + c_c + theta * S2 + kappa_resid * T,
    theta = m * kappa_hat / zeta,   kappa_resid = (1 - m) * kappa_hat,
so the total treatment coefficient on the conversion index is kappa_hat for
every mediation share m in {0, 0.5, 1}. At m = 1 the rich index
(visit, S2) satisfies surrogacy exactly; the poor index (visit only) never
does.

For each m and each index version (poor = visit; rich = visit + S2):
LO / SI / PPI++ and the two-sided diagnostic, standard protocol
(n = 64,000, pi_L = 0.20, OLS prediction with 5-fold cross-fitting).

True tau per m computed by Gauss-Hermite quadrature over S2 on the
covariate pool.

Output (in --outdir, default results/tables):
  multisurrogate_raw.csv    one row per (replication, m, index, method); for
                            method DIAG, `bias` holds T_n and `covers` the
                            two-sided rejection flag at alpha = 0.05
  multisurrogate_rows.json  summary result rows, one per (m, index, method)
  multisurrogate.md         the table
`--from-raw` rebuilds the rows and the table from an existing raw CSV.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.config import make_result_row, spec_by_key
from src.utils.registry import metrics_from_errors, write_rows
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



# The covariate pool is a regenerable intermediate written by
# scripts/run_semisynthetic.py into the (git-ignored) cache directory.
POOL_PATH = os.path.join(PROJECT_ROOT, "results", "cache", "semisynth_pool.npz")
PARAM_PATH = os.path.join(PROJECT_ROOT, "results", "tables", "semisynth_params.json")

ZETA = 0.5
N = 64_000
PI_L = 0.20
M_GRID = [0.0, 0.5, 1.0]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def true_tau(params, X_pool, m):
    """tau by averaging fitted surfaces; GH quadrature over S2 ~ N(zeta*T, 1)."""
    av = np.asarray(params["visit_coef"][:-1]); bv = params["visit_coef"][-1]
    cv = params["visit_intercept"]
    ac = np.asarray(params["conv_coef"][:-1]); cc = params["conv_intercept"]
    kap = params["kappa_hat"]
    theta = m * kap / ZETA
    kr = (1 - m) * kap

    nodes, weights = np.polynomial.hermite_e.hermegauss(31)  # N(0,1) quadrature
    weights = weights / weights.sum()

    lin_v = X_pool @ av + cv
    lin_c = X_pool @ ac + cc
    # E[Y | T, X] = P(visit | T, X) * E_{S2}[ sigmoid(lin_c + theta*S2 + kr*T) ]
    def mean_conv(t):
        s2 = ZETA * t + nodes  # quadrature points of N(zeta*t, 1)
        p = np.zeros_like(lin_c)
        for s2k, wk in zip(s2, weights):
            p += wk * sigmoid(lin_c + theta * s2k + kr * t)
        return p
    m1 = sigmoid(lin_v + bv) * mean_conv(1.0)
    m0 = sigmoid(lin_v) * mean_conv(0.0)
    return float(np.mean(m1 - m0))


def run_one(args):
    rep, m, params, tau = args
    rng = np.random.default_rng(660_000 + rep * 10 + int(m * 4))

    X_pool = np.load(POOL_PATH)["X"].astype(np.float64)
    X = X_pool[rng.choice(len(X_pool), size=N, replace=True)]

    av = np.asarray(params["visit_coef"][:-1]); bv = params["visit_coef"][-1]
    cv = params["visit_intercept"]
    ac = np.asarray(params["conv_coef"][:-1]); cc = params["conv_intercept"]
    kap = params["kappa_hat"]
    theta = m * kap / ZETA
    kr = (1 - m) * kap

    T = (rng.random(N) < 0.5).astype(np.float64)
    S1 = (rng.random(N) < sigmoid(X @ av + cv + bv * T)).astype(np.float64)
    S2 = ZETA * T + rng.standard_normal(N)
    pY = sigmoid(X @ ac + cc + theta * S2 + kr * T)
    Y = np.where(S1 == 1, (rng.random(N) < pY).astype(np.float64), 0.0)

    lm = np.zeros(N, dtype=bool)
    lm[rng.choice(N, size=int(PI_L * N), replace=False)] = True

    out = []
    for version, S_use in [("poor", S1), ("rich", np.column_stack([S1, S2]))]:
        Yh, design = train_prediction_model(S_use, X, Y, lm, n_folds=5,
                                    rng=np.random.default_rng(rep + 7777),
            protocol=PROTOCOL, return_design=True,
        )
        res = {}
        for m_id, name in [(0, "LO"), (2, "SI"), (3, "PPI++")]:
            kw = {"design": design} if m_id in (2, 5) else {}
            r = estimate(m_id, T, S_use, Y, Yh, lm, protocol=PROTOCOL, **kw)
            res[m_id] = r
            out.append(dict(rep=rep, m=m, version=version, method=name,
                            true_tau=tau, bias=r["tau_hat"] - tau,
                            covers=int(r["ci_lower"] <= tau <= r["ci_upper"])))
        cov_sp = estimate_cov_si_ppi(T, Y, Yh, lm,
                                     res[3].get("lambda_hat", 0.0), design=design)
        t2 = surrogacy_test(res[2]["tau_hat"], res[3]["tau_hat"],
                            res[2]["var_hat"], res[3]["var_hat"],
                            cov_sp, alternative="two-sided")
        out.append(dict(rep=rep, m=m, version=version, method="DIAG",
                        true_tau=tau, bias=t2["T_n"], covers=int(t2["p_value"] < 0.05)))
    return out


METHOD_KEYS = {"LO": "lo", "SI": "si", "PPI++": "ppi"}
DIAG_LABEL = "SI--PPI++ diagnostic (two-sided)"


def summary_rows(df: pd.DataFrame, taus, protocol: str,
                 seed: int = 42) -> list:
    """One result row per (m, index version, method) plus the diagnostic.

    Estimator rows: bias, relative bias against the cell's true tau, RMSE,
    coverage, RE = RMSE(LO) / RMSE(method), MC SEs.  Diagnostic row: the
    two-sided rejection rate at alpha = 0.05 and the mean T_n.
    """
    rows = []
    for m in M_GRID:
        for version in ("poor", "rich"):
            sub = df[(df.m == m) & (df.version == version)]
            if sub.empty:
                continue
            common = dict(dgp="multisurrogate",
                          config_name=f"m={m}; index={version}",
                          params={"m": m, "version": version, "zeta": ZETA},
                          pi_L=PI_L, n=N, protocol=protocol, seed=seed,
                          target="ATE")
            extra = {"analysis": "multisurrogate", "m": m,
                     "version": version}
            lo = sub[sub.method == "LO"]
            lo_rmse = float(np.sqrt((lo.bias ** 2).mean()))
            for meth, key in METHOD_KEYS.items():
                ms = sub[sub.method == meth]
                if ms.empty:
                    continue
                rows.append(make_result_row(
                    spec=spec_by_key(key), R=ms.rep.nunique(), **common,
                    metrics=metrics_from_errors(
                        ms.bias.to_numpy(float), true_tau=taus[m],
                        covers=ms.covers.to_numpy(float),
                        rmse_baseline=lo_rmse),
                    extra={"method_key": key, "table_label": meth, **extra},
                ))
            dg = sub[sub.method == "DIAG"]
            if not dg.empty:
                rows.append(make_result_row(
                    method_id=-1, method_label=DIAG_LABEL,
                    R=dg.rep.nunique(), **common,
                    metrics=metrics_from_errors(
                        None, true_tau=taus[m],
                        reject=dg.covers.to_numpy(float)),
                    extra={"method_key": "diagnostic",
                           "mean_T_n": float(dg.bias.mean()), **extra},
                ))
    return rows


def build_markdown(df: pd.DataFrame, taus, kappa_hat: float, R: int) -> list:
    lines = ["# Multi-dimensional surrogate on the Criteo-calibrated funnel\n",
             f"n = {N}, pi_L = {PI_L}, R = {R}, zeta = {ZETA}, "
             f"kappa_hat = {kappa_hat:.4f} (total violation fixed "
             "at dial 1 for every mediation share m)\n\n",
             "| m | true tau | Index | Method | Bias | RelBias% | Coverage |\n"
             "|---|---|---|---|---|---|---|\n"]
    for m in M_GRID:
        for version in ("poor", "rich"):
            sub = df[(df.m == m) & (df.version == version)]
            for meth in ("LO", "SI", "PPI++"):
                ms = sub[sub.method == meth]
                lines.append(
                    f"| {m} | {taus[m]:.6f} | {version} | {meth} "
                    f"| {ms.bias.mean():+.6f} "
                    f"| {100 * ms.bias.mean() / taus[m]:+.1f} "
                    f"| {ms.covers.mean():.3f} |\n")
            dg = sub[sub.method == "DIAG"]
            lines.append(f"| {m} | | {version} | DIAG rejection "
                         f"| {dg.covers.mean():.3f} | mean T_n = "
                         f"{dg.bias.mean():.2f} | |\n")
    return lines


def _taus_from(df: pd.DataFrame, params) -> dict:
    """True tau per m: from the raw file when recorded, else recomputed from
    the fitted parameters and the covariate pool (deterministic)."""
    if "true_tau" in df.columns and df["true_tau"].notna().all():
        return {m: float(df.loc[df.m == m, "true_tau"].iloc[0])
                for m in M_GRID if (df.m == m).any()}
    X_pool = np.load(POOL_PATH)["X"].astype(np.float64)
    return {m: true_tau(params, X_pool, m) for m in M_GRID}


def write_outputs(df, outdir, taus, params, R, protocol, write_raw=True):
    os.makedirs(outdir, exist_ok=True)
    if write_raw:
        df.to_csv(os.path.join(outdir, "multisurrogate_raw.csv"), index=False)
    write_rows(
        os.path.join(outdir, "multisurrogate_rows.json"),
        summary_rows(df, taus, protocol),
        generated_by="scripts/run_multisurrogate.py", protocol=protocol,
    )
    with open(os.path.join(outdir, "multisurrogate.md"), "w") as f:
        f.writelines(build_markdown(df, taus, params["kappa_hat"], R))


def main():
    import argparse

    global PROTOCOL
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--R", type=int, default=500,
                     help="Monte Carlo replications (smoke runs use a few)")
    _ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    _ap.add_argument("--outdir",
                     default=os.path.join(PROJECT_ROOT, "results", "tables"))
    _ap.add_argument("--from-raw", metavar="CSV", default=None,
                     help="rebuild rows + md from an existing raw CSV; "
                          "no simulation")
    _args = _ap.parse_args()
    PROTOCOL = _args.protocol

    with open(PARAM_PATH) as f:
        params = json.load(f)

    if _args.from_raw:
        df = pd.read_csv(_args.from_raw, float_precision="round_trip")
        R = int(df.groupby(["m", "version", "method"])["rep"].nunique().max())
        write_outputs(df, _args.outdir, _taus_from(df, params), params, R,
                      PROTOCOL, write_raw=False)
        return

    t0 = time.time()
    R = _args.R
    X_pool = np.load(POOL_PATH)["X"].astype(np.float64)
    taus = {m: true_tau(params, X_pool, m) for m in M_GRID}
    print("true tau by mediation share:",
          {m: round(t, 6) for m, t in taus.items()}, flush=True)

    tasks = [(rep, m, params, taus[m]) for m in M_GRID for rep in range(R)]
    print(f"{len(tasks)} tasks", flush=True)
    with multiprocessing.Pool(8, initializer=_set_protocol,
                              initargs=(PROTOCOL,)) as pool:
        nested = pool.map(run_one, tasks)
    df = pd.DataFrame([r for b in nested for r in b])
    write_outputs(df, _args.outdir, taus, params, R, PROTOCOL)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
