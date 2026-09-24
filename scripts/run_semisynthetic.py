#!/usr/bin/env python3
"""
Criteo-calibrated semi-synthetic funnel DGP (DGP-C).

Construction:
  1. Fit, on the real Criteo uplift data, a logistic visit model
     P(S=1 | X, T) and a logistic conversion-given-visit model
     P(Y=1 | S=1, X, T); the fitted T coefficient kappa_hat in the
     conversion model is the empirically observed within-funnel
     violation. Conversion without a visit has probability zero, as in
     the real data.
  2. Generate semi-synthetic experiments by resampling real covariate
     rows X, drawing T ~ Bernoulli(0.5), S from the fitted visit model,
     and Y (for visitors) from the fitted conversion model with the
     violation dialed: kappa = dial * kappa_hat,
     dial in {0, 0.5, 1, 1.5, 2}.
  3. True tau(dial) is computed exactly by averaging the fitted response
     surfaces over the covariate pool.
  4. For each dial: R replications of the standard protocol
     (n = 64,000, pi_L = 0.20, OLS prediction model with 5-fold
     cross-fitting), reporting LO / SI / PPI++ and the two-sided
     diagnostic.

Usage:
    python scripts/run_semisynthetic.py --data <criteo csv> [--fit-sub 2000000]
    python scripts/run_semisynthetic.py --from-raw results/tables/semisynthetic_raw.csv

Output (in --outdir, default results/tables):
  semisynthetic_raw.csv    one row per (replication, dial, method); for method
                           DIAG, `tau_hat` holds D_hat, `bias` holds T_n and
                           `covers` the two-sided rejection flag (alpha 0.05)
  semisynthetic_rows.json  summary result rows, one per (dial, method)
  semisynthetic.md         the table
"""

from __future__ import annotations

import argparse
import json
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

from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.config import default_spec_for_id, make_result_row
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



FEATURES = [f"f{i}" for i in range(12)]
# The covariate pool is a large regenerable intermediate, so it lives in the
# (git-ignored) cache directory rather than alongside the committed tables.
CACHE_DIR = os.path.join(PROJECT_ROOT, "results", "cache")
POOL_PATH = os.path.join(CACHE_DIR, "semisynth_pool.npz")
PARAM_PATH = os.path.join(PROJECT_ROOT, "results", "tables", "semisynth_params.json")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_models(data_path: str, fit_sub: int, pool_size: int, seed: int = 7):
    from sklearn.linear_model import LogisticRegression

    usecols = FEATURES + ["treatment", "conversion", "visit"]
    dtypes = {c: np.float32 for c in FEATURES}
    dtypes.update({"treatment": np.int8, "conversion": np.int8, "visit": np.int8})
    df = pd.read_csv(data_path, usecols=usecols, dtype=dtypes)
    rng = np.random.default_rng(seed)

    X = df[FEATURES].to_numpy(np.float64)
    T = df["treatment"].to_numpy(np.float64)
    S = df["visit"].to_numpy(np.float64)
    Y = df["conversion"].to_numpy(np.float64)

    # Visit model on a subsample (visits are ~4-5%, subsample keeps it fast)
    idx = rng.choice(len(df), size=min(fit_sub, len(df)), replace=False)
    Zv = np.column_stack([X[idx], T[idx]])
    lr_v = LogisticRegression(max_iter=2000, C=1e6)
    lr_v.fit(Zv, S[idx])

    # Conversion-given-visit model on all visitors
    vis = S == 1
    Zc = np.column_stack([X[vis], T[vis]])
    lr_c = LogisticRegression(max_iter=2000, C=1e6)
    lr_c.fit(Zc, Y[vis])

    kappa_hat = float(lr_c.coef_[0][-1])

    # Covariate pool for resampling
    pool_idx = rng.choice(len(df), size=pool_size, replace=False)
    X_pool = X[pool_idx].astype(np.float32)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(POOL_PATH, X=X_pool)

    params = {
        "visit_coef": lr_v.coef_[0].tolist(),
        "visit_intercept": float(lr_v.intercept_[0]),
        "conv_coef": lr_c.coef_[0].tolist(),
        "conv_intercept": float(lr_c.intercept_[0]),
        "kappa_hat": kappa_hat,
        "fit_sub": int(len(idx)),
        "n_visitors_fit": int(vis.sum()),
        "pool_size": pool_size,
    }
    with open(PARAM_PATH, "w") as f:
        json.dump(params, f, indent=2)
    print(json.dumps({k: v for k, v in params.items()
                      if not k.endswith("coef")}, indent=2), flush=True)
    return params


def true_tau(params: Dict, X_pool: np.ndarray, dial: float) -> float:
    """Exact tau under the semi-synthetic model, averaged over the pool."""
    av = np.asarray(params["visit_coef"][:-1])
    bv = params["visit_coef"][-1]
    cv = params["visit_intercept"]
    ac = np.asarray(params["conv_coef"][:-1])
    kap = dial * params["kappa_hat"]
    cc = params["conv_intercept"]

    lin_v = X_pool @ av + cv
    lin_c = X_pool @ ac + cc
    m1 = sigmoid(lin_v + bv) * sigmoid(lin_c + kap)   # E[Y | T=1, X]
    m0 = sigmoid(lin_v) * sigmoid(lin_c)              # E[Y | T=0, X]
    return float(np.mean(m1 - m0))


# Estimators run per replication. (0, 2, 3) is the default trio; 9 is the
# correctly tuned PPI++ comparator (exact-variance tuning rule + corrected
# variance), selectable with --methods.
METHOD_CHOICES = {0: "LO", 1: "Naive", 2: "SI", 3: "PPI++", 4: "GREG",
                  6: "AIPW", 9: "PPI++ exact"}
DEFAULT_METHODS = (0, 2, 3)


def run_one(args: Tuple) -> list:
    rep, dial, n, pi_L, params, tau, method_ids = args
    rng = np.random.default_rng(555000 + rep * 100 + int(dial * 10))

    dat = np.load(POOL_PATH)
    X_pool = dat["X"].astype(np.float64)
    rows = rng.choice(len(X_pool), size=n, replace=True)
    X = X_pool[rows]

    av = np.asarray(params["visit_coef"][:-1]); bv = params["visit_coef"][-1]
    cv = params["visit_intercept"]
    ac = np.asarray(params["conv_coef"][:-1]); cc = params["conv_intercept"]
    kap = dial * params["kappa_hat"]

    T = (rng.random(n) < 0.5).astype(np.float64)
    pS = sigmoid(X @ av + cv + bv * T)
    S = (rng.random(n) < pS).astype(np.float64)
    pY = sigmoid(X @ ac + cc + kap * T)
    Y = np.where(S == 1, (rng.random(n) < pY).astype(np.float64), 0.0)

    n_L = int(np.floor(pi_L * n))
    lm = np.zeros(n, dtype=bool)
    lm[rng.choice(n, size=n_L, replace=False)] = True

    Y_hat, design = train_prediction_model(S, X, Y, lm, n_folds=5,
                                   rng=np.random.default_rng(rep + 7777),
        protocol=PROTOCOL, return_design=True,
    )
    out = []
    res = {}
    for m_id in method_ids:
        name = METHOD_CHOICES[m_id]
        kwargs = {"design": design} if m_id in (2, 5) else {}
        r = estimate(m_id, T, S, Y, Y_hat, lm, protocol=PROTOCOL, **kwargs)
        res[m_id] = r
        out.append(dict(rep=rep, dial=dial, method=name, method_id=m_id,
                        true_tau=tau, tau_hat=r["tau_hat"],
                        covers=int(r["ci_lower"] <= tau <= r["ci_upper"]),
                        bias=r["tau_hat"] - tau))
    if 2 in res and 3 in res:
        lam = res[3].get("lambda_hat", 0.0)
        cov_sp = estimate_cov_si_ppi(T, Y, Y_hat, lm, lam, design=design)
        t2 = surrogacy_test(res[2]["tau_hat"], res[3]["tau_hat"],
                            res[2]["var_hat"], res[3]["var_hat"],
                            cov_sp, alternative="two-sided")
        out.append(dict(rep=rep, dial=dial, method="DIAG", method_id=-1,
                        true_tau=tau, tau_hat=t2["D_hat"], covers=int(t2["p_value"] < 0.05),
                        bias=t2["T_n"]))
    return out


DIALS = [0.0, 0.5, 1.0, 1.5, 2.0]
DIAG_LABEL = "SI--PPI++ diagnostic (two-sided)"
_NAME_TO_ID = {v: k for k, v in METHOD_CHOICES.items()}


def summary_rows(df: pd.DataFrame, taus, n: int, pi_L: float,
                 protocol: str, seed: int = 42) -> list:
    """One result row per (dial, method) plus the diagnostic.

    Estimator rows: bias, relative bias against the exact model tau, RMSE,
    coverage, RE = RMSE(LO) / RMSE(method), MC SEs.  Diagnostic row: the
    two-sided rejection rate at alpha = 0.05, mean T_n and mean D_hat.
    """
    rows = []
    for d in DIALS:
        sub = df[df.dial == d]
        if sub.empty:
            continue
        common = dict(dgp="semisynthetic", config_name=f"dial={d}",
                      params={"dial": d}, pi_L=pi_L, n=n,
                      protocol=protocol, seed=seed, target="ATE")
        extra = {"analysis": "semisynthetic", "dial": d}
        lo = sub[sub.method == "LO"]
        lo_rmse = (float(np.sqrt((lo.bias ** 2).mean())) if not lo.empty
                   else float("nan"))
        for name in pd.unique(sub.loc[sub.method != "DIAG", "method"]):
            ms = sub[sub.method == name]
            m_id = _NAME_TO_ID[name]
            spec = default_spec_for_id(m_id)
            rows.append(make_result_row(
                spec=spec, R=ms.rep.nunique(), **common,
                metrics=metrics_from_errors(
                    ms.bias.to_numpy(float), true_tau=taus[d],
                    covers=ms.covers.to_numpy(float), rmse_baseline=lo_rmse),
                extra={"method_key": spec.key if m_id != 9 else "ppi_id9",
                       "table_label": name, **extra},
            ))
        dg = sub[sub.method == "DIAG"]
        if not dg.empty:
            rows.append(make_result_row(
                method_id=-1, method_label=DIAG_LABEL, R=dg.rep.nunique(),
                **common,
                metrics=metrics_from_errors(
                    None, true_tau=taus[d], reject=dg.covers.to_numpy(float)),
                extra={"method_key": "diagnostic",
                       "mean_T_n": float(dg.bias.mean()),
                       "mean_D_hat": float(dg.tau_hat.mean()), **extra},
            ))
    return rows


def build_markdown(df, taus, method_names, n, pi_L, R, kappa_hat) -> list:
    lines = ["# Criteo-calibrated semi-synthetic results\n",
             f"n = {n}, pi_L = {pi_L}, R = {R}, "
             f"kappa_hat = {kappa_hat:.4f}\n\n",
             "| dial | true tau | Method | Bias | RelBias% | Coverage | RE |\n"
             "|---|---|---|---|---|---|---|\n"]
    for d in DIALS:
        sub = df[(df.dial == d) & (df.method != "DIAG")]
        lo_mse = (sub[sub.method == "LO"].bias ** 2).mean()
        for m in method_names:
            ms = sub[sub.method == m]
            bias = ms.bias.mean()
            cov = ms.covers.mean()
            re = np.sqrt(lo_mse / (ms.bias ** 2).mean())
            lines.append(f"| {d} | {taus[d]:.6f} | {m} | {bias:.6f} "
                         f"| {100*bias/taus[d]:.1f} | {cov:.3f} | {re:.2f} |\n")
        dg = df[(df.dial == d) & (df.method == "DIAG")]
        lines.append(f"| {d} | | DIAG rejection | {dg.covers.mean():.3f} "
                     f"| | mean T_n = {dg.bias.mean():.2f} | |\n")
    return lines


def write_outputs(df, outdir, taus, method_names, n, pi_L, R, params,
                  write_raw=True):
    os.makedirs(outdir, exist_ok=True)
    if write_raw:
        df.to_csv(os.path.join(outdir, "semisynthetic_raw.csv"), index=False)
    write_rows(
        os.path.join(outdir, "semisynthetic_rows.json"),
        summary_rows(df, taus, n, pi_L, PROTOCOL),
        generated_by="scripts/run_semisynthetic.py", protocol=PROTOCOL,
    )
    with open(os.path.join(outdir, "semisynthetic.md"), "w") as f:
        f.writelines(build_markdown(df, taus, method_names, n, pi_L, R,
                                    params["kappa_hat"]))


def main():
    global PROTOCOL
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=os.environ.get("CRITEO_CSV"),
        help="Path to criteo-uplift-v2.1.csv. Defaults to $CRITEO_CSV. "
             "Not needed with --skip-fit. See README.md.",
    )
    ap.add_argument("--fit-sub", type=int, default=2_000_000)
    ap.add_argument("--pool", type=int, default=500_000)
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--n", type=int, default=64_000)
    ap.add_argument("--pi-L", type=float, default=0.20)
    ap.add_argument("--skip-fit", action="store_true")
    ap.add_argument(
        "--methods", type=str,
        default=",".join(str(m) for m in DEFAULT_METHODS),
        help=f"Comma-separated method ids. Choices: "
             f"{sorted(METHOD_CHOICES)} (default: "
             f"{','.join(str(m) for m in DEFAULT_METHODS)}). Method 9 is the "
             "exact-rule PPI++ comparator.",
    )
    ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    ap.add_argument("--outdir",
                    default=os.path.join(PROJECT_ROOT, "results", "tables"))
    ap.add_argument("--from-raw", metavar="CSV", default=None,
                    help="rebuild rows + md from an existing raw CSV (uses "
                         "the stored parameters and covariate pool for the "
                         "true tau when the CSV does not record it; pass the "
                         "--n / --pi-L the run used); no simulation")
    args = ap.parse_args()
    PROTOCOL = args.protocol

    if args.from_raw:
        df = pd.read_csv(args.from_raw, float_precision="round_trip")
        with open(PARAM_PATH) as f:
            params = json.load(f)
        if "true_tau" in df.columns and df["true_tau"].notna().all():
            taus = {d: float(df.loc[df.dial == d, "true_tau"].iloc[0])
                    for d in DIALS if (df.dial == d).any()}
        else:
            X_pool = np.load(POOL_PATH)["X"].astype(np.float64)
            taus = {d: true_tau(params, X_pool, d) for d in DIALS}
        names = list(pd.unique(df.loc[df.method != "DIAG", "method"]))
        R = int(df.groupby(["dial", "method"])["rep"].nunique().max())
        write_outputs(df, args.outdir, taus, names, args.n, args.pi_L, R,
                      params, write_raw=False)
        return

    method_ids = tuple(int(x) for x in args.methods.split(",") if x.strip())
    bad = [m for m in method_ids if m not in METHOD_CHOICES]
    if bad:
        ap.error(f"unknown method id(s) {bad}; choose from "
                 f"{sorted(METHOD_CHOICES)}")
    method_names = [METHOD_CHOICES[m] for m in method_ids]

    t0 = time.time()
    if not args.skip_fit:
        assert args.data, "--data required unless --skip-fit"
        print("Fitting calibration models on Criteo...", flush=True)
        params = fit_models(args.data, args.fit_sub, args.pool)
    else:
        with open(PARAM_PATH) as f:
            params = json.load(f)

    X_pool = np.load(POOL_PATH)["X"].astype(np.float64)
    taus = {d: true_tau(params, X_pool, d) for d in DIALS}
    print("true tau by dial:", {d: round(t, 6) for d, t in taus.items()},
          flush=True)

    tasks = [(rep, d, args.n, args.pi_L, params, taus[d], method_ids)
             for d in DIALS for rep in range(args.R)]
    print(f"{len(tasks)} tasks", flush=True)
    with multiprocessing.Pool(8, initializer=_set_protocol,
                              initargs=(PROTOCOL,)) as pool:
        nested = pool.map(run_one, tasks)
    rows = [r for batch in nested for r in batch]
    df = pd.DataFrame(rows)
    write_outputs(df, args.outdir, taus, method_names, args.n, args.pi_L,
                  args.R, params)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
