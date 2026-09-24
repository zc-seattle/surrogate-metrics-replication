#!/usr/bin/env python3
"""
Criteo uplift dataset (v2.1) analysis for the surrogate metrics paper.

Mirrors the Hillstrom pipeline at industrial scale:
  Part A: descriptive statistics (funnel structure, within-visitor shift)
  Part B: estimator comparison at pi_L in {0.05, 0.20} with R random splits
  Part C: diagnostic detection vs sample size (subsample scale curve)

Surrogate S = visit, outcome Y = conversion, both binary and defined for
every randomized unit. Ground truth = full-sample difference in means.

Usage:
    python scripts/run_criteo.py --data /path/to/criteo-uplift-v2.1.csv
    python scripts/run_criteo.py --from-raw results/tables/criteo_main_raw.csv

Part B outputs (in --outdir, default results/tables):
  criteo_main_raw.csv    one row per (split, pi_L, method); for method DIAG,
                         `tau_hat` holds D_hat, `bias` holds T_n and `covers`
                         the two-sided rejection flag at alpha = 0.05
  criteo_main_rows.json  summary result rows, one per (pi_L, method)
  criteo_main.md         the table
`--from-raw` rebuilds the Part B rows and table from an existing raw CSV and
criteo_descriptive.json, without the 3.2 GB data file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

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


def load_criteo(path: str) -> Dict[str, np.ndarray]:
    usecols = FEATURES + ["treatment", "conversion", "visit"]
    dtypes = {c: np.float32 for c in FEATURES}
    dtypes.update({"treatment": np.int8, "conversion": np.int8, "visit": np.int8})
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes)
    return {
        "T": df["treatment"].to_numpy(np.int8),
        "S": df["visit"].to_numpy(np.float64),
        "Y": df["conversion"].to_numpy(np.float64),
        "X": df[FEATURES].to_numpy(np.float32),
    }


def descriptives(T, S, Y) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["n"] = int(len(T))
    out["n_treated"] = int((T == 1).sum())
    out["n_control"] = int((T == 0).sum())
    for t in (0, 1):
        m = T == t
        out[f"visit_rate_T{t}"] = float(S[m].mean())
        out[f"conv_rate_T{t}"] = float(Y[m].mean())
        mv = m & (S == 1)
        out[f"conv_given_visit_T{t}"] = float(Y[mv].mean())
        mnv = m & (S == 0)
        out[f"conv_given_novisit_T{t}"] = float(Y[mnv].mean())
    out["tau_S"] = out["visit_rate_T1"] - out["visit_rate_T0"]
    out["tau"] = out["conv_rate_T1"] - out["conv_rate_T0"]
    out["within_visitor_shift"] = (
        out["conv_given_visit_T1"] - out["conv_given_visit_T0"]
    )
    out["visit_conv_ratio"] = float(S.mean() / Y.mean())
    return out


# Estimators run by Part B / Part C.  (0, 2, 3) are the paper's default
# trio; 9 is the correctly tuned PPI++ comparator (exact-variance tuning
# rule + corrected variance), selectable with --methods.
METHOD_CHOICES = {0: "LO", 1: "Naive", 2: "SI", 3: "PPI++", 4: "GREG",
                  6: "AIPW", 9: "PPI++ exact"}
DEFAULT_METHODS = (0, 2, 3)


def _method_kwargs(m_id: int) -> Dict[str, Any]:
    if m_id == 3:
        return {"corrected_variance": True}
    if m_id == 6:
        return {"propensity_model": "constant"}
    return {}


def one_split(rep: int, pi_L: float, T, S, Y, X, true_ate: float,
              run_diag: bool = True,
              method_ids=DEFAULT_METHODS) -> List[Dict[str, Any]]:
    n = len(T)
    rng = np.random.default_rng(90000 + rep * 1000 + int(pi_L * 10000))
    n_L = max(int(np.floor(pi_L * n)), 50)
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[rng.choice(n, size=n_L, replace=False)] = True

    cf_rng = np.random.default_rng(rep + 7777)
    Y_hat, design = train_prediction_model(S, X, Y, labeled_mask, n_folds=5, rng=cf_rng, protocol=PROTOCOL, return_design=True)

    rows = []
    res_by_id = {}
    for m_id in method_ids:
        name = METHOD_CHOICES[m_id]
        mkwargs = dict(_method_kwargs(m_id))
        if m_id in (2, 5):
            mkwargs["design"] = design
        res = estimate(m_id, T, S, Y, Y_hat, labeled_mask,
                       protocol=PROTOCOL, **mkwargs)
        res_by_id[m_id] = res
        rows.append(dict(
            rep=rep, pi_L=pi_L, method=name,
            tau_hat=res["tau_hat"], var_hat=res["var_hat"],
            covers=int(res["ci_lower"] <= true_ate <= res["ci_upper"]),
            bias=res["tau_hat"] - true_ate,
        ))

    if run_diag and 2 in res_by_id and 3 in res_by_id:
        lam = res_by_id[3].get("lambda_hat", 0.0)
        cov_sp = estimate_cov_si_ppi(T, Y, Y_hat, labeled_mask, lam, design=design)
        test = surrogacy_test(
            res_by_id[2]["tau_hat"], res_by_id[3]["tau_hat"],
            res_by_id[2]["var_hat"], res_by_id[3]["var_hat"],
            cov_sp, alternative="two-sided",
        )
        rows.append(dict(
            rep=rep, pi_L=pi_L, method="DIAG",
            tau_hat=test["D_hat"], var_hat=test["var_D"],
            covers=int(test["p_value"] < 0.05),  # 'covers' = reject flag
            bias=test["T_n"],
        ))
    return rows


PI_L_MAIN = (0.05, 0.20)
DIAG_LABEL = "SI--PPI++ diagnostic (two-sided)"
_NAME_TO_ID = {v: k for k, v in METHOD_CHOICES.items()}


def main_summary_rows(df: pd.DataFrame, true_ate: float, n: int,
                      protocol: str, seed: int = 42) -> List[Dict[str, Any]]:
    """Part B result rows, one per (pi_L, method) plus the diagnostic.

    The target is the masking target (full-sample difference in means).
    Estimator rows: bias, relative bias, RMSE, coverage, RE = RMSE(LO) /
    RMSE(method), MC SEs.  Diagnostic row: two-sided rejection rate at
    alpha = 0.05, mean T_n, mean D_hat.
    """
    rows: List[Dict[str, Any]] = []
    for pi_L in PI_L_MAIN:
        sub = df[df.pi_L == pi_L]
        if sub.empty:
            continue
        common = dict(dgp="criteo", config_name=f"pi_L={pi_L}",
                      params={"pi_L": pi_L}, pi_L=pi_L, n=n,
                      protocol=protocol, seed=seed, target="masking_target")
        extra = {"analysis": "criteo_main"}
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
                    ms.bias.to_numpy(float), true_tau=true_ate,
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
                    None, true_tau=true_ate,
                    reject=dg.covers.to_numpy(float)),
                extra={"method_key": "diagnostic",
                       "mean_T_n": float(dg.bias.mean()),
                       "mean_D_hat": float(dg.tau_hat.mean()), **extra},
            ))
    return rows


def main_markdown(df: pd.DataFrame, method_names, true_ate: float, n: int,
                  R: int) -> List[str]:
    lines = ["# Criteo main results\n",
             f"true ATE (full DIM) = {true_ate:.6f}, n = {n:,}, "
             f"R = {R}\n\n"]
    for pi_L in PI_L_MAIN:
        sub = df[(df.pi_L == pi_L) & (df.method != "DIAG")]
        lo_mse = (sub[sub.method == "LO"].bias ** 2).mean()
        lines.append(f"## pi_L = {pi_L}\n\n"
                     "| Method | Bias | RelBias% | RMSE | Coverage | RE |\n"
                     "|---|---|---|---|---|---|\n")
        for m in method_names:
            ms = sub[sub.method == m]
            bias = ms.bias.mean()
            rmse = np.sqrt((ms.bias ** 2).mean())
            cov = ms.covers.mean()
            re = np.sqrt(lo_mse / (ms.bias ** 2).mean())
            lines.append(f"| {m} | {bias:.6f} | {100*bias/true_ate:.1f} "
                         f"| {rmse:.6f} | {cov:.3f} | {re:.2f} |\n")
        diag = df[(df.pi_L == pi_L) & (df.method == "DIAG")]
        lines.append(f"\nDiagnostic two-sided rejection rate: "
                     f"{diag.covers.mean():.3f} "
                     f"(mean T_n = {diag.bias.mean():.2f})\n\n")
    return lines


def write_main_outputs(df, outdir, method_names, true_ate, n, R, protocol,
                       write_raw=True) -> None:
    os.makedirs(outdir, exist_ok=True)
    if write_raw:
        df.to_csv(os.path.join(outdir, "criteo_main_raw.csv"), index=False)
    write_rows(
        os.path.join(outdir, "criteo_main_rows.json"),
        main_summary_rows(df, true_ate, n, protocol),
        generated_by="scripts/run_criteo.py", protocol=protocol,
    )
    with open(os.path.join(outdir, "criteo_main.md"), "w") as f:
        f.writelines(main_markdown(df, method_names, true_ate, n, R))


def main():
    global PROTOCOL
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=os.environ.get("CRITEO_CSV"),
        help="Path to criteo-uplift-v2.1.csv. Defaults to $CRITEO_CSV. "
             "See README.md for how to obtain the file.",
    )
    ap.add_argument("--R-main", type=int, default=100)
    ap.add_argument(
        "--methods", type=str, default=",".join(str(m) for m in DEFAULT_METHODS),
        help="Comma-separated method ids for Parts B and C. "
             f"Choices: {sorted(METHOD_CHOICES)} (default: "
             f"{','.join(str(m) for m in DEFAULT_METHODS)}). Method 9 is the "
             "exact-rule PPI++ comparator.",
    )
    ap.add_argument("--skip-main", action="store_true")
    ap.add_argument("--skip-scale", action="store_true")
    ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    ap.add_argument("--outdir",
                    default=os.path.join(PROJECT_ROOT, "results", "tables"))
    ap.add_argument("--from-raw", metavar="CSV", default=None,
                    help="rebuild the Part B rows + md from an existing "
                         "criteo_main_raw.csv and criteo_descriptive.json "
                         "(read from --outdir, else results/tables); no data "
                         "file needed")
    args = ap.parse_args()
    PROTOCOL = args.protocol

    if args.from_raw:
        df = pd.read_csv(args.from_raw, float_precision="round_trip")
        desc_path = os.path.join(args.outdir, "criteo_descriptive.json")
        if not os.path.exists(desc_path):
            desc_path = os.path.join(PROJECT_ROOT, "results", "tables",
                                     "criteo_descriptive.json")
        with open(desc_path) as f:
            desc = json.load(f)
        names = list(pd.unique(df.loc[df.method != "DIAG", "method"]))
        R = int(df.groupby(["pi_L", "method"])["rep"].nunique().max())
        write_main_outputs(df, args.outdir, names, desc["tau"], desc["n"], R,
                           PROTOCOL, write_raw=False)
        return

    method_ids = tuple(int(x) for x in args.methods.split(",") if x.strip())
    bad = [m for m in method_ids if m not in METHOD_CHOICES]
    if bad:
        ap.error(f"unknown method id(s) {bad}; choose from "
                 f"{sorted(METHOD_CHOICES)}")
    method_names = [METHOD_CHOICES[m] for m in method_ids]

    if not args.data:
        ap.error(
            "no Criteo CSV given: pass --data /path/to/criteo-uplift-v2.1.csv "
            "or set the CRITEO_CSV environment variable "
            "(see README.md)."
        )

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print("Loading Criteo...", flush=True)
    t0 = time.time()
    d = load_criteo(args.data)
    T, S, Y, X = d["T"], d["S"], d["Y"], d["X"]
    print(f"Loaded n={len(T):,} in {time.time()-t0:.0f}s", flush=True)

    # ---- Part A: descriptives ----
    desc = descriptives(T, S, Y)
    true_ate = desc["tau"]
    with open(os.path.join(outdir, "criteo_descriptive.json"), "w") as f:
        json.dump(desc, f, indent=2)
    print(json.dumps(desc, indent=2), flush=True)

    # ---- Part B: estimator comparison at full n ----
    if not args.skip_main:
        rows = []
        for pi_L in (0.05, 0.20):
            for rep in range(args.R_main):
                rows.extend(one_split(rep, pi_L, T, S, Y, X, true_ate,
                                      method_ids=method_ids))
                if rep % 10 == 0:
                    print(f"main pi_L={pi_L} rep={rep} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        write_main_outputs(df, outdir, method_names, true_ate, len(T),
                           args.R_main, PROTOCOL)
        print("Part B done", flush=True)

    # ---- Part C: detection vs scale ----
    if not args.skip_scale:
        n_grid = [10_000, 30_000, 100_000, 300_000, 1_000_000,
                  3_000_000, len(T)]
        R_grid = [300, 300, 300, 200, 100, 60, 40]
        pi_L = 0.20
        rows = []
        rng0 = np.random.default_rng(31415)
        for n_sub, R in zip(n_grid, R_grid):
            for rep in range(R):
                if n_sub >= len(T):
                    idx = np.arange(len(T))
                else:
                    idx = rng0.choice(len(T), size=n_sub, replace=False)
                r = one_split(rep, pi_L, T[idx], S[idx], Y[idx], X[idx],
                              true_ate, method_ids=method_ids)
                for row in r:
                    row["n_sub"] = n_sub
                rows.extend(r)
            done = pd.DataFrame([x for x in rows if x["n_sub"] == n_sub])
            rej = done[done.method == "DIAG"].covers.mean()
            si_cov = done[done.method == "SI"].covers.mean()
            print(f"scale n={n_sub:,}: reject={rej:.3f} SI_cov={si_cov:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(outdir, "criteo_scale_raw.csv"), index=False)

        lines = ["# Criteo detection vs scale (pi_L = 0.20)\n\n",
                 "| n | R | SI bias | SI coverage | PPI++ coverage | "
                 "diagnostic rejection |\n|---|---|---|---|---|---|\n"]
        for n_sub, R in zip(n_grid, R_grid):
            sub = df[df.n_sub == n_sub]
            si = sub[sub.method == "SI"]
            ppi = sub[sub.method == "PPI++"]
            dg = sub[sub.method == "DIAG"]
            lines.append(f"| {n_sub:,} | {R} | {si.bias.mean():.6f} "
                         f"| {si.covers.mean():.3f} | {ppi.covers.mean():.3f} "
                         f"| {dg.covers.mean():.3f} |\n")
        with open(os.path.join(outdir, "criteo_scale.md"), "w") as f:
            f.writelines(lines)
        print("Part C done", flush=True)

    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
