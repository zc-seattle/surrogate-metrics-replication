#!/usr/bin/env python3
"""
Paired detection-damage analysis.

This script measures every quantity on the SAME Monte Carlo draws, with the
same fitted index, under the all-units cross-fitting protocol, and with the
joint sandwich covariance of (tau_SI, tau_PPI) for the learned index.

Design
------
DGP 2 (partial mediation), pi_L = 0.20,
rho in {0, .02, .04, ..., .20, .25, .30, .35, .40, .50, .60},
n in {10,000, 100,000}.

rho = 0 is the null: the rejection rates in that row are the diagnostic's
size, and they are also written out on their own as the null-size panel that
sits beside the detection-damage figure.

Per replication, all of the following come from one draw and one fitted index:
  * SI coverage under the joint sandwich variance
  * SI coverage under the plug-in variance (predictions treated as fixed)
  * SI coverage under the delta-method first-stage variance
  * PPI++ coverage under the exact (overlap-corrected) variance
  * two-sided diagnostic rejection at alpha = 0.05 and alpha = 0.10

SE(D_hat) in the diagnostic
---------------------------
D_hat = tau_PPI - tau_SI and
    Var(D_hat) = Var(tau_PPI) + Var(tau_SI) - 2 Cov(tau_SI, tau_PPI).

  * `sandwich`  -- all three terms from `joint_influence_cov`, the joint
                   estimating-equation covariance for the LEARNED index.
                   This is the headline diagnostic.
  * `fixedf`    -- the fixed-predictor SE: plug-in SI variance,
                   exact PPI++ variance, and `estimate_cov_si_ppi`.  Kept as
                   the ablation column so the two conventions can be compared
                   on identical draws.
  * `delta`     -- fixed-f covariance with the delta-method first-stage term
                   added to Var(tau_SI).

Output
------
results/tables/detection_damage_paired.csv / .md      (with MC standard errors)
results/tables/detection_damage_paired_raw.csv        (per replication)
results/tables/detection_damage_null_size.csv / .md   (the rho = 0 panel)
results/tables/detection_damage_paired_rows.json      (result rows)

Refined grid
---------------------------------------------------------
A local grid around the damage crossing at each n, with the SAME seed
stream: rho in {0, 0.005, ..., 0.06} at n = 100,000 and
{0.06, 0.065, ..., 0.14} at n = 10,000, R = 2,000 at rho = 0 and 1,000
elsewhere (``--R-null``, ``--R-fine``).  A fine-grid point that coincides
with a primary grid point reuses the primary draws for its first R_primary
replications (the seed depends only on (rho, n, rep)), so the refined run
extends the primary one rather than replacing it.  Outputs, leaving the
primary files untouched:

results/tables/detection_damage_refined_raw.csv         (per replication)
results/tables/detection_damage_refined_grid.csv / .md  (per grid point)
results/tables/detection_damage_refined_grid_rows.json  (result rows)

The edges, the grid-resolution bracket and the interpolation-conditional
bootstrap intervals are computed by scripts/detection_damage_edge_ci.py.

Usage:
    python scripts/run_detection_damage_paired.py [--R 500] [--R-large 300]
    python scripts/run_detection_damage_paired.py --R 5 --R-large 5   # smoke
    python scripts/run_detection_damage_paired.py --refine [--R-null 2000] \
        [--R-fine 1000]
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dgps.dgps import generate_dgp2
from src.methods import estimate_cov_si_ppi
from src.simulations.simulation import derive_seed
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS, make_result_row
from src.utils.estimator_api import (
    estimate,
    joint_influence_cov,
    surrogacy_test,
    train_prediction_model,
)
from src.utils.registry import write_rows

TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")

MASTER_SEED = 42
SEED_TAG = 250          # distinct from every core-grid DGP id
N_WORKERS = 8

PI_L = 0.20
RHO_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
            0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
N_VALUES = [10_000, 100_000]
ALPHAS = [0.05, 0.10]

#: Refined local grids around the damage crossing (``--refine``).
REFINE_GRID = {
    100_000: [round(0.005 * i, 3) for i in range(13)],       # 0 .. 0.06
    10_000: [round(0.06 + 0.005 * i, 3) for i in range(17)],  # 0.06 .. 0.14
}

#: SE(D) conventions, headline first.
SE_TAGS = ("sandwich", "fixedf", "delta")

PROTOCOL = DEFAULT_PROTOCOL      # set from --protocol in main()


def run_one(args: Tuple) -> Dict[str, Any]:
    rep, n, rho, protocol = args
    seed = derive_seed(MASTER_SEED, SEED_TAG, int(round(rho * 1000)),
                       int(round(np.log10(n) * 100)), rep)

    data = generate_dgp2(n=n, pi_L=PI_L, rho=rho, seed=seed)
    T, X, S, Y = data["T"], data["X"], data["S"], data["Y"]
    lm = data["labeled_mask"]
    tau = data["true_tau"]

    Y_hat, design = train_prediction_model(
        S, X, Y, lm, n_folds=5, rng=np.random.default_rng(seed + 7777),
        protocol=protocol, return_design=True,
    )

    ppi = estimate(3, T, S, Y, Y_hat, lm, protocol=protocol)
    lam = ppi.get("lambda_hat", 0.0)

    si_s = estimate(2, T, S, Y, Y_hat, lm, protocol=protocol,
                    si_variance="sandwich", design=design, lambda_hat=lam)
    si_p = estimate(2, T, S, Y, Y_hat, lm, protocol=protocol,
                    si_variance="plugin")
    si_d = estimate(2, T, S, Y, Y_hat, lm, protocol=protocol,
                    si_variance="delta", design=design)

    jc = joint_influence_cov(T, Y, lm, design, lam)
    cov_fixedf = estimate_cov_si_ppi(T, Y, Y_hat, lm, lam)

    out: Dict[str, Any] = dict(
        rep=rep, n=n, rho=rho, true_tau=tau, protocol=protocol,
        si_bias=si_s["tau_hat"] - tau,
        ppi_bias=ppi["tau_hat"] - tau,
        si_cov_sandwich=int(si_s["ci_lower"] <= tau <= si_s["ci_upper"]),
        si_cov_plugin=int(si_p["ci_lower"] <= tau <= si_p["ci_upper"]),
        si_cov_delta=int(si_d["ci_lower"] <= tau <= si_d["ci_upper"]),
        ppi_cov_exact=int(ppi["ci_lower"] <= tau <= ppi["ci_upper"]),
        si_var_sandwich=si_s["var_hat"],
        si_var_plugin=si_p["var_hat"],
        si_var_delta=si_d["var_hat"],
        ppi_var_exact=ppi["var_hat"],
        cov_si_ppi_sandwich=jc["cov_si_ppi"],
        cov_si_ppi_fixedf=cov_fixedf,
        var_D_sandwich=jc["var_D"],
        lambda_hat=lam,
    )

    # Three SE(D) conventions on the same draw.
    variants = {
        "sandwich": (jc["var_si"], jc["var_ppi"], jc["cov_si_ppi"]),
        "fixedf": (si_p["var_hat"], ppi["var_hat"], cov_fixedf),
        "delta": (si_d["var_hat"], ppi["var_hat"], cov_fixedf),
    }
    for tag, (var_si, var_ppi, cov) in variants.items():
        test = surrogacy_test(
            tau_si=si_s["tau_hat"], tau_ppi=ppi["tau_hat"],
            var_si=var_si, var_ppi=var_ppi, cov_si_ppi=cov,
            alternative="two-sided",
        )
        out[f"T_n_{tag}"] = test["T_n"]
        out[f"SE_D_{tag}"] = test["SE_D"]
        out[f"p_{tag}"] = test["p_value"]
        for a in ALPHAS:
            out[f"rej_{tag}_a{int(a * 100):02d}"] = int(test["p_value"] < a)
    return out


def _rate(x: np.ndarray) -> Tuple[float, float]:
    p = float(np.mean(x))
    return p, float(np.sqrt(p * (1 - p) / len(x)))


COVERAGE_COLS = (
    "si_cov_sandwich", "si_cov_plugin", "si_cov_delta", "ppi_cov_exact",
)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n, rho), g in raw.groupby(["n", "rho"]):
        R = len(g)
        row: Dict[str, Any] = dict(
            n=int(n), rho=float(rho), pi_L=PI_L, R=R,
            protocol=str(g.protocol.iloc[0]),
            true_tau=float(g.true_tau.iloc[0]),
            si_bias=float(g.si_bias.mean()),
            ppi_bias=float(g.ppi_bias.mean()),
            mean_lambda=float(g.lambda_hat.mean()),
        )
        for col in COVERAGE_COLS:
            p, se = _rate(g[col].to_numpy())
            row[col] = p
            row[f"{col}_mcse"] = se
        for tag in SE_TAGS:
            for a in ALPHAS:
                key = f"rej_{tag}_a{int(a * 100):02d}"
                p, se = _rate(g[key].to_numpy())
                row[key] = p
                row[f"{key}_mcse"] = se
            row[f"mean_T_n_{tag}"] = float(g[f"T_n_{tag}"].mean())
            row[f"mean_SE_D_{tag}"] = float(g[f"SE_D_{tag}"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n", "rho"]).reset_index(drop=True)


def null_size_panel(summ: pd.DataFrame) -> pd.DataFrame:
    """The rho = 0 rows: the diagnostic's size at both levels and both n.

    The detection-damage figure annotates its power curves with these, so that
    a reader can read power and size off the same experiment.
    """
    null = summ[summ.rho == 0.0].copy()
    rows = []
    for _, r in null.iterrows():
        for tag in SE_TAGS:
            for a in ALPHAS:
                key = f"rej_{tag}_a{int(a * 100):02d}"
                rows.append(dict(
                    n=int(r.n), pi_L=PI_L, rho=0.0, R=int(r.R),
                    protocol=r.protocol, se_convention=tag, alpha=a,
                    size=float(r[key]), mc_se=float(r[f"{key}_mcse"]),
                    nominal=a,
                    within_mc_band=bool(
                        abs(float(r[key]) - a) <= 1.96 * float(r[f"{key}_mcse"])
                    ),
                ))
    return pd.DataFrame(rows)


def write_markdown(summ: pd.DataFrame, path: str, refined: bool = False) -> None:
    title = ("# Paired detection-damage analysis, refined local grid (DGP 2)\n\n"
             if refined else "# Paired detection-damage analysis (DGP 2)\n\n")
    rho_fmt = ".3f" if refined else ".2f"
    lines = [
        title,
        f"pi_L = {PI_L}. Every quantity in a row is measured on the SAME "
        "Monte Carlo draws and the same fitted index, under the all-units "
        "cross-fitting protocol. `SI cov (sandwich)` is the interval "
        "for the learned index; `SI cov (plug-in)` treats the predictions as "
        "fixed and `SI cov (delta)` adds the delta-method term d' V_beta d. "
        "PPI++ uses the exact (overlap-corrected) variance. Diagnostic "
        "columns are the two-sided rejection rate: the `sandwich` block "
        "builds SE(D_hat) from the joint sandwich covariance (the headline "
        "diagnostic) and the `fixed-f` block from the "
        "fixed-predictor covariance. The rho = 0 row is the null: those "
        "rejection rates are the diagnostic's size. Parenthesised figures are "
        "Monte Carlo standard errors.\n\n",
    ]
    if refined:
        lines.append(
            "Refined local grid around the damage crossing, same seed stream "
            "as the primary run (`derive_seed(42, 250, 1000 rho, 100 log10 n, "
            "rep)`): at a rho the primary grid also carries, the first "
            "R_primary replications here are the primary draws. The R column "
            "gives each row's replication count.\n\n")
    for n in sorted(summ.n.unique()):
        sub = summ[summ.n == n]
        Rs = sorted(set(int(x) for x in sub.R))
        R_txt = (f"R = {Rs[0]:,}" if len(Rs) == 1
                 else "R = " + " or ".join(f"{x:,}" for x in Rs))
        lines.append(f"## n = {n:,} ({R_txt})\n\n")
        lines.append(
            ("| rho | R " if refined else "| rho ") +
            "| true tau | SI bias | SI cov (sandwich) | "
            "SI cov (plug-in) | SI cov (delta) | PPI++ cov (exact) | "
            "Reject a=.05 (sandwich) | Reject a=.10 (sandwich) | "
            "Reject a=.05 (fixed-f) | Reject a=.10 (fixed-f) |\n"
            + ("|---:" if refined else "") +
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for _, r in sub.iterrows():
            lead = (f"| {r.rho:{rho_fmt}} | {int(r.R):,} " if refined
                    else f"| {r.rho:{rho_fmt}} ")
            lines.append(
                lead + f"| {r.true_tau:.4f} | {r.si_bias:+.5f} "
                f"| {r.si_cov_sandwich:.3f} ({r.si_cov_sandwich_mcse:.3f}) "
                f"| {r.si_cov_plugin:.3f} ({r.si_cov_plugin_mcse:.3f}) "
                f"| {r.si_cov_delta:.3f} ({r.si_cov_delta_mcse:.3f}) "
                f"| {r.ppi_cov_exact:.3f} ({r.ppi_cov_exact_mcse:.3f}) "
                f"| {r.rej_sandwich_a05:.3f} ({r.rej_sandwich_a05_mcse:.3f}) "
                f"| {r.rej_sandwich_a10:.3f} ({r.rej_sandwich_a10_mcse:.3f}) "
                f"| {r.rej_fixedf_a05:.3f} ({r.rej_fixedf_a05_mcse:.3f}) "
                f"| {r.rej_fixedf_a10:.3f} ({r.rej_fixedf_a10_mcse:.3f}) |\n"
            )
        lines.append("\n")
    with open(path, "w") as f:
        f.writelines(lines)


def write_null_markdown(null: pd.DataFrame, path: str) -> None:
    lines = [
        "# Null size of the SI--PPI++ estimator-disagreement diagnostic\n\n",
        "DGP 2 at rho = 0, the same draws as the detection-damage table. "
        "`within MC band` marks a size within 1.96 Monte Carlo standard "
        "errors of its nominal level.\n\n",
        "| n | SE convention | alpha | size | MC SE | within MC band |\n"
        "|---:|---|---:|---:|---:|---|\n",
    ]
    for _, r in null.iterrows():
        lines.append(
            f"| {int(r.n):,} | {r.se_convention} | {r.alpha:.2f} "
            f"| {r['size']:.3f} | {r.mc_se:.3f} "
            f"| {'yes' if r.within_mc_band else 'NO'} |\n"
        )
    with open(path, "w") as f:
        f.writelines(lines)


def registry_rows(summ: pd.DataFrame, protocol: str,
                  prefix: str = "detection_damage", rho_fmt: str = ".2f",
                  analysis: str = "detection_damage_paired",
                  ) -> List[Dict[str, Any]]:
    """One row per (cell, SE convention, alpha) for the result rows."""
    rows: List[Dict[str, Any]] = []
    for _, r in summ.iterrows():
        params = {"rho": float(r.rho), "dgp": 2}
        cname = f"{prefix}_rho{r.rho:{rho_fmt}}_n{int(r.n)}"
        # Coverage rows, one per SI/PPI++ variance convention.
        for label, cov_col, var_col, si_var, var_rule in (
            ("SI", "si_cov_sandwich", "si_var_sandwich", "sandwich", None),
            ("SI (plug-in variance)", "si_cov_plugin", "si_var_plugin",
             "plugin", None),
            ("SI (delta-method first stage)", "si_cov_delta", "si_var_delta",
             "delta", None),
            ("PPI++", "ppi_cov_exact", "ppi_var_exact", None, "exact"),
        ):
            rows.append(make_result_row(
                dgp=2, config_name=cname,
                params=params, pi_L=PI_L, n=int(r.n), R=int(r.R),
                protocol=protocol,
                method_id=2 if si_var else 3, method_label=label,
                alpha=0.05, target="ATE", seed=MASTER_SEED,
                metrics={
                    "coverage": float(r[cov_col]),
                    "mc_se_coverage": float(r[f"{cov_col}_mcse"]),
                    "true_tau": float(r.true_tau),
                },
                extra={"analysis": analysis},
            ))
            rows[-1]["si_variance"] = si_var
            rows[-1]["variance"] = var_rule
        # Diagnostic rejection rows.
        for tag in SE_TAGS:
            for a in ALPHAS:
                key = f"rej_{tag}_a{int(a * 100):02d}"
                rows.append(make_result_row(
                    dgp=2,
                    config_name=cname,
                    params=params, pi_L=PI_L, n=int(r.n), R=int(r.R),
                    protocol=protocol, method_id=-1,
                    method_label=f"SI--PPI++ diagnostic ({tag} SE)",
                    alpha=a, target="ATE", seed=MASTER_SEED,
                    metrics={
                        "rejection": float(r[key]),
                        "mc_se_rejection": float(r[f"{key}_mcse"]),
                    },
                    extra={
                        "analysis": analysis,
                        "se_convention": tag,
                        "is_null": bool(r.rho == 0.0),
                        "mean_SE_D": float(r[f"mean_SE_D_{tag}"]),
                    },
                ))
    return rows


def run_refine(args: argparse.Namespace) -> None:
    """The refined local grid; writes detection_damage_refined_* only."""
    t0 = time.time()
    tasks: List[Tuple] = []
    for n, grid in REFINE_GRID.items():
        for rho in grid:
            R = args.R_null if rho == 0.0 else args.R_fine
            tasks += [(rep, n, rho, args.protocol) for rep in range(R)]
    print(f"{len(tasks)} tasks (refined grid; R = {args.R_null} at rho = 0, "
          f"{args.R_fine} elsewhere; protocol={args.protocol})", flush=True)
    with multiprocessing.Pool(args.workers) as pool:
        rows = pool.map(run_one, tasks, chunksize=5)

    raw = pd.DataFrame(rows)
    summ = summarize(raw)
    out = args.outdir
    os.makedirs(out, exist_ok=True)
    raw.to_csv(os.path.join(out, "detection_damage_refined_raw.csv"),
               index=False)
    summ.to_csv(os.path.join(out, "detection_damage_refined_grid.csv"),
                index=False)
    write_markdown(summ, os.path.join(out, "detection_damage_refined_grid.md"),
                   refined=True)
    write_rows(
        os.path.join(out, "detection_damage_refined_grid_rows.json"),
        registry_rows(summ, args.protocol, prefix="detection_damage_refined",
                      rho_fmt=".3f", analysis="detection_damage_refined"),
        generated_by="scripts/run_detection_damage_paired.py --refine",
        protocol=args.protocol,
    )
    print("\n" + summ[["n", "rho", "R", "si_cov_sandwich",
                       "si_cov_sandwich_mcse", "rej_sandwich_a05"]]
          .to_string(index=False))
    print(f"\nWrote detection_damage_refined_{{raw.csv,grid.csv,grid.md,"
          f"grid_rows.json}} in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=500,
                    help="replications at n = 10,000")
    ap.add_argument("--R-large", type=int, default=300,
                    help="replications at n = 100,000 (slower)")
    ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--refine", action="store_true",
                    help="run the refined local grid around the damage "
                         "crossing (see the module docstring)")
    ap.add_argument("--R-null", type=int, default=2000,
                    help="--refine: replications at rho = 0")
    ap.add_argument("--R-fine", type=int, default=1000,
                    help="--refine: replications at rho > 0")
    ap.add_argument("--outdir", default=TABLES_DIR)
    args = ap.parse_args()

    if args.refine:
        run_refine(args)
        return

    t0 = time.time()
    tasks: List[Tuple] = []
    for n in N_VALUES:
        R = args.R if n == 10_000 else args.R_large
        for rho in RHO_GRID:
            tasks += [(rep, n, rho, args.protocol) for rep in range(R)]
    print(f"{len(tasks)} tasks "
          f"(n=10,000: R={args.R}; n=100,000: R={args.R_large}; "
          f"protocol={args.protocol})", flush=True)

    with multiprocessing.Pool(args.workers) as pool:
        rows = pool.map(run_one, tasks, chunksize=5)

    raw = pd.DataFrame(rows)
    summ = summarize(raw)
    null = null_size_panel(summ)

    os.makedirs(TABLES_DIR, exist_ok=True)
    raw.to_csv(os.path.join(TABLES_DIR, "detection_damage_paired_raw.csv"),
               index=False)
    summ.to_csv(os.path.join(TABLES_DIR, "detection_damage_paired.csv"),
                index=False)
    write_markdown(summ, os.path.join(TABLES_DIR,
                                      "detection_damage_paired.md"))
    null.to_csv(os.path.join(TABLES_DIR, "detection_damage_null_size.csv"),
                index=False)
    write_null_markdown(null, os.path.join(TABLES_DIR,
                                           "detection_damage_null_size.md"))
    write_rows(
        os.path.join(TABLES_DIR, "detection_damage_paired_rows.json"),
        registry_rows(summ, args.protocol),
        generated_by="scripts/run_detection_damage_paired.py",
        protocol=args.protocol,
    )

    cols = ["n", "rho", "si_cov_sandwich", "si_cov_plugin", "ppi_cov_exact",
            "rej_sandwich_a05", "rej_sandwich_a10", "rej_fixedf_a05",
            "rej_fixedf_a10"]
    print("\n" + summ[cols].to_string(index=False))
    print("\nNull size (rho = 0):")
    print(null.to_string(index=False))
    print(f"\nWrote results/tables/detection_damage_paired.{{csv,md}}, "
          f"detection_damage_null_size.{{csv,md}} and "
          f"detection_damage_paired_rows.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
