#!/usr/bin/env python3
"""
The MCAR sandwich under nonrandom labels: a two-cell check.

Proposition 3's joint sandwich is derived for MCAR labels.  Two reported
designs label units nonrandomly and still use it, with the labeled-design
Gram matrix Q-hat = (1/n_L) sum_{labeled} g g' in place of E[g g']:

  * DGP 5, MAR variant at q = 0.10: P(L = 1 | S) = expit(eta_0 + 0.3 S)
    (core grid, `run_dgp.py --dgp 5`, config `DGP5_q0.10_MAR`);
  * DGP 11, mix drift d = 2: the labeled set is the earliest-enrolled 20%
    and the covariate mix drifts with enrollment time
    (`scripts/run_enrollment_drift.py`, config "Mix drift d=2").

Surrogacy holds and the linear index is correctly specified in both, so both
estimators are consistent and the diagnostic's null holds: the question is
only whether the variance formulas are the right size.

Per replication (same draws as the source runs, so the point estimates
reproduce them):
  * the joint sandwich (`joint_influence_cov`): Var(SI), Var(PPI++), Cov,
    Var(D), with the labeled-design Q-hat, exactly as the source runs use it;
  * PPI++'s own exact variance (the reported PPI++ interval);
  * a joint paired bootstrap, B draws, stratified by arm, labeled flags
    carried along, the OLS index refit under the all-units protocol and both
    estimators recomputed (`estimator_api.paired_bootstrap`);
  * coverage of tau by the SI sandwich and bootstrap intervals and the PPI++
    exact, sandwich and bootstrap intervals; the diagnostic's two-sided
    rejection (its size here) with SE(D) from the sandwich and the bootstrap.

The Monte Carlo variance across the R replications is the target.  The
bootstrap treats units as i.i.d. with their labels attached, which is exact
for MAR and approximate for the enrollment rule (a rank threshold on
enrollment time, i.e. L = 1{e <= e_(n_L)}).

Output: results/tables/nonrandom_label_variance_check.{md,csv} and _raw.csv

Usage:
    python scripts/run_nonrandom_label_variance_check.py [--R 500] [--B 200]
    python scripts/run_nonrandom_label_variance_check.py --from-raw
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

from src.dgps.dgps import generate_dgp5
from src.simulations.simulation import _build_design_matrix, derive_seed
from src.utils.estimator_api import (
    estimate,
    joint_influence_cov,
    paired_bootstrap,
    train_prediction_model,
)

TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
MASTER_SEED = 42
N_WORKERS = 8
Z = float(stats.norm.ppf(0.975))
ALPHAS = (0.05, 0.10)

#: DGP 5 MAR, q = 0.10: the core-grid parameters and seed stream
#: (run_dgp.py: config_id = 1 + 100, pi_L_id = 10).
DGP5_PARAMS = dict(n=10_000, alpha_S=5.0, beta_SX=1.0, gamma_S=0.3,
                   sigma_S=2.0, alpha_Y=0.0, beta_YS=0.5, beta_YX=0.2,
                   sigma_Y=1.0, q=0.10, missingness="MAR", eta_S=0.3)
DGP5_CONFIG_ID, DGP5_PIL_ID = 101, 10

#: DGP 11 "Mix drift d=2" (run_enrollment_drift.py CONFIGS[3]).
DRIFT_N, DRIFT_PI_L, DRIFT_D, DRIFT_GX, DRIFT_CFG = 10_000, 0.20, 2.0, 0.3, 3

CELLS = {
    1: "DGP 5 MAR, q = 0.10",
    2: "DGP 11 mix drift d = 2 (earliest 20% labeled)",
}


def _data(cell: int, rep: int):
    """(T, S, X, Y, lm, tau, fold_rng_seed, seed) for one replication."""
    if cell == 1:
        seed = derive_seed(MASTER_SEED, 5, DGP5_CONFIG_ID, DGP5_PIL_ID, rep)
        d = generate_dgp5(seed=seed, **DGP5_PARAMS)
        return (d["T"], d["S"], d["X"], d["Y"], d["labeled_mask"],
                float(d["true_tau"]), seed + 7777, seed)
    # Transcribed from scripts/run_enrollment_drift.run_one (same rng use).
    rng = np.random.default_rng(770_000 + rep * 10 + DRIFT_CFG)
    N = DRIFT_N
    e = rng.uniform(0, 1, N)
    X = (DRIFT_D * (e - 0.5) + rng.standard_normal(N)).reshape(-1, 1)
    T = (rng.random(N) < 0.5).astype(np.float64)
    gamma = 0.5 + DRIFT_GX * X[:, 0]
    S = 5.0 + X[:, 0] + gamma * T + rng.normal(0, 2.0, N)
    Y = 0.3 * S + 0.2 * X[:, 0] + rng.normal(0, 1.0, N)
    tau = 0.3 * float(gamma.mean())
    lm = np.zeros(N, dtype=bool)
    lm[np.argsort(e)[:int(DRIFT_PI_L * N)]] = True
    boot_seed = derive_seed(MASTER_SEED, 11, DRIFT_CFG, 20, rep)
    return T, S, X, Y, lm, tau, rep + 7777, boot_seed


def run_one(args: Tuple[int, int, int]) -> Dict[str, Any]:
    cell, rep, B = args
    T, S, X, Y, lm, tau, fold_seed, seed = _data(cell, rep)
    Y_hat, design = train_prediction_model(
        S, X, Y, lm, n_folds=5, rng=np.random.default_rng(fold_seed),
        return_design=True,
    )
    ppi = estimate(3, T, S, Y, Y_hat, lm)
    lam = float(ppi["lambda_hat"])
    si = estimate(2, T, S, Y, Y_hat, lm, si_variance="sandwich",
                  design=design, lambda_hat=lam)
    J = joint_influence_cov(T, Y, lm, design, lam)
    tau_si, tau_ppi = float(si["tau_hat"]), float(ppi["tau_hat"])
    D = tau_ppi - tau_si

    row: Dict[str, Any] = dict(
        cell=cell, rep=rep, true_tau=tau, tau_si=tau_si, tau_ppi=tau_ppi,
        D_hat=D, lambda_hat=lam, n_L=int(lm.sum()),
        var_si_sandwich=float(J["var_si"]), var_si_reported=float(si["var_hat"]),
        var_ppi_sandwich=float(J["var_ppi"]),
        var_ppi_exact=float(ppi["var_hat"]),
        cov_sandwich=float(J["cov_si_ppi"]), var_D_sandwich=float(J["var_D"]),
    )
    for tag, stat, var in (
        ("si_sandwich", tau_si, J["var_si"]),
        ("ppi_exact", tau_ppi, ppi["var_hat"]),
        ("ppi_sandwich", tau_ppi, J["var_ppi"]),
    ):
        row[f"cov_{tag}"] = int(abs(stat - tau) <= Z * np.sqrt(max(var, 0.0)))
    for a in ALPHAS:
        crit = stats.norm.ppf(1 - a / 2)
        row[f"rej_sandwich_a{int(a * 100):02d}"] = int(
            J["var_D"] > 0 and abs(D) / np.sqrt(J["var_D"]) > crit)

    G = _build_design_matrix(S, X)
    bs = paired_bootstrap(G, Y, T, lm, B, seed + 991)
    row.update(bs)
    row["B"] = B
    row["cov_si_boot"] = int(abs(tau_si - tau) <= Z * np.sqrt(bs["boot_var_si"]))
    row["cov_ppi_boot"] = int(abs(tau_ppi - tau)
                              <= Z * np.sqrt(bs["boot_var_ppi"]))
    for a in ALPHAS:
        crit = stats.norm.ppf(1 - a / 2)
        row[f"rej_boot_a{int(a * 100):02d}"] = int(
            bs["boot_var_D"] > 0 and abs(D) / np.sqrt(bs["boot_var_D"]) > crit)
    return row


# ---------------------------------------------------------------------------
# Summaries with Monte Carlo standard errors
# ---------------------------------------------------------------------------

def _mean_se(x: np.ndarray) -> Tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def _mcvar_se(x: np.ndarray) -> Tuple[float, float]:
    v = float(x.var(ddof=1))
    m4 = float(np.mean((x - x.mean()) ** 4))
    return v, float(np.sqrt(max(m4 - v ** 2, 0.0) / len(x)))


def _ratio_se(est: np.ndarray, stat: np.ndarray, seed: int,
              reps: int = 2000) -> Tuple[float, float]:
    point = float(est.mean() / stat.var(ddof=1))
    rng = np.random.default_rng(seed)
    R = len(est)
    draws = np.empty(reps)
    for b in range(reps):
        i = rng.integers(0, R, R)
        draws[b] = est[i].mean() / stat[i].var(ddof=1)
    return point, float(draws.std(ddof=1))


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cell, g in raw.groupby("cell"):
        R, B = len(g), int(g["B"].iloc[0])
        base = dict(cell=CELLS[int(cell)], n=10_000, R=R, B=B,
                    mean_n_L=float(g.n_L.mean()), diagnostic_null=True)
        for stat_col, name, ests in (
            ("tau_si", "Var(SI)", (("MCAR sandwich", "var_si_sandwich"),
                                   ("paired bootstrap", "boot_var_si"))),
            ("tau_ppi", "Var(PPI++)", (("MCAR sandwich", "var_ppi_sandwich"),
                                       ("exact (reported)", "var_ppi_exact"),
                                       ("paired bootstrap", "boot_var_ppi"))),
            ("D_hat", "Var(D)", (("MCAR sandwich", "var_D_sandwich"),
                                 ("paired bootstrap", "boot_var_D"))),
        ):
            stat = g[stat_col].to_numpy()
            mc, mc_se = _mcvar_se(stat)
            rows.append(dict(base, quantity=name, estimator="Monte Carlo",
                             value=mc, mc_se=mc_se, ratio_to_mc=1.0,
                             ratio_mc_se=0.0))
            for k, (tag, col) in enumerate(ests):
                v, se = _mean_se(g[col].to_numpy())
                ratio, rse = _ratio_se(g[col].to_numpy(), stat,
                                       seed=1000 * int(cell) + 10 * len(rows) + k)
                rows.append(dict(base, quantity=name, estimator=tag, value=v,
                                 mc_se=se, ratio_to_mc=ratio,
                                 ratio_mc_se=rse))
        for col, name, tag in (
            ("cov_si_sandwich", "SI coverage", "MCAR sandwich"),
            ("cov_si_boot", "SI coverage", "paired bootstrap"),
            ("cov_ppi_exact", "PPI++ coverage", "exact (reported)"),
            ("cov_ppi_sandwich", "PPI++ coverage", "MCAR sandwich"),
            ("cov_ppi_boot", "PPI++ coverage", "paired bootstrap"),
        ):
            p = float(g[col].mean())
            rows.append(dict(base, quantity=name, estimator=tag, value=p,
                             mc_se=float(np.sqrt(p * (1 - p) / R))))
        for a in ALPHAS:
            for tag, col in (("MCAR sandwich", f"rej_sandwich_a{int(a*100):02d}"),
                             ("paired bootstrap", f"rej_boot_a{int(a*100):02d}")):
                p = float(g[col].mean())
                rows.append(dict(base,
                                 quantity=f"diagnostic size (alpha = {a:.2f})",
                                 estimator=tag, value=p,
                                 mc_se=float(np.sqrt(p * (1 - p) / R))))
        for col, name in (("tau_si", "SI bias"), ("tau_ppi", "PPI++ bias")):
            err = g[col] - g.true_tau
            rows.append(dict(base, quantity=name, estimator="",
                             value=float(err.mean()),
                             mc_se=float(err.std(ddof=1) / np.sqrt(R))))
    return pd.DataFrame(rows)


def crosscheck(raw: pd.DataFrame) -> str:
    """Reproduction check against the source runs' per-replication output."""
    msgs = []
    drift_raw = os.path.join(TABLES_DIR, "enrollment_drift_raw.csv")
    g = raw[raw.cell == 2]
    if len(g) and os.path.exists(drift_raw):
        src = pd.read_csv(drift_raw)
        src = src[src.config == "Mix drift d=2"]
        for m, col in (("SI", "tau_si"), ("PPI++", "tau_ppi")):
            s = src[src.method == m].set_index("rep")["bias"]
            mine = (g.set_index("rep")[col] - g.set_index("rep")["true_tau"])
            common = mine.index.intersection(s.index)
            if len(common):
                msgs.append(f"DGP 11 {m} bias vs enrollment_drift_raw.csv: max "
                            f"|diff| = {np.max(np.abs(mine.loc[common] - s.loc[common])):.1e}"
                            f" over {len(common)} replications")
    g = raw[raw.cell == 1]
    if len(g):
        msgs.append("DGP 5 MAR draws use the core-grid seed stream "
                    "(derive_seed(42, 5, 101, 10, rep)); the core grid "
                    "stores summaries only, so the check there is the "
                    "coverage in dgp5_eval.json")
    return "; ".join(msgs) + "."


def write_markdown(summ: pd.DataFrame, path: str, note: str) -> None:
    R, B = int(summ.R.iloc[0]), int(summ.B.iloc[0])
    band = {500: "[0.931, 0.969]", 200: "[0.920, 0.980]",
            1000: "[0.936, 0.964]"}.get(R, "")
    lines = [
        "# The MCAR sandwich under nonrandom labels\n\n",
        f"R = {R} replications per cell; joint paired bootstrap with B = {B} "
        "draws per replication (units resampled within arm with their labels "
        "attached, OLS index refit under the all-units protocol, SI and "
        "PPI++ recomputed). `MCAR sandwich` is Proposition 3's joint "
        "covariance computed with the labeled-design Q-hat, the variance the "
        "reported DGP 5 MAR and DGP 11 rows use; `exact (reported)` is "
        "PPI++'s own interval. Monte Carlo = the variance of the statistic "
        "across replications (MC SE by the delta method); variance-estimator "
        "rows are means over replications (MC SE = SD / sqrt(R)); `ratio` is "
        "that mean over the Monte Carlo variance (MC SE by a bootstrap over "
        "replications). Surrogacy holds and the index is correctly specified "
        "in both cells, so the diagnostic rows are its size. Coverage and "
        "rejection MC SEs are binomial."
        + (f" Monte Carlo band for 95% coverage at R = {R}: {band}." if band
           else "") + "\n\n",
        f"Reproduction: {note}\n\n",
    ]
    for cell in summ.cell.unique():
        sub = summ[summ.cell == cell]
        lines.append(f"## {cell} (mean n_L = {sub.mean_n_L.iloc[0]:.0f})\n\n")
        lines.append("| Quantity | Estimator | Value (MC SE) | Ratio to MC (MC SE) |\n"
                     "|---|---|---:|---:|\n")
        for _, r in sub.iterrows():
            if r.quantity.startswith("Var"):
                val = f"{r.value:.3e} ({r.mc_se:.1e})"
            elif "bias" in r.quantity:
                val = f"{r.value:+.5f} ({r.mc_se:.5f})"
            else:
                val = f"{r.value:.3f} ({r.mc_se:.3f})"
            ratio = ("" if pd.isna(r.get("ratio_to_mc", np.nan))
                     else f"{r.ratio_to_mc:.3f} ({r.ratio_mc_se:.3f})")
            lines.append(f"| {r.quantity} | {r.estimator} | {val} | {ratio} |\n")
        lines.append("\n")
    with open(path, "w") as f:
        f.writelines(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--from-raw", action="store_true")
    ap.add_argument("--outdir", default=TABLES_DIR)
    args = ap.parse_args()

    raw_path = os.path.join(args.outdir, "nonrandom_label_variance_check_raw.csv")
    t0 = time.time()
    if args.from_raw:
        raw = pd.read_csv(raw_path)
    else:
        tasks = [(c, r, args.B) for c in CELLS for r in range(args.R)]
        print(f"{len(tasks)} tasks (R = {args.R}, B = {args.B})", flush=True)
        with multiprocessing.Pool(args.workers) as pool:
            out = pool.map(run_one, tasks, chunksize=4)
        raw = pd.DataFrame(out).sort_values(["cell", "rep"])
        os.makedirs(args.outdir, exist_ok=True)
        raw.to_csv(raw_path, index=False)

    summ = summarize(raw)
    summ.to_csv(os.path.join(args.outdir,
                             "nonrandom_label_variance_check.csv"), index=False)
    write_markdown(summ, os.path.join(args.outdir,
                                      "nonrandom_label_variance_check.md"),
                   crosscheck(raw))
    print(summ[["cell", "quantity", "estimator", "value", "mc_se",
                "ratio_to_mc"]].to_string(index=False))
    print(f"\nWrote nonrandom_label_variance_check.{{csv,md}} in "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
