#!/usr/bin/env python3
"""Run the adaptive hybrid estimator on the Hillstrom dataset.

This script reports hybrid bias / RMSE / coverage / weight distribution
alongside SI and PPI++ at every pi_L, so the paper can honestly characterise
hybrid behaviour on a real violation.
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.real_data.hillstrom import load_hillstrom, compute_ground_truth_ate
from src.simulations.simulation import derive_seed
from src.simulations.simulation import METHOD_DISPLAY_NAMES
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.registry import write_table_rows
from src.utils.estimator_api import (
    estimate,
    estimate_cov_si_ppi,
    hybrid_estimator,
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

PI_L_VALUES = [0.05, 0.10, 0.20, 0.50]
R = 500
MASTER_SEED = 42
N_WORKERS = 8
C_HYBRID = 1.5
ALPHA = 0.05

_DATA = load_hillstrom()
_TRUE_TAU = compute_ground_truth_ate(_DATA["T"], _DATA["Y"])
_T = _DATA["T"]
_S = _DATA["S"]
_Y = _DATA["Y"]
_X = _DATA["X"]
_N = len(_T)


#: The SI-variance conventions the --si-variance comparison runs, in table
#: order.  "sandwich" is the default and is what the hybrid and the
#: diagnostic use everywhere else in the paper; it already carries the
#: first-stage uncertainty for the learned index.
SI_VARIANCE_CONVENTIONS = ("sandwich", "plugin", "delta")

SI_VARIANCE_LABELS = {
    "sandwich": "sandwich (default)",
    "plugin": "plug-in",
    "delta": "delta-method first stage",
}


def run_one_rep(args: Tuple) -> Dict[str, Any]:
    """One replication.

    args = (pi_L, rep_idx) or (pi_L, rep_idx, si_variance).

    ``si_variance`` names the SI variance handed to the diagnostic and to the
    hybrid, so it enters both Sigma_hat and SE(D_hat): "sandwich", "plugin", or "delta"
    (plug-in plus d' V_beta d).  Point estimates, seeds and the labeled mask
    are identical for all three.
    """
    if len(args) == 3:
        pi_L, rep_idx, si_variance = args
        if isinstance(si_variance, bool):
            si_variance = "delta" if si_variance else "sandwich"
    else:
        pi_L, rep_idx = args
        si_variance = "sandwich"
    pi_L_id = int(round(pi_L * 100))
    seed = derive_seed(MASTER_SEED, dgp_id=100, config_id=0, pi_L_id=pi_L_id, rep=rep_idx)
    rng = np.random.default_rng(seed)

    n_L = int(np.floor(pi_L * _N))
    perm = rng.permutation(_N)
    labeled_mask = np.zeros(_N, dtype=bool)
    labeled_mask[perm[:n_L]] = True

    cf_rng = np.random.default_rng(seed + 7777)
    Y_hat, design = train_prediction_model(
        _S, _X, _Y, labeled_mask,
        n_folds=5, rng=cf_rng, prediction_model="ols", seed=seed,
        protocol=PROTOCOL, return_design=True,
    )

    si_res = estimate(2, _T, _S, _Y, Y_hat, labeled_mask, protocol=PROTOCOL,
                      design=design, si_variance=si_variance)
    ppi_res = estimate(3, _T, _S, _Y, Y_hat, labeled_mask, protocol=PROTOCOL)

    var_si = si_res["var_hat"]
    var_si_plugin = si_res.get("var_plugin", si_res["var_hat"])

    lambda_hat = ppi_res.get("lambda_hat", 0.0)
    cov = estimate_cov_si_ppi(_T, _Y, Y_hat, labeled_mask, lambda_hat, design=design)

    test_res = surrogacy_test(
        tau_si=si_res["tau_hat"],
        tau_ppi=ppi_res["tau_hat"],
        var_si=var_si,
        var_ppi=ppi_res["var_hat"],
        cov_si_ppi=cov,
        alternative="two-sided",
    )

    hyb = hybrid_estimator(
        tau_si=si_res["tau_hat"],
        tau_ppi=ppi_res["tau_hat"],
        var_si=var_si,
        var_ppi=ppi_res["var_hat"],
        cov_si_ppi=cov,
        c=C_HYBRID,
        alpha=ALPHA,
        M=10_000,
        rng_seed=seed + 31337,
    )

    return {
        "pi_L": pi_L,
        "replication": rep_idx,
        "si_variance": si_variance,
        "var_si": var_si,
        "var_si_plugin": var_si_plugin,
        "tau_si": si_res["tau_hat"],
        "tau_ppi": ppi_res["tau_hat"],
        "tau_hybrid": hyb["tau_hybrid"],
        "weight": hyb["w"],
        "ci_lower": hyb["ci_lower"],
        "ci_upper": hyb["ci_upper"],
        "T_n": test_res["T_n"],
        "p_value": test_res["p_value"],
        "test_reject": int(test_res["p_value"] < ALPHA),
        "true_tau": _TRUE_TAU,
    }


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pi_L, grp in df.groupby("pi_L"):
        true_tau = grp["true_tau"].iloc[0]
        th = grp["tau_hybrid"].values
        cl = grp["ci_lower"].values
        cu = grp["ci_upper"].values
        n = len(th)
        bias = th.mean() - true_tau
        rmse = float(np.sqrt(np.mean((th - true_tau) ** 2)))
        coverage = float(np.mean((cl <= true_tau) & (true_tau <= cu)))
        mc_bias = float(np.std(th, ddof=1) / np.sqrt(n))
        mc_cov = float(np.sqrt(coverage * (1 - coverage) / n))
        rej = float(grp["test_reject"].mean())
        rows.append({
            "pi_L": pi_L,
            "n_reps": n,
            "hybrid_bias": bias,
            "hybrid_mc_bias": mc_bias,
            "hybrid_rmse": rmse,
            "hybrid_coverage": coverage,
            "hybrid_mc_cov": mc_cov,
            "weight_mean": float(grp["weight"].mean()),
            "weight_median": float(grp["weight"].median()),
            "weight_p10": float(np.percentile(grp["weight"], 10)),
            "weight_p90": float(np.percentile(grp["weight"], 90)),
            "diagnostic_reject": rej,
            "abs_rel_bias_pct": 100 * abs(bias / true_tau),
        })
    return pd.DataFrame(rows)


def main_firststage(pi_L: float = 0.20, reps: int = R):
    """Compare the SI-variance conventions inside the hybrid.

    Same replications, same seeds, same point estimates; only the SI variance
    fed to Sigma_hat and to SE(D_hat) differs.  The default is the
    joint sandwich, which already carries the first-stage uncertainty of the
    learned index, so the row that used to be labeled "plug-in" is now the
    sandwich and the comparison is against the genuinely plug-in convention
    and against the delta-method first-stage term.  Writes
    results/tables/hillstrom_hybrid_firststage.md / .csv and leaves the
    existing hillstrom_hybrid.* artifacts untouched.
    """
    print(f"Hillstrom hybrid SI-variance comparison: n={_N}, "
          f"true_tau={_TRUE_TAU:.6f}, pi_L={pi_L}, R={reps}, c={C_HYBRID}")

    out_rows = []
    raw_frames = []
    for si_variance in SI_VARIANCE_CONVENTIONS:
        label = SI_VARIANCE_LABELS[si_variance]
        tasks = [(pi_L, r, si_variance) for r in range(reps)]
        t0 = time.time()
        with Pool(N_WORKERS) as pool:
            results = pool.map(run_one_rep, tasks, chunksize=20)
        print(f"  {label}: {time.time() - t0:.1f}s")
        df = pd.DataFrame(results)
        raw_frames.append(df)
        summ = summarise(df).iloc[0].to_dict()
        summ["si_variance"] = label
        summ["si_variance_key"] = si_variance
        summ["mean_var_si"] = float(df["var_si"].mean())
        summ["mean_var_si_plugin"] = float(df["var_si_plugin"].mean())
        out_rows.append(summ)

    summary = pd.DataFrame(out_rows)
    out_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    pd.concat(raw_frames, ignore_index=True).to_csv(
        os.path.join(out_dir, "hillstrom_hybrid_firststage_raw.csv"),
        index=False)
    summary.to_csv(
        os.path.join(out_dir, "hillstrom_hybrid_firststage.csv"), index=False)

    md = [
        "# Hillstrom hybrid: the SI-variance conventions inside Sigma_hat\n",
        "\n",
        f"n={_N}, true ATE={_TRUE_TAU:.6f}, pi_L={pi_L:.2f}, R={reps}, "
        f"c={C_HYBRID}, alpha={ALPHA}\n",
        "\n",
        "All rows use identical seeds, identical labeled masks and identical "
        "point estimates (tau_SI, tau_PPI++, tau_hybrid differ only through "
        "the weight w). Each row substitutes its Var(tau_SI) into BOTH the "
        "hybrid's Sigma_hat and SE(D_hat) of the estimator-disagreement "
        "diagnostic. `sandwich (default)` is the default and is what "
        "the paper's hybrid results use: the joint sandwich for the learned "
        "index, which already carries the first-stage uncertainty. `plug-in` "
        "treats the predictions as fixed. `delta-method first stage` is the "
        "plug-in variance plus d' V_beta d.\n",
        "\n",
        "| SI variance | Mean w | w p10 / p90 | Hybrid bias (MC SE) | "
        "Abs rel bias (%) | Hybrid RMSE | Hybrid coverage (MC SE) | "
        "Diagnostic rejection | Mean Var(tau_SI) |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in summary.iterrows():
        md.append(
            f"| {r['si_variance']} "
            f"| {r['weight_mean']:.3f} "
            f"| {r['weight_p10']:.3f} / {r['weight_p90']:.3f} "
            f"| {r['hybrid_bias']:+.6f} ({r['hybrid_mc_bias']:.6f}) "
            f"| {r['abs_rel_bias_pct']:.1f} "
            f"| {r['hybrid_rmse']:.6f} "
            f"| {r['hybrid_coverage']:.3f} ({r['hybrid_mc_cov']:.4f}) "
            f"| {r['diagnostic_reject']:.3f} "
            f"| {r['mean_var_si']:.4e} |\n")
    band = {2000: "[0.940, 0.960]", 1000: "[0.936, 0.964]",
            500: "[0.931, 0.969]", 200: "[0.920, 0.980]"}.get(reps)
    band_txt = (f" The Monte Carlo band for coverage at R={reps} is {band}."
                if band else "")
    md.append(
        "\nNotes: Monte Carlo standard errors in parentheses."
        f"{band_txt} `Diagnostic rejection` is the two-sided "
        "SI--PPI++ estimator-disagreement diagnostic at the 5% level. "
        "`Mean Var(tau_SI)` is the mean over replications of the SI variance "
        "actually used in that row.\n")
    with open(os.path.join(out_dir, "hillstrom_hybrid_firststage.md"),
              "w") as f:
        f.writelines(md)

    print("\n=== Hillstrom hybrid: SI-variance comparison ===")
    cols = ["si_variance", "weight_mean", "abs_rel_bias_pct",
            "hybrid_coverage", "diagnostic_reject", "hybrid_rmse",
            "mean_var_si", "mean_var_si_plugin"]
    print(summary[cols].to_string(index=False))


def main(R_override=None):
    global R
    if R_override is not None:
        R = R_override
    print(f"Hillstrom hybrid run: n={_N}, true_tau={_TRUE_TAU:.6f}, R={R}, c={C_HYBRID}")
    tasks = [(p, r) for p in PI_L_VALUES for r in range(R)]
    print(f"Tasks: {len(tasks)}")

    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        results = pool.map(run_one_rep, tasks, chunksize=20)
    print(f"Done in {time.time() - t0:.1f}s")

    df = pd.DataFrame(results)
    out_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "hillstrom_hybrid_raw.csv"), index=False)

    summary = summarise(df)
    summary.to_csv(os.path.join(out_dir, "hillstrom_hybrid.csv"), index=False)
    write_table_rows(
        os.path.join(out_dir, "hillstrom_hybrid_rows.json"), summary,
        dgp="hillstrom", analysis="hillstrom_hybrid", protocol=PROTOCOL,
        config_cols=("pi_L",), R=R, target="masking_target",
        generated_by="scripts/hillstrom_hybrid.py",
    )

    md_lines = ["# Hillstrom Hybrid Estimator Results\n",
                f"\nn={_N}, true ATE={_TRUE_TAU:.6f}, R={R}, c={C_HYBRID}\n\n",
                "| pi_L | Hybrid bias (MC SE) | Abs Rel Bias (%) | Hybrid RMSE | Coverage (MC SE) | "
                "Mean weight | Weight p10 / p90 | Diagnostic reject |\n",
                "|---|---|---|---|---|---|---|---|\n"]
    for _, r in summary.iterrows():
        md_lines.append(
            f"| {r['pi_L']:.2f} "
            f"| {r['hybrid_bias']:+.6f} ({r['hybrid_mc_bias']:.6f}) "
            f"| {r['abs_rel_bias_pct']:.1f} "
            f"| {r['hybrid_rmse']:.6f} "
            f"| {r['hybrid_coverage']:.3f} ({r['hybrid_mc_cov']:.4f}) "
            f"| {r['weight_mean']:.3f} "
            f"| {r['weight_p10']:.3f} / {r['weight_p90']:.3f} "
            f"| {r['diagnostic_reject']:.3f} |\n"
        )
    with open(os.path.join(out_dir, "hillstrom_hybrid.md"), "w") as f:
        f.writelines(md_lines)

    print("\n=== Hillstrom hybrid summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--first-stage-variance", "--si-variance", dest="first_stage_variance",
        action="store_true",
        help="run the pi_L=0.20 comparison of the sandwich (default), "
             "plug-in and delta-method SI variance inside the hybrid, "
             "writing results/tables/hillstrom_hybrid_firststage.md",
    )
    ap.add_argument("--pi-L", type=float, default=0.20,
                    help="pi_L for --first-stage-variance (default 0.20)")
    ap.add_argument("-R", "--R", "--reps", dest="reps", type=int, default=R,
                    help=f"replications (default {R})")
    cli = ap.parse_args()

    if cli.first_stage_variance:
        main_firststage(pi_L=cli.pi_L, reps=cli.reps)
    else:
        main(R_override=cli.reps)
