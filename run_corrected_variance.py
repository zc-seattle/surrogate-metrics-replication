#!/usr/bin/env python3
"""
Simulation 4: Corrected Variance Verification.

DGP 1, pi_L = {0.05, 0.10, 0.20, 0.50, 0.80, 1.00}, R = 2000.

The overlap correction is a property of the (tuning rule, variance) PAIR, not
of the variance alone, so the run reports two blocks on the same draws:

  Block A -- the plug-in-rule estimator: plug-in tuning rule, coefficient clipped to
    [0, 1]. Its three intervals are the plug-in (uncorrected) variance, the
    exact (overlap-corrected) variance, and the percentile bootstrap.

  Block B -- the primary PPI++: exact tuning rule, common unclipped
    coefficient.  Same two analytic variances.  At the exact rule the
    correction is numerically zero, so the two columns coincide; the block is
    here to show that, not to add a result.

Every replication runs all five configurations on one draw and one fitted
prediction model, so the blocks are paired cell by cell.

Saves results/tables/corrected_variance_verification.{md,csv} and the method table
rows in corrected_variance_verification_rows.json.
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

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dgps.dgps import generate_dgp1
from src.simulations.simulation import derive_seed
from src.utils.config import (
    DEFAULT_PROTOCOL,
    PROTOCOLS,
    default_spec_for_id,
    make_result_row,
    spec_by_key,
)
from src.utils.registry import write_rows
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

Z_ALPHA = stats.norm.ppf(0.975)

PI_L_VALUES = [0.05, 0.10, 0.20, 0.50, 0.80, 1.00]

#: The five combinations. "pluginrule_*" is the plug-in-rule estimator (plug-in
#: rule, clipped); "primary_*" is the primary (exact rule, common unclipped
#: coefficient).
COMBINATIONS: Tuple[Tuple[str, int, str, Dict[str, Any]], ...] = (
    ("pluginrule_plugin", 3, "ppi_pluginrule",
     {"lambda_rule": "plugin", "clip": True, "per_arm": False,
      "variance": "plugin"}),
    ("pluginrule_exact", 8, "",
     {"lambda_rule": "plugin", "clip": True, "per_arm": False,
      "variance": "exact"}),
    ("pluginrule_boot", 7, "", {}),
    ("primary_plugin", 3, "ppi_plugin_var",
     {"lambda_rule": "exact", "clip": False, "per_arm": False,
      "variance": "plugin"}),
    ("primary_exact", 3, "ppi",
     {"lambda_rule": "exact", "clip": False, "per_arm": False,
      "variance": "exact"}),
)

COLUMN_ALIASES = {
    "plugin": "pluginrule_plugin",
    "corrected": "pluginrule_exact",
    "boot": "pluginrule_boot",
}


def _spec_for(key: str, method_id: int):
    """The method table MethodSpec a combination's rows are filed under."""
    if key:
        return spec_by_key(key)
    return default_spec_for_id(method_id)


def run_one_rep(args: Tuple) -> Dict[str, Any]:
    rep, pi_L, master_seed = args
    rep_seed = derive_seed(master_seed, 1, 0, int(pi_L * 100), rep)

    data = generate_dgp1(n=10000, pi_L=pi_L, seed=rep_seed)
    true_tau = data["true_tau"]

    cf_rng = np.random.default_rng(rep_seed + 7777)
    Y_hat, design = train_prediction_model(
        data["S"], data["X"], data["Y"],
        data["labeled_mask"], n_folds=5, rng=cf_rng,
        protocol=PROTOCOL, return_design=True,
    )

    T, S, Y, lm = data["T"], data["S"], data["Y"], data["labeled_mask"]

    result: Dict[str, Any] = {
        "rep": rep,
        "pi_L": pi_L,
        "true_tau": true_tau,
    }

    for col, method_id, _key, cfg in COMBINATIONS:
        if method_id == 7:
            res = estimate(7, T, S, Y, Y_hat, lm, protocol=PROTOCOL,
                           B=200, boot_seed=rep_seed + 1111)
        else:
            res = estimate(method_id, T, S, Y, Y_hat, lm, protocol=PROTOCOL,
                           **cfg)

        ci_lo, ci_hi = res["ci_lower"], res["ci_upper"]
        result[f"tau_hat_{col}"] = res["tau_hat"]
        result[f"var_hat_{col}"] = res["var_hat"]
        result[f"ci_lower_{col}"] = ci_lo
        result[f"ci_upper_{col}"] = ci_hi
        result[f"covers_{col}"] = 1 if (ci_lo <= true_tau <= ci_hi) else 0
        result[f"ci_width_{col}"] = ci_hi - ci_lo
        if "lambda_hat" in res:
            result[f"lambda_hat_{col}"] = res["lambda_hat"]
        if "delta_V" in res:
            result[f"delta_V_{col}"] = res["delta_V"]

    # delta_V of the plug-in-rule estimator keeps its historical bare name.
    if "delta_V_pluginrule_exact" in result:
        result["delta_V"] = result["delta_V_pluginrule_exact"]

    for old, new in COLUMN_ALIASES.items():
        for stem in ("tau_hat", "var_hat", "ci_lower", "ci_upper", "covers",
                     "ci_width"):
            result[f"{stem}_{old}"] = result[f"{stem}_{new}"]

    return result


def _block_lines(df: pd.DataFrame, cols: Tuple[str, ...],
                 headers: Tuple[str, ...]) -> List[str]:
    """One coverage / CI-width table over the pi_L grid."""
    head = "| pi_L | " + " | ".join(
        [f"{h} Coverage" for h in headers]
        + [f"{h} CI Width" for h in headers]
    ) + " |\n"
    sep = "|------|" + "|".join("------" for _ in range(2 * len(headers))) + "|\n"
    lines = [head, sep]
    for pi_L in PI_L_VALUES:
        sub = df[df["pi_L"] == pi_L]
        if sub.empty:
            continue
        cov = []
        wid = []
        for c in cols:
            if f"covers_{c}" not in sub.columns:
                cov.append("---")
                wid.append("---")
                continue
            cov.append(f"{sub[f'covers_{c}'].mean():.3f}")
            wid.append(f"{sub[f'ci_width_{c}'].mean():.4f}")
        lines.append(f"| {pi_L} | " + " | ".join(cov + wid) + " |\n")
    return lines


def _registry_rows(df: pd.DataFrame, R: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pi_L in PI_L_VALUES:
        sub = df[df["pi_L"] == pi_L]
        if sub.empty:
            continue
        truth = float(sub["true_tau"].mean())
        for col, method_id, key, _cfg in COMBINATIONS:
            spec = _spec_for(key, method_id)
            err = sub[f"tau_hat_{col}"].values - sub["true_tau"].values
            n_valid = int(np.isfinite(err).sum())
            bias = float(np.nanmean(err))
            rmse = float(np.sqrt(np.nanmean(err ** 2)))
            cov = float(sub[f"covers_{col}"].mean())
            width = float(sub[f"ci_width_{col}"].mean())
            rows.append(make_result_row(
                dgp=1, config_name=f"DGP1_piL{pi_L:.2f}",
                params={"pi_L": pi_L, "block": col.split("_")[0]},
                pi_L=pi_L, n=10000, R=R, protocol=PROTOCOL, spec=spec,
                alpha=0.05, target="ATE", seed=42,
                metrics={
                    "true_tau": truth, "n_valid": n_valid, "bias": bias,
                    "rel_bias": (100.0 * bias / truth) if truth else float("nan"),
                    "rmse": rmse, "coverage": cov,
                    "mc_se_coverage": float(
                        np.sqrt(cov * (1 - cov) / max(n_valid, 1))),
                    "mc_se_bias": float(
                        np.nanstd(err, ddof=1) / np.sqrt(max(n_valid, 1))),
                    "ci_width": width,
                },
                extra={
                    "analysis": "corrected_variance_verification",
                    "column": col,
                    "mean_delta_V": (
                        float(sub[f"delta_V_{col}"].mean())
                        if f"delta_V_{col}" in sub.columns else float("nan")
                    ),
                    "mean_var_hat": float(sub[f"var_hat_{col}"].mean()),
                },
            ))
    return rows


def main():
    global PROTOCOL
    parser = argparse.ArgumentParser(description="Corrected variance verification")
    parser.add_argument("--R", type=int, default=2000)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", choices=PROTOCOLS,
                        default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    R = args.R
    cores = args.cores or min(8, os.cpu_count() or 1)
    master_seed = args.seed
    PROTOCOL = args.protocol

    tasks = []
    for pi_L in PI_L_VALUES:
        for r in range(R):
            tasks.append((r, pi_L, master_seed))

    print(f"Corrected Variance Verification: {len(tasks)} tasks, R={R}, {cores} cores")

    t0 = time.time()
    with multiprocessing.Pool(cores, initializer=_set_protocol,
                              initargs=(PROTOCOL,)) as pool:
        all_results = pool.map(run_one_rep, tasks)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    df = pd.DataFrame(all_results)

    output_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    lines = ["# Corrected Variance Verification\n\n"]
    lines.append(f"DGP 1, n=10000, R={R}\n\n")

    lines.append("## Block A: the plug-in-rule estimator (plug-in tuning rule, "
                 "coefficient clipped to [0, 1])\n\n")
    lines.append("Plug-in = the uncorrected analytic variance. Corrected = "
                 "the exact (overlap-corrected) analytic variance. Bootstrap "
                 "= the percentile bootstrap of method id 7 (B=200), whose "
                 "loop re-estimates lambda under the same plug-in rule.\n\n")
    lines.extend(_block_lines(
        df, ("pluginrule_plugin", "pluginrule_exact", "pluginrule_boot"),
        ("Plug-in", "Corrected", "Bootstrap"),
    ))

    lines.append("\n## Block B: the primary PPI++ (exact tuning "
                 "rule, common unclipped coefficient)\n\n")
    lines.append("Same draws and the same fitted prediction model as block A. "
                 "At the exact rule the overlap correction is numerically "
                 "zero, so the two columns coincide. There is no bootstrap "
                 "column: method id 7 re-estimates lambda with the plug-in "
                 "rule and clips it, so its bootstrap belongs to block A.\n\n")
    lines.extend(_block_lines(
        df, ("primary_plugin", "primary_exact", "__none__"),
        ("Plug-in", "Corrected", "Bootstrap"),
    ))

    lines.append("\n## Detailed Delta_V Statistics\n\n")
    lines.append("delta_V is the overlap correction added to the plug-in "
                 "variance, reported for both tuning rules.\n\n")
    lines.append("| pi_L | Rule | Mean delta_V | Median delta_V | "
                 "Mean var_plugin | Mean var_corrected | "
                 "Ratio (delta_V / var_plugin) |\n")
    lines.append("|------|------|-------------|----------------|"
                 "-----------------|--------------------|--------------|\n")

    for pi_L in PI_L_VALUES:
        sub = df[df["pi_L"] == pi_L]
        if sub.empty:
            continue
        for rule, plug_col, exact_col in (
            ("plug-in", "pluginrule_plugin", "pluginrule_exact"),
            ("exact", "primary_plugin", "primary_exact"),
        ):
            dv_col = f"delta_V_{exact_col}"
            if dv_col not in sub.columns:
                lines.append(f"| {pi_L} | {rule} | N/A | N/A | - | - | - |\n")
                continue
            dv = sub[dv_col].dropna()
            if dv.empty:
                lines.append(f"| {pi_L} | {rule} | N/A | N/A | - | - | - |\n")
                continue
            mean_vp = sub[f"var_hat_{plug_col}"].mean()
            mean_vc = sub[f"var_hat_{exact_col}"].mean()
            ratio = dv.mean() / mean_vp if mean_vp > 0 else np.nan
            ratio_str = f"{ratio:.4f}" if not np.isnan(ratio) else "-"
            lines.append(
                f"| {pi_L} | {rule} | {dv.mean():.6f} | {dv.median():.6f} "
                f"| {mean_vp:.6f} | {mean_vc:.6f} | {ratio_str} |\n")

    lines.append(
        "\nNotes: R = "
        f"{R} replications per pi_L, DGP 1, n = 10000, nominal coverage 0.95. "
        "The Monte Carlo band at R = 2,000 is [0.940, 0.960]. Blocks A and B "
        "run on the same draws and the same fitted prediction model, so they "
        "are paired cell by cell. Method labels: block A is "
        "`PPI++ (plug-in lambda, plug-in variance)` and "
        "`PPI++ (plug-in lambda, clipped)` plus `PPI++ (bootstrap variance)`; "
        "block B is `PPI++ (plug-in variance)` and `PPI++`.\n")

    output_path = os.path.join(output_dir, "corrected_variance_verification.md")
    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"Results saved to {output_path}")

    csv_path = os.path.join(output_dir, "corrected_variance_verification.csv")
    df.to_csv(csv_path, index=False)
    write_rows(
        csv_path.replace(".csv", "_rows.json"), _registry_rows(df, R),
        generated_by="run_corrected_variance.py", protocol=PROTOCOL,
    )
    print(f"Raw results saved to {csv_path}")


if __name__ == "__main__":
    main()
