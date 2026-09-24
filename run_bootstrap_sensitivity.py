"""
Bootstrap sensitivity analysis for PPI++ variance correction.

Tests how the bootstrap PPI++ coverage and CI width vary with:
  - pi_L (labeled fraction): {0.05, 0.10, 0.20, 0.50, 0.80, 1.00}
  - B (bootstrap draws):     {50, 100, 200, 500, 1000}

Uses DGP 1 (valid surrogate) with n=10000, R=500 replications per cell.
True tau = 0.15 (beta_YS * gamma_S = 0.5 * 0.3).

The overlap correction belongs to the (tuning rule, variance) pair, so the
analytic columns are reported for two tuning rules on the same draws:

  Block A -- the plug-in-rule estimator (plug-in tuning rule, coefficient clipped to
    [0, 1]), which is also the rule the bootstrap loop of method id 7
    re-estimates lambda under, so the B columns belong beside it.
  Block B -- the primary PPI++ (exact tuning rule, common unclipped
    coefficient), where the correction is numerically zero and the plug-in and
    corrected columns coincide.

Parallelized with multiprocessing to use all cores.
"""

from __future__ import annotations

import hashlib
import os
import time
from multiprocessing import Pool, cpu_count
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.dgps.dgps import generate_dgp1
from src.methods import ppi_plus, ppi_plus_bootstrap
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.registry import write_table_rows
from src.utils.estimator_api import (
    train_prediction_model,
)


#: The four analytic (tuning rule, variance) combinations, in column order.
#: "pluginrule_*" is the plug-in-rule estimator (plug-in rule, clipped to [0, 1]);
#: "primary_*" is the primary PPI++ (exact rule, common unclipped
#: coefficient).
ANALYTIC_COMBINATIONS = (
    ("pluginrule_plugin", {"lambda_rule": "plugin", "clip": True,
                     "per_arm": False, "variance": "plugin"}),
    ("pluginrule_corrected", {"lambda_rule": "plugin", "clip": True,
                        "per_arm": False, "variance": "exact"}),
    ("primary_plugin", {"lambda_rule": "exact", "clip": False,
                        "per_arm": False, "variance": "plugin"}),
    ("primary_corrected", {"lambda_rule": "exact", "clip": False,
                           "per_arm": False, "variance": "exact"}),
)

#: Method label per analytic column.
COMBINATION_LABELS = {
    "pluginrule_plugin": "PPI++ (plug-in lambda, plug-in variance)",
    "pluginrule_corrected": "PPI++ (plug-in lambda, clipped)",
    "primary_plugin": "PPI++ (plug-in variance)",
    "primary_corrected": "PPI++",
    "boot": "PPI++ (bootstrap variance)",
}


#: Prediction protocol; set from --protocol in main() and propagated to the
#: worker processes by `_set_protocol`.
PROTOCOL = DEFAULT_PROTOCOL


def _set_protocol(protocol: str) -> None:
    """Pool initializer: workers re-import this module under spawn."""
    global PROTOCOL
    PROTOCOL = protocol


# ── Configuration ──────────────────────────────────────────────────────────

PI_L_VALUES = [0.05, 0.10, 0.20, 0.50, 0.80, 1.00]
B_VALUES = [50, 100, 200, 500, 1000]
R = 500            # replications per cell
N = 10_000         # sample size
TRUE_TAU = 0.15    # beta_YS * gamma_S = 0.5 * 0.3
MASTER_SEED = 42


def derive_seed(master: int, pi_L_idx: int, rep: int) -> int:
    key = f"{master}-boot-{pi_L_idx}-{rep}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16)


def run_single_rep(args):
    """Run a single replication for a given (pi_L, B) cell."""
    pi_L, pi_idx, B, r = args
    seed = derive_seed(MASTER_SEED, pi_idx, r)

    # Generate data
    data = generate_dgp1(n=N, pi_L=pi_L, seed=seed)
    T = data["T"]
    S = data["S"]
    Y = data["Y"]
    X = data["X"]
    labeled_mask = data["labeled_mask"]

    # Train prediction model
    cf_rng = np.random.default_rng(seed + 7777)
    Y_hat, design = train_prediction_model(
        S, X, Y, labeled_mask, n_folds=5, rng=cf_rng,
        protocol=PROTOCOL, return_design=True,
    )

    # The four analytic (tuning rule, variance) combinations, on this draw.
    out = {"pi_L": pi_L, "B": B}
    for col, cfg in ANALYTIC_COMBINATIONS:
        res = ppi_plus(T, S, Y, Y_hat, labeled_mask, **cfg)
        out[f"covers_{col}"] = int(
            res["ci_lower"] <= TRUE_TAU <= res["ci_upper"])
        out[f"width_{col}"] = res["ci_upper"] - res["ci_lower"]

    # Bootstrap PPI++.  Its loop re-estimates lambda with the plug-in rule and
    # clips it, so it is the bootstrap interval of the plug-in-rule estimator.
    res_boot = ppi_plus_bootstrap(
        T, S, Y, Y_hat, labeled_mask,
        B=B, ci_method="percentile", boot_seed=seed + 9999,
    )
    out["covers_boot"] = int(
        res_boot["ci_lower"] <= TRUE_TAU <= res_boot["ci_upper"])
    out["width_boot"] = res_boot["ci_upper"] - res_boot["ci_lower"]
    return out


def run_sensitivity() -> pd.DataFrame:
    # Build task list: all (pi_L, B, rep) combinations
    tasks = []
    for pi_idx, pi_L in enumerate(PI_L_VALUES):
        for B in B_VALUES:
            for r in range(R):
                tasks.append((pi_L, pi_idx, B, r))

    n_workers = min(cpu_count(), 8)
    print(f"Running {len(tasks)} total replications across {n_workers} workers...")
    t0 = time.time()

    with Pool(n_workers) as pool:
        results = pool.map(run_single_rep, tasks, chunksize=10)

    elapsed = time.time() - t0
    print(f"All done in {elapsed:.1f}s")

    # Aggregate results by (pi_L, B)
    cols = [c for c, _ in ANALYTIC_COMBINATIONS] + ["boot"]
    raw = pd.DataFrame(results)
    grouped = raw.groupby(["pi_L", "B"], as_index=False).mean()

    rows = []
    for _, g in grouped.iterrows():
        row = {"pi_L": g["pi_L"], "B": int(g["B"])}
        for c in cols:
            row[f"coverage_{c}"] = g[f"covers_{c}"]
            row[f"width_{c}"] = g[f"width_{c}"]
        row["coverage_plugin"] = row["coverage_pluginrule_plugin"]
        row["width_plugin"] = row["width_pluginrule_plugin"]
        row["coverage_corrected"] = row["coverage_pluginrule_corrected"]
        row["width_corrected"] = row["width_pluginrule_corrected"]
        rows.append(row)

    return pd.DataFrame(rows)


def _block(df: pd.DataFrame, plug_col: str, corr_col: str,
           stem: str, fmt: str, with_boot: bool) -> list:
    """One table: the two analytic columns of a block, optionally + the B grid."""
    sep = "|------|---------|-----------|" + "|".join(
        "------" for _ in B_VALUES) + "|"
    header = ("| pi_L | Plug-in | Corrected |"
              + "|".join(f" B={B} " for B in B_VALUES) + "|")
    if not with_boot:
        header = "| pi_L | Plug-in | Corrected |"
        sep = "|------|---------|-----------|"
    lines = [header, sep]
    for pi_L in PI_L_VALUES:
        sub = df[df["pi_L"] == pi_L]
        if sub.empty:
            continue
        a = format(sub[f"{stem}_{plug_col}"].iloc[0], fmt)
        c = format(sub[f"{stem}_{corr_col}"].iloc[0], fmt)
        row = f"| {pi_L:.2f} | {a} | {c} |"
        if with_boot:
            for B in B_VALUES:
                cell = sub[sub["B"] == B]
                if cell.empty:
                    row += " --- |"
                    continue
                row += f" {format(cell[f'{stem}_boot'].iloc[0], fmt)} |"
        lines.append(row)
    lines.append("")
    return lines


def format_table(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Bootstrap Sensitivity Analysis: PPI++ Variance Correction")
    lines.append("")
    lines.append(f"DGP 1 (valid surrogate), n=10000, R={R} replications per cell.")
    lines.append(f"True tau = {TRUE_TAU}. Nominal coverage = 95%.")
    lines.append("")
    lines.append("Plug-in = PPI++ analytic variance without the overlap "
                 "correction (ppi_plus, variance=\"plugin\").")
    lines.append("Corrected = PPI++ exact (overlap-corrected) analytic "
                 "variance.")
    lines.append("B=... = PPI++ percentile bootstrap with B resamples.")
    lines.append("")
    lines.append("The correction is a property of the (tuning rule, variance) "
                 "pair, so both analytic columns are reported under two tuning "
                 "rules on the same draws. Block A is the plug-in-rule estimator: "
                 "plug-in tuning rule, coefficient clipped to [0, 1]. Block B "
                 "is the primary PPI++: exact tuning rule, common "
                 "unclipped coefficient, where the correction is numerically "
                 "zero and the two columns coincide. The B columns sit in "
                 "block A because the bootstrap loop re-estimates lambda "
                 "under the plug-in rule and clips it.")
    lines.append("")

    for stem, fmt, title in (("coverage", ".3f", "Coverage (95% CI)"),
                             ("width", ".4f", "Mean CI Width")):
        lines.append(f"## Block A, {title}: plug-in-rule estimator "
                     "(plug-in rule, clipped)")
        lines.append("")
        lines += _block(df, "pluginrule_plugin", "pluginrule_corrected", stem, fmt, True)
        lines.append(f"## Block B, {title}: primary PPI++ "
                     "(exact rule, unclipped)")
        lines.append("")
        lines += _block(df, "primary_plugin", "primary_corrected", stem, fmt,
                        False)

    # ── Combined table (block A) ──
    lines.append("## Combined, block A: Coverage (CI Width)")
    lines.append("")
    header = ("| pi_L | Plug-in | Corrected |"
              + "|".join(f" B={B} " for B in B_VALUES) + "|")
    sep = "|------|---------|-----------|" + "|".join(
        "------" for _ in B_VALUES) + "|"
    lines.append(header)
    lines.append(sep)

    for pi_L in PI_L_VALUES:
        sub = df[df["pi_L"] == pi_L]
        if sub.empty:
            continue
        row = (f"| {pi_L:.2f} "
               f"| {sub['coverage_pluginrule_plugin'].iloc[0]:.3f} "
               f"({sub['width_pluginrule_plugin'].iloc[0]:.4f}) "
               f"| {sub['coverage_pluginrule_corrected'].iloc[0]:.3f} "
               f"({sub['width_pluginrule_corrected'].iloc[0]:.4f}) |")
        for B in B_VALUES:
            cell = sub[sub["B"] == B]
            if cell.empty:
                row += " --- |"
                continue
            row += (f" {cell['coverage_boot'].iloc[0]:.3f} "
                    f"({cell['width_boot'].iloc[0]:.4f}) |")
        lines.append(row)

    lines.append("")
    lines.append("Notes: the analytic columns do not depend on B; they are "
                 "constant across the B grid by construction and are printed "
                 "once. Method labels: `PPI++ (plug-in lambda, plug-in "
                 "variance)` and `PPI++ (plug-in lambda, clipped)` for block "
                 "A, `PPI++ (plug-in variance)` and `PPI++` for block B, "
                 "`PPI++ (bootstrap variance)` for the B columns.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--R", type=int, default=R,
                     help="Monte Carlo replications (smoke runs use a few)")
    _ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    _args = _ap.parse_args()
    R = _args.R
    PROTOCOL = _args.protocol

    print("Bootstrap sensitivity analysis (parallelized)")
    print("=" * 60)
    df = run_sensitivity()

    table = format_table(df)
    print("\n" + table)

    out_path = "results/tables/bootstrap_sensitivity.md"
    # Result rows: one per (pi_L, B, configuration), labeled by the paper
    # label of that (tuning rule, variance) pair.  The analytic columns do not
    # depend on B, so they are filed once, at B = nan.
    long_rows = []
    for _, r in df.iterrows():
        if int(r["B"]) == B_VALUES[0]:
            for col, _cfg in ANALYTIC_COMBINATIONS:
                long_rows.append({
                    "pi_L": r["pi_L"], "B": float("nan"), "n": N,
                    "method_label": COMBINATION_LABELS[col],
                    "coverage": r[f"coverage_{col}"],
                    "ci_width": r[f"width_{col}"],
                })
        long_rows.append({
            "pi_L": r["pi_L"], "B": int(r["B"]), "n": N,
            "method_label": COMBINATION_LABELS["boot"],
            "coverage": r["coverage_boot"],
            "ci_width": r["width_boot"],
        })
    write_table_rows(
        "results/tables/bootstrap_sensitivity_rows.json",
        pd.DataFrame(long_rows),
        dgp=1, analysis="bootstrap_sensitivity", protocol=PROTOCOL,
        label_col="method_label", config_cols=("pi_L", "B"),
        R=R, generated_by="run_bootstrap_sensitivity.py",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(table)
    print(f"\nSaved to {out_path}")
