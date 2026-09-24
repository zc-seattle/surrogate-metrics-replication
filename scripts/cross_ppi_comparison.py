#!/usr/bin/env python3
"""Cross-PPI comparison simulation.

Implements Cross-PPI and compares to PPI++ (with cross-fitting) on
DGPs 1, 2 (ρ=0, 0.2, 0.4) at π_L=0.20, R=500.

Cross-PPI splits the labeled data into two halves:
  - Half 1: train f (prediction model)
  - Half 2: compute the PPI correction (rectifier)
Then swaps roles and averages.

Uses 8-core multiprocessing.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.dgps.dgps import generate_dgp1, generate_dgp2
from src.simulations.simulation import derive_seed
from src.methods import labeled_only, ppi_plus, cross_ppi
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.registry import write_table_rows
from src.utils.estimator_api import (
    train_prediction_model,
)


#: Prediction protocol; set from --protocol in main() and propagated to the
#: worker processes by `_set_protocol`.
PROTOCOL = DEFAULT_PROTOCOL


def _set_protocol(protocol: str) -> None:
    """Pool initializer: workers re-import this module under spawn."""
    global PROTOCOL
    PROTOCOL = protocol

# ── Configuration ──────────────────────────────────────────────────────────

DGP_CONFIGS = [
    (1, "DGP1", {}),
    (2, "DGP2_rho0.0", {"rho": 0.0}),
    (2, "DGP2_rho0.2", {"rho": 0.2}),
    (2, "DGP2_rho0.4", {"rho": 0.4}),
]

DGP_FUNCS = {1: generate_dgp1, 2: generate_dgp2}

PI_L = 0.20
N = 10_000
R = 500
MASTER_SEED = 42
N_WORKERS = 8


# ── Single-replication worker ─────────────────────────────────────────────

def run_one_rep(args: Tuple) -> Dict[str, Any]:
    """Run one replication for a given (dgp_config, rep_id)."""
    dgp_id, dgp_label, dgp_extra, rep_id = args

    config_id = int(hashlib.sha256(dgp_label.encode()).hexdigest()[:8], 16) % 10000
    pi_L_id = int(PI_L * 100)
    seed = derive_seed(MASTER_SEED, dgp_id, config_id, pi_L_id, rep_id)

    # Generate data
    dgp_func = DGP_FUNCS[dgp_id]
    dgp_kwargs = {"n": N, "pi_L": PI_L, "seed": seed}
    dgp_kwargs.update(dgp_extra)
    data = dgp_func(**dgp_kwargs)

    T = data["T"]
    S = data["S"]
    Y = data["Y"]
    X = data["X"]
    labeled_mask = data["labeled_mask"]
    true_tau = data["true_tau"]

    # --- Standard PPI++ (with cross-fitting in the prediction model) ---
    cf_rng = np.random.default_rng(seed + 7777)
    Y_hat, design = train_prediction_model(S, X, Y, labeled_mask, n_folds=5, rng=cf_rng, protocol=PROTOCOL, return_design=True)

    res_ppi = ppi_plus(T, S, Y, Y_hat, labeled_mask)
    res_lo = labeled_only(T, S, Y, Y_hat, labeled_mask)

    # --- Cross-PPI ---
    res_cppi = cross_ppi(T, S, X, Y, labeled_mask, seed=seed)

    ppi_covers = 1.0 if res_ppi["ci_lower"] <= true_tau <= res_ppi["ci_upper"] else 0.0
    cppi_covers = 1.0 if res_cppi["ci_lower"] <= true_tau <= res_cppi["ci_upper"] else 0.0
    lo_covers = 1.0 if res_lo["ci_lower"] <= true_tau <= res_lo["ci_upper"] else 0.0

    return {
        "dgp_label": dgp_label,
        "rep": rep_id,
        "true_tau": true_tau,
        # PPI++
        "ppi_tau": res_ppi["tau_hat"],
        "ppi_var": res_ppi["var_hat"],
        "ppi_covers": ppi_covers,
        # Cross-PPI
        "cppi_tau": res_cppi["tau_hat"],
        "cppi_var": res_cppi["var_hat"],
        "cppi_covers": cppi_covers,
        # LO
        "lo_tau": res_lo["tau_hat"],
        "lo_var": res_lo["var_hat"],
        "lo_covers": lo_covers,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse

    global R, PROTOCOL
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--R", type=int, default=R,
                     help="Monte Carlo replications (smoke runs use a few)")
    _ap.add_argument("--protocol", choices=PROTOCOLS,
                     default=DEFAULT_PROTOCOL)
    _args = _ap.parse_args()
    R = _args.R
    PROTOCOL = _args.protocol

    print("Cross-PPI Comparison")
    print(f"  DGP configs: {[c[1] for c in DGP_CONFIGS]}")
    print(f"  pi_L={PI_L}, n={N}, R={R}")
    print()

    # Build task list
    tasks = []
    for dgp_id, dgp_label, dgp_extra in DGP_CONFIGS:
        for r in range(R):
            tasks.append((dgp_id, dgp_label, dgp_extra, r))

    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks}")

    t0 = time.time()

    with Pool(N_WORKERS) as pool:
        results = pool.map(run_one_rep, tasks, chunksize=10)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    df = pd.DataFrame(results)

    # Aggregate
    summary_rows = []
    for dgp_label, grp in df.groupby("dgp_label"):
        true_tau = grp["true_tau"].iloc[0]

        # Per-cell labeled-only RMSE is the baseline for relative efficiency.
        # Paper convention: RE = RMSE_LO / RMSE_method (>1 means more efficient).
        rmse_lo = np.sqrt(((grp["lo_tau"] - true_tau) ** 2).mean())

        for method, tau_col, var_col, cov_col in [
            ("LO", "lo_tau", "lo_var", "lo_covers"),
            ("PPI++", "ppi_tau", "ppi_var", "ppi_covers"),
            ("Cross-PPI", "cppi_tau", "cppi_var", "cppi_covers"),
        ]:
            bias = grp[tau_col].mean() - true_tau
            rmse = np.sqrt(((grp[tau_col] - true_tau) ** 2).mean())
            coverage = grp[cov_col].mean()
            mean_var = grp[var_col].mean()

            # RE vs LO (paper convention: ratio of RMSEs against per-cell LO).
            # ESS multiplier is the squared version (variance-scale ratio).
            re_vs_lo = rmse_lo / rmse if rmse > 0 else np.nan
            ess_vs_lo = re_vs_lo ** 2 if np.isfinite(re_vs_lo) else np.nan

            summary_rows.append({
                "DGP": dgp_label,
                "Method": method,
                "true_tau": round(true_tau, 4),
                "Bias": round(bias, 5),
                "RMSE": round(rmse, 5),
                "Coverage": round(coverage, 3),
                "Mean_Var": round(mean_var, 6),
                "RE_vs_LO": round(re_vs_lo, 3),
                "ESS_multiplier": round(ess_vs_lo, 3),
            })

    summary = pd.DataFrame(summary_rows)

    # Save
    out_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "cross_ppi_comparison.csv")
    md_path = os.path.join(out_dir, "cross_ppi_comparison.md")

    summary.to_csv(csv_path, index=False)
    write_table_rows(
        csv_path.replace(".csv", "_rows.json"), summary,
        dgp="cross_ppi",
        analysis="cross_ppi_comparison",
        protocol=PROTOCOL,
        label_col="Method",
        config_cols=("DGP",),
        R=R,
        generated_by="scripts/cross_ppi_comparison.py",
    )

    # Markdown
    lines = [
        "# Cross-PPI Comparison",
        "",
        f"n={N}, pi_L={PI_L}, R={R}",
        "",
        "RE_vs_LO = RMSE(labeled-only) / RMSE(method), computed against the "
        "per-cell labeled-only RMSE; >1 means the method is more efficient.",
        "ESS multiplier = RE_vs_LO^2 (effective-sample-size scale).",
        "",
        "| DGP | Method | true_tau | Bias | RMSE | Coverage | Mean_Var | RE_vs_LO | ESS multiplier |",
        "|-----|--------|----------|------|------|----------|----------|----------|----------------|",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['DGP']} | {r['Method']} "
            f"| {r['true_tau']:.4f} | {r['Bias']:.5f} | {r['RMSE']:.5f} "
            f"| {r['Coverage']:.3f} | {r['Mean_Var']:.6f} | {r['RE_vs_LO']:.3f} "
            f"| {r['ESS_multiplier']:.3f} |"
        )

    md_text = "\n".join(lines) + "\n"
    with open(md_path, "w") as f:
        f.write(md_text)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved MD:  {md_path}")
    print()
    print(md_text)


if __name__ == "__main__":
    main()
