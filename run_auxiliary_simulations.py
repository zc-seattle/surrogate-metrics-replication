"""
Produces results for:
  A. AIPW comparisons
  B. Bootstrap PPI++ fix
  C. Multi-surrogate
  D. Nonlinear DGP (limitation)
  E. Mixed-validity portfolio (limitation)
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from src.utils.registry import write_table_rows

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dgps.dgps import DGP_GENERATORS
from src.evaluation import evaluate_simulation_results
from src.simulations.simulation import (
    run_simulation,
    estimate_composite_weight_from_historical,
    estimate_composite_weight_from_historical_multi,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: run a single simulation cell (for multiprocessing)
# ---------------------------------------------------------------------------

def _run_cell(args: Tuple) -> Tuple[str, pd.DataFrame]:
    """Run one simulation cell. Returns (label, raw_df)."""
    (label, dgp_id, dgp_params, method_ids, R, master_seed, hist_exps,
     protocol) = args
    t0 = time.time()
    df = run_simulation(
        dgp_id=dgp_id,
        dgp_params=dgp_params,
        method_ids=method_ids,
        R=R,
        master_seed=master_seed,
        historical_experiments=hist_exps,
        protocol=protocol,
    )
    elapsed = time.time() - t0
    print(f"  [{elapsed:5.1f}s] {label}  (R={R}, methods={method_ids})")
    return label, df


# ---------------------------------------------------------------------------
# A. AIPW comparisons
# ---------------------------------------------------------------------------

def build_aipw_cells() -> List[Tuple]:
    cells = []
    methods = [0, 2, 3, 6]
    R = 500

    # DGP 1 at varying pi_L
    for pi_L in [0.05, 0.20, 0.50]:
        label = f"A_dgp1_piL{pi_L}"
        params = dict(n=10000, alpha_S=5.0, beta_SX=1.0, gamma_S=0.3,
                      sigma_S=2.0, alpha_Y=0.0, beta_YS=0.5, beta_YX=0.2,
                      sigma_Y=1.0, pi_L=pi_L)
        cells.append((label, 1, params, methods, R, 42, None))

    # DGP 2 at varying rho
    for rho in [0.0, 0.2, 0.4]:
        label = f"A_dgp2_rho{rho}"
        params = dict(n=10000, alpha_S=5.0, beta_SX=1.0, gamma_S=0.3,
                      sigma_S=2.0, alpha_Y=0.0, beta_YS=0.5, beta_YX=0.2,
                      sigma_Y=1.0, rho=rho, pi_L=0.20)
        cells.append((label, 2, params, methods, R, 42, None))

    # DGP 5 MAR
    label = "A_dgp5_MAR"
    params = dict(n=10000, q=0.10, missingness="MAR", alpha_S=5.0,
                  beta_SX=1.0, gamma_S=0.3, sigma_S=2.0, alpha_Y=0.0,
                  beta_YS=0.5, beta_YX=0.2, sigma_Y=1.0, eta_S=0.3)
    cells.append((label, 5, params, methods, R, 42, None))

    return cells


# ---------------------------------------------------------------------------
# B. Bootstrap PPI++ fix
# ---------------------------------------------------------------------------

def build_bootstrap_cells() -> List[Tuple]:
    cells = []
    methods = [0, 3, 7]
    R = 500

    for pi_L in [0.05, 0.20, 0.50, 0.80, 1.00]:
        label = f"B_dgp1_piL{pi_L}"
        params = dict(n=10000, alpha_S=5.0, beta_SX=1.0, gamma_S=0.3,
                      sigma_S=2.0, alpha_Y=0.0, beta_YS=0.5, beta_YX=0.2,
                      sigma_Y=1.0, pi_L=pi_L)
        cells.append((label, 1, params, methods, R, 42, None))

    return cells


# ---------------------------------------------------------------------------
# C. Multi-surrogate
# ---------------------------------------------------------------------------

def build_multi_surrogate_cells() -> List[Tuple]:
    cells = []
    R = 500

    # Generate historical experiments for composite proxy (method 5)
    hist_exps = estimate_composite_weight_from_historical_multi(
        K_hist=30, J=3, seed=999,
    )

    for pi_L in [0.05, 0.20, 0.50]:
        label = f"C_dgp7_piL{pi_L}"
        params = dict(n=10000, gamma_1=0.3, gamma_2=0.2, gamma_3=0.1,
                      beta_1=0.3, beta_2=0.4, beta_3=0.2, beta_YX=0.2,
                      sigma_1=2.0, sigma_2=np.sqrt(2.0), sigma_3=1.0,
                      sigma_Y=1.0, pi_L=pi_L)
        # Methods: labeled-only, surrogate index, PPI++, composite proxy
        cells.append((label, 7, params, [0, 2, 3, 5], R, 42, hist_exps))

    return cells


# ---------------------------------------------------------------------------
# D. Nonlinear DGP (limitation)
# ---------------------------------------------------------------------------

def build_nonlinear_cells() -> List[Tuple]:
    cells = []
    methods = [0, 2, 3, 6]
    R = 500

    for pi_L in [0.05, 0.20, 0.50]:
        label = f"D_dgp8_piL{pi_L}"
        params = dict(n=10000, alpha_S=5.0, beta_SX=1.0, gamma_S=0.3,
                      sigma_S=2.0, beta_YS_linear=0.5, beta_YS_quad=-0.03,
                      beta_YX=0.2, sigma_Y=1.0, pi_L=pi_L)
        cells.append((label, 8, params, methods, R, 42, None))

    return cells


# ---------------------------------------------------------------------------
# E. Mixed-validity portfolio (limitation)
# ---------------------------------------------------------------------------

def build_mixed_validity_cells() -> List[Tuple]:
    cells = []
    methods = [0, 1, 2, 3]
    R = 200

    label = "E_dgp6_rhoMix0.3"
    params = dict(K=200, n_min=500, n_max=5000, pi_0=0.5, sigma_tau=0.10,
                  alpha_S=5.0, beta_SX=1.0, sigma_S=2.0, alpha_Y=0.0,
                  beta_YS=0.5, beta_YX=0.2, sigma_Y=1.0, alpha_decision=0.05,
                  rho_mix=0.3, pi_L=0.20)
    cells.append((label, 6, params, methods, R, 42, None))

    return cells


# ---------------------------------------------------------------------------
# Run all and produce results
# ---------------------------------------------------------------------------

def _dgp_of_label(label):
    """Recover the DGP number from an artifact label like 'C_dgp7_piL0.2'."""
    import re as _re
    m = _re.search(r"dgp(\d+)", str(label))
    return int(m.group(1)) if m else str(label)


def run_all(R_override=None, protocol="allunits_crossfit"):
    t_start = time.time()

    # Collect all cells
    all_cells = []
    all_cells.extend(build_aipw_cells())
    all_cells.extend(build_bootstrap_cells())
    all_cells.extend(build_multi_surrogate_cells())
    all_cells.extend(build_nonlinear_cells())
    all_cells.extend(build_mixed_validity_cells())

    # --R and --protocol apply uniformly to every cell.  A cell is the tuple
    # (label, dgp_id, dgp_params, method_ids, R, master_seed, hist_exps),
    # which this extends with the protocol.
    all_cells = [
        (c[0], c[1], c[2], c[3],
         R_override if R_override is not None else c[4],
         c[5], c[6], protocol)
        for c in all_cells
    ]

    print(f"Running {len(all_cells)} simulation cells...")

    # Run with multiprocessing (4 workers)
    results: Dict[str, pd.DataFrame] = {}
    with Pool(processes=4) as pool:
        for label, df in pool.imap_unordered(_run_cell, all_cells):
            results[label] = df

    print(f"\nAll simulations complete in {time.time() - t_start:.1f}s")

    # Evaluate and save
    eval_results: Dict[str, pd.DataFrame] = {}
    raw_dir = os.path.join(os.path.dirname(RESULTS_DIR), "raw")
    for label, df in sorted(results.items()):
        eval_df = evaluate_simulation_results(df)
        eval_df.insert(0, "config", label)
        # R is the replication count.  n_valid counts finite point estimates,
        # which is 0 for the portfolio cell (it is scored by regret), so it
        # cannot stand in for R in the result rows.
        eval_df.insert(1, "R", int(df["replication"].nunique()))
        eval_results[label] = eval_df

        # Keep the per-replication draws of the portfolio cell so its regret
        # summary can be re-aggregated without a re-run.
        if _dgp_of_label(label) == 6:
            os.makedirs(raw_dir, exist_ok=True)
            save_cols = [c for c in df.columns
                         if c not in ("decisions", "true_taus")]
            df[save_cols].to_parquet(
                os.path.join(raw_dir, f"{label}.parquet"), index=False
            )

        # Save raw eval CSV
        csv_path = os.path.join(RESULTS_DIR, f"{label}_eval.csv")
        eval_df.to_csv(csv_path, index=False)
        write_table_rows(
            csv_path.replace(".csv", "_rows.json"), eval_df,
            dgp=_dgp_of_label(label),
            analysis="auxiliary_simulations", protocol=protocol,
            label_col="method_name", config_cols=("config",),
            # `config` already holds the artifact label (A_dgp1_piL0.05, ...),
            # so the cell stays identifiable after the dgp field is numeric.
            R_col="R", generated_by="run_auxiliary_simulations.py",
        )

    # Build summary report
    report = build_report(eval_results)
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "new_simulation_results.md"
    )
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    return eval_results


def build_report(eval_results: Dict[str, pd.DataFrame]) -> str:
    lines = ["# New Simulation Results\n"]

    # --- Section A: AIPW comparisons ---
    lines.append("## A. AIPW Comparisons\n")

    # DGP 1 table
    lines.append("### DGP 1 (Valid Surrogate) — AIPW vs. baselines\n")
    lines.append(_make_table(eval_results, "A_dgp1",
                             param_col="pi_L", param_extractor=_extract_piL))

    # DGP 2 table
    lines.append("\n### DGP 2 (Partial Mediation) — AIPW under direct effects\n")
    lines.append(_make_table(eval_results, "A_dgp2",
                             param_col="rho", param_extractor=_extract_rho))

    # DGP 5 MAR table
    lines.append("\n### DGP 5 (MAR missingness, q=0.10)\n")
    a_mar = eval_results.get("A_dgp5_MAR")
    if a_mar is not None:
        lines.append(_format_eval_df(a_mar))

    # --- Section B: Bootstrap PPI++ ---
    lines.append("\n## B. Bootstrap PPI++ Fix\n")
    lines.append("### DGP 1 — Coverage: PPI++ vs PPI++ Bootstrap\n")
    lines.append(_make_table(eval_results, "B_dgp1",
                             param_col="pi_L", param_extractor=_extract_piL))

    # --- Section C: Multi-surrogate ---
    lines.append("\n## C. Multi-Surrogate DGP 7\n")
    lines.append(_make_table(eval_results, "C_dgp7",
                             param_col="pi_L", param_extractor=_extract_piL))

    # --- Section D: Nonlinear ---
    lines.append("\n## D. Nonlinear DGP 8\n")
    lines.append(_make_table(eval_results, "D_dgp8",
                             param_col="pi_L", param_extractor=_extract_piL))

    # --- Section E: Mixed-validity portfolio ---
    lines.append("\n## E. Mixed-Validity Portfolio (DGP 6, rho_mix=0.3)\n")
    e_mix = eval_results.get("E_dgp6_rhoMix0.3")
    if e_mix is not None:
        lines.append(_format_eval_df_portfolio(e_mix))

    return "\n".join(lines)


def _extract_piL(label: str) -> str:
    # e.g., "A_dgp1_piL0.05" -> "0.05"
    return label.split("piL")[-1]


def _extract_rho(label: str) -> str:
    return label.split("rho")[-1]


def _make_table(eval_results, prefix, param_col, param_extractor):
    """Build a markdown table combining multiple configs with a varying parameter."""
    matching = {k: v for k, v in eval_results.items() if k.startswith(prefix)}
    if not matching:
        return "(no results)\n"

    rows = []
    for label in sorted(matching.keys()):
        param_val = param_extractor(label)
        edf = matching[label]
        for _, row in edf.iterrows():
            rows.append({
                param_col: param_val,
                "Method": row["method_name"],
                "Bias": f"{row['bias']:.4f}",
                "RMSE": f"{row['rmse']:.4f}",
                "Coverage": f"{row['coverage']:.3f}",
                "RE": f"{row['relative_efficiency']:.2f}",
            })

    table_df = pd.DataFrame(rows)
    return table_df.to_markdown(index=False) + "\n"


def _format_eval_df(edf: pd.DataFrame) -> str:
    """Format a single evaluation DataFrame as markdown table."""
    display = edf[["method_name", "bias", "rmse", "coverage", "relative_efficiency"]].copy()
    display.columns = ["Method", "Bias", "RMSE", "Coverage", "RE"]
    display["Bias"] = display["Bias"].map(lambda x: f"{x:.4f}")
    display["RMSE"] = display["RMSE"].map(lambda x: f"{x:.4f}")
    display["Coverage"] = display["Coverage"].map(lambda x: f"{x:.3f}")
    display["RE"] = display["RE"].map(lambda x: f"{x:.2f}")
    return display.to_markdown(index=False) + "\n"


def _format_eval_df_portfolio(edf: pd.DataFrame) -> str:
    """Format portfolio evaluation DataFrame."""
    cols = ["method_name"]
    display_cols = ["Method"]

    if "mean_cumul_regret" in edf.columns:
        cols.append("mean_cumul_regret")
        display_cols.append("Mean Regret")
    if "rel_regret" in edf.columns:
        cols.append("rel_regret")
        display_cols.append("Rel Regret (%)")
    if "mc_se_rel_regret" in edf.columns:
        cols.append("mc_se_rel_regret")
        display_cols.append("MC SE Rel Regret")
    if "oracle_gain" in edf.columns:
        cols.append("oracle_gain")
        display_cols.append("Mean Oracle Gain")

    display = edf[cols].copy()
    display.columns = display_cols

    for c in display.columns:
        if c != "Method":
            display[c] = display[c].map(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")

    return display.to_markdown(index=False) + "\n"


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--R", type=int, default=None,
                     help="override every block's replication count")
    _ap.add_argument("--protocol", default="allunits_crossfit",
                     choices=("allunits_crossfit", "mixed_fit"))
    _args = _ap.parse_args()
    run_all(R_override=_args.R, protocol=_args.protocol)
