"""
CUPED-adjusted baseline comparison simulation.

Runs DGPs 1, 2 (rho=0, 0.2, 0.4), and 8 with pi_L in {0.05, 0.20, 0.50}.
For each configuration: R=500 replications comparing labeled-only, CUPED-adjusted
labeled-only, surrogate index, and PPI++.

Reports bias, RMSE, coverage, and RE (relative to CUPED-adjusted baseline).
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.dgps.dgps import DGP_GENERATORS
from src.methods import labeled_only_cuped
from src.simulations.simulation import (
    derive_seed,
)
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.registry import write_table_rows
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


# ---- Configuration ----

R = 500
MASTER_SEED = 42
PI_L_VALUES = [0.05, 0.20, 0.50]

# DGP configurations: (dgp_id, config_label, extra_params, config_id_int)
DGP_CONFIGS: List[Tuple[int, str, Dict[str, Any], int]] = [
    (1, "DGP 1 (Valid Surrogate)", {}, 0),
    (2, "DGP 2 (rho=0.0)", {"rho": 0.0}, 0),
    (2, "DGP 2 (rho=0.2)", {"rho": 0.2}, 1),
    (2, "DGP 2 (rho=0.4)", {"rho": 0.4}, 2),
    (8, "DGP 8 (Nonlinear)", {}, 0),
]

# Methods to compare via estimate(): labeled-only(0), surrogate index(2), PPI++(3)
METHOD_IDS = [0, 2, 3]
METHOD_NAMES = {0: "Labeled-Only", 2: "Surrogate Index", 3: "PPI++"}


def run_single_rep(
    dgp_id: int,
    dgp_params: Dict[str, Any],
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Run one replication. Returns dict mapping method_name -> result dict."""
    dgp_func = DGP_GENERATORS[dgp_id]
    dgp_kwargs = dict(dgp_params)
    dgp_kwargs["seed"] = seed

    data = dgp_func(**dgp_kwargs)

    T = data["T"]
    S = data["S"]
    Y = data["Y"]
    X = data["X"]
    labeled_mask = data["labeled_mask"]
    true_tau = data["true_tau"]

    # Train prediction model for surrogate-based methods
    cf_rng = np.random.default_rng(seed + 7777)
    Y_hat, design = train_prediction_model(
        S, X, Y, labeled_mask, n_folds=5, rng=cf_rng,
        protocol=PROTOCOL, return_design=True,
    )

    results = {}

    # Standard methods via estimate()
    for m_id in METHOD_IDS:
        # Method 2 (SI) needs the design dict for the sandwich variance.
        mkwargs = {"design": design} if m_id in (2, 5) else {}
        res = estimate(m_id, T, S, Y, Y_hat, labeled_mask,
                       protocol=PROTOCOL, **mkwargs)
        results[METHOD_NAMES[m_id]] = {
            "tau_hat": res["tau_hat"],
            "var_hat": res["var_hat"],
            "ci_lower": res["ci_lower"],
            "ci_upper": res["ci_upper"],
            "true_tau": true_tau,
        }

    # CUPED-adjusted labeled-only (pass X)
    res_cuped = labeled_only_cuped(T, S, Y, Y_hat, labeled_mask, X=X)
    results["CUPED Labeled-Only"] = {
        "tau_hat": res_cuped["tau_hat"],
        "var_hat": res_cuped["var_hat"],
        "ci_lower": res_cuped["ci_lower"],
        "ci_upper": res_cuped["ci_upper"],
        "true_tau": true_tau,
    }

    return results


def compute_metrics(
    records: List[Dict[str, float]],
) -> Dict[str, float]:
    """Compute bias, RMSE, coverage, and MSE from a list of replication results."""
    tau_hats = np.array([r["tau_hat"] for r in records])
    true_taus = np.array([r["true_tau"] for r in records])
    ci_lowers = np.array([r["ci_lower"] for r in records])
    ci_uppers = np.array([r["ci_upper"] for r in records])

    # Filter out NaN replications
    valid = np.isfinite(tau_hats) & np.isfinite(ci_lowers) & np.isfinite(ci_uppers)
    if valid.sum() < 10:
        return {"bias": np.nan, "rmse": np.nan, "coverage": np.nan, "mse": np.nan}

    tau_hats = tau_hats[valid]
    true_taus = true_taus[valid]
    ci_lowers = ci_lowers[valid]
    ci_uppers = ci_uppers[valid]

    errors = tau_hats - true_taus
    bias = float(errors.mean())
    mse = float((errors ** 2).mean())
    rmse = float(np.sqrt(mse))
    coverage = float(np.mean((ci_lowers <= true_taus) & (true_taus <= ci_uppers)))

    return {"bias": bias, "rmse": rmse, "coverage": coverage, "mse": mse}


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

    all_rows = []
    total_configs = len(DGP_CONFIGS) * len(PI_L_VALUES)
    config_count = 0

    for dgp_id, dgp_label, extra_params, config_id in DGP_CONFIGS:
        for pi_L_idx, pi_L in enumerate(PI_L_VALUES):
            config_count += 1
            print(f"[{config_count}/{total_configs}] {dgp_label}, pi_L={pi_L}")

            dgp_params = dict(extra_params)
            dgp_params["pi_L"] = pi_L

            # Collect results per method
            method_records: Dict[str, List[Dict[str, float]]] = {
                name: [] for name in list(METHOD_NAMES.values()) + ["CUPED Labeled-Only"]
            }

            t0 = time.time()
            for r in range(R):
                seed = derive_seed(MASTER_SEED, dgp_id, config_id, pi_L_idx, r)
                rep_results = run_single_rep(dgp_id, dgp_params, seed)

                for method_name, res in rep_results.items():
                    method_records[method_name].append(res)

            elapsed = time.time() - t0
            print(f"  {R} reps in {elapsed:.1f}s")

            # Compute metrics for each method
            method_metrics = {}
            for method_name, records in method_records.items():
                method_metrics[method_name] = compute_metrics(records)

            # Relative efficiency against the CUPED-adjusted baseline.
            # Paper convention: RE = RMSE(baseline) / RMSE(method).
            cuped_rmse = method_metrics["CUPED Labeled-Only"]["rmse"]

            for method_name, metrics in method_metrics.items():
                re_vs_cuped = (
                    cuped_rmse / metrics["rmse"] if metrics["rmse"] > 0 else np.nan
                )
                ess_vs_cuped = (
                    re_vs_cuped ** 2 if np.isfinite(re_vs_cuped) else np.nan
                )

                all_rows.append({
                    "DGP": dgp_label,
                    "pi_L": pi_L,
                    "Method": method_name,
                    "Bias": metrics["bias"],
                    "RMSE": metrics["rmse"],
                    "Coverage": metrics["coverage"],
                    "RE vs CUPED": re_vs_cuped,
                    "ESS multiplier": ess_vs_cuped,
                })

    df = pd.DataFrame(all_rows)

    # Format as markdown table
    lines = []
    lines.append("# CUPED-Adjusted Baseline Comparison")
    lines.append("")
    lines.append(f"R = {R} replications. RE = RMSE(CUPED) / RMSE(method); >1 means method is more efficient than CUPED.")
    lines.append("ESS multiplier = RE^2 (effective-sample-size scale).")
    lines.append("")
    lines.append("| DGP | pi_L | Method | Bias | RMSE | Coverage | RE vs CUPED | ESS multiplier |")
    lines.append("|-----|------|--------|------|------|----------|-------------|----------------|")

    for _, row in df.iterrows():
        lines.append(
            f"| {row['DGP']} | {row['pi_L']:.2f} | {row['Method']} | "
            f"{row['Bias']:.4f} | {row['RMSE']:.4f} | "
            f"{row['Coverage']:.3f} | {row['RE vs CUPED']:.3f} | "
            f"{row['ESS multiplier']:.3f} |"
        )

    md_content = "\n".join(lines) + "\n"

    output_path = "results/tables/cuped_comparison.md"
    write_table_rows(
        "results/tables/cuped_comparison_rows.json", df,
        dgp="cuped", analysis="cuped_comparison", protocol=PROTOCOL,
        label_col="Method", config_cols=("DGP", "pi_L"), R=R,
        generated_by="run_cuped_comparison.py",
    )
    with open(output_path, "w") as f:
        f.write(md_content)

    print(f"\nResults saved to {output_path}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
