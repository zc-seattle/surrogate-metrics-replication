"""
Comprehensive computational timing benchmarks for surrogate metrics paper.

Times each component (prediction model training, estimation method, CI construction)
across sample sizes and prediction models. Includes method 8 (corrected variance)
and the hybrid estimator's simulation-calibrated CI.

Usage:
    python scripts/computational_timing.py [--reps 20] [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dgps.dgps import generate_dgp1
from src.methods import (
    METHODS,
)
from src.simulations.simulation import (
    estimate_composite_weight_from_historical,
)
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.registry import write_table_rows
from src.utils.estimator_api import (
    estimate,
    estimate_cov_si_ppi,
    hybrid_estimator,
    train_prediction_model,
)


#: Prediction protocol; set from --protocol in main() and propagated to the
#: worker processes by `_set_protocol`.
PROTOCOL = DEFAULT_PROTOCOL


def _set_protocol(protocol: str) -> None:
    """Pool initializer: workers re-import this module under spawn."""
    global PROTOCOL
    PROTOCOL = protocol


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]
PREDICTION_MODELS = ["ols", "gbt"]

# Methods 0-8 (all main methods including corrected variance)
METHOD_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]

METHOD_LABELS = {
    0: "LO",
    1: "NS",
    2: "SI",
    3: "PPI++",
    4: "GREG",
    5: "CP",
    6: "AIPW",
    7: "PPI++ Boot",
    8: "PPI++ Corr",
}

# DGP 1 defaults for timing
DGP_KWARGS_BASE = dict(
    pi_L=0.20,
    alpha_S=5.0,
    beta_SX=1.0,
    gamma_S=0.3,
    sigma_S=2.0,
    alpha_Y=0.0,
    beta_YS=0.5,
    beta_YX=0.2,
    sigma_Y=1.0,
)


def _generate_data(n: int, seed: int) -> Dict[str, Any]:
    """Generate DGP 1 data for a given sample size."""
    return generate_dgp1(n=n, seed=seed, **DGP_KWARGS_BASE)


def _time_prediction(
    data: Dict[str, Any], prediction_model: str, seed: int
) -> Tuple[np.ndarray, Dict[str, Any], float]:
    """Time the prediction-model fit.

    Returns (Y_hat, design, elapsed_seconds); the design dictionary is what
    the surrogate index needs for its sandwich variance, so it is carried out
    of the timed block rather than recomputed.
    """
    cf_rng = np.random.default_rng(seed + 7777)
    t0 = time.perf_counter()
    Y_hat, design = train_prediction_model(
        data["S"], data["X"], data["Y"],
        data["labeled_mask"], n_folds=5, rng=cf_rng,
        prediction_model=prediction_model, seed=seed,
        protocol=PROTOCOL, return_design=True,
    )
    elapsed = time.perf_counter() - t0
    return Y_hat, design, elapsed


def _time_method(
    method_id: int,
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    historical_experiments: Optional[List[Dict]] = None,
    design: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], float]:
    """Time a single estimation method. Returns (result_dict, elapsed_seconds)."""
    mkwargs: Dict[str, Any] = {"protocol": PROTOCOL}
    if method_id in (2, 5) and design is not None:
        if design.get("g") is not None:
            mkwargs["design"] = design
        else:
            # GBT index: no linear design, so the joint sandwich is
            # unavailable.  Time the plug-in variance instead and say so.
            mkwargs["si_variance"] = "plugin"
    if method_id == 5 and historical_experiments is not None:
        mkwargs["historical_experiments"] = historical_experiments

    t0 = time.perf_counter()
    res = estimate(method_id, T, S, Y, Y_hat, labeled_mask, **mkwargs)
    elapsed = time.perf_counter() - t0
    return res, elapsed


def _time_hybrid_ci(
    tau_si: float,
    tau_ppi: float,
    var_si: float,
    var_ppi: float,
    cov_si_ppi: float,
    M: int = 10_000,
) -> float:
    """Time the hybrid estimator's simulation-calibrated CI construction."""
    t0 = time.perf_counter()
    hybrid_estimator(
        tau_si=tau_si,
        tau_ppi=tau_ppi,
        var_si=var_si,
        var_ppi=var_ppi,
        cov_si_ppi=cov_si_ppi,
        M=M,
        rng_seed=42,
    )
    elapsed = time.perf_counter() - t0
    return elapsed


def run_timing_benchmarks(
    R: int = 20,
    quick: bool = False,
) -> pd.DataFrame:
    """
    Run comprehensive timing benchmarks.

    Parameters
    ----------
    R : int
        Number of replications for median timing.
    quick : bool
        If True, use smaller sample sizes and fewer reps for testing.

    Returns
    -------
    DataFrame with columns:
        n, prediction_model, component, median_time_s, mean_time_s,
        min_time_s, max_time_s
    """
    sample_sizes = [1_000, 5_000, 10_000] if quick else SAMPLE_SIZES
    reps = min(R, 5) if quick else R

    # Pre-compute historical experiments for composite proxy
    _, hist_exps = estimate_composite_weight_from_historical(
        K_hist=30, beta_YS=0.5, gamma_S=0.3, seed=999,
    )

    all_records: List[Dict[str, Any]] = []

    for pred_model in PREDICTION_MODELS:
        for n in sample_sizes:
            print(f"\n{'='*60}")
            print(f"  n={n:,}, prediction_model={pred_model}, R={reps}")
            print(f"{'='*60}")

            # Storage for timing data
            pred_times: List[float] = []
            method_times: Dict[int, List[float]] = {m: [] for m in METHOD_IDS}
            hybrid_ci_times: List[float] = []

            for r in range(reps):
                seed = 42 * 1000 + r
                data = _generate_data(n, seed)

                # 1. Time prediction model
                Y_hat, design, pred_elapsed = _time_prediction(
                    data, pred_model, seed)
                pred_times.append(pred_elapsed)

                T = data["T"]
                S = data["S"]
                Y = data["Y"]
                labeled_mask = data["labeled_mask"]

                # 2. Time each estimation method
                results_cache = {}
                for m_id in METHOD_IDS:
                    res, m_elapsed = _time_method(
                        m_id, T, S, Y, Y_hat, labeled_mask,
                        historical_experiments=hist_exps, design=design,
                    )
                    method_times[m_id].append(m_elapsed)
                    results_cache[m_id] = res

                # 3. Time hybrid CI construction (M=10000)
                res_si = results_cache[2]   # surrogate index
                res_ppi = results_cache[3]  # PPI++
                lambda_hat = res_ppi.get("lambda_hat", 0.0)

                # The joint sandwich needs a linear design, which a GBT
                # index does not have; for those cells fall back to the
                # fixed-predictor covariance.  This is a timing benchmark, so
                # the covariance only has to be the right shape of
                # computation, not the reported number.
                cov_val = estimate_cov_si_ppi(
                    T, Y, Y_hat, labeled_mask, lambda_hat,
                    design=design if design.get("g") is not None else None,
                )

                hybrid_elapsed = _time_hybrid_ci(
                    tau_si=res_si["tau_hat"],
                    tau_ppi=res_ppi["tau_hat"],
                    var_si=res_si["var_hat"],
                    var_ppi=res_ppi["var_hat"],
                    cov_si_ppi=cov_val,
                    M=10_000,
                )
                hybrid_ci_times.append(hybrid_elapsed)

                if (r + 1) % 5 == 0:
                    print(f"    rep {r+1}/{reps} done")

            # Record prediction model timing
            all_records.append(dict(
                n=n,
                prediction_model=pred_model,
                component="prediction",
                median_time_s=float(np.median(pred_times)),
                mean_time_s=float(np.mean(pred_times)),
                min_time_s=float(np.min(pred_times)),
                max_time_s=float(np.max(pred_times)),
            ))

            # Record method timings
            for m_id in METHOD_IDS:
                label = METHOD_LABELS[m_id]
                times = method_times[m_id]
                all_records.append(dict(
                    n=n,
                    prediction_model=pred_model,
                    component=label,
                    median_time_s=float(np.median(times)),
                    mean_time_s=float(np.mean(times)),
                    min_time_s=float(np.min(times)),
                    max_time_s=float(np.max(times)),
                ))

            # Record hybrid CI timing
            all_records.append(dict(
                n=n,
                prediction_model=pred_model,
                component="Hybrid CI (M=10k)",
                median_time_s=float(np.median(hybrid_ci_times)),
                mean_time_s=float(np.mean(hybrid_ci_times)),
                min_time_s=float(np.min(hybrid_ci_times)),
                max_time_s=float(np.max(hybrid_ci_times)),
            ))

    return pd.DataFrame(all_records)


def _format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f}us"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.1f}ms"
    else:
        return f"{seconds:.2f}s"


def save_results(df: pd.DataFrame, output_dir: str) -> None:
    """Save timing results to CSV and markdown."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "computational_timing.csv")
    df.to_csv(csv_path, index=False)
    write_table_rows(
        csv_path.replace(".csv", "_rows.json"), df,
        dgp="timing",
        analysis="computational_timing",
        protocol=PROTOCOL,
        config_cols=("n", "prediction_model", "component"),
        n_col="n",
        generated_by="scripts/computational_timing.py",
    )
    print(f"\nSaved CSV to {csv_path}")

    # Build markdown table
    md_lines = [
        "# Computational Timing Benchmarks",
        "",
        f"Median wall-clock time per operation (over R={df.attrs.get('R', 20)} reps), "
        "DGP 1, pi_L=0.20",
        "",
    ]

    for pred_model in PREDICTION_MODELS:
        sub = df[df["prediction_model"] == pred_model]
        sample_sizes = sorted(sub["n"].unique())
        components = sub["component"].unique()

        md_lines.append(f"## Prediction Model: {pred_model.upper()}")
        md_lines.append("")

        # Header row
        header = "| Component |"
        sep = "|-----------|"
        for n in sample_sizes:
            header += f" n={n:,} |"
            sep += " --------|"
        md_lines.append(header)
        md_lines.append(sep)

        # Data rows
        for comp in components:
            row = f"| {comp} |"
            for n in sample_sizes:
                cell = sub[(sub["n"] == n) & (sub["component"] == comp)]
                if len(cell) > 0:
                    t = cell["median_time_s"].values[0]
                    row += f" {_format_time(t)} |"
                else:
                    row += " - |"
            md_lines.append(row)

        md_lines.append("")

    md_path = os.path.join(output_dir, "computational_timing.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved Markdown to {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Computational timing benchmarks for surrogate metrics."
    )
    parser.add_argument(
        "--reps", "--R", dest="reps", type=int, default=20,
        help="Number of replications for median timing (default: 20)",
    )
    parser.add_argument("--protocol", choices=PROTOCOLS,
                        default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: smaller sample sizes and fewer reps",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Computational Timing Benchmarks")
    print(f"  Replications: {args.reps}")
    print(f"  Quick mode: {args.quick}")
    print("=" * 60)

    df = run_timing_benchmarks(R=args.reps, quick=args.quick)
    df.attrs["R"] = args.reps

    output_dir = str(PROJECT_ROOT / "results" / "tables")
    save_results(df, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (median times)")
    print("=" * 60)
    for pred_model in PREDICTION_MODELS:
        sub = df[df["prediction_model"] == pred_model]
        print(f"\n--- {pred_model.upper()} ---")
        for _, row in sub.iterrows():
            print(f"  n={row['n']:>7,}  {row['component']:>20s}  "
                  f"{_format_time(row['median_time_s'])}")


if __name__ == "__main__":
    main()
