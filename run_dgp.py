"""
Parallelized simulation runner for a single DGP (core grid, DGPs 1-6).

Usage:
    python run_dgp.py --dgp 1 --R 2000
    python run_dgp.py --dgp 6 --R 2000          # portfolio

Each DGP's configurations are run in parallel using multiprocessing.

Output
------
results/raw/<label>.parquet
    One row per (replication, method configuration).
results/tables/dgp<N>_eval.json
    Results schema version 2: {"registry_version": 2, "rows": [ResultRow,
    ...]}. Every row carries dgp, config_name, params, pi_L, n, R, protocol,
    method_id, method_label, the four configuration axes, alpha, target and
    seed, plus the Monte Carlo metrics.

Both the primary and the ablation configurations are evaluated in the same
replication, on the same draws and the same predictions, so the ablation
tables are paired with the headline tables.  Pass --no-ablation to skip them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dgps.dgps import DGP_GENERATORS
from src.simulations.simulation import (
    run_simulation,
    estimate_composite_weight_from_historical,
)
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS, resolve_specs
from src.utils.registry import rows_from_replications, write_rows


# The core grid runs the paper's primary configurations (LO, NS, SI, PPI++,
# GREG, CP, AIPW) plus, unless --no-ablation is given, the six ablation
# configurations, all on the same draws.
CORE_METHOD_IDS = (0, 1, 2, 3, 4, 5, 6)


# ---- Configuration per DGP ----

def get_dgp_configs(dgp_id: int, R: int = 2000) -> List[Dict[str, Any]]:
    """Return list of simulation configs for a DGP."""

    # Common defaults
    BASE = dict(
        n=10000, alpha_S=5.0, beta_SX=1.0, gamma_S=0.3,
        sigma_S=2.0, alpha_Y=0.0, beta_YS=0.5, beta_YX=0.2, sigma_Y=1.0,
    )

    configs = []

    if dgp_id == 1:
        pi_Ls = [0.05, 0.20, 0.50, 1.00]
        for pi_L in pi_Ls:
            params = {**BASE, "pi_L": pi_L}
            configs.append(dict(
                dgp_id=1, dgp_params=params, R=R, pi_L=pi_L,
                config_id=0, pi_L_id=int(pi_L * 100),
                label=f"DGP1_piL{pi_L:.2f}",
            ))

    elif dgp_id == 2:
        rhos = [0.0, 0.2, 0.4, 0.6, 0.8]
        pi_Ls = [0.10, 0.20, 0.50, 1.00]
        for i, rho in enumerate(rhos):
            for pi_L in pi_Ls:
                params = {**BASE, "pi_L": pi_L, "rho": rho}
                configs.append(dict(
                    dgp_id=2, dgp_params=params, R=R, pi_L=pi_L,
                    config_id=i, pi_L_id=int(pi_L * 100),
                    label=f"DGP2_rho{rho:.1f}_piL{pi_L:.2f}",
                ))

    elif dgp_id == 3:
        beta_YS_lows = [0.0, 0.2, 0.4]
        pi_Ls = [0.10, 0.20, 0.50]
        for i, b_low in enumerate(beta_YS_lows):
            for pi_L in pi_Ls:
                # DGP 3 uses different params than BASE: no gamma_S or beta_YS
                params = {
                    "n": BASE["n"], "pi_L": pi_L,
                    "alpha_S": BASE["alpha_S"], "beta_SX": BASE["beta_SX"],
                    "sigma_S": BASE["sigma_S"], "alpha_Y": BASE["alpha_Y"],
                    "beta_YX": BASE["beta_YX"], "sigma_Y": BASE["sigma_Y"],
                    "pi_group": 0.3,
                    "beta_YS_high": 0.8, "beta_YS_low": b_low,
                    "gamma_S_high": 0.5, "gamma_S_low": 0.2,
                }
                configs.append(dict(
                    dgp_id=3, dgp_params=params, R=R, pi_L=pi_L,
                    config_id=i, pi_L_id=int(pi_L * 100),
                    label=f"DGP3_blow{b_low:.1f}_piL{pi_L:.2f}",
                ))

    elif dgp_id == 4:
        delta_betas = [0.0, 0.1, 0.2, 0.4]
        pi_Ls = [0.10, 0.20, 0.50]
        for i, db in enumerate(delta_betas):
            for pi_L in pi_Ls:
                # DGP 4 uses beta_YS_0 instead of beta_YS
                params = {
                    "n": BASE["n"], "pi_L": pi_L,
                    "alpha_S": BASE["alpha_S"], "beta_SX": BASE["beta_SX"],
                    "gamma_S": BASE["gamma_S"], "sigma_S": BASE["sigma_S"],
                    "alpha_Y": BASE["alpha_Y"], "beta_YS_0": BASE["beta_YS"],
                    "beta_YX": BASE["beta_YX"], "sigma_Y": BASE["sigma_Y"],
                    "delta_beta": db, "n_cal": 50000,
                    "mu_shift": 0.0,
                }
                configs.append(dict(
                    dgp_id=4, dgp_params=params, R=R, pi_L=pi_L,
                    config_id=i, pi_L_id=int(pi_L * 100),
                    label=f"DGP4_db{db:.1f}_piL{pi_L:.2f}",
                ))

    elif dgp_id == 5:
        qs = [0.05, 0.10, 0.20, 0.30]
        for i, q in enumerate(qs):
            params = {**BASE, "q": q, "missingness": "MCAR"}
            configs.append(dict(
                dgp_id=5, dgp_params=params, R=R, pi_L=q,
                config_id=i, pi_L_id=int(q * 100),
                label=f"DGP5_q{q:.2f}_MCAR",
            ))
            # Also MAR variant
            params_mar = {**BASE, "q": q, "missingness": "MAR", "eta_S": 0.3}
            configs.append(dict(
                dgp_id=5, dgp_params=params_mar, R=R, pi_L=q,
                config_id=i + 100, pi_L_id=int(q * 100),
                label=f"DGP5_q{q:.2f}_MAR",
            ))

    elif dgp_id == 6:
        Ks = [50, 100, 200]
        pi_Ls = [0.20, 1.00]
        R_portfolio = min(R, 500)  # Fewer reps for portfolio
        for i, K in enumerate(Ks):
            for pi_L in pi_Ls:
                # DGP 6 generator doesn't accept n or gamma_S from BASE
                base6 = {k: v for k, v in BASE.items() if k not in ("n", "gamma_S")}
                params = {
                    **base6, "pi_L": pi_L,
                    "K": K, "n_min": 500, "n_max": 5000,
                    "pi_0": 0.5, "sigma_tau": 0.10,
                    "alpha_decision": 0.05,
                }
                configs.append(dict(
                    dgp_id=6, dgp_params=params, R=R_portfolio, pi_L=pi_L,
                    config_id=i, pi_L_id=int(pi_L * 100),
                    label=f"DGP6_K{K}_piL{pi_L:.2f}",
                ))

    return configs


def run_one_config(cfg: Dict[str, Any]) -> Tuple[str, pd.DataFrame, List[Dict[str, Any]]]:
    """Run a single simulation config. Returns (label, raw_df, registry_rows)."""
    dgp_id = cfg["dgp_id"]
    dgp_params = cfg["dgp_params"]
    R = cfg["R"]
    label = cfg["label"]
    protocol = cfg.get("protocol", DEFAULT_PROTOCOL)
    include_ablation = cfg.get("include_ablation", True)
    master_seed = cfg.get("master_seed", 42)
    alpha = cfg.get("alpha", 0.05)

    # Generate historical experiments for composite proxy (method 5)
    hist_exp = None
    if dgp_id != 6:  # DGP 6 handles this internally or skips
        _, hist_exp = estimate_composite_weight_from_historical(
            K_hist=30,
            beta_YS=dgp_params.get("beta_YS", 0.5),
            gamma_S=dgp_params.get("gamma_S", 0.3),
            sigma_tau=0.10,
            seed=12345,
        )

    t0 = time.time()
    df = run_simulation(
        dgp_id=dgp_id,
        dgp_params=dgp_params,
        method_ids=CORE_METHOD_IDS,
        R=R,
        master_seed=master_seed,
        config_id=cfg["config_id"],
        pi_L_id=cfg["pi_L_id"],
        historical_experiments=hist_exp,
        verbose=False,
        protocol=protocol,
        include_ablation=include_ablation,
    )
    elapsed = time.time() - t0

    rows = rows_from_replications(
        df,
        dgp=dgp_id,
        config_name=label,
        params=dgp_params,
        pi_L=cfg.get("pi_L", float("nan")),
        n=dgp_params.get("n", float("nan")),
        R=R,
        protocol=protocol,
        alpha=alpha,
        target="ATE",
        seed=master_seed,
    )
    for row in rows:
        row["elapsed_sec"] = elapsed

    return label, df, rows


def main():
    parser = argparse.ArgumentParser(description="Run simulations for one DGP")
    parser.add_argument("--dgp", type=int, required=True, help="DGP number (1-6)")
    parser.add_argument("--R", type=int, default=2000, help="Number of MC replications")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (default: CPU count)")
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL,
                        help="Prediction protocol")
    parser.add_argument("--no-ablation", action="store_true",
                        help="Run only the primary configurations")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Test level behind the rejection and CDR columns")
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--out-suffix", default="",
                        help="Suffix for the eval JSON, e.g. '_mixed_fit'")
    args = parser.parse_args()

    dgp_id = args.dgp
    R = args.R

    # Create output dirs
    raw_dir = os.path.join(os.path.dirname(__file__), "results", "raw")
    table_dir = os.path.join(os.path.dirname(__file__), "results", "tables")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    configs = get_dgp_configs(dgp_id, R=R)
    for cfg in configs:
        cfg["protocol"] = args.protocol
        cfg["include_ablation"] = not args.no_ablation
        cfg["alpha"] = args.alpha
        cfg["master_seed"] = args.master_seed
    n_configs = len(configs)
    max_workers = args.workers or min(os.cpu_count() or 4, n_configs)

    n_specs = len(resolve_specs(
        CORE_METHOD_IDS, include_ablation=not args.no_ablation
    ))
    print(f"=== DGP {dgp_id}: {n_configs} configs x {n_specs} method "
          f"configurations, R={R}, protocol={args.protocol}, "
          f"{max_workers} workers ===")

    all_rows: List[Dict[str, Any]] = []
    completed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one_config, cfg): cfg["label"] for cfg in configs}

        for future in as_completed(futures):
            label = futures[future]
            try:
                label_out, df, rows = future.result()
                completed += 1

                # Save raw results incrementally
                raw_path = os.path.join(raw_dir, f"{label_out}.parquet")
                # Drop non-serializable columns for parquet
                save_cols = [c for c in df.columns if c not in ("decisions", "true_taus")]
                df[save_cols].to_parquet(raw_path, index=False)

                all_rows.extend(rows)
                elapsed = rows[0]["elapsed_sec"] if rows else float("nan")
                print(f"  [{completed}/{n_configs}] {label_out} done in {elapsed:.1f}s")

            except Exception as e:
                completed += 1
                print(f"  [{completed}/{n_configs}] {label} FAILED: {e}")

    total_time = time.time() - t_start

    # Save the result rows
    eval_path = os.path.join(
        table_dir, f"dgp{dgp_id}_eval{args.out_suffix}.json"
    )
    write_rows(
        eval_path, all_rows,
        generated_by=f"run_dgp.py --dgp {dgp_id} --R {R}",
        protocol=args.protocol,
        extra_meta={"wall_time_sec": round(total_time, 1)},
    )

    print(f"\n=== DGP {dgp_id} complete: {total_time:.1f}s total ===")
    print(f"Raw results: {raw_dir}/DGP{dgp_id}_*.parquet")
    print(f"Result rows ({len(all_rows)}): {eval_path}")


if __name__ == "__main__":
    main()
