#!/usr/bin/env python3
"""LaLonde Negative Control Reframing.

RE75 (1975 earnings) is a PRE-TREATMENT covariate, not a post-treatment
surrogate. The NSW job training program ran 1976-1977, so RE75 cannot
mediate the treatment effect on RE78 (1978 earnings).

This analysis demonstrates a negative control: the surrogate framework
correctly identifies that RE75 is not a valid surrogate via the
surrogacy diagnostic test (high rejection rates).

  - pi_L in {0.10, 0.20, 0.50}
  - Full method suite + surrogacy diagnostic test
  - Monte Carlo SEs on all metrics
  - R=500, 8-core multiprocessing
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

from src.real_data.lalonde import load_lalonde, compute_ground_truth_ate
from src.simulations.simulation import derive_seed
from src.methods import (
    surrogate_index,
    ppi_plus,
)
from src.simulations.simulation import METHOD_DISPLAY_NAMES
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.registry import rows_from_summary, write_rows
from src.utils.estimator_api import (
    estimate,
    estimate_cov_si_ppi,
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

# ── Configuration ─────────────────────────────────────────────────────────
PI_L_VALUES = [0.10, 0.20, 0.50]
METHOD_IDS = [0, 1, 2, 3, 4, 5]
R = 500
MASTER_SEED = 42
N_WORKERS = 8
ALPHA_TEST = 0.05

# Load data once at module level
_DATA = load_lalonde()
_TRUE_TAU = compute_ground_truth_ate(_DATA["T"], _DATA["Y"])
_T = _DATA["T"]
_S = _DATA["S"]  # RE75 — pre-treatment (invalid surrogate)
_Y = _DATA["Y"]  # RE78 — post-treatment outcome
_X = _DATA["X"]
_N = len(_T)


def run_one_rep(args: Tuple[float, int]) -> List[Dict[str, Any]]:
    """Run one replication for a given pi_L on the LaLonde dataset."""
    pi_L, rep_idx = args

    pi_L_id = int(round(pi_L * 100))
    seed = derive_seed(MASTER_SEED, dgp_id=200, config_id=0, pi_L_id=pi_L_id, rep=rep_idx)
    rng = np.random.default_rng(seed)

    # Draw labeled mask
    n_L = int(np.floor(pi_L * _N))
    n_L = min(n_L, _N)
    perm = rng.permutation(_N)
    labeled_mask = np.zeros(_N, dtype=bool)
    labeled_mask[perm[:n_L]] = True

    # Ensure at least 10 labeled per arm
    for t_val in [0, 1]:
        t_idx = np.where(_T == t_val)[0]
        t_labeled = np.where(labeled_mask & (_T == t_val))[0]
        if len(t_labeled) < 10:
            unlabeled_t = np.where(~labeled_mask & (_T == t_val))[0]
            need = min(10 - len(t_labeled), len(unlabeled_t))
            if need > 0:
                extra = rng.choice(unlabeled_t, size=need, replace=False)
                labeled_mask[extra] = True

    # Train prediction model
    cf_rng = np.random.default_rng(seed + 7777)
    Y_hat, design = train_prediction_model(
        _S, _X, _Y, labeled_mask,
        n_folds=5, rng=cf_rng, prediction_model="ols", seed=seed,
        protocol=PROTOCOL, return_design=True,
    )

    # Run all methods
    method_results = {}
    for m_id in METHOD_IDS:
        try:
            mkwargs = {"design": design} if m_id in (2, 5) else {}
            res = estimate(m_id, _T, _S, _Y, Y_hat, labeled_mask,
                           protocol=PROTOCOL, **mkwargs)
        except Exception:
            res = {"tau_hat": np.nan, "var_hat": np.nan,
                   "ci_lower": np.nan, "ci_upper": np.nan, "se_hat": np.nan}
        method_results[m_id] = res

    # Surrogacy diagnostic test
    si_res = method_results[2]
    ppi_res = method_results[3]

    lambda_hat = ppi_res.get("lambda_hat", 0.0)
    try:
        cov = estimate_cov_si_ppi(_T, _Y, Y_hat, labeled_mask, lambda_hat, design=design)
        test_res = surrogacy_test(
            tau_si=si_res["tau_hat"],
            tau_ppi=ppi_res["tau_hat"],
            var_si=si_res["var_hat"],
            var_ppi=ppi_res["var_hat"],
            cov_si_ppi=cov,
            alternative="two-sided",
        )
        test_reject = 1 if test_res["p_value"] < ALPHA_TEST else 0
        test_p = test_res["p_value"]
        test_D = test_res["D_hat"]
    except Exception:
        test_reject = 0
        test_p = np.nan
        test_D = np.nan

    rows = []
    for m_id in METHOD_IDS:
        r = method_results[m_id]
        rows.append({
            "pi_L": pi_L,
            "replication": rep_idx,
            "method_id": m_id,
            "method_name": METHOD_DISPLAY_NAMES.get(m_id, f"Method {m_id}"),
            "tau_hat": r["tau_hat"],
            "var_hat": r["var_hat"],
            "ci_lower": r["ci_lower"],
            "ci_upper": r["ci_upper"],
            "true_tau": _TRUE_TAU,
            "test_D_hat": test_D,
            "test_p_value": test_p,
            "test_reject": test_reject,
        })

    return rows


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary metrics with Monte Carlo SEs."""
    rows = []
    for (pi_L, m_id), grp in df.groupby(["pi_L", "method_id"]):
        true_tau = grp["true_tau"].iloc[0]
        th = grp["tau_hat"].dropna().values
        cl = grp["ci_lower"].dropna().values
        cu = grp["ci_upper"].dropna().values
        n_valid = len(th)

        if n_valid == 0:
            continue

        bias = np.mean(th) - true_tau
        variance = np.var(th, ddof=1)
        rmse = np.sqrt(np.mean((th - true_tau) ** 2))
        coverage = np.mean((cl <= true_tau) & (true_tau <= cu))
        rejection_rate = np.mean(np.sign(cl) == np.sign(cu))

        mc_se_bias = np.std(th, ddof=1) / np.sqrt(n_valid)
        mc_se_coverage = np.sqrt(coverage * (1 - coverage) / n_valid)
        mc_se_rmse = np.std((th - true_tau) ** 2, ddof=1) / (2 * rmse * np.sqrt(n_valid)) if rmse > 0 else np.nan
        mc_se_rejection = np.sqrt(rejection_rate * (1 - rejection_rate) / n_valid)

        lo_grp = df[(df["pi_L"] == pi_L) & (df["method_id"] == 0)]
        lo_th = lo_grp["tau_hat"].dropna().values
        lo_rmse = np.sqrt(np.mean((lo_th - true_tau) ** 2)) if len(lo_th) else np.nan
        # Paper convention: RE = RMSE_LO / RMSE_method (an RMSE ratio).
        # The squared version is the ESS multiplier, reported separately.
        rel_eff = lo_rmse / rmse if rmse > 0 and not np.isnan(lo_rmse) else np.nan
        ess_mult = rel_eff ** 2 if not np.isnan(rel_eff) else np.nan

        # Surrogacy diagnostic test rejection rate
        test_rej_rate = grp["test_reject"].mean()
        mc_se_test = np.sqrt(test_rej_rate * (1 - test_rej_rate) / n_valid)

        rows.append({
            "pi_L": pi_L,
            "method_id": int(m_id),
            "method_name": METHOD_DISPLAY_NAMES.get(int(m_id), f"Method {int(m_id)}"),
            "true_tau": true_tau,
            "n_valid": n_valid,
            "bias": bias,
            "mc_se_bias": mc_se_bias,
            "rmse": rmse,
            "mc_se_rmse": mc_se_rmse,
            "variance": variance,
            "coverage": coverage,
            "mc_se_coverage": mc_se_coverage,
            "rejection_rate": rejection_rate,
            "mc_se_rejection": mc_se_rejection,
            "relative_efficiency": rel_eff,
            "ess_multiplier": ess_mult,
            "surrogacy_test_rejection_rate": test_rej_rate,
            "mc_se_surrogacy_test": mc_se_test,
        })

    return pd.DataFrame(rows)


def main():
    print("LaLonde Negative Control Reframing")
    print(f"  n={_N}, true_tau (RE78)={_TRUE_TAU:.2f}")
    print(f"  'Surrogate' = RE75 (PRE-TREATMENT — invalid!)")
    print(f"  Methods: {METHOD_IDS}")
    print(f"  pi_L values: {PI_L_VALUES}")
    print(f"  R={R}, Workers={N_WORKERS}")
    print()

    tasks: List[Tuple[float, int]] = []
    for pi_L in PI_L_VALUES:
        for r in range(R):
            tasks.append((pi_L, r))

    print(f"Total tasks: {len(tasks)}")

    t0 = time.time()

    with Pool(N_WORKERS) as pool:
        all_results_nested = pool.map(run_one_rep, tasks, chunksize=10)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    all_results = []
    for batch in all_results_nested:
        all_results.extend(batch)

    df_raw = pd.DataFrame(all_results)
    df_summary = compute_metrics(df_raw)

    # Print
    print("\n" + "=" * 80)
    print("LaLonde Negative Control: Summary")
    print("=" * 80)
    for pi_L in PI_L_VALUES:
        sub = df_summary[df_summary["pi_L"] == pi_L]
        test_rej = sub["surrogacy_test_rejection_rate"].iloc[0]
        print(f"\npi_L = {pi_L} (surrogacy test rejection rate: {test_rej:.3f}):")
        print(sub[["method_name", "bias", "mc_se_bias", "rmse", "coverage",
                    "mc_se_coverage", "relative_efficiency"]].to_string(index=False))

    # Save CSV
    csv_path = os.path.join(PROJECT_ROOT, "results", "tables", "lalonde_negative_control.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Result rows.  target="masking_target": the estimand here is the
    # full-sample difference in means that the label masking hides, not an ATE
    # from a data-generating process.
    rows_path = os.path.join(PROJECT_ROOT, "results", "tables",
                             "lalonde_negative_control_rows.json")
    write_rows(
        rows_path,
        rows_from_summary(
            df_summary, dgp="lalonde", target="masking_target",
            protocol=PROTOCOL, n=_N, config_name_col=None,
            config_name="lalonde", params_cols=("pi_L",),
        ),
        generated_by="scripts/lalonde_negative_control.py", protocol=PROTOCOL,
    )
    print(f"Saved result rows: {rows_path}")

    # Save markdown
    md_path = os.path.join(PROJECT_ROOT, "results", "tables", "lalonde_negative_control.md")
    with open(md_path, "w") as f:
        f.write("# LaLonde Negative Control Analysis\n\n")
        f.write(f"**LaLonde NSW Dataset**, n={_N}, "
                f"true ATE (RE78, full-sample)={_TRUE_TAU:.2f}, R={R}\n\n")
        f.write("**Critical context:** RE75 is a PRE-TREATMENT covariate "
                "(1975 earnings), not a post-treatment surrogate. "
                "The NSW program ran 1976-1977. This analysis serves as "
                "a **negative control** — the diagnostic test should reject.\n\n")

        for pi_L in PI_L_VALUES:
            sub = df_summary[df_summary["pi_L"] == pi_L]
            test_rej = sub["surrogacy_test_rejection_rate"].iloc[0]
            mc_se_t = sub["mc_se_surrogacy_test"].iloc[0]

            f.write(f"## pi_L = {pi_L}\n\n")
            f.write(f"**Surrogacy diagnostic test rejection rate: "
                    f"{test_rej:.3f}** (MC SE: {mc_se_t:.4f})\n\n")
            f.write("| Method | Bias | MC SE(Bias) | RMSE | MC SE(RMSE) | "
                    "Coverage | MC SE(Cov) | Power | Rel. Eff. | ESS multiplier |\n")
            f.write("|--------|-----:|------------:|-----:|------------:|"
                    "---------:|-----------:|------:|----------:|---------------:|\n")
            for _, row in sub.iterrows():
                f.write(f"| {row['method_name']} "
                        f"| {row['bias']:.2f} "
                        f"| {row['mc_se_bias']:.2f} "
                        f"| {row['rmse']:.2f} "
                        f"| {row['mc_se_rmse']:.2f} "
                        f"| {row['coverage']:.3f} "
                        f"| {row['mc_se_coverage']:.4f} "
                        f"| {row['rejection_rate']:.3f} "
                        f"| {row['relative_efficiency']:.3f} "
                        f"| {row['ess_multiplier']:.3f} |\n")
            f.write("\n")

    print(f"Saved MD:  {md_path}")


if __name__ == "__main__":
    main()
