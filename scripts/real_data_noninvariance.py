#!/usr/bin/env python3
"""
Covariate-adjusted evidence of conditional-mean noninvariance.

For each dataset, among VISITORS (S = 1) only, fit

    logit P(Y = 1 | T, X) = b0 + b_T T + X' b

and report:
  * b_T, its heteroskedasticity-robust (HC1) standard error and p-value --
    a nonzero b_T is direct evidence that E[Y | S = 1, X, T] depends on T,
    i.e. conditional-mean noninvariance;
  * the covariate-adjusted within-visitor conversion difference, computed as
    the average marginal effect (AME):
        AME = mean_i [ P(Y=1 | T=1, X_i) - P(Y=1 | T=0, X_i) ]
    over the visitor sample, with a delta-method standard error using the
    model's robust covariance;
  * the UNADJUSTED within-visitor difference in means, for comparison.

Covariates
  Hillstrom: recency, history, mens, womens, newbie, plus channel and
             zip_code dummies (drop-first).
  Criteo:    f0 - f11.

Criteo needs the raw CSV: pass --criteo or set CRITEO_CSV (see
README.md).  Without it the Criteo block is skipped and the table says
so.

Output: results/tables/real_data_noninvariance.md (+ .csv)

Usage:
    python scripts/real_data_noninvariance.py [--criteo /path/to/criteo.csv]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.registry import write_table_rows        # noqa: E402

TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
FEATURES_CRITEO = [f"f{i}" for i in range(12)]


def fit_and_summarize(name: str, T: np.ndarray, Y: np.ndarray,
                      X: np.ndarray, covariate_names: List[str]
                      ) -> Dict[str, Any]:
    """Logistic Y ~ T + X on the given (visitor) sample; robust inference."""
    import statsmodels.api as sm
    from scipy import stats as sp_stats

    T = np.asarray(T, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    # Drop zero-variance covariates (they make the design singular).
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    covariate_names = [c for c, k in zip(covariate_names, keep) if k]

    # Standardize covariates for conditioning; this does not change b_T.
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Xs = (X - mu) / sd

    design = np.column_stack([np.ones(len(T)), T, Xs])
    model = sm.Logit(Y, design)
    res = model.fit(disp=0, maxiter=200, method="newton")

    # HC1-style robust covariance for the MLE (Huber sandwich).
    robust = model.fit(disp=0, maxiter=200, method="newton", cov_type="HC1")

    beta = robust.params
    b_T = float(beta[1])
    se_T = float(robust.bse[1])
    z_T = b_T / se_T if se_T > 0 else np.nan
    p_T = float(2.0 * (1.0 - sp_stats.norm.cdf(abs(z_T))))

    # Average marginal effect of T, with a delta-method SE.
    def _p(t_val: float) -> np.ndarray:
        d = design.copy()
        d[:, 1] = t_val
        return 1.0 / (1.0 + np.exp(-(d @ beta)))

    d1 = design.copy(); d1[:, 1] = 1.0
    d0 = design.copy(); d0[:, 1] = 0.0
    p1, p0 = _p(1.0), _p(0.0)
    ame = float(np.mean(p1 - p0))

    # d AME / d beta = mean_i [ p1_i(1-p1_i) x1_i - p0_i(1-p0_i) x0_i ]
    g = (((p1 * (1 - p1))[:, None] * d1)
         - ((p0 * (1 - p0))[:, None] * d0)).mean(axis=0)
    V = np.asarray(robust.cov_params())
    se_ame = float(np.sqrt(max(g @ V @ g, 0.0)))
    z_ame = ame / se_ame if se_ame > 0 else np.nan
    p_ame = float(2.0 * (1.0 - sp_stats.norm.cdf(abs(z_ame))))

    # Unadjusted within-sample difference in means.
    y1, y0 = Y[T == 1], Y[T == 0]
    unadj = float(y1.mean() - y0.mean())
    se_unadj = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    z_un = unadj / se_unadj if se_unadj > 0 else np.nan
    p_unadj = float(2.0 * (1.0 - sp_stats.norm.cdf(abs(z_un))))

    return dict(
        dataset=name,
        n_visitors=int(len(T)),
        n_visitors_treated=int((T == 1).sum()),
        n_visitors_control=int((T == 0).sum()),
        conv_rate_T1=float(y1.mean()), conv_rate_T0=float(y0.mean()),
        n_covariates=len(covariate_names),
        beta_T=b_T, se_beta_T=se_T, z_beta_T=z_T, p_beta_T=p_T,
        odds_ratio_T=float(np.exp(b_T)),
        ame_adjusted=ame, se_ame=se_ame, p_ame=p_ame,
        diff_unadjusted=unadj, se_unadjusted=se_unadj, p_unadjusted=p_unadj,
        pseudo_r2=float(res.prsquared),
    )


def hillstrom_block() -> Dict[str, Any]:
    from src.real_data.hillstrom import load_hillstrom

    d = load_hillstrom()
    df = d["df"]
    S = d["S"]
    vis = S == 1

    cont = ["recency", "history", "mens", "womens", "newbie"]
    parts = [df.loc[vis, cont].to_numpy(np.float64)]
    names = list(cont)
    for col in ("channel", "zip_code"):
        dum = pd.get_dummies(df.loc[vis, col], prefix=col, drop_first=True)
        parts.append(dum.to_numpy(np.float64))
        names += list(dum.columns)
    X = np.hstack(parts)

    return fit_and_summarize(
        "Hillstrom (visit -> conversion)",
        df.loc[vis, "T"].to_numpy(np.float64),
        df.loc[vis, "conversion"].to_numpy(np.float64),
        X, names,
    )


def criteo_block(path: str) -> Dict[str, Any]:
    usecols = FEATURES_CRITEO + ["treatment", "conversion", "visit"]
    dtypes = {c: np.float32 for c in FEATURES_CRITEO}
    dtypes.update({"treatment": np.int8, "conversion": np.int8,
                   "visit": np.int8})
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes)
    vis = df["visit"].to_numpy() == 1
    return fit_and_summarize(
        "Criteo (visit -> conversion)",
        df.loc[vis, "treatment"].to_numpy(np.float64),
        df.loc[vis, "conversion"].to_numpy(np.float64),
        df.loc[vis, FEATURES_CRITEO].to_numpy(np.float64),
        list(FEATURES_CRITEO),
    )


def write_markdown(rows: List[Dict[str, Any]], skipped: Optional[str],
                   path: str) -> None:
    lines = [
        "# Covariate-adjusted evidence of conditional-mean noninvariance\n\n",
        "Among VISITORS only (S = 1), logistic regression of conversion on "
        "the treatment indicator and the covariates. A nonzero treatment "
        "coefficient is direct evidence that E[Y | S = 1, X, T] depends on "
        "T, i.e. that surrogacy (Assumption 2) fails CONDITIONAL on X -- the "
        "Standard errors are "
        "heteroskedasticity-robust (HC1 / Huber sandwich); covariates are "
        "standardized, which leaves the treatment coefficient unchanged. "
        "`AME` is the covariate-adjusted within-visitor conversion "
        "difference, mean_i [P(Y=1|T=1,X_i) - P(Y=1|T=0,X_i)], with a "
        "delta-method standard error. `Unadjusted` is the raw within-visitor "
        "difference in means (what the paper currently reports).\n\n"
        "Covariates -- Hillstrom: recency, history, mens, womens, newbie, "
        "plus channel and zip_code dummies (drop-first). Criteo: f0-f11.\n\n",
        "| Dataset | Visitors | Conv. rate (T=1) | Conv. rate (T=0) | "
        "# covariates | T coefficient | SE | p-value | Odds ratio | "
        "Adjusted diff. (AME) | SE | p-value | Unadjusted diff. | SE | "
        "p-value |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|\n",
    ]

    def pfmt(p: float) -> str:
        return "< 1e-16" if p < 1e-16 else f"{p:.3g}"

    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['n_visitors']:,} "
            f"| {r['conv_rate_T1']:.5f} | {r['conv_rate_T0']:.5f} "
            f"| {r['n_covariates']} | {r['beta_T']:+.5f} "
            f"| {r['se_beta_T']:.5f} | {pfmt(r['p_beta_T'])} "
            f"| {r['odds_ratio_T']:.4f} | {r['ame_adjusted']:+.6f} "
            f"| {r['se_ame']:.6f} | {pfmt(r['p_ame'])} "
            f"| {r['diff_unadjusted']:+.6f} | {r['se_unadjusted']:.6f} "
            f"| {pfmt(r['p_unadjusted'])} |\n"
        )
    if skipped:
        lines.append(f"\n{skipped}\n")
    with open(path, "w") as f:
        f.writelines(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteo", default=os.environ.get("CRITEO_CSV"),
                    help="Path to criteo-uplift-v2.1.csv. Defaults to "
                         "$CRITEO_CSV. See README.md.")
    ap.add_argument("--skip-hillstrom", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rows: List[Dict[str, Any]] = []

    if not args.skip_hillstrom:
        print("Hillstrom...", flush=True)
        rows.append(hillstrom_block())

    skipped = None
    if args.criteo and os.path.exists(args.criteo):
        print("Criteo...", flush=True)
        rows.append(criteo_block(args.criteo))
    else:
        skipped = ("**Criteo not run**: no Criteo CSV available on this "
                   "machine (pass `--criteo /path/to/criteo-uplift-v2.1.csv` "
                   "or set `CRITEO_CSV`; see README.md).")
        print(skipped, flush=True)

    os.makedirs(TABLES_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(TABLES_DIR, "real_data_noninvariance.csv"), index=False)
    write_table_rows(
        os.path.join(TABLES_DIR, "real_data_noninvariance_rows.json"),
        pd.DataFrame(rows),
        dgp="real_data", analysis="conditional_mean_noninvariance",
        config_cols=("dataset",), target="masking_target",
        generated_by="scripts/real_data_noninvariance.py",
    )
    write_markdown(rows, skipped,
                   os.path.join(TABLES_DIR, "real_data_noninvariance.md"))

    for r in rows:
        print(f"\n{r['dataset']}: n_visitors={r['n_visitors']:,}")
        print(f"  T coefficient  = {r['beta_T']:+.5f} "
              f"(SE {r['se_beta_T']:.5f}, z {r['z_beta_T']:.2f}, "
              f"p {r['p_beta_T']:.3g}), odds ratio {r['odds_ratio_T']:.4f}")
        print(f"  adjusted AME   = {r['ame_adjusted']:+.6f} "
              f"(SE {r['se_ame']:.6f}, p {r['p_ame']:.3g})")
        print(f"  unadjusted dif = {r['diff_unadjusted']:+.6f} "
              f"(SE {r['se_unadjusted']:.6f}, p {r['p_unadjusted']:.3g})")
    print(f"\nWrote results/tables/real_data_noninvariance.md "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
