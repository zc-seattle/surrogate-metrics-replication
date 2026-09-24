#!/usr/bin/env python3
"""
Two-cell check of the surrogate-index interval with a gradient-boosted-tree
index.

What the GBT comparison reports
-------------------------------
`run_gbt_comparison.py` fits the index by gradient-boosted trees under the
all-units cross-fitting protocol.  A tree index has no linear design, so the
joint sandwich of Proposition 3 is unavailable and every GBT surrogate-index
row carries ``si_variance = "plugin"``: the Neyman variance of the imputed
outcomes, which treats the fitted trees as fixed.  The SI--PPI++ diagnostic
on such an index can only use the fixed-predictor covariance
(`estimate_cov_si_ppi` without a design).  Both ignore first-stage
uncertainty.  This script measures how much that matters.

Design
------
Two cells of the GBT comparison, on the SAME draws that script uses (seed
`derive_seed(42, dgp, sha256(str(overrides)) % 1000, 100 pi_L, rep)`,
fold rng `seed + 7777`, tree `random_state = seed`), so the point estimates
here reproduce the comparison's GBT rows replication by replication:

  * DGP 1,            n = 10,000, pi_L = 0.20 (valid surrogate; diagnostic null)
  * DGP 2, rho = 0.2, n = 10,000, pi_L = 0.20 (violation; diagnostic power)

Per replication:
  * tau_SI, tau_PPI++ (primary configuration) and D = tau_PPI++ - tau_SI
  * plug-in Var(SI); exact Var(PPI++); fixed-predictor Cov and Var(D)
  * a joint paired bootstrap, B draws, stratified by arm: units resampled with
    replacement, labeled flags carried along, the trees REFIT under the
    all-units protocol and both estimators recomputed, giving bootstrap
    Var(SI), Var(PPI++), Cov and Var(D)
  * coverage of tau by the SI plug-in interval, the SI bootstrap interval, the
    PPI++ exact interval and the PPI++ bootstrap interval
  * two-sided diagnostic rejection at alpha = 0.05 and 0.10 with SE(D) from
    the fixed-predictor covariance and from the bootstrap

The Monte Carlo variance of each statistic across the R replications is the
target both variance estimates are aiming at.

Output
------
results/tables/gbt_variance_check.csv       (one row per cell and quantity)
results/tables/gbt_variance_check.md
results/tables/gbt_variance_check_raw.csv   (per replication)

With ``--clustered-folds`` the bootstrap assigns folds by ORIGINAL unit (all
copies of a resampled unit share a fold) and the run writes
results/tables/gbt_bootstrap_clustered.{md,csv} (+ _raw.csv): the clustered
bootstrap beside the row-fold bootstrap of gbt_variance_check_raw.csv on the
same replications.

Usage:
    python scripts/run_gbt_variance_check.py [--R 200] [--B 100]
    python scripts/run_gbt_variance_check.py --from-raw   # rebuild tables
    python scripts/run_gbt_variance_check.py --R 2 --B 3  # smoke
    python scripts/run_gbt_variance_check.py --clustered-folds --cells 1 \
        --R 200 --B 50
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dgps.dgps import generate_dgp1, generate_dgp2
from src.methods import estimate_cov_si_ppi
from src.simulations.simulation import (
    _get_gbt_class, allunits_fold_ids, derive_seed, unit_clustered_fold_ids,
)
from src.utils.config import DGP_DEFAULTS
from src.utils.estimator_api import estimate, train_prediction_model

TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
MASTER_SEED = 42
N_WORKERS = 8
N_FOLDS = 5
PI_L = 0.20
Z = float(stats.norm.ppf(0.975))
ALPHAS = (0.05, 0.10)

#: (dgp id, overrides, label, diagnostic null holds)
CELLS: Dict[int, Tuple[int, Dict[str, Any], str, bool]] = {
    1: (1, {}, "DGP 1, pi_L = 0.20", True),
    2: (2, {"rho": 0.2}, "DGP 2 rho = 0.2, pi_L = 0.20", False),
}


def _rep_seed(dgp_id: int, overrides: Dict[str, Any], rep: int) -> int:
    """The seed run_gbt_comparison.py uses for this (cell, replication)."""
    cfg_id = int(hashlib.sha256(str(overrides).encode()).hexdigest()[:8],
                 16) % 1000
    return derive_seed(MASTER_SEED, dgp_id, cfg_id, int(PI_L * 100), rep)


def _generate(dgp_id: int, overrides: Dict[str, Any], seed: int):
    kw = dict(DGP_DEFAULTS[dgp_id])
    kw.update(overrides)
    kw["seed"] = seed
    kw["pi_L"] = PI_L
    return (generate_dgp1 if dgp_id == 1 else generate_dgp2)(**kw)


def _fit_pair(S, X, Y, T, lm, rng, tree_seed, fold_ids=None):
    """GBT index under all-units cross-fitting; both estimators.

    With ``fold_ids=None`` the folds come from `train_prediction_model` (one
    permutation of the n rows, the pipeline's own path).  With an explicit
    fold vector the same trees are fit on the labeled units outside each
    fold by `gbt_allunits_predict`, which the clustered bootstrap uses.
    """
    if fold_ids is None:
        Y_hat = train_prediction_model(
            S, X, Y, lm, n_folds=N_FOLDS, rng=rng, model="gbt",
            seed=tree_seed,
        )
    else:
        Y_hat = gbt_allunits_predict(S, X, Y, lm, fold_ids, tree_seed)
    t1 = T == 1
    tau_si = float(Y_hat[t1].mean() - Y_hat[~t1].mean())
    ppi = estimate(3, T, S, Y, Y_hat, lm)
    return tau_si, ppi, Y_hat


def _gbt_features(S, X):
    """[S, S^2, X], the GBT feature set of `_train_prediction_model_gbt`."""
    S_cols = S.reshape(-1, 1) if S.ndim == 1 else S
    parts = [S_cols, S_cols ** 2]
    if X is not None and X.shape[1] > 0:
        parts.append(X)
    return np.hstack(parts)


def gbt_allunits_predict(S, X, Y, lm, fold_ids, tree_seed) -> np.ndarray:
    """All-units cross-fitted GBT predictions for a GIVEN fold vector.

    The all-units branch of `src.simulations.simulation.
    _train_prediction_model_gbt` (same features, hyperparameters and
    random_state) with the fold assignment supplied by the caller;
    `_self_check` verifies that it reproduces `train_prediction_model` when
    handed that function's own folds.
    """
    GBT = _get_gbt_class()
    lab = np.where(lm)[0]
    Y_hat = np.empty(len(Y), dtype=np.float64)
    for k in range(N_FOLDS):
        test_idx = np.where(fold_ids == k)[0]
        train_idx = lab[fold_ids[lab] != k]
        if len(test_idx) == 0:
            continue
        if len(train_idx) < 2:
            raise ValueError(f"degenerate fold {k}: {len(train_idx)} labeled "
                             "training units")
        model = GBT(n_estimators=100, max_depth=4, learning_rate=0.1,
                    random_state=tree_seed)
        model.fit(_gbt_features(S[train_idx], X[train_idx]), Y[train_idx])
        Y_hat[test_idx] = model.predict(_gbt_features(S[test_idx],
                                                      X[test_idx]))
    return Y_hat


def gbt_paired_bootstrap(S, X, Y, T, lm, B: int, seed: int,
                         tree_seed: int,
                         clustered: bool = False) -> Dict[str, float]:
    """Joint paired bootstrap of (tau_SI, tau_PPI++) refitting the trees.

    Resamples n_1 treated and n_0 control units with replacement (labeled
    flags travel with the units), refits the GBT index under the all-units
    protocol on each resample and recomputes both estimators, so the
    covariance and Var(D) are joint bootstrap quantities.  The GBT analogue
    of `scripts/run_joint_cov_check.paired_bootstrap`, which refits OLS on a
    prebuilt design.

    ``clustered=False`` (the gbt_variance_check.md run) splits the resampled
    ROWS into folds, as the pipeline does.  ``clustered=True`` assigns folds
    by original unit (`unit_clustered_fold_ids`), so no copy of a unit is ever
    predicted by a tree trained on another copy of the same unit.
    """
    rng = np.random.default_rng(seed)
    idx1 = np.where(T == 1)[0]
    idx0 = np.where(T == 0)[0]
    draws = np.empty((B, 2))
    for b in range(B):
        take = np.concatenate([idx1[rng.integers(0, len(idx1), len(idx1))],
                               idx0[rng.integers(0, len(idx0), len(idx0))]])
        folds = (unit_clustered_fold_ids(take, N_FOLDS, rng) if clustered
                 else None)
        tau_si, ppi, _ = _fit_pair(S[take], X[take], Y[take], T[take],
                                   lm[take], rng, tree_seed, fold_ids=folds)
        draws[b] = (tau_si, float(ppi["tau_hat"]))
    cov = np.cov(draws[:, 0], draws[:, 1], ddof=1)
    return dict(boot_var_si=float(cov[0, 0]), boot_var_ppi=float(cov[1, 1]),
                boot_cov=float(cov[0, 1]),
                boot_var_D=float(np.var(draws[:, 1] - draws[:, 0], ddof=1)))


def run_one(args: Tuple[int, int, int, bool]) -> Dict[str, Any]:
    cell, rep, B, clustered = args
    dgp_id, overrides, _, _ = CELLS[cell]
    seed = _rep_seed(dgp_id, overrides, rep)
    d = _generate(dgp_id, overrides, seed)
    T, S, X, Y, lm = d["T"], d["S"], d["X"], d["Y"], d["labeled_mask"]
    tau = float(d["true_tau"])

    # Same fold rng and tree seed as run_single_replication.
    tau_si, ppi, Y_hat = _fit_pair(S, X, Y, T, lm,
                                   np.random.default_rng(seed + 7777), seed)
    si_p = estimate(2, T, S, Y, Y_hat, lm, si_variance="plugin")
    lam = float(ppi["lambda_hat"])
    var_si_p = float(si_p["var_hat"])
    var_ppi = float(ppi["var_hat"])
    cov_ff = float(estimate_cov_si_ppi(T, Y, Y_hat, lm, lam))
    var_D_ff = var_si_p + var_ppi - 2.0 * cov_ff
    tau_ppi = float(ppi["tau_hat"])
    D = tau_ppi - tau_si

    row: Dict[str, Any] = dict(
        cell=cell, rep=rep, seed=seed, true_tau=tau, tau_si=tau_si,
        tau_ppi=tau_ppi, D_hat=D, lambda_hat=lam,
        var_si_plugin=var_si_p, var_ppi_exact=var_ppi,
        cov_fixedf=cov_ff, var_D_fixedf=var_D_ff,
        cov_si_plugin=int(abs(tau_si - tau) <= Z * np.sqrt(var_si_p)),
        cov_ppi_exact=int(abs(tau_ppi - tau) <= Z * np.sqrt(var_ppi)),
    )
    for a in ALPHAS:
        crit = stats.norm.ppf(1 - a / 2)
        row[f"rej_fixedf_a{int(a * 100):02d}"] = int(
            var_D_ff > 0 and abs(D) / np.sqrt(var_D_ff) > crit)

    bs = gbt_paired_bootstrap(S, X, Y, T, lm, B, seed + 991, seed,
                              clustered=clustered)
    row.update(bs)
    row["B"] = B
    row["folds"] = "clustered" if clustered else "rows"
    row["cov_si_boot"] = int(abs(tau_si - tau) <= Z * np.sqrt(bs["boot_var_si"]))
    row["cov_ppi_boot"] = int(abs(tau_ppi - tau)
                              <= Z * np.sqrt(bs["boot_var_ppi"]))
    for a in ALPHAS:
        crit = stats.norm.ppf(1 - a / 2)
        row[f"rej_boot_a{int(a * 100):02d}"] = int(
            bs["boot_var_D"] > 0 and abs(D) / np.sqrt(bs["boot_var_D"]) > crit)
    return row


# ---------------------------------------------------------------------------
# Summaries with Monte Carlo standard errors
# ---------------------------------------------------------------------------

def _mean_se(x: np.ndarray) -> Tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def _mcvar_se(x: np.ndarray) -> Tuple[float, float]:
    """Sample variance and its delta-method SE sqrt((m4 - s^4) / R)."""
    v = float(x.var(ddof=1))
    m4 = float(np.mean((x - x.mean()) ** 4))
    return v, float(np.sqrt(max(m4 - v ** 2, 0.0) / len(x)))


def _ratio_se(est: np.ndarray, stat: np.ndarray, reps: int = 2000,
              seed: int = 0) -> Tuple[float, float]:
    """mean(est) / Var(stat) and its SE by a bootstrap over replications."""
    point = float(est.mean() / stat.var(ddof=1))
    rng = np.random.default_rng(seed)
    R = len(est)
    draws = np.empty(reps)
    for b in range(reps):
        i = rng.integers(0, R, R)
        draws[b] = est[i].mean() / stat[i].var(ddof=1)
    return point, float(draws.std(ddof=1))


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cell, g in raw.groupby("cell"):
        dgp_id, overrides, label, null = CELLS[int(cell)]
        R = len(g)
        B = int(g["B"].iloc[0])
        base = dict(cell=label, dgp=dgp_id, rho=overrides.get("rho", ""),
                    n=10_000, pi_L=PI_L, R=R, B=B, learner="gbt",
                    diagnostic_null=null)
        for stat_col, name, ests in (
            ("tau_si", "Var(SI)", (("plug-in", "var_si_plugin"),
                                   ("paired bootstrap", "boot_var_si"))),
            ("tau_ppi", "Var(PPI++)", (("exact", "var_ppi_exact"),
                                       ("paired bootstrap", "boot_var_ppi"))),
            ("D_hat", "Var(D)", (("fixed-predictor", "var_D_fixedf"),
                                 ("paired bootstrap", "boot_var_D"))),
        ):
            stat = g[stat_col].to_numpy()
            mc, mc_se = _mcvar_se(stat)
            rows.append(dict(base, quantity=name, estimator="Monte Carlo",
                             value=mc, mc_se=mc_se, ratio_to_mc=1.0,
                             ratio_mc_se=0.0))
            for k, (tag, col) in enumerate(ests):
                v, se = _mean_se(g[col].to_numpy())
                ratio, rse = _ratio_se(g[col].to_numpy(), stat,
                                       seed=1000 * int(cell) + 10 * len(rows)
                                       + k)
                rows.append(dict(base, quantity=name, estimator=tag,
                                 value=v, mc_se=se, ratio_to_mc=ratio,
                                 ratio_mc_se=rse))
        for col, name, tag in (
            ("cov_si_plugin", "SI coverage", "plug-in"),
            ("cov_si_boot", "SI coverage", "paired bootstrap"),
            ("cov_ppi_exact", "PPI++ coverage", "exact"),
            ("cov_ppi_boot", "PPI++ coverage", "paired bootstrap"),
        ):
            p, se = _mean_se(g[col].to_numpy().astype(float))
            rows.append(dict(base, quantity=name, estimator=tag, value=p,
                             mc_se=float(np.sqrt(p * (1 - p) / R))))
        for a in ALPHAS:
            for tag, col in (("fixed-predictor", f"rej_fixedf_a{int(a*100):02d}"),
                             ("paired bootstrap", f"rej_boot_a{int(a*100):02d}")):
                p = float(g[col].mean())
                rows.append(dict(
                    base,
                    quantity=(f"diagnostic {'size' if null else 'power'} "
                              f"(alpha = {a:.2f})"),
                    estimator=tag, value=p,
                    mc_se=float(np.sqrt(p * (1 - p) / R))))
        rows.append(dict(base, quantity="SI bias", estimator="",
                         value=float((g.tau_si - g.true_tau).mean()),
                         mc_se=float(g.tau_si.std(ddof=1) / np.sqrt(R))))
        rows.append(dict(base, quantity="true tau", estimator="",
                         value=float(g.true_tau.iloc[0]), mc_se=0.0))
    return pd.DataFrame(rows)


def write_markdown(summ: pd.DataFrame, path: str, crosscheck: str) -> None:
    R = int(summ.R.iloc[0])
    B = int(summ.B.iloc[0])
    band = {200: "[0.920, 0.980]", 100: "[0.907, 0.993]",
            500: "[0.931, 0.969]"}.get(R, "")
    lines = [
        "# Surrogate-index interval with a gradient-boosted-tree index\n\n",
        f"n = 10,000, pi_L = 0.20, R = {R} replications per cell, joint "
        f"paired bootstrap with B = {B} draws per replication (units "
        "resampled within arm, labeled flags carried along, trees refit "
        "under the all-units protocol, SI and PPI++ recomputed). The draws "
        "are the GBT comparison's (`run_gbt_comparison.py`, same seeds), so "
        "the point estimates reproduce its GBT rows. Every GBT SI row in "
        "that comparison uses si_variance = \"plugin\" (the Neyman variance "
        "of the imputed outcomes, trees treated as fixed); the diagnostic "
        "on a tree index can only use the fixed-predictor covariance. "
        "Monte Carlo = the variance of the statistic across the R "
        "replications (MC SE by the delta method). Plug-in, exact, "
        "fixed-predictor and bootstrap rows are the mean of the per-"
        "replication variance estimate (MC SE = SD / sqrt(R)); `ratio` is "
        "that mean over the Monte Carlo variance (MC SE by a bootstrap over "
        "replications). Coverage and rejection MC SEs are binomial."
        + (f" Monte Carlo band for 95% coverage at R = {R}: {band}."
           if band else "") + "\n\n",
        crosscheck + "\n\n",
        _reading(summ) + "\n\n",
    ]
    for cell in summ.cell.unique():
        sub = summ[summ.cell == cell]
        lines.append(f"## {cell}\n\n")
        lines.append("| Quantity | Estimator | Value (MC SE) | Ratio to MC (MC SE) |\n"
                     "|---|---|---:|---:|\n")
        for _, r in sub.iterrows():
            nd = 6 if r.quantity.startswith("Var") else 4 \
                if r.quantity in ("SI bias", "true tau") else 3
            val = f"{r.value:.{nd}f} ({r.mc_se:.{nd}f})"
            if r.quantity.startswith("Var"):
                val = f"{r.value:.3e} ({r.mc_se:.1e})"
            ratio = ("" if pd.isna(r.get("ratio_to_mc", np.nan))
                     else f"{r.ratio_to_mc:.3f} ({r.ratio_mc_se:.3f})")
            lines.append(f"| {r.quantity} | {r.estimator} | {val} | {ratio} |\n")
        lines.append("\n")
    with open(path, "w") as f:
        f.writelines(lines)


def _reading(summ: pd.DataFrame) -> str:
    """The verdict, built from the numbers so it cannot drift from them."""
    def r(cell, q, est):
        row = summ[(summ.cell == cell) & (summ.quantity == q)
                   & (summ.estimator == est)].iloc[0]
        return row.ratio_to_mc, row.ratio_mc_se
    parts = []
    for cell in summ.cell.unique():
        a, ase = r(cell, "Var(SI)", "plug-in")
        b, bse = r(cell, "Var(SI)", "paired bootstrap")
        c, cse = r(cell, "Var(D)", "fixed-predictor")
        d, dse = r(cell, "Var(PPI++)", "paired bootstrap")
        e, ese = r(cell, "Var(D)", "paired bootstrap")
        parts.append(
            f"{cell}: plug-in Var(SI) / MC = {a:.2f} ({ase:.2f}), bootstrap "
            f"Var(SI) / MC = {b:.2f} ({bse:.2f}); fixed-predictor Var(D) / MC "
            f"= {c:.2f} ({cse:.2f}); bootstrap Var(PPI++) / MC = {d:.2f} "
            f"({dse:.2f}), bootstrap Var(D) / MC = {e:.2f} ({ese:.2f})")
    return (
        "Reading (ratios to the Monte Carlo variance, MC SE in parentheses). "
        + "; ".join(parts) + ". The Monte Carlo variance is the benchmark. "
        "The tree-refitting bootstrap understates the PPI++ and D variances. "
        "The likely cause (not tested here): a resample contains duplicated "
        "units, the all-units fold split can put copies of one unit on both "
        "sides of a fold, and a tree fit then predicts the held-out copy "
        "partly from its own label, shrinking the labeled residuals that "
        "drive the PPI++ rectifier. OLS is too rigid for this to matter, "
        "consistent with the OLS paired bootstrap (run_joint_cov_check.py) "
        "tracking the sandwich.")


def crosscheck_comparison(raw: pd.DataFrame) -> str:
    """Compare the point estimates with gbt_comparison.csv, rep by rep."""
    path = os.path.join(TABLES_DIR, "gbt_comparison.csv")
    if not os.path.exists(path):
        return "Cross-check against gbt_comparison.csv: file not found."
    comp = pd.read_csv(path)
    comp = comp[(comp.prediction_model == "gbt") & np.isclose(comp.pi_L, PI_L)]
    msgs = []
    for cell, g in raw.groupby("cell"):
        dgp_id, overrides, label, _ = CELLS[int(cell)]
        c = comp[(comp.dgp_id == dgp_id) & (comp.overrides == str(overrides))]
        out = []
        for mid, col in ((2, "tau_si"), (3, "tau_ppi")):
            cm = c[c.method_id == mid].set_index("replication")["tau_hat"]
            common = g.set_index("rep")[col].index.intersection(cm.index)
            if len(common) == 0:
                out.append(f"method {mid}: no common replications")
                continue
            diff = np.max(np.abs(g.set_index("rep")[col].loc[common]
                                 - cm.loc[common]))
            out.append(f"{'SI' if mid == 2 else 'PPI++'} max |diff| = "
                       f"{diff:.1e} over {len(common)} replications")
        msgs.append(f"{label}: " + "; ".join(out))
    return ("Cross-check against the GBT comparison's per-replication "
            "estimates: " + " | ".join(msgs) + ".")


def _self_check() -> None:
    """`gbt_allunits_predict` on the pipeline's folds == the pipeline."""
    d = _generate(1, {}, _rep_seed(1, {}, 0))
    S, X, Y, lm = d["S"], d["X"], d["Y"], d["labeled_mask"]
    m = 3000
    S, X, Y, lm = S[:m], X[:m], Y[:m], lm[:m]
    ref = train_prediction_model(S, X, Y, lm, n_folds=N_FOLDS,
                                 rng=np.random.default_rng(5), model="gbt",
                                 seed=11)
    folds = allunits_fold_ids(m, N_FOLDS, np.random.default_rng(5))
    mine = gbt_allunits_predict(S, X, Y, lm, folds, 11)
    err = float(np.max(np.abs(ref - mine)))
    assert err == 0.0, f"gbt_allunits_predict deviates by {err:g}"
    ids = np.array([4, 4, 9, 1, 9, 9, 2, 7, 1, 3])
    f = unit_clustered_fold_ids(ids, 3, np.random.default_rng(0))
    for u in np.unique(ids):
        assert len(set(f[ids == u])) == 1, "copies split across folds"
    print(f"self-check ok: gbt_allunits_predict reproduces "
          f"train_prediction_model (max |dY_hat| = {err:.1e}); clustered "
          f"folds keep copies together", flush=True)


# ---------------------------------------------------------------------------
# Clustered-fold comparison
# ---------------------------------------------------------------------------

CLUSTERED_STEM = "gbt_bootstrap_clustered"
CELLS_BY_LABEL = {v[2]: v for v in CELLS.values()}


def compare_clustered(clu: pd.DataFrame, rows: pd.DataFrame,
                      full_R: int,
                      rows_summary: pd.DataFrame = None) -> pd.DataFrame:
    """Clustered-fold bootstrap beside the row-fold bootstrap, same draws.

    ``clu`` is the clustered run, ``rows`` the row-fold run of
    gbt_variance_check_raw.csv.  Both are restricted to their common
    replications, where the point estimates coincide (checked), so the
    Monte Carlo variance is shared and the ratio of the two mean bootstrap
    variances is a paired comparison.  Rows for the first ``R_sub``
    replications are added when the run is longer than the planned
    R = 100.

    When the common replications are the whole row-fold run and its
    summary (gbt_variance_check.csv) is given, the row-fold rows (plug-in,
    exact, fixed-predictor, row-fold bootstrap) take their ratio MC SEs from
    that summary, so the same quantity carries one SE in both artifacts; the
    point values are recomputed and must agree.
    """
    out: List[Dict[str, Any]] = []
    for cell, gc in clu.groupby("cell"):
        dgp_id, overrides, label, null = CELLS[int(cell)]
        gr = rows[rows.cell == cell].set_index("rep")
        gc = gc.set_index("rep")
        common = gc.index.intersection(gr.index)
        gc, gr = gc.loc[common], gr.loc[common]
        for col in ("tau_si", "tau_ppi", "D_hat", "var_si_plugin",
                    "var_ppi_exact", "var_D_fixedf"):
            diff = float(np.max(np.abs(gc[col] - gr[col])))
            assert diff < 1e-12, f"{col} differs between runs by {diff:g}"
        subsets = [("all", common)]
        if len(common) > 100:
            subsets.append(("first 100", common[common < 100]))
        for sub_tag, reps in subsets:
            c, r = gc.loc[reps], gr.loc[reps]
            R = len(reps)
            base = dict(cell=label, dgp=dgp_id, rho=overrides.get("rho", ""),
                        n=10_000, pi_L=PI_L, R=R, reps=sub_tag,
                        B_clustered=int(c.B.iloc[0]),
                        B_rows=int(r.B.iloc[0]), learner="gbt",
                        diagnostic_null=null)
            k = 0
            for stat_col, name, ests in (
                ("tau_si", "Var(SI)", (("plug-in", r, "var_si_plugin"),
                                       ("bootstrap, row folds", r,
                                        "boot_var_si"),
                                       ("bootstrap, clustered folds", c,
                                        "boot_var_si"))),
                ("tau_ppi", "Var(PPI++)", (("exact", r, "var_ppi_exact"),
                                           ("bootstrap, row folds", r,
                                            "boot_var_ppi"),
                                           ("bootstrap, clustered folds", c,
                                            "boot_var_ppi"))),
                ("D_hat", "Var(D)", (("fixed-predictor", r, "var_D_fixedf"),
                                     ("bootstrap, row folds", r,
                                      "boot_var_D"),
                                     ("bootstrap, clustered folds", c,
                                      "boot_var_D"))),
            ):
                stat = c[stat_col].to_numpy()
                mc, mc_se = _mcvar_se(stat)
                out.append(dict(base, quantity=name, estimator="Monte Carlo",
                                value=mc, mc_se=mc_se, ratio_to_mc=1.0,
                                ratio_mc_se=0.0))
                for tag, src, col in ests:
                    k += 1
                    v, se = _mean_se(src[col].to_numpy())
                    ratio, rse = _ratio_se(src[col].to_numpy(), stat,
                                           seed=7000 * int(cell) + k)
                    if (rows_summary is not None and src is r
                            and R == full_R):
                        old_tag = ("paired bootstrap"
                                   if tag == "bootstrap, row folds" else tag)
                        o = rows_summary[(rows_summary.cell == label)
                                         & (rows_summary.quantity == name)
                                         & (rows_summary.estimator
                                            == old_tag)].iloc[0]
                        assert abs(o.ratio_to_mc - ratio) < 1e-12, (
                            f"{name} {tag}: ratio {ratio} vs "
                            f"gbt_variance_check.csv {o.ratio_to_mc}")
                        rse = float(o.ratio_mc_se)
                    out.append(dict(base, quantity=name, estimator=tag,
                                    value=v, mc_se=se, ratio_to_mc=ratio,
                                    ratio_mc_se=rse))
                # Paired: clustered over row-fold mean bootstrap variance.
                boot_col = {"tau_si": "boot_var_si", "tau_ppi": "boot_var_ppi",
                            "D_hat": "boot_var_D"}[stat_col]
                a = c[boot_col].to_numpy()
                b = r[boot_col].to_numpy()
                rng = np.random.default_rng(9000 * int(cell) + k)
                dr = np.empty(2000)
                for j in range(2000):
                    i = rng.integers(0, R, R)
                    dr[j] = a[i].mean() / b[i].mean()
                out.append(dict(base, quantity=name,
                                estimator="clustered / row-fold bootstrap",
                                value=float(a.mean() / b.mean()),
                                mc_se=float(dr.std(ddof=1))))
            for col, name, tag, src in (
                ("cov_si_plugin", "SI coverage", "plug-in", r),
                ("cov_si_boot", "SI coverage", "bootstrap, row folds", r),
                ("cov_si_boot", "SI coverage", "bootstrap, clustered folds",
                 c),
                ("cov_ppi_exact", "PPI++ coverage", "exact", r),
                ("cov_ppi_boot", "PPI++ coverage", "bootstrap, row folds", r),
                ("cov_ppi_boot", "PPI++ coverage",
                 "bootstrap, clustered folds", c),
            ):
                p = float(src[col].mean())
                out.append(dict(base, quantity=name, estimator=tag, value=p,
                                mc_se=float(np.sqrt(p * (1 - p) / R))))
            for a_ in ALPHAS:
                sfx = f"a{int(a_ * 100):02d}"
                for tag, src, col in (
                    ("fixed-predictor", r, f"rej_fixedf_{sfx}"),
                    ("bootstrap, row folds", r, f"rej_boot_{sfx}"),
                    ("bootstrap, clustered folds", c, f"rej_boot_{sfx}"),
                ):
                    p = float(src[col].mean())
                    out.append(dict(
                        base,
                        quantity=(f"diagnostic {'size' if null else 'power'}"
                                  f" (alpha = {a_:.2f})"),
                        estimator=tag, value=p,
                        mc_se=float(np.sqrt(p * (1 - p) / R))))
    return pd.DataFrame(out)


def _clustered_reading(cmp_: pd.DataFrame) -> str:
    """Verdict computed from the table, so it cannot drift from it."""
    parts, verdicts = [], []
    for cell in cmp_.cell.unique():
        sub = cmp_[(cmp_.cell == cell) & (cmp_.reps == "all")]

        def g(q, e):
            row = sub[(sub.quantity == q) & (sub.estimator == e)].iloc[0]
            return row

        seg = []
        for q in ("Var(SI)", "Var(PPI++)", "Var(D)"):
            rr = g(q, "bootstrap, row folds")
            cc = g(q, "bootstrap, clustered folds")
            pr = g(q, "clustered / row-fold bootstrap")
            seg.append(
                f"{q}: row folds {rr.ratio_to_mc:.2f} ({rr.ratio_mc_se:.2f}),"
                f" clustered {cc.ratio_to_mc:.2f} ({cc.ratio_mc_se:.2f}), "
                f"clustered over row-fold {pr.value:.2f} ({pr.mc_se:.2f})")
            if q != "Var(SI)":
                verdicts.append(abs(cc.ratio_to_mc - 1.0)
                                <= 2.0 * cc.ratio_mc_se)
        for a_ in ALPHAS:
            q = (f"diagnostic {'size' if CELLS_BY_LABEL[cell][3] else 'power'}"
                 f" (alpha = {a_:.2f})")
            rr = g(q, "bootstrap, row folds")
            cc = g(q, "bootstrap, clustered folds")
            seg.append(f"{q}: row folds {rr.value:.3f} ({rr.mc_se:.3f}), "
                       f"clustered {cc.value:.3f} ({cc.mc_se:.3f})")
        parts.append(f"{cell}, R = {int(sub.R.iloc[0])}: " + "; ".join(seg))
    gone = all(verdicts)
    concl = (
        "Assigning folds by original unit removes the understatement: the "
        "clustered-fold bootstrap Var(PPI++) and Var(D) are within two Monte "
        "Carlo standard errors of the Monte Carlo variance, so the duplicated-"
        "copy mechanism accounts for the row-fold shortfall, and the "
        "clustered-fold bootstrap is a usable validation reference for the "
        "tree index."
        if gone else
        "Assigning folds by original unit does not remove the "
        "understatement: at least one of the clustered-fold bootstrap "
        "Var(PPI++) and Var(D) ratios is more than two Monte Carlo standard "
        "errors below one, so the duplicated-copy mechanism does not account "
        "for the whole shortfall and the tree-refitting bootstrap stays "
        "exploratory and unvalidated.")
    return ("Reading (ratio of the mean bootstrap variance to the Monte Carlo "
            "variance, MC SE in parentheses; the clustered-over-row-fold "
            "ratio is paired on the same replications, so it does not carry "
            "the Monte Carlo variance's noise). " + " | ".join(parts) + ". "
            + concl)


def write_clustered_markdown(cmp_: pd.DataFrame, path: str, wall: float,
                             workers: int, full_R: int, full_B: int) -> None:
    Bc = int(cmp_.B_clustered.iloc[0])
    lines = [
        "# Tree-refitting bootstrap with unit-clustered folds\n\n",
        f"GBT index, n = 10,000, pi_L = 0.20, all-units cross-fitting. The "
        f"joint paired bootstrap (units resampled within arm, labeled flags "
        f"carried along, trees refit, SI and PPI++ recomputed) is run twice "
        f"on the same draws. Row folds: the fold split permutes the "
        f"resampled rows, as the pipeline does, so copies of one original "
        f"unit can land on both sides of a fold (B = {full_B}, from "
        f"`gbt_variance_check_raw.csv`, R = {full_R}). Clustered folds: the "
        f"distinct original units are permuted and cut into five blocks and "
        f"every copy inherits its unit's fold, so no copy is predicted by a "
        f"tree trained on another copy (B = {Bc}, `--clustered-folds`). "
        "The point estimates, the plug-in, exact and fixed-predictor "
        "variances, and hence the Monte Carlo variance are identical in the "
        "two runs (checked replication by replication). Monte Carlo = the "
        "variance of the statistic across replications (MC SE by the delta "
        "method); other variance rows are the mean per-replication estimate "
        "(MC SE = SD / sqrt(R)); `ratio` is that mean over the Monte Carlo "
        "variance (MC SE by a bootstrap over replications). The mean of a "
        "bootstrap sample variance does not depend on B, so the different B "
        "of the two runs affects only the per-replication noise (coverage "
        "and size), not the ratio's expectation. Coverage and rejection MC "
        "SEs are binomial. Monte Carlo band for 95% coverage: "
        "[0.920, 0.980] at R = 200, [0.907, 0.993] at R = 100.\n\n",
        _clustered_reading(cmp_) + "\n\n",
    ]
    for cell in cmp_.cell.unique():
        for sub_tag in ("all", "first 100"):
            sub = cmp_[(cmp_.cell == cell) & (cmp_.reps == sub_tag)]
            if sub.empty:
                continue
            R = int(sub.R.iloc[0])
            title = (f"{cell}, replications 0-{R - 1} (R = {R})")
            lines.append(f"## {title}\n\n")
            lines.append("| Quantity | Estimator | Value (MC SE) "
                         "| Ratio to MC (MC SE) |\n|---|---|---:|---:|\n")
            for _, r in sub.iterrows():
                if r.estimator == "clustered / row-fold bootstrap":
                    val = f"{r.value:.3f} ({r.mc_se:.3f})"
                elif r.quantity.startswith("Var"):
                    val = f"{r.value:.3e} ({r.mc_se:.1e})"
                else:
                    val = f"{r.value:.3f} ({r.mc_se:.3f})"
                ratio = ("" if pd.isna(r.get("ratio_to_mc", np.nan))
                         else f"{r.ratio_to_mc:.3f} ({r.ratio_mc_se:.3f})")
                lines.append(f"| {r.quantity} | {r.estimator} | {val} "
                             f"| {ratio} |\n")
            lines.append("\n")
    lines.append(f"Wall time of the clustered run: {wall:.0f} s on "
                 f"{workers} cores.\n")
    with open(path, "w") as f:
        f.writelines(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--cells", default="1,2",
                    help="comma-separated cell ids (1 = DGP 1, 2 = DGP 2 "
                         "rho = 0.2)")
    ap.add_argument("--clustered-folds", action="store_true",
                    help="assign bootstrap folds by original unit and write "
                         f"{CLUSTERED_STEM}.{{md,csv}} (+ _raw.csv) beside "
                         "the row-fold run instead of gbt_variance_check.*")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--from-raw", action="store_true",
                    help="rebuild the tables from the raw csv")
    ap.add_argument("--outdir", default=TABLES_DIR)
    args = ap.parse_args()

    stem = CLUSTERED_STEM if args.clustered_folds else "gbt_variance_check"
    raw_path = os.path.join(args.outdir, f"{stem}_raw.csv")
    cells = [int(c) for c in args.cells.split(",") if c.strip()]
    t0 = time.time()
    if args.from_raw:
        raw = pd.read_csv(raw_path)
        wall = float("nan")
    else:
        _self_check()
        tasks = [(c, r, args.B, args.clustered_folds)
                 for c in cells for r in range(args.R)]
        print(f"{len(tasks)} tasks (R = {args.R}, B = {args.B}, "
              f"folds = {'clustered' if args.clustered_folds else 'rows'}, "
              f"{args.workers} workers)", flush=True)
        with multiprocessing.Pool(args.workers) as pool:
            out = []
            for i, row in enumerate(pool.imap_unordered(run_one, tasks)):
                out.append(row)
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1}/{len(tasks)} done, "
                          f"{time.time() - t0:.0f}s", flush=True)
        raw = pd.DataFrame(out).sort_values(["cell", "rep"])
        os.makedirs(args.outdir, exist_ok=True)
        raw.to_csv(raw_path, index=False)
        wall = time.time() - t0

    if args.clustered_folds:
        rows = pd.read_csv(os.path.join(TABLES_DIR,
                                        "gbt_variance_check_raw.csv"))
        full_R = int(rows.groupby("cell").size().max())
        full_B = int(rows.B.iloc[0])
        if args.from_raw:
            md_old = os.path.join(args.outdir, f"{stem}.md")
            if os.path.exists(md_old):
                for line in open(md_old):
                    if line.startswith("Wall time of the clustered run:"):
                        wall = float(line.split(":")[1].split("s")[0])
        rows_summary = pd.read_csv(os.path.join(TABLES_DIR,
                                                "gbt_variance_check.csv"))
        cmp_ = compare_clustered(raw, rows, full_R, rows_summary)
        cmp_.to_csv(os.path.join(args.outdir, f"{stem}.csv"), index=False)
        write_clustered_markdown(cmp_, os.path.join(args.outdir,
                                                    f"{stem}.md"),
                                 wall, args.workers, full_R, full_B)
        print(_clustered_reading(cmp_))
        print(f"\nWrote {stem}.{{csv,md}} in {time.time() - t0:.0f}s")
        return

    summ = summarize(raw)
    summ.to_csv(os.path.join(args.outdir, "gbt_variance_check.csv"),
                index=False)
    write_markdown(summ, os.path.join(args.outdir, "gbt_variance_check.md"),
                   crosscheck_comparison(raw))
    print(summ[["cell", "quantity", "estimator", "value", "mc_se",
                "ratio_to_mc"]].to_string(index=False))
    print(f"\nWrote gbt_variance_check.{{csv,md}} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
