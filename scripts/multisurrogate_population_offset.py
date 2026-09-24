#!/usr/bin/env python3
"""
Population restriction d_0' beta_0 = tau on the multi-surrogate funnel.

The SI--PPI++ diagnostic's null needs statistical surrogacy AND the
population restriction d_0' beta_0 = tau, where

  beta_0 = E[g g']^{-1} E[g Y]   population OLS of Y on the index basis g
                                 over the labeled population (MCAR here, so
                                 the full population),
  d_0    = E[g | T = 1] - E[g | T = 0].

d_0' beta_0 is the probability limit of the surrogate index.  Correct
conditional-mean specification is sufficient for the restriction but not
necessary.  On the Criteo-calibrated funnel of scripts/run_multisurrogate.py
the conversion stage is logistic,

  S1 | T, X  ~ Bernoulli(expit(X a_v + c_v + b_v T))            (visit)
  S2 | T     ~ N(zeta T, 1),  zeta = 0.5                          (engagement)
  Y  | S1, S2, T, X = S1 * Bernoulli(expit(X a_c + c_c + theta S2 + kappa_r T)),
  theta = m kappa_hat / zeta,  kappa_r = (1 - m) kappa_hat,

while the index is OLS on the basis

  rich:  g = (1, S1, S2, S1^2, S2^2, X)        poor:  g = (1, S1, S1^2, X),

with S1^2 = S1 (binary), so the design is rank deficient by one column.
The fitted values, and hence d_0' beta_0, do not depend on which spanning
subset is kept; the computation below drops the S1^2 column.

Method (primary): exact expectations over the population the DGP samples
from.  X is drawn with replacement from the 500,000-row covariate pool, so
the population X law is the pool's empirical law; T ~ Bernoulli(1/2); S1 is
summed out exactly; S2 is integrated by Gauss-Hermite quadrature (the Gram
matrix is polynomial of degree 4 in S2 and is exact with 7 nodes; E[g Y]
involves the logistic link and uses 121 nodes, checked against 61).  tau is
`run_multisurrogate.true_tau` (31-node quadrature; recomputed here with 121
nodes as a check).

Method (check): one simulated sample of n = 2,000,000 units per m, all units
used in a full-sample OLS fit (MCAR labeling leaves the population
restriction unchanged); d-hat' beta-hat minus tau with a batch-means SE over
20 batches of 100,000.

Output: results/tables/multisurrogate_population_offset.{md,csv}
Usage:  python scripts/multisurrogate_population_offset.py [--n-sim 2000000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (PROJECT_ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_multisurrogate as ms  # noqa: E402  (POOL_PATH, PARAM_PATH, true_tau)
from src.simulations.simulation import derive_seed  # noqa: E402

TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
M_GRID = [0.0, 0.5, 1.0]
ZETA = ms.ZETA
SEED_TAG = 340


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _gh(k: int) -> Tuple[np.ndarray, np.ndarray]:
    nodes, w = np.polynomial.hermite_e.hermegauss(k)
    return nodes, w / w.sum()


def _coefs(params: Dict[str, Any], m: float):
    av = np.asarray(params["visit_coef"][:-1]); bv = params["visit_coef"][-1]
    cv = params["visit_intercept"]
    ac = np.asarray(params["conv_coef"][:-1]); cc = params["conv_intercept"]
    kap = params["kappa_hat"]
    return av, bv, cv, ac, cc, m * kap / ZETA, (1 - m) * kap


def _basis(version: str, s1, s2, X):
    """Index basis with the duplicate S1^2 column dropped."""
    n = X.shape[0]
    s1 = np.broadcast_to(np.asarray(s1, float), (n,))
    one = np.ones(n)
    if version == "poor":
        return np.column_stack([one, s1, X])
    s2 = np.broadcast_to(np.asarray(s2, float), (n,))
    return np.column_stack([one, s1, s2, s2 ** 2, X])


def population_quadrature(params, X_pool, m: float, version: str,
                          k_gram: int = 7, k_mean: int = 121
                          ) -> Dict[str, float]:
    """Exact population d_0, beta_0 and tau by summation and quadrature."""
    av, bv, cv, ac, cc, theta, kr = _coefs(params, m)
    lin_v = X_pool @ av + cv
    lin_c = X_pool @ ac + cc
    N = X_pool.shape[0]
    q = _basis(version, 0.0, 0.0, X_pool[:1]).shape[1]

    G = np.zeros((q, q))
    gY = np.zeros(q)
    gbar = {0: np.zeros(q), 1: np.zeros(q)}
    EY = {0: 0.0, 1: 0.0}

    for t in (0, 1):
        pv = sigmoid(lin_v + bv * t)                    # P(S1 = 1 | T, X)
        for s1 in (0, 1):
            w_s1 = (pv if s1 == 1 else 1.0 - pv) / N    # over X, given T
            # Gram and conditional means: polynomial in S2 -> k_gram nodes
            nodes, wk = _gh(k_gram) if version == "rich" else (np.zeros(1),
                                                              np.ones(1))
            for z, w in zip(nodes, wk):
                s2 = ZETA * t + z
                g = _basis(version, s1, s2, X_pool)
                wt = 0.5 * w * w_s1
                G += (g * wt[:, None]).T @ g
                gbar[t] += 2.0 * (g * wt[:, None]).sum(axis=0)   # E[g | T=t]
            if s1 == 0:
                continue                                # Y = 0 when S1 = 0
            nodes, wk = _gh(k_mean)
            for z, w in zip(nodes, wk):
                s2 = ZETA * t + z
                py = sigmoid(lin_c + theta * s2 + kr * t)
                g = _basis(version, s1, s2, X_pool)
                wt = 0.5 * w * w_s1 * py
                gY += (g * wt[:, None]).sum(axis=0)
                EY[t] += 2.0 * wt.sum()
    beta0 = np.linalg.solve(G, gY)
    d0 = gbar[1] - gbar[0]
    return dict(si_limit=float(d0 @ beta0), tau=float(EY[1] - EY[0]),
                cond_G=float(np.linalg.cond(G)))


def simulate(params, X_pool, m: float, n: int, seed: int,
             n_batches: int = 20) -> Dict[str, Tuple[float, float]]:
    """d-hat' beta-hat - tau on one large sample, both index versions."""
    rng = np.random.default_rng(seed)
    av, bv, cv, ac, cc, theta, kr = _coefs(params, m)
    X = X_pool[rng.choice(len(X_pool), size=n, replace=True)]
    T = (rng.random(n) < 0.5).astype(float)
    S1 = (rng.random(n) < sigmoid(X @ av + cv + bv * T)).astype(float)
    S2 = ZETA * T + rng.standard_normal(n)
    pY = sigmoid(X @ ac + cc + theta * S2 + kr * T)
    Y = np.where(S1 == 1, (rng.random(n) < pY).astype(float), 0.0)
    out = {}
    for version in ("poor", "rich"):
        g = _basis(version, S1, S2, X)

        def si(idx):
            gi, Yi, Ti = g[idx], Y[idx], T[idx]
            beta = np.linalg.lstsq(gi, Yi, rcond=None)[0]
            d = gi[Ti == 1].mean(0) - gi[Ti == 0].mean(0)
            return float(d @ beta)

        full = si(np.arange(n))
        batches = np.array_split(rng.permutation(n), n_batches)
        vals = np.array([si(b) for b in batches])
        out[version] = (full, float(vals.std(ddof=1) / np.sqrt(n_batches)))
    return out


def observed_bias(raw_path: str) -> Dict[Tuple[float, str], Tuple[float, ...]]:
    """Mean SI bias with MC SE, and the Monte Carlo SD of D = tau_PPI - tau_SI,
    from multisurrogate_raw.csv (per replication, D = bias_PPI - bias_SI)."""
    if not os.path.exists(raw_path):
        return {}
    raw = pd.read_csv(raw_path)
    out = {}
    for (m, v), g in raw.groupby(["m", "version"]):
        si = g[g.method == "SI"].set_index("rep")["bias"]
        ppi = g[g.method == "PPI++"].set_index("rep")["bias"]
        D = (ppi - si).dropna()
        out[(float(m), v)] = (float(si.mean()),
                              float(si.std(ddof=1) / np.sqrt(len(si))),
                              int(len(si)), float(D.std(ddof=1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sim", type=int, default=2_000_000)
    ap.add_argument("--outdir", default=TABLES_DIR)
    args = ap.parse_args()
    t0 = time.time()

    with open(ms.PARAM_PATH) as f:
        params = json.load(f)
    X_pool = np.load(ms.POOL_PATH)["X"].astype(np.float64)
    obs = observed_bias(os.path.join(TABLES_DIR, "multisurrogate_raw.csv"))

    rows: List[Dict[str, Any]] = []
    for m in M_GRID:
        tau_ms = ms.true_tau(params, X_pool, m)
        sim = (simulate(params, X_pool, m, args.n_sim,
                        derive_seed(42, SEED_TAG, int(m * 10), 0, 0))
               if args.n_sim > 0 else {})
        for version in ("poor", "rich"):
            pop = population_quadrature(params, X_pool, m, version)
            chk = population_quadrature(params, X_pool, m, version,
                                        k_gram=9, k_mean=61)
            off = pop["si_limit"] - tau_ms
            row = dict(
                m=m, index=version, tau=tau_ms,
                tau_quadrature_121=pop["tau"],
                si_limit=pop["si_limit"],
                offset=off, rel_offset_pct=100.0 * off / tau_ms,
                quadrature_change_61_vs_121=abs(chk["si_limit"]
                                                - pop["si_limit"]),
                gram_condition=pop["cond_G"],
            )
            if version in sim:
                full, se = sim[version]
                # tau of the simulated sample is the population tau (the
                # restriction is a population statement); report the gap.
                row.update(sim_n=args.n_sim, sim_offset=full - tau_ms,
                           sim_offset_se=se)
            if (m, version) in obs:
                b, se, R, sdD = obs[(m, version)]
                row.update(observed_si_bias=b, observed_si_bias_mcse=se,
                           observed_R=R, mc_sd_D=sdD,
                           offset_over_sd_D=off / sdD,
                           observed_rel_bias_pct=100.0 * b / tau_ms,
                           observed_minus_offset=b - off,
                           observed_minus_offset_z=(b - off) / se)
            rows.append(row)
            print({k: (f"{v:.3e}" if isinstance(v, float) else v)
                   for k, v in row.items()}, flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir,
                           "multisurrogate_population_offset.csv"), index=False)

    rich1 = df[(df.m == 1.0) & (df["index"] == "rich")].iloc[0]
    lines = [
        "# Population restriction d_0' beta_0 - tau on the multi-surrogate "
        "funnel\n\n",
        "Rich-funnel outcome equation: S1 | T, X ~ Bernoulli(expit(X a_v + "
        "c_v + b_v T)); S2 | T ~ N(0.5 T, 1); Y = S1 * Bernoulli(expit(X a_c "
        "+ c_c + theta S2 + (1 - m) kappa_hat T)), theta = m kappa_hat / 0.5, "
        f"kappa_hat = {params['kappa_hat']:.4f} "
        "(`scripts/run_multisurrogate.py`, parameters in "
        "`results/tables/semisynth_params.json`). Index basis: rich g = (1, "
        "S1, S2, S1^2, S2^2, X), poor g = (1, S1, S1^2, X), X the 12 Criteo "
        "covariates; S1^2 = S1, so one column is redundant and is dropped "
        "(fitted values are unchanged). beta_0 = E[g g']^{-1} E[g Y] is the "
        "population OLS coefficient over the labeled population (MCAR, so the "
        "full population), d_0 = E[g | T = 1] - E[g | T = 0], and d_0' beta_0 "
        "is the probability limit of the surrogate index. Expectations are "
        "exact over the 500,000-row covariate pool (the DGP's X law) and T, "
        "with S1 summed out and S2 integrated by Gauss-Hermite quadrature "
        "(Gram matrix 7 nodes, exact; E[g Y] 121 nodes; `quad change` is the "
        "change from 61 to 121 nodes and 7 to 9). tau is "
        "`run_multisurrogate.true_tau`. `sim` is d-hat' beta-hat - tau from "
        f"one simulated sample of n = {args.n_sim:,} (full-sample OLS), SE by "
        "batch means over 20 batches. `observed SI bias` is the mean SI error "
        "in `multisurrogate_raw.csv` (n = 64,000, pi_L = 0.20, R = 500, "
        "all-units cross-fitting) with its MC SE; z = (observed - offset) / "
        "MC SE. `Offset / SD(D)` divides the offset by the Monte Carlo SD of "
        "D = tau_PPI - tau_SI in the same run: the diagnostic's noncentrality "
        "at n = 64,000 induced by the offset.\n\n",
        "| m | Index | tau | d_0' beta_0 | Offset d_0' beta_0 - tau | Offset % "
        "of tau | Quad change | Sim offset (SE) | Observed SI bias (MC SE) "
        "| Observed % | z | Offset / SD(D) |\n"
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in df.iterrows():
        sim = (f"{r.sim_offset:+.2e} ({r.sim_offset_se:.1e})"
               if "sim_offset" in r and pd.notna(r.get("sim_offset")) else "---")
        ob = (f"{r.observed_si_bias:+.2e} ({r.observed_si_bias_mcse:.1e})"
              if pd.notna(r.get("observed_si_bias", np.nan)) else "---")
        lines.append(
            f"| {r.m:.1f} | {r['index']} | {r.tau:.6f} | {r.si_limit:.6f} "
            f"| {r.offset:+.3e} | {r.rel_offset_pct:+.2f} "
            f"| {r.quadrature_change_61_vs_121:.1e} | {sim} | {ob} "
            f"| {r.get('observed_rel_bias_pct', np.nan):+.1f} "
            f"| {r.get('observed_minus_offset_z', np.nan):+.2f} "
            f"| {r.get('offset_over_sd_D', np.nan):+.3f} |\n")
    lines.append(
        "\nNotes: at m = 1 the rich index satisfies statistical surrogacy by "
        "construction, but the OLS basis is not the logistic conditional "
        f"mean, and the restriction holds only approximately: the offset is "
        f"{rich1.offset:+.2e} ({rich1.rel_offset_pct:+.2f}% of tau), far "
        "above quadrature precision "
        f"({rich1.quadrature_change_61_vs_121:.0e}). The cell is a near-null, "
        "not an exact null; the offset is "
        f"{abs(rich1.offset_over_sd_D):.3f} Monte Carlo SDs of D at n = 64,000, "
        "so it moves the diagnostic's rejection rate by a negligible amount "
        "at that sample size.\n")
    with open(os.path.join(args.outdir,
                           "multisurrogate_population_offset.md"), "w") as f:
        f.writelines(lines)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
