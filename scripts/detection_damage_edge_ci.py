#!/usr/bin/env python3
"""
Monte Carlo intervals for the detection-damage band edges.

The paper quotes the two band edges to three decimals:

  * the **damage edge**, the smallest rho at which SI coverage falls through
    0.90, and
  * the **detection edge**, the smallest rho at which the two-sided SI--PPI++
    estimator-disagreement diagnostic has power above 0.50,

both read off `results/tables/detection_damage_paired.csv` by linear
interpolation between bracketing grid points (the same `first_crossing` rule
that `scripts/make_detection_damage_figure.py` uses for the shaded band).
Neither edge carried a Monte Carlo interval, and an independent grid
(`results/tables/frontier_coverage.csv`) puts the damage edge somewhere else
entirely.  This script supplies the missing uncertainty.

Method
------
Nonparametric bootstrap over REPLICATIONS of the paired run, reading the
per-replication file `results/tables/detection_damage_paired_raw.csv`.  Each
(n, rho) cell is its own independent Monte Carlo experiment (the seeds are
`derive_seed(42, 250, rho, n, rep)`, so no draw is shared across cells), so
each cell is resampled independently with replacement to its own R.  For every
bootstrap resample the whole SI coverage curve and the whole two-sided power
curve are recomputed on the rho grid and both edges are re-interpolated.  The
reported intervals are 95% percentile intervals over B resamples of

  * the damage edge (coverage 0.90; it does not depend on alpha),
  * the detection edge (power 0.50) at each alpha,
  * the band width (detection edge minus damage edge), and
  * the two shrink factors, edge(n = 10,000) / edge(n = 100,000), one per
    edge, which is the quantity the paper's "both edges shrink at close to
    the sqrt(n) rate" reading rests on (sqrt(10) = 3.162).

Degenerate resamples are counted rather than dropped:

  * `left_edge_frac` -- share of resamples in which the curve is already past
    the threshold at the first grid point, so the edge is reported at rho = 0
    (the band opens at the left edge of the grid);
  * `no_crossing_frac` -- share of resamples in which the curve never crosses
    on the grid, which contribute NaN and are excluded from the percentiles.

The same interpolation is also applied to `results/tables/frontier_coverage.csv`
-- an independent run on a coarser rho grid at R = 500 with no diagnostic
columns -- and printed side by side. That file stores only per-cell coverage
proportions, so its interval is a PARAMETRIC (binomial) bootstrap:
`si_coverage` at each grid point is redrawn as Binomial(R, p_hat)/R.

Refined grid and grid-resolution bracket
-----------------------------------------------------------------------
A bootstrap over replications at FIXED grid points cannot see the error of
replacing the curved coverage function by the interpolant between them.
`scripts/run_detection_damage_paired.py --refine` adds a local grid around
the damage crossing (steps of 0.005; R = 2,000 at rho = 0 and 1,000
elsewhere), written to `detection_damage_refined_raw.csv`.  The FINE curve
at each n is the refined points plus the primary points outside the refined
range (at a shared rho the refined cell contains the primary draws and more).
The damage edge is then computed on the coarse and on the fine curve under
two interpolants:

  * `linear`  -- the rule above;
  * `pchip`   -- monotone piecewise-cubic Hermite interpolation
                 (scipy PchipInterpolator, Fritsch-Carlson), which is monotone
                 on every interval whose end values are, so the crossing
                 inside the first bracketing interval is unique.

The spread of the four point estimates is the grid-resolution bracket.  Each
bootstrap interval is conditional on its grid and interpolant: it measures
Monte Carlo error at the grid points and nothing else.  The detection edges
are read off the coarse grid by linear interpolation as before (the refined
grid does not reach them).

Output
------
results/tables/detection_damage_edges_ci.csv   (adds the fine-grid block)
results/tables/detection_damage_edges_ci.md
results/tables/detection_damage_refined.csv    (damage edges: grid x interp.)
results/tables/detection_damage_refined.md

Usage:
    python scripts/detection_damage_edge_ci.py [--B 1000]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.simulations.simulation import derive_seed

TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
RAW_SRC = os.path.join(TABLES_DIR, "detection_damage_paired_raw.csv")
REFINED_SRC = os.path.join(TABLES_DIR, "detection_damage_refined_raw.csv")
FRONTIER_SRC = os.path.join(TABLES_DIR, "frontier_coverage.csv")

MASTER_SEED = 42
SEED_TAG = 251           # one past run_detection_damage_paired.py's 250

RHO_MAX = 0.60
COVERAGE_THRESHOLD = 0.90
POWER_THRESHOLD = 0.50
ALPHAS = [0.05, 0.10]
SQRT10 = float(np.sqrt(10.0))

#: DGP 2 holds the mediated effect beta_YS * gamma_S = 0.5 * 0.3 = 0.15 fixed
#: and sets the direct effect delta = rho / (1 - rho) * 0.15
#: (`src.dgps.dgps.generate_dgp2`; paper Section 3, "delta = rho tau with
#: tau = 0.15 / (1 - rho)"; the `true tau` column of
#: detection_damage_paired.md is 0.15 + delta).
MEDIATED = 0.15


def rho_to_delta(rho: Optional[float]) -> Optional[float]:
    """Direct effect delta on the DGP 2 dial rho (None and inf pass through)."""
    if rho is None:
        return None
    rho = float(rho)
    if not np.isfinite(rho):
        return rho
    return MEDIATED * rho / (1.0 - rho)


def _identity(x: Optional[float]) -> Optional[float]:
    return x


SCALES = (("rho", _identity), ("delta", rho_to_delta))


def _ratio(small, large):
    """Shrink-factor draws small / large.  A resample whose n = 100,000 edge
    sits at the left edge of the grid (0 on either scale) has an unbounded
    ratio: +inf, kept in the order statistics."""
    out = []
    for s, l in zip(small, large):
        if s is None or l is None:
            out.append(None)
        elif l > 1e-12:
            out.append(s / l)
        else:
            out.append(float("inf") if s > 1e-12 else None)
    return out


def _ratio_point(s, l):
    if s is None or l is None:
        return None
    return _ratio([s], [l])[0]


#: Column carrying the headline (joint sandwich) SI coverage indicator.
SI_COV_COL = "si_cov_sandwich"
#: Two-sided rejection indicator of the headline (sandwich SE) diagnostic.
REJ_COL = {0.05: "rej_sandwich_a05", 0.10: "rej_sandwich_a10"}


# ---------------------------------------------------------------------------
# Interpolation (identical rule to make_detection_damage_figure.py, with an
# explicit convention for a curve that is already past the threshold at the
# first grid point)
# ---------------------------------------------------------------------------

#: Returned in place of an edge when the curve starts past the threshold.
LEFT_EDGE = 0.0


def first_crossing(
    x: Sequence[float], y: Sequence[float], level: float, direction: str
) -> Optional[float]:
    """Linearly interpolate the first x at which y crosses `level`.

    ``direction == "down"``: first x where y falls from >= level to < level.
    ``direction == "up"``:   first x where y rises from <= level to > level.

    Returns ``LEFT_EDGE`` when the curve is ALREADY past the threshold at the
    first grid point (coverage below 0.90, or power above 0.50, at rho = 0),
    and ``None`` when it never crosses on the grid.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if direction == "down" and y[0] < level:
        return LEFT_EDGE
    if direction == "up" and y[0] > level:
        return LEFT_EDGE
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        if direction == "down" and y0 >= level > y1:
            pass
        elif direction == "up" and y0 <= level < y1:
            pass
        else:
            continue
        if y1 == y0:
            return float(x[i])
        frac = (level - y0) / (y1 - y0)
        return float(x[i] + frac * (x[i + 1] - x[i]))
    return None


INTERPS = ("linear", "pchip")


def crossing(
    x: Sequence[float], y: Sequence[float], level: float, direction: str,
    interp: str = "linear",
) -> Optional[float]:
    """First crossing of `level` under the chosen interpolant.

    The bracketing interval is found exactly as in `first_crossing` (so both
    interpolants agree on which interval holds the crossing, on LEFT_EDGE and
    on None); ``interp="pchip"`` then solves the monotone cubic
    (PchipInterpolator) for the crossing inside that interval.
    """
    lin = first_crossing(x, y, level, direction)
    if interp == "linear" or lin is None or lin == LEFT_EDGE:
        return lin
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    i = int(np.searchsorted(x, lin, side="right") - 1)
    i = min(max(i, 0), len(x) - 2)
    if y[i] == level:
        return float(x[i])
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq
    f = PchipInterpolator(x, y)
    g = lambda t: float(f(t)) - level  # noqa: E731
    a, b = float(x[i]), float(x[i + 1])
    if g(a) * g(b) > 0:          # cannot happen for a bracketing interval
        return lin
    return float(brentq(g, a, b, xtol=1e-10))


def _summarize(
    draws: Sequence[Optional[float]],
    point: Optional[float],
    label: str,
    **extra: Any,
) -> Dict[str, Any]:
    arr = np.asarray(
        [np.nan if d is None else float(d) for d in draws], dtype=float
    )
    B = len(arr)
    usable = arr[~np.isnan(arr)]          # may contain +inf (ratio draws)
    finite = usable[np.isfinite(usable)]

    def _pct(q: float) -> float:
        # Order statistic of the usable draws, +inf included: a shrink-factor
        # draw whose n = 100,000 edge sits at the left edge of the grid is an
        # unbounded ratio and must push the upper limit up, not be dropped.
        if not len(usable):
            return np.nan
        srt = np.sort(usable)
        k = int(np.clip(np.ceil(q / 100.0 * len(srt)) - 1, 0, len(srt) - 1))
        return float(srt[k])

    row: Dict[str, Any] = dict(
        quantity=label,
        point=np.nan if point is None else float(point),
        ci_lo=_pct(2.5),
        ci_hi=_pct(97.5),
        boot_median=_pct(50.0),
        boot_sd=float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
        B=B,
        n_usable=int(len(usable)),
        no_crossing_frac=float(np.mean(np.isnan(arr))),
        left_edge_frac=float(np.mean(usable == LEFT_EDGE))
        if len(usable) else np.nan,
        unbounded_frac=float(np.mean(np.isinf(usable)))
        if len(usable) else np.nan,
    )
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Nonparametric bootstrap over the paired run's replications
# ---------------------------------------------------------------------------

def bootstrap_paired(raw: pd.DataFrame, B: int, return_draws: bool = False):
    raw = raw[raw["rho"] <= RHO_MAX + 1e-9]
    n_values = sorted(raw["n"].unique())

    # Per (n, rho) cell: the indicator vectors, in rho order.
    cells: Dict[int, Dict[str, Any]] = {}
    for n in n_values:
        sub = raw[raw["n"] == n]
        rhos = np.array(sorted(sub["rho"].unique()), dtype=float)
        cov_by_rho, rej_by_rho = [], {a: [] for a in ALPHAS}
        for rho in rhos:
            g = sub[np.isclose(sub["rho"], rho)]
            cov_by_rho.append(g[SI_COV_COL].to_numpy(dtype=float))
            for a in ALPHAS:
                rej_by_rho[a].append(g[REJ_COL[a]].to_numpy(dtype=float))
        cells[n] = dict(rho=rhos, cov=cov_by_rho, rej=rej_by_rho,
                        R=int(len(cov_by_rho[0])))

    # Point estimates on the observed curves.
    point: Dict[Any, Optional[float]] = {}
    for n in n_values:
        c = cells[n]
        point[("damage", n)] = first_crossing(
            c["rho"], [v.mean() for v in c["cov"]],
            COVERAGE_THRESHOLD, "down",
        )
        for a in ALPHAS:
            point[("detect", n, a)] = first_crossing(
                c["rho"], [v.mean() for v in c["rej"][a]],
                POWER_THRESHOLD, "up",
            )

    # Bootstrap.
    draws: Dict[Any, List[Optional[float]]] = {k: [] for k in point}
    for b in range(B):
        for n in n_values:
            c = cells[n]
            rng = np.random.default_rng(
                derive_seed(MASTER_SEED, SEED_TAG, int(n), 0, b)
            )
            idx = [
                rng.integers(0, len(v), size=len(v)) for v in c["cov"]
            ]
            cov_curve = [v[i].mean() for v, i in zip(c["cov"], idx)]
            draws[("damage", n)].append(
                first_crossing(c["rho"], cov_curve, COVERAGE_THRESHOLD, "down")
            )
            for a in ALPHAS:
                rej_curve = [
                    v[i].mean() for v, i in zip(c["rej"][a], idx)
                ]
                draws[("detect", n, a)].append(
                    first_crossing(c["rho"], rej_curve, POWER_THRESHOLD, "up")
                )

    rows: List[Dict[str, Any]] = []
    for scale, fn in SCALES:
        # The map rho -> delta is monotone, so each edge draw maps draw by
        # draw; widths and shrink factors are recomputed on the new scale.
        dr = {k: [fn(x) for x in v] for k, v in draws.items()}
        pt = {k: fn(v) for k, v in point.items()}
        tag = dict(source="paired", scale=scale)
        for n in n_values:
            R = cells[n]["R"]
            rows.append(_summarize(
                dr[("damage", n)], pt[("damage", n)],
                "damage edge (SI coverage 0.90)", n=n, alpha="", R=R, **tag,
            ))
            for a in ALPHAS:
                rows.append(_summarize(
                    dr[("detect", n, a)], pt[("detect", n, a)],
                    "detection edge (power 0.50)", n=n, alpha=a, R=R, **tag,
                ))
                width_draws = [
                    (d - dm) if (d is not None and dm is not None) else None
                    for d, dm in zip(dr[("detect", n, a)], dr[("damage", n)])
                ]
                pd_, pm_ = pt[("detect", n, a)], pt[("damage", n)]
                pw = (pd_ - pm_) if (pd_ is not None and pm_ is not None) \
                    else None
                rows.append(_summarize(
                    width_draws, pw, "band width (detection - damage)",
                    n=n, alpha=a, R=R, **tag,
                ))

        # Shrink factors: the n = 10,000 edge over the n = 100,000 edge.
        if 10_000 in cells and 100_000 in cells:
            rows.append(_summarize(
                _ratio(dr[("damage", 10_000)], dr[("damage", 100_000)]),
                _ratio_point(pt[("damage", 10_000)], pt[("damage", 100_000)]),
                "shrink factor, damage edge (n=1e4 / n=1e5)",
                n="", alpha="", R="", **tag,
            ))
            for a in ALPHAS:
                rows.append(_summarize(
                    _ratio(dr[("detect", 10_000, a)],
                           dr[("detect", 100_000, a)]),
                    _ratio_point(pt[("detect", 10_000, a)],
                                 pt[("detect", 100_000, a)]),
                    "shrink factor, detection edge (n=1e4 / n=1e5)",
                    n="", alpha=a, R="", **tag,
                ))
            for a in ALPHAS:
                w10 = [(d - m) if (d is not None and m is not None) else None
                       for d, m in zip(dr[("detect", 10_000, a)],
                                       dr[("damage", 10_000)])]
                w100 = [(d - m) if (d is not None and m is not None) else None
                        for d, m in zip(dr[("detect", 100_000, a)],
                                        dr[("damage", 100_000)])]

                def _pw(n_):
                    d_, m_ = pt[("detect", n_, a)], pt[("damage", n_)]
                    return (d_ - m_) if (d_ is not None and m_ is not None) \
                        else None

                rows.append(_summarize(
                    _ratio(w10, w100), _ratio_point(_pw(10_000), _pw(100_000)),
                    "shrink factor, band width (n=1e4 / n=1e5)",
                    n="", alpha=a, R="", **tag,
                ))

    out = pd.DataFrame(rows)
    out["grid"] = "coarse"
    out["interpolation"] = "linear"
    if return_draws:
        return out, draws, point, cells
    return out


# ---------------------------------------------------------------------------
# Damage edge on the coarse and the refined grid, linear and monotone spline
# ---------------------------------------------------------------------------

def _cov_cells(raw: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """Per n: the rho grid, the SI coverage indicators and R per cell."""
    raw = raw[raw["rho"] <= RHO_MAX + 1e-9]
    cells: Dict[int, Dict[str, Any]] = {}
    for n in sorted(raw["n"].unique()):
        sub = raw[raw["n"] == n]
        rhos = np.array(sorted(sub["rho"].unique()), dtype=float)
        cov = [sub[np.isclose(sub["rho"], r)][SI_COV_COL].to_numpy(float)
               for r in rhos]
        cells[int(n)] = dict(rho=rhos, cov=cov,
                             R=np.array([len(v) for v in cov]))
    return cells


def fine_curve_raw(coarse: pd.DataFrame, refined: pd.DataFrame) -> pd.DataFrame:
    """Refined cells plus the coarse cells at rho values the refined grid
    does not carry (a shared rho takes the refined cell, which contains the
    primary draws as its first R_primary replications)."""
    keep = []
    for n, sub in coarse.groupby("n"):
        fine_rhos = refined.loc[refined["n"] == n, "rho"].unique()
        mask = ~np.isclose(sub["rho"].to_numpy()[:, None],
                           fine_rhos[None, :]).any(axis=1) \
            if len(fine_rhos) else np.ones(len(sub), bool)
        keep.append(sub[mask])
    cols = [c for c in coarse.columns if c in refined.columns]
    return pd.concat([pd.concat(keep)[cols], refined[cols]],
                     ignore_index=True)


def bootstrap_damage(cells: Dict[int, Dict[str, Any]], B: int,
                     stream: int) -> Dict[Any, Any]:
    """Point and bootstrap draws of the damage edge under both interpolants.

    Every (n, rho) cell is resampled independently to its own R; the two
    interpolants are applied to the SAME resampled curve, so their draws are
    paired.  `stream` separates the coarse (0, the stream the paired block
    above uses, so coarse draws coincide with it) and fine (3) seeds.
    """
    point, draws, R_bracket = {}, {}, {}
    for n, c in cells.items():
        curve = [v.mean() for v in c["cov"]]
        for it in INTERPS:
            point[(n, it)] = crossing(c["rho"], curve, COVERAGE_THRESHOLD,
                                      "down", it)
            draws[(n, it)] = []
        lin = point[(n, "linear")]
        if lin is None:
            R_bracket[n] = ""
        else:
            i = int(np.searchsorted(c["rho"], lin, side="right") - 1)
            i = min(max(i, 0), len(c["rho"]) - 2)
            R_bracket[n] = (f"{c['rho'][i]:.3f}:{int(c['R'][i])}, "
                            f"{c['rho'][i + 1]:.3f}:{int(c['R'][i + 1])}")
        for b in range(B):
            rng = np.random.default_rng(
                derive_seed(MASTER_SEED, SEED_TAG, int(n), stream, b))
            curve_b = [v[rng.integers(0, len(v), size=len(v))].mean()
                       for v in c["cov"]]
            for it in INTERPS:
                draws[(n, it)].append(crossing(
                    c["rho"], curve_b, COVERAGE_THRESHOLD, "down", it))
    return dict(point=point, draws=draws, R_bracket=R_bracket)


def frontier_damage(B: int) -> Dict[Any, Any]:
    """The frontier grid's damage edge under both interpolants (parametric
    binomial bootstrap, the same draws for both)."""
    if not os.path.exists(FRONTIER_SRC):
        return dict(point={}, draws={}, R_bracket={})
    df = pd.read_csv(FRONTIER_SRC)
    df = df[df["rho"] <= RHO_MAX + 1e-9].sort_values(["n", "rho"])
    point, draws, R_bracket = {}, {}, {}
    for n in sorted(df["n"].unique()):
        sub = df[df["n"] == n]
        rho = sub["rho"].to_numpy(float)
        p_hat = sub["si_coverage"].to_numpy(float)
        R = int(sub["R"].iloc[0])
        for it in INTERPS:
            point[(int(n), it)] = crossing(rho, p_hat, COVERAGE_THRESHOLD,
                                           "down", it)
            draws[(int(n), it)] = []
        R_bracket[int(n)] = f"{R} per point"
        rng = np.random.default_rng(
            derive_seed(MASTER_SEED, SEED_TAG, int(n), 1, 0))
        for _ in range(B):
            p_b = rng.binomial(R, p_hat) / R
            for it in INTERPS:
                draws[(int(n), it)].append(crossing(
                    rho, p_b, COVERAGE_THRESHOLD, "down", it))
    return dict(point=point, draws=draws, R_bracket=R_bracket)


def refined_table(coarse: Dict[Any, Any], fine: Dict[Any, Any],
                  frontier: Dict[Any, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scale, fn in SCALES:
        for src, grid, res in (("paired", "coarse", coarse),
                               ("paired", "fine", fine),
                               ("frontier (parametric binomial)", "frontier",
                                frontier)):
            for (n, it), d in res["draws"].items():
                rows.append(_summarize(
                    [fn(x) for x in d], fn(res["point"][(n, it)]),
                    "damage edge (SI coverage 0.90)", n=n, alpha="",
                    R=res["R_bracket"].get(n, ""), source=src, scale=scale,
                    grid=grid, interpolation=it,
                ))
    return pd.DataFrame(rows)


def bracket_table(ref: pd.DataFrame) -> pd.DataFrame:
    """Per (scale, n): the four paired point estimates and their range."""
    rows = []
    for (scale, n), g in ref[ref.source == "paired"].groupby(["scale", "n"]):
        pts = {(r.grid, r.interpolation): r.point for _, r in g.iterrows()}
        vals = np.array(list(pts.values()), dtype=float)
        rows.append(dict(
            scale=scale, n=n,
            coarse_linear=pts.get(("coarse", "linear")),
            coarse_pchip=pts.get(("coarse", "pchip")),
            fine_linear=pts.get(("fine", "linear")),
            fine_pchip=pts.get(("fine", "pchip")),
            bracket_lo=float(np.nanmin(vals)), bracket_hi=float(np.nanmax(vals)),
            fine_pchip_minus_linear=pts[("fine", "pchip")]
            - pts[("fine", "linear")],
            fine_minus_coarse_linear=pts[("fine", "linear")]
            - pts[("coarse", "linear")],
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The independent frontier grid (per-cell proportions only)
# ---------------------------------------------------------------------------

def fine_block(fine: Dict[Any, Any], det_draws: Dict[Any, List],
               det_point: Dict[Any, Optional[float]],
               R_of: Dict[int, int]) -> pd.DataFrame:
    """edges_ci rows with the damage edge from the refined grid.

    Detection edges stay the coarse-grid linear ones (unchanged); band widths
    and shrink factors pair bootstrap draw b of the fine damage edge with
    draw b of the coarse detection edge (independent cells).
    """
    rows: List[Dict[str, Any]] = []
    n_values = sorted(fine["R_bracket"])
    for scale, fn in SCALES:
        for it in INTERPS:
            tag = dict(source="paired", scale=scale, grid="fine",
                       interpolation=it)
            dm = {n: [fn(x) for x in fine["draws"][(n, it)]] for n in n_values}
            pm = {n: fn(fine["point"][(n, it)]) for n in n_values}
            for n in n_values:
                rows.append(_summarize(
                    dm[n], pm[n], "damage edge (SI coverage 0.90)", n=n,
                    alpha="", R=fine["R_bracket"][n], **tag))
                for a in ALPHAS:
                    dd = [fn(x) for x in det_draws[("detect", n, a)]]
                    pdt = fn(det_point[("detect", n, a)])
                    w = [(d - m) if (d is not None and m is not None) else None
                         for d, m in zip(dd, dm[n])]
                    pw = (pdt - pm[n]) if (pdt is not None and pm[n] is not None) \
                        else None
                    rows.append(_summarize(
                        w, pw, "band width (detection - damage)", n=n,
                        alpha=a, R=R_of.get(n, ""), **tag))
            if 10_000 in dm and 100_000 in dm:
                rows.append(_summarize(
                    _ratio(dm[10_000], dm[100_000]),
                    _ratio_point(pm[10_000], pm[100_000]),
                    "shrink factor, damage edge (n=1e4 / n=1e5)",
                    n="", alpha="", R="", **tag))
                for a in ALPHAS:
                    ws = {}
                    for n in (10_000, 100_000):
                        dd = [fn(x) for x in det_draws[("detect", n, a)]]
                        ws[n] = ([(d - m) if (d is not None and m is not None)
                                  else None for d, m in zip(dd, dm[n])],
                                 None if (det_point[("detect", n, a)] is None
                                          or pm[n] is None)
                                 else fn(det_point[("detect", n, a)]) - pm[n])
                    rows.append(_summarize(
                        _ratio(ws[10_000][0], ws[100_000][0]),
                        _ratio_point(ws[10_000][1], ws[100_000][1]),
                        "shrink factor, band width (n=1e4 / n=1e5)",
                        n="", alpha=a, R="", **tag))
    return pd.DataFrame(rows)


def bootstrap_frontier(B: int) -> pd.DataFrame:
    if not os.path.exists(FRONTIER_SRC):
        return pd.DataFrame()
    df = pd.read_csv(FRONTIER_SRC)
    df = df[df["rho"] <= RHO_MAX + 1e-9].sort_values(["n", "rho"])

    rows: List[Dict[str, Any]] = []
    edge_draws: Dict[int, List[Optional[float]]] = {}
    edge_point: Dict[int, Optional[float]] = {}
    R_of: Dict[int, int] = {}
    for n in sorted(df["n"].unique()):
        sub = df[df["n"] == n]
        rho = sub["rho"].to_numpy(dtype=float)
        p_hat = sub["si_coverage"].to_numpy(dtype=float)
        R_of[int(n)] = int(sub["R"].iloc[0])
        edge_point[int(n)] = first_crossing(
            rho, p_hat, COVERAGE_THRESHOLD, "down")

        rng = np.random.default_rng(
            derive_seed(MASTER_SEED, SEED_TAG, int(n), 1, 0)
        )
        draws = []
        for _ in range(B):
            p_b = rng.binomial(R_of[int(n)], p_hat) / R_of[int(n)]
            draws.append(
                first_crossing(rho, p_b, COVERAGE_THRESHOLD, "down")
            )
        edge_draws[int(n)] = draws

    # Shrink factor on that grid: its own independent stream of paired
    # (n = 10,000, n = 100,000) edge draws, as before.
    ratio_pairs: List[Any] = []
    if 10_000 in edge_draws and 100_000 in edge_draws:
        sub10 = df[df["n"] == 10_000]
        sub100 = df[df["n"] == 100_000]
        R10, R100 = R_of[10_000], R_of[100_000]
        rng = np.random.default_rng(
            derive_seed(MASTER_SEED, SEED_TAG, 0, 2, 0)
        )
        for _ in range(B):
            e10 = first_crossing(
                sub10["rho"].to_numpy(dtype=float),
                rng.binomial(R10, sub10["si_coverage"].to_numpy()) / R10,
                COVERAGE_THRESHOLD, "down",
            )
            e100 = first_crossing(
                sub100["rho"].to_numpy(dtype=float),
                rng.binomial(R100, sub100["si_coverage"].to_numpy()) / R100,
                COVERAGE_THRESHOLD, "down",
            )
            ratio_pairs.append((e10, e100))

    tag0 = dict(source="frontier (parametric binomial)")
    for scale, fn in SCALES:
        for n in sorted(edge_draws):
            rows.append(_summarize(
                [fn(x) for x in edge_draws[n]], fn(edge_point[n]),
                "damage edge (SI coverage 0.90)", n=n, alpha="",
                R=R_of[n], scale=scale, **tag0,
            ))
        if ratio_pairs:
            rows.append(_summarize(
                _ratio([fn(a) for a, _ in ratio_pairs],
                       [fn(b) for _, b in ratio_pairs]),
                _ratio_point(fn(edge_point[10_000]), fn(edge_point[100_000])),
                "shrink factor, damage edge (n=1e4 / n=1e5)",
                n="", alpha="", R="", scale=scale, **tag0,
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

COLS = ["source", "grid", "interpolation", "scale", "quantity", "n", "alpha",
        "R", "point", "ci_lo", "ci_hi",
        "boot_median", "boot_sd", "B", "n_usable", "no_crossing_frac",
        "left_edge_frac", "unbounded_frac"]


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
        return "---"
    if isinstance(v, float) and np.isinf(v):
        return "inf"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return str(v)


def write_markdown(df: pd.DataFrame, path: str, B: int) -> None:
    lines = [
        "# Monte Carlo intervals for the detection-damage band edges\n\n",
        "Nonparametric bootstrap over replications of the paired run "
        "(`results/tables/detection_damage_paired_raw.csv`), B = "
        f"{B:,}; each (n, rho) cell resampled independently to its own R, "
        "the SI coverage curve and the two-sided power curve recomputed, "
        "and both edges re-interpolated on the rho grid. Intervals are 95% "
        "percentile intervals. The damage edge does not depend on alpha. "
        "The `frontier` block is an INDEPENDENT run on a coarser rho grid "
        "(`results/tables/frontier_coverage.csv`, R = 500) that stores only "
        "per-cell coverage proportions, so its interval is a parametric "
        "binomial bootstrap and is not term-by-term comparable with the "
        "nonparametric ones.\n\n",
        "Each quantity is reported on two scales. `rho` is the DGP 2 dial. "
        "`delta` is the direct effect, delta = 0.15 rho / (1 - rho) (the "
        "mediated effect beta_YS gamma_S = 0.15 is held fixed, so "
        "tau = 0.15 / (1 - rho)); every bootstrap draw of an edge is mapped "
        "to delta, and widths and shrink factors are recomputed from the "
        "mapped draws, so a delta-scale width or shrink factor is not a "
        "transform of its rho-scale counterpart. The `shrink factor, band "
        "width` rows divide the n = 10,000 band width by the n = 100,000 "
        "one.\n\n",
        "Notes: `point` is the edge read off the observed curve, the value "
        "the paper quotes. `left_edge_frac` is the share of resamples whose "
        "curve is already past the threshold at rho = 0, reported at rho = "
        "0. `no_crossing_frac` is the share with no crossing anywhere on the "
        "grid; those resamples are excluded from the percentiles. "
        "`unbounded_frac` is the share of shrink-factor resamples whose "
        "n = 100,000 edge (or width) is 0, an unbounded ratio kept as +inf "
        "in the order statistics (so an upper limit of `inf` means more "
        "than 2.5% of resamples are unbounded). Percentiles are order "
        "statistics of the B draws. The "
        f"sqrt(n) reference for a shrink factor is {SQRT10:.3f}.\n\n",
        "Grid and interpolation: `coarse` rows use the primary paired grid "
        "(and the frontier grid for the frontier block), `fine` rows take "
        "the damage edge from the refined local grid "
        "(`detection_damage_refined_raw.csv`, steps of 0.005, R = 2,000 at "
        "rho = 0 and 1,000 elsewhere; for a fine row R lists the two "
        "bracketing grid points as rho:R); detection edges are always the "
        "coarse-grid linear ones. `linear` is the linear interpolant, "
        "`pchip` the monotone piecewise-cubic one. Every interval is "
        "CONDITIONAL on its grid and interpolant: it resamples Monte Carlo "
        "replications at fixed grid points and cannot see the error of "
        "interpolating a curved coverage function. The spread across grids "
        "and interpolants is the grid-resolution bracket in "
        "`detection_damage_refined.md`.\n\n",
        "| Source | Grid | Interp. | Scale | Quantity | n | alpha | R | Point "
        "| 95% CI (conditional) "
        "| Boot median | Boot SD | Left-edge frac | No-crossing frac "
        "| Unbounded frac |\n"
        "|---|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in df.iterrows():
        nd = 4 if (r.scale == "delta" and "shrink" not in r.quantity) else 3
        lines.append(
            f"| {r.source} | {r.grid} | {r.interpolation} "
            f"| {r.scale} | {r.quantity} | {_fmt(r['n'])} "
            f"| {_fmt(r['alpha'], 2)} | {_fmt(r['R'])} "
            f"| {_fmt(r['point'], nd)} "
            f"| [{_fmt(r['ci_lo'], nd)}, {_fmt(r['ci_hi'], nd)}] "
            f"| {_fmt(r['boot_median'], nd)} | {_fmt(r['boot_sd'], nd)} "
            f"| {_fmt(r['left_edge_frac'], 3)} "
            f"| {_fmt(r['no_crossing_frac'], 3)} "
            f"| {_fmt(r['unbounded_frac'], 3)} |\n"
        )
    with open(path, "w") as f:
        f.writelines(lines)


def _draws_damage(det_draws: Dict[Any, List], n: int) -> List:
    return det_draws[("damage", n)]


def write_refined_markdown(ref: pd.DataFrame, br: pd.DataFrame,
                           det_point: Dict[Any, Optional[float]],
                           pcells: Dict[int, Dict[str, Any]], path: str,
                           B: int) -> None:
    lines = [
        "# Damage edge: refined grid and grid-resolution bracket\n\n",
        "The damage edge is the smallest rho (DGP 2, pi_L = 0.20) at which "
        "SI coverage under the joint sandwich falls through 0.90. `coarse` "
        "is the primary paired grid (`detection_damage_paired_raw.csv`: "
        "rho step 0.02, R = 500 at n = 10,000 and 300 at n = 100,000). "
        "`fine` adds the refined local grid "
        "(`detection_damage_refined_raw.csv`: rho in {0, 0.005, ..., 0.06} "
        "at n = 100,000 and {0.06, 0.065, ..., 0.14} at n = 10,000; R = 2,000 "
        "at rho = 0 and 1,000 elsewhere; same seed stream, so a shared rho "
        "contains the primary draws). `frontier` is the independent "
        "`frontier_coverage.csv` grid (rho step 0.05, R = 500; parametric "
        "binomial bootstrap). `linear` and `pchip` (monotone piecewise-cubic "
        "Hermite) interpolate the coverage curve between grid points; both "
        "use the same bracketing interval. R lists the bracketing grid "
        f"points as rho:R. Intervals are 95% percentile intervals over B = "
        f"{B:,} resamples of the replications, CONDITIONAL on the grid and "
        "the interpolant; they do not include interpolation error. The "
        "bracket is the range of the four paired point estimates (coarse and "
        "fine grid, linear and pchip).\n\n",
        "## Grid-resolution bracket\n\n",
        "| Scale | n | Coarse linear | Coarse pchip | Fine linear | Fine pchip "
        "| Bracket | Fine: pchip - linear | Fine - coarse (linear) |\n"
        "|---|---:|---:|---:|---:|---:|---|---:|---:|\n",
    ]
    for _, r in br.iterrows():
        nd = 4 if r.scale == "delta" else 3
        lines.append(
            f"| {r.scale} | {int(r.n):,} | {r.coarse_linear:.{nd}f} "
            f"| {r.coarse_pchip:.{nd}f} | {r.fine_linear:.{nd}f} "
            f"| {r.fine_pchip:.{nd}f} "
            f"| [{r.bracket_lo:.{nd}f}, {r.bracket_hi:.{nd}f}] "
            f"| {r.fine_pchip_minus_linear:+.{nd}f} "
            f"| {r.fine_minus_coarse_linear:+.{nd}f} |\n")
    lines += [
        "\n## Damage edge by grid and interpolant\n\n",
        "| Source | Grid | Interp. | Scale | n | R (bracketing points) | Point "
        "| 95% CI (conditional) | Boot SD | Left-edge frac "
        "| No-crossing frac |\n"
        "|---|---|---|---|---:|---|---:|---|---:|---:|---:|\n",
    ]
    for _, r in ref.iterrows():
        nd = 4 if r.scale == "delta" else 3
        lines.append(
            f"| {r.source} | {r.grid} | {r.interpolation} | {r.scale} "
            f"| {int(r.n):,} | {r.R} | {_fmt(r.point, nd)} "
            f"| [{_fmt(r.ci_lo, nd)}, {_fmt(r.ci_hi, nd)}] "
            f"| {_fmt(r.boot_sd, nd)} | {_fmt(r.left_edge_frac, 3)} "
            f"| {_fmt(r.no_crossing_frac, 3)} |\n")
    lines += [
        "\n## Detection edges (unchanged: coarse paired grid, linear)\n\n",
        "| n | alpha | Detection edge (rho) | Detection edge (delta) |\n"
        "|---:|---:|---:|---:|\n",
    ]
    for n in sorted(pcells):
        for a in ALPHAS:
            d = det_point[("detect", n, a)]
            lines.append(f"| {n:,} | {a:.2f} | {_fmt(d, 3)} "
                         f"| {_fmt(rho_to_delta(d), 4)} |\n")
    lines.append("\nIntervals for the detection edges, band widths and "
                 "shrink factors are in `detection_damage_edges_ci.md`.\n")
    with open(path, "w") as f:
        f.writelines(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000,
                    help="bootstrap resamples (default 1000)")
    args = ap.parse_args()

    if not os.path.exists(RAW_SRC):
        raise SystemExit(
            f"{RAW_SRC} not found -- run "
            "scripts/run_detection_damage_paired.py first."
        )
    raw = pd.read_csv(RAW_SRC)

    paired, det_draws, det_point, pcells = bootstrap_paired(
        raw, args.B, return_draws=True)
    frontier = bootstrap_frontier(args.B)
    frontier["grid"] = "coarse"
    frontier["interpolation"] = "linear"
    parts = [paired, frontier]

    have_fine = os.path.exists(REFINED_SRC)
    if have_fine:
        refined = pd.read_csv(REFINED_SRC)
        coarse_res = bootstrap_damage(_cov_cells(raw), args.B, stream=0)
        fine_res = bootstrap_damage(
            _cov_cells(fine_curve_raw(raw, refined)), args.B, stream=3)
        front_res = frontier_damage(args.B)
        # The coarse linear draws must reproduce the paired block exactly.
        for n in coarse_res["R_bracket"]:
            assert coarse_res["draws"][(n, "linear")] == \
                [x for x in _draws_damage(det_draws, n)], \
                "coarse damage draws out of step with the paired block"
        parts.insert(1, fine_block(fine_res, det_draws, det_point,
                                   {n: c["R"] for n, c in pcells.items()}))
    else:
        print(f"{REFINED_SRC} not found: fine-grid block skipped "
              "(run scripts/run_detection_damage_paired.py --refine)")

    out = pd.concat(parts, ignore_index=True)[COLS]
    os.makedirs(TABLES_DIR, exist_ok=True)
    csv_path = os.path.join(TABLES_DIR, "detection_damage_edges_ci.csv")
    md_path = os.path.join(TABLES_DIR, "detection_damage_edges_ci.md")
    out.to_csv(csv_path, index=False)
    write_markdown(out, md_path, args.B)
    print(out.to_string(index=False))
    print(f"\nWrote {csv_path} and {md_path}")

    if have_fine:
        ref = refined_table(coarse_res, fine_res, front_res)
        br = bracket_table(ref)
        ref_cols = ["source", "grid", "interpolation", "scale", "quantity",
                    "n", "R", "point", "ci_lo", "ci_hi", "boot_median",
                    "boot_sd", "B", "n_usable", "no_crossing_frac",
                    "left_edge_frac"]
        ref = ref[ref_cols]
        rcsv = os.path.join(TABLES_DIR, "detection_damage_refined.csv")
        rmd = os.path.join(TABLES_DIR, "detection_damage_refined.md")
        both = pd.concat([ref.assign(table="edges"),
                          br.assign(table="bracket")], ignore_index=True)
        both.to_csv(rcsv, index=False)
        write_refined_markdown(ref, br, det_point, pcells, rmd, args.B)
        print(br.to_string(index=False))
        print(f"Wrote {rcsv} and {rmd}")


if __name__ == "__main__":
    main()
