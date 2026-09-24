#!/usr/bin/env python3
"""
Local-asymptotic (limit-experiment) risk of the Cauchy-kernel hybrid.

Under local alternatives delta_n = h / sqrt(n),
  sqrt(n) (tau_PPI - tau)  ->  G_P ~ N(0, sig_P^2)
  sqrt(n) (tau_SI  - tau)  ->  G_S ~ N(-h, sig_S^2),  Cov(G_S, G_P) = sig_SP
  sqrt(n) D_hat            ->  V = G_P - G_S ~ N(h, sig_D^2)
  T_n                      ->  V / sig_D
  sqrt(n) (tau_hyb - tau)  ->  H = G_P - w(V/sig_D) V,   w(t) = c/(c+t^2)

Writing gamma = Cov(G_P, V) / sig_D^2 = (sig_P^2 - sig_SP) / sig_D^2 and
using E[G_P | V] = gamma (V - h), the asymptotic risk is the 1-D integral

  r(h; c) = sig_P^2 - gamma^2 sig_D^2
            + E_V[ (gamma (V - h) - w(V/sig_D) V)^2 ],   V ~ N(h, sig_D^2).

This script:
  1. Calibrates (sig_S, sig_P, sig_SP) by simulating (tau_SI, tau_PPI)
     under DGP 2 with delta = 0 at n = 10,000, pi_L = 0.20 (the paper's
     featured cell), scaling by sqrt(n).
  2. Computes r(h; c) by dense trapezoidal quadrature over V ~ N(h, sig_D^2)
     (truncated at +/- 10 sd) on a grid of c and h;
     reports the limit-minimax c* over the excess risk
     r(h;c) - min(sig_P^2, sig_S^2 + h^2).
  3. Monte Carlo check: finite-sample n * MSE of the implemented hybrid
     at n = 10,000 under delta = h/sqrt(n), compared with r(h; c=1.5).

Outputs: results/tables/limit_experiment.json, limit_risk_curves.csv,
         and figure results/figures/figure_limit_risk.png (+ paper copies).
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dgps.dgps import generate_dgp2
from src.utils.config import DEFAULT_PROTOCOL, PROTOCOLS
from src.utils.registry import write_table_rows
from src.utils.estimator_api import (
    estimate,
    hybrid_estimator,
    hybrid_sigma,
    train_prediction_model,
)

N_CAL = 10_000
PI_L = 0.20

#: Prediction protocol; overridden by --protocol in main().
PROTOCOL = DEFAULT_PROTOCOL

def _set_protocol(protocol: str) -> None:
    """Pool initializer: propagate --protocol into the worker processes.

    Workers re-import this module under the spawn start method, so a global
    set in main() would not reach them.
    """
    global PROTOCOL
    PROTOCOL = protocol


# ---------- Step 1: calibrate the limit covariance ----------

def _one_cal(rep: int) -> Tuple[float, float]:
    d = generate_dgp2(n=N_CAL, pi_L=PI_L, rho=0.0, seed=910_000 + rep)
    Yh, design = train_prediction_model(
        d["S"], d["X"], d["Y"], d["labeled_mask"], n_folds=5,
        rng=np.random.default_rng(rep + 7777),
        protocol=PROTOCOL, return_design=True,
    )
    pp = estimate(3, d["T"], d["S"], d["Y"], Yh, d["labeled_mask"],
                  protocol=PROTOCOL)
    si = estimate(2, d["T"], d["S"], d["Y"], Yh, d["labeled_mask"],
                  protocol=PROTOCOL, design=design,
                  lambda_hat=pp.get("lambda_hat", 0.0))
    return si["tau_hat"] - d["true_tau"], pp["tau_hat"] - d["true_tau"]


def calibrate(R: int = 600) -> dict:
    with multiprocessing.Pool(8, initializer=_set_protocol,
                              initargs=(PROTOCOL,)) as pool:
        res = pool.map(_one_cal, range(R))
    e_si = np.array([r[0] for r in res]) * np.sqrt(N_CAL)
    e_pp = np.array([r[1] for r in res]) * np.sqrt(N_CAL)
    sig_S = float(e_si.std(ddof=1))
    sig_P = float(e_pp.std(ddof=1))
    sig_SP = float(np.cov(e_si, e_pp, ddof=1)[0, 1])
    return {"sig_S": sig_S, "sig_P": sig_P, "sig_SP": sig_SP, "R_cal": R}


# ---------- Step 2: limit risk by quadrature ----------

def limit_risk(h: float, c: float, sig_S: float, sig_P: float,
               sig_SP: float, n_gh: int = 400) -> float:
    sig_D2 = sig_P**2 + sig_S**2 - 2 * sig_SP
    sig_D = np.sqrt(sig_D2)
    gamma = (sig_P**2 - sig_SP) / sig_D2
    # dense trapezoid over V ~ N(h, sig_D^2), +/- 10 sd
    V = np.linspace(h - 10 * sig_D, h + 10 * sig_D, 4001)
    dens = np.exp(-0.5 * ((V - h) / sig_D) ** 2) / (sig_D * np.sqrt(2 * np.pi))
    w = c / (c + (V / sig_D) ** 2)
    integrand = (gamma * (V - h) - w * V) ** 2
    Ev = float(np.trapz(integrand * dens, V))
    return sig_P**2 - gamma**2 * sig_D2 + Ev


# ---------- Step 3: finite-sample MC check ----------

def _one_mc(args) -> float:
    rep, h = args
    n = N_CAL
    delta = h / np.sqrt(n)
    # DGP 2 parameterization: rho = delta / (0.15 + delta)
    rho = delta / (0.15 + delta)
    d = generate_dgp2(n=n, pi_L=PI_L, rho=rho, seed=920_000 + rep)
    Yh, design = train_prediction_model(
        d["S"], d["X"], d["Y"], d["labeled_mask"], n_folds=5,
        rng=np.random.default_rng(rep + 7777),
        protocol=PROTOCOL, return_design=True,
    )
    pp = estimate(3, d["T"], d["S"], d["Y"], Yh, d["labeled_mask"],
                  protocol=PROTOCOL)
    lam = pp.get("lambda_hat", 0.0)
    si = estimate(2, d["T"], d["S"], d["Y"], Yh, d["labeled_mask"],
                  protocol=PROTOCOL, design=design, lambda_hat=lam)
    # Sigma_hat: the joint sandwich covariance for the learned index.
    sigma = hybrid_sigma(d["T"], d["Y"], d["labeled_mask"], design, lam)
    var_si_j, cov_sp, var_ppi_j = (
        float(sigma[0, 0]), float(sigma[0, 1]), float(sigma[1, 1])
    )
    hyb = hybrid_estimator(si["tau_hat"], pp["tau_hat"], var_si_j,
                           var_ppi_j, cov_sp, c=1.5,
                           rng_seed=rep)
    return (hyb["tau_hybrid"] - d["true_tau"]) ** 2


def mc_check(h_values, R: int = 500) -> dict:
    out = {}
    for h in h_values:
        with multiprocessing.Pool(8, initializer=_set_protocol,
                                  initargs=(PROTOCOL,)) as pool:
            sq = pool.map(_one_mc, [(rep, h) for rep in range(R)])
        out[h] = float(N_CAL * np.mean(sq))
    return out


def make_figure(df, sig_S, sig_P, sig_SP, sig_D, c_star, mc, h_check,
                figdir) -> None:
    """figure_limit_risk.{pdf,png}: hybrid risk curves against theta.

    Legend labels carry curve names only; the calibration, the sample size
    of the finite-sample check and the c* derivation are in the caption
    note.  Also callable from the saved artifacts (``--figure-only``).
    """
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType (Type 42), no Type 3 fonts
    import matplotlib.pyplot as plt
    h_grid = np.sort(df[df.c == df.c.iloc[0]].h.to_numpy())
    oracle = np.minimum(sig_P**2, sig_S**2 + h_grid**2)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    theta_grid = h_grid / sig_D
    curve_cs = [0.5, 1.5, 1.6, 3.0]
    curve_colors = ["#7bafd4", "#c0392b", "#1a7a3c", "#8e6bb0"]
    for c, color in zip(curve_cs, curve_colors):
        r = df[np.isclose(df.c, c)].sort_values("h").risk.to_numpy()
        lw = 2.4 if c == 1.5 else 1.6
        is_star = abs(c - c_star) < 1e-9
        lbl = (f"hybrid, $c={c}$ ($c^*$)" if is_star
               else f"hybrid, $c={c}$")
        ax.plot(theta_grid, r, lw=lw, color=color,
                linestyle="-." if is_star else "-", label=lbl)
    ax.plot(theta_grid, np.full_like(theta_grid, sig_P**2), "--",
            color="#2c7fb8", lw=1.4, label="PPI++")
    ax.plot(theta_grid, sig_S**2 + h_grid**2, ":", color="#c05020",
            lw=1.6, label="SI")
    ax.plot(theta_grid, oracle, color="0.35", lw=1.0, alpha=0.8,
            label="oracle envelope")
    # MC check points
    for h in h_check:
        ax.plot(h / sig_D, mc[h], "k*", ms=11,
                label="finite-sample check" if h == 0 else None)
    ax.set_xlabel(r"local violation $\theta = h/\sigma_D$")
    ax.set_ylabel(r"asymptotic risk of $\sqrt{n}(\hat\tau - \tau)$")
    ax.set_ylim(0, sig_P**2 * 2.6)
    ax.legend(fontsize=8.5, loc="upper right")
    plt.tight_layout()
    # Vector PDF alongside the raster copies.
    plt.savefig(os.path.join(figdir, "figure_limit_risk.pdf"),
                bbox_inches="tight")
    png_path = os.path.join(figdir, "figure_limit_risk.png")
    plt.savefig(png_path, dpi=300)
    plt.close(fig)
    # Copy into the paper figure directories when they exist (they are absent
    # in the public replication package).
    import shutil
    for d in (os.path.join(PROJECT_ROOT, "paper", "arxiv", "figures"),
              os.path.join(PROJECT_ROOT, "paper", "ijds", "figures")):
        if os.path.isdir(d):
            shutil.copyfile(png_path, os.path.join(d, "figure_limit_risk.png"))
            shutil.copyfile(os.path.join(figdir, "figure_limit_risk.pdf"),
                            os.path.join(d, "figure_limit_risk.pdf"))


def figure_from_artifacts() -> None:
    """Redraw the figure from limit_risk_curves.csv and
    limit_experiment.json without recalibrating or rerunning the MC check."""
    outdir = os.path.join(PROJECT_ROOT, "results", "tables")
    figdir = os.path.join(PROJECT_ROOT, "results", "figures")
    df = pd.read_csv(os.path.join(outdir, "limit_risk_curves.csv"))
    with open(os.path.join(outdir, "limit_experiment.json")) as f:
        res = json.load(f)
    cal = res["cal"]
    mc = {float(k): float(v) for k, v in res["mc_check"].items()}
    h_check = sorted(mc)
    make_figure(df, cal["sig_S"], cal["sig_P"], cal["sig_SP"],
                float(res["sig_D"]), float(res["c_star"]), mc, h_check,
                figdir)
    print("redrew figure_limit_risk from saved artifacts", flush=True)


def main():
    import argparse

    global PROTOCOL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--R", type=int, default=None,
                    help="override the calibration and Monte Carlo R")
    ap.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw figure_limit_risk from the saved "
                         "limit_risk_curves.csv and limit_experiment.json")
    args = ap.parse_args()
    if args.figure_only:
        figure_from_artifacts()
        return
    PROTOCOL = args.protocol

    t0 = time.time()
    outdir = os.path.join(PROJECT_ROOT, "results", "tables")
    figdir = os.path.join(PROJECT_ROOT, "results", "figures")

    print("calibrating limit covariance...", flush=True)
    cal = calibrate(**({"R": args.R} if args.R else {}))
    sig_S, sig_P, sig_SP = cal["sig_S"], cal["sig_P"], cal["sig_SP"]
    rho_SP = sig_SP / (sig_S * sig_P)
    sig_D = np.sqrt(sig_P**2 + sig_S**2 - 2 * sig_SP)
    print(f"sig_S={sig_S:.3f} sig_P={sig_P:.3f} rho={rho_SP:.3f} "
          f"sig_D={sig_D:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # Risk curves
    # c = 1.6 is the limit-minimax c* (see c_star below); it is on the grid so
    # that the risk curve and the minimax table can both report it directly.
    c_grid = [0.5, 1.0, 1.5, 1.6, 1.7, 2.0, 3.0]
    h_grid = np.linspace(0, 8 * sig_D, 81)
    rows = []
    for c in c_grid:
        for h in h_grid:
            rows.append(dict(c=c, h=float(h), theta=float(h / sig_D),
                             risk=limit_risk(h, c, sig_S, sig_P, sig_SP)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "limit_risk_curves.csv"), index=False)
    write_table_rows(
        os.path.join(outdir, "limit_risk_curves_rows.json"), df,
        dgp=2,
        analysis="limit_experiment",
        protocol=PROTOCOL,
        config_cols=("c", "h"),
        generated_by="scripts/run_limit_experiment.py",
    )

    # Oracle envelope and minimax excess
    oracle = np.minimum(sig_P**2, sig_S**2 + h_grid**2)
    minimax = {}
    for c in c_grid:
        r = df[df.c == c].sort_values("h").risk.to_numpy()
        minimax[c] = float(np.max(r - oracle))
    c_dense = np.arange(0.3, 4.01, 0.05)
    mm_dense = []
    for c in c_dense:
        r = np.array([limit_risk(h, c, sig_S, sig_P, sig_SP)
                      for h in h_grid])
        mm_dense.append(float(np.max(r - oracle)))
    c_star = float(c_dense[int(np.argmin(mm_dense))])
    print(f"limit-minimax c* = {c_star:.2f}; excess at c=1.5: "
          f"{minimax[1.5]:.3f} vs at c*: {min(mm_dense):.3f}", flush=True)

    # MC check at c = 1.5
    h_check = [0.0, float(2 * sig_D), float(4 * sig_D)]
    print("running finite-sample MC check...", flush=True)
    mc = mc_check(h_check)
    theory = {h: limit_risk(h, 1.5, sig_S, sig_P, sig_SP) for h in h_check}
    for h in h_check:
        print(f"h={h:.2f}: n*MSE (MC) = {mc[h]:.3f}, limit risk = "
              f"{theory[h]:.3f}", flush=True)

    results = dict(cal=cal, rho_SP=rho_SP, sig_D=float(sig_D),
                   minimax_by_c=minimax, c_star=c_star,
                   minimax_at_cstar=float(min(mm_dense)),
                   mc_check={str(k): v for k, v in mc.items()},
                   theory_at_check={str(k): v for k, v in theory.items()})
    with open(os.path.join(outdir, "limit_experiment.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Small companion table: worst-case excess risk over the oracle envelope
    # by c, with the ratio to the limit-minimax c* value.
    mm_lines = [
        "# Limit-experiment minimax: worst-case excess risk by c",
        "",
        "Excess risk = max over the local-violation grid h of "
        "`r(h; c) - min(sig_P^2, sig_S^2 + h^2)`, i.e. the worst-case gap "
        "between the Cauchy-kernel hybrid's asymptotic risk and the oracle "
        "envelope that switches between PPI++ and SI.",
        "",
        f"Calibrated at DGP 2, n = {N_CAL:,}, pi_L = {PI_L:.2f} "
        f"(sig_S = {sig_S:.3f}, sig_P = {sig_P:.3f}, sig_D = {sig_D:.3f}); "
        f"h grid = {len(h_grid)} points on [0, {8 * sig_D:.2f}].",
        "",
        f"Limit-minimax c* = {c_star:.2f}, worst-case excess "
        f"{min(mm_dense):.3f}.",
        "",
        "| c | Worst-case excess risk | Ratio to c* value |",
        "|---|---|---|",
    ]
    mm_star = float(min(mm_dense))
    for c in c_grid:
        mark = " (c\\*)" if abs(c - c_star) < 1e-9 else ""
        mm_lines.append(
            f"| {c:.1f}{mark} | {minimax[c]:.3f} | {minimax[c] / mm_star:.3f} |"
        )
    mm_lines.append("")
    with open(os.path.join(outdir, "limit_minimax_by_c.md"), "w") as f:
        f.write("\n".join(mm_lines))
    print("wrote results/tables/limit_minimax_by_c.md", flush=True)

    make_figure(df, sig_S, sig_P, sig_SP, sig_D, c_star, mc, h_check,
                figdir)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
