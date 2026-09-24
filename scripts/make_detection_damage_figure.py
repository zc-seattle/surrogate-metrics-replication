"""
Generate figure_detection_damage.png / .pdf from the PAIRED simulation
(results/tables/detection_damage_paired.csv).

This replaces the figure produced by scripts/make_detection_damage_figure.py,
which is kept for reference: that version read SI/PPI++ coverage from
frontier_coverage.csv and diagnostic power from power_curve_extended.csv --
two separate simulations, with the diagnostic built on the uncorrected PPI++
variance.  Everything here comes from one set of Monte Carlo draws with the
corrected joint covariance (see scripts/run_detection_damage_paired.py).

What the figure shows, per panel (one panel per n):
  * SI coverage with a +/- 2 MC-SE band
  * PPI++ corrected coverage
  * two-sided diagnostic power at alpha = 0.05 (solid) and alpha = 0.10
    (dot-dashed), SE(D_hat) from the joint sandwich
  * the detection-damage band for the DEFAULT rule -- from the rho where SI
    coverage first falls below 0.90 to the rho where power at alpha = 0.05
    first exceeds 0.50 -- shaded; the legend names it only, and the
    thresholds, levels, replication counts and interpolation are stated in
    the caption note
  * when results/tables/detection_damage_refined_grid.csv exists (the
    `run_detection_damage_paired.py --refine` run), the refined-grid SI
    coverage points around the damage crossing (open markers with +/- 2 MC SE
    bars), and the band's left edge read off the FINE curve (refined points
    plus the primary points outside the refined range, linear
    interpolation); the detection edge stays the primary-grid one

Also writes results/tables/detection_damage_sensitivity.md: band edges under
every (coverage threshold) x (power threshold) x alpha x n combination, and
separately for the plug-in and the first-stage-aware SI variance.

Outputs
-------
results/figures/figure_detection_damage.png   (dpi 300)
results/figures/figure_detection_damage.pdf   (vector)
results/tables/detection_damage_sensitivity.md
paper/{arxiv,ijds}/figures/figure_detection_damage.{png,pdf}

Usage
-----
    python scripts/make_detection_damage_figure.py
"""

from __future__ import annotations

import os
import shutil
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType (Type 42), no Type 3 fonts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
PAPER_FIGURE_DIRS = [
    os.path.join(PROJECT_ROOT, "paper", "arxiv", "figures"),
    os.path.join(PROJECT_ROOT, "paper", "ijds", "figures"),
]

SRC = os.path.join(TABLES_DIR, "detection_damage_paired.csv")
REFINED_SRC = os.path.join(TABLES_DIR, "detection_damage_refined_grid.csv")
COLOR_SI_FINE = "#7b1b12"

RHO_MAX = 0.60

# Default rule shown as the shaded band.
COVERAGE_THRESHOLD = 0.90
POWER_THRESHOLD = 0.50
DEFAULT_ALPHA = 0.05

# Sensitivity grid.
COVERAGE_THRESHOLDS = [0.85, 0.90, 0.95]
POWER_THRESHOLDS = [0.50, 0.80]
ALPHAS = [0.05, 0.10]
#: SE(D) conventions in the sensitivity table, headline first.  "sandwich" is
#: the joint estimating-equation covariance for the learned index; "fixedf" is
#: the fixed-predictor covariance, kept as the ablation.
SI_VARIANTS = [("sandwich", "joint sandwich SI variance"),
               ("fixedf", "fixed-predictor SI variance (ablation)"),
               ("delta", "delta-method first-stage SI variance")]

#: Which coverage column each convention reads.
SI_COVERAGE_COL = {
    "sandwich": "si_cov_sandwich",
    "fixedf": "si_cov_plugin",
    "delta": "si_cov_delta",
}

COLOR_SI = "#c0392b"
COLOR_PPI = "#2c7fb8"
COLOR_POWER = "#2d7f2d"

# Larger fonts throughout.
FS_TICK = 13
FS_LABEL = 15
FS_TITLE = 16
FS_LEGEND = 11.5
FS_ANNOT = 12


def first_crossing(x: Sequence[float], y: Sequence[float],
                   level: float, direction: str) -> Optional[float]:
    """Linearly interpolate the first x at which y crosses `level`.

    direction == "down": first x where y goes from >= level to < level.
    direction == "up":   first x where y goes from <= level to > level.
    Returns None if no such crossing occurs on the grid.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
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


def load() -> pd.DataFrame:
    if not os.path.exists(SRC):
        raise SystemExit(
            f"{SRC} not found -- run scripts/run_detection_damage_paired.py first."
        )
    df = pd.read_csv(SRC)
    return df[df["rho"] <= RHO_MAX + 1e-9].sort_values(["n", "rho"])


def load_refined() -> Optional[pd.DataFrame]:
    if not os.path.exists(REFINED_SRC):
        return None
    return pd.read_csv(REFINED_SRC).sort_values(["n", "rho"])


def fine_si_curve(sub: pd.DataFrame, fine: pd.DataFrame):
    """SI coverage on the fine curve: refined cells, plus primary cells at rho
    values the refined grid does not carry."""
    fr = fine["rho"].to_numpy()
    keep = sub[~np.isclose(sub["rho"].to_numpy()[:, None],
                           fr[None, :]).any(axis=1)]
    both = pd.concat([keep[["rho", "si_cov_sandwich"]],
                      fine[["rho", "si_cov_sandwich"]]]).sort_values("rho")
    return both["rho"].to_numpy(), both["si_cov_sandwich"].to_numpy()


def make_figure(df: pd.DataFrame, refined: Optional[pd.DataFrame] = None):
    n_values = sorted(df["n"].unique())
    fig, axes = plt.subplots(1, len(n_values), figsize=(13.0, 5.2),
                             sharey=True)
    if len(n_values) == 1:
        axes = [axes]

    band_edges = {}

    for ax, n in zip(axes, n_values):
        sub = df[df["n"] == n]
        R = int(sub["R"].iloc[0])

        rho = sub["rho"].to_numpy()
        si = sub["si_cov_sandwich"].to_numpy()
        si_se = sub["si_cov_sandwich_mcse"].to_numpy()
        ppi = sub["ppi_cov_exact"].to_numpy()
        p05 = sub["rej_sandwich_a05"].to_numpy()
        p10 = sub["rej_sandwich_a10"].to_numpy()

        fine = (refined[refined["n"] == n] if refined is not None
                else pd.DataFrame())
        if len(fine):
            rho_f, si_f = fine_si_curve(sub, fine)
            rho_lo = first_crossing(rho_f, si_f, COVERAGE_THRESHOLD, "down")
        else:
            rho_lo = first_crossing(rho, si, COVERAGE_THRESHOLD, "down")
        rho_hi = first_crossing(rho, p05, POWER_THRESHOLD, "up")
        band_edges[n] = (rho_lo, rho_hi)

        if rho_lo is not None and rho_hi is not None and rho_hi > rho_lo:
            ax.axvspan(rho_lo, rho_hi, color="gray", alpha=0.28, zorder=0,
                       label="detection-damage band")

        for level in (0.95, COVERAGE_THRESHOLD, POWER_THRESHOLD):
            ax.axhline(level, color="0.3", linestyle=":", linewidth=1.0,
                       zorder=1)

        # SI coverage with +/- 2 MC-SE band.  Legend labels carry method and curve names only; the variance
        # estimators, the +/- 2 MC SE ribbon and bars, replication counts,
        # band thresholds and interpolation are in the caption note.  The
        # ribbon shares the SI line's legend entry.
        ribbon = ax.fill_between(rho, si - 2 * si_se, si + 2 * si_se,
                                 color=COLOR_SI, alpha=0.22, linewidth=0,
                                 zorder=2)
        (si_line,) = ax.plot(rho, si, marker="o", color=COLOR_SI,
                             linewidth=2.4, markersize=6, zorder=4)
        si_handle = (ribbon, si_line)
        if len(fine):
            fr = fine["rho"].to_numpy()
            fs = fine["si_cov_sandwich"].to_numpy()
            fse = fine["si_cov_sandwich_mcse"].to_numpy()
            ax.errorbar(fr, fs, yerr=2 * fse, fmt="D", markersize=4.5,
                        markerfacecolor="white", markeredgecolor=COLOR_SI_FINE,
                        ecolor=COLOR_SI_FINE, elinewidth=0.9, capsize=1.8,
                        linewidth=0, zorder=5,
                        label="SI coverage, refined grid")
        ax.plot(rho, ppi, marker="s", color=COLOR_PPI, linewidth=2.4,
                linestyle="--", markersize=6,
                label="PPI++ coverage", zorder=4)
        ax.plot(rho, p05, marker="^", color=COLOR_POWER, linewidth=2.4,
                markersize=7, label=r"diagnostic power, $\alpha = 0.05$",
                zorder=4)
        ax.plot(rho, p10, marker="v", color=COLOR_POWER, linewidth=2.2,
                linestyle="-.", markersize=7,
                label=r"diagnostic power, $\alpha = 0.10$", zorder=4)

        # Null-size line: the rho = 0 row of the same experiment is the
        # diagnostic's size, so power and size are read off one panel.  It
        # goes in the panel title, where it covers no curve.
        null = sub[sub["rho"] == 0.0]
        size_line = ""
        if len(null):
            r0 = null.iloc[0]
            size_line = (
                "\n" + r"size at $\rho = 0$: "
                + r"{:.3f} ($\alpha=0.05$), {:.3f} ($\alpha=0.10$)".format(
                    float(r0["rej_sandwich_a05"]),
                    float(r0["rej_sandwich_a10"]))
                + r"; MC SE {:.3f}".format(float(r0["rej_sandwich_a05_mcse"])))

        if rho_lo is not None and rho_hi is not None and rho_hi > rho_lo:
            ax.annotate(
                r"$\rho \in [{:.3f},\, {:.3f}]$".format(rho_lo, rho_hi),
                xy=(0.985, 0.5), xycoords="axes fraction",
                ha="right", va="center", fontsize=FS_ANNOT, zorder=6,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="0.55", alpha=0.95),
            )

        ax.set_xlim(0.0, RHO_MAX)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel(r"direct-effect share $\rho$", fontsize=FS_LABEL)
        n_tex = "{:,}".format(n).replace(",", "{,}")
        ax.set_title(r"$n = " + n_tex + r"$, $\pi_L = 0.20$, $R = "
                     + str(R) + r"$", fontsize=FS_TITLE, pad=24)
        if size_line:
            ax.text(0.5, 1.005, size_line.strip(), transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=FS_ANNOT - 1,
                    color="0.25")
        ax.tick_params(axis="both", labelsize=FS_TICK)

    axes[0].set_ylabel("coverage / rejection rate", fontsize=FS_LABEL)

    handles, labels = axes[0].get_legend_handles_labels()
    handles = [si_handle] + handles
    labels = ["SI coverage"] + labels
    # Order: SI, SI refined, PPI++, power 0.05, power 0.10, band.
    order = [labels.index(x) for x in (
        "SI coverage", "SI coverage, refined grid", "PPI++ coverage",
        r"diagnostic power, $\alpha = 0.05$",
        r"diagnostic power, $\alpha = 0.10$",
        "detection-damage band") if x in labels]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=FS_LEGEND, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.105, 1, 1))
    return fig, band_edges


def sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in sorted(df["n"].unique()):
        sub = df[df["n"] == n]
        rho = sub["rho"].to_numpy()
        for variant, variant_label in SI_VARIANTS:
            si = sub[SI_COVERAGE_COL[variant]].to_numpy()
            for cov_thr in COVERAGE_THRESHOLDS:
                # If SI coverage is already at or below the threshold at the
                # smallest rho on the grid, the band opens at rho = 0 rather
                # than "never": first_crossing needs a strict crossing.
                already = bool(si[0] <= cov_thr)
                lo = (float(rho[0]) if already
                      else first_crossing(rho, si, cov_thr, "down"))
                for alpha in ALPHAS:
                    pw = sub[f"rej_{variant}_a{int(alpha * 100):02d}"].to_numpy()
                    for pow_thr in POWER_THRESHOLDS:
                        hi = first_crossing(rho, pw, pow_thr, "up")
                        if lo is None or hi is None:
                            width = np.nan
                        else:
                            width = hi - lo
                        rows.append(dict(
                            n=int(n), si_variance=variant_label,
                            coverage_threshold=cov_thr, power_threshold=pow_thr,
                            alpha=alpha, rho_lo=lo, rho_hi=hi,
                            band_width=width,
                            si_already_below_at_rho0=already,
                            nonempty=(lo is not None and hi is not None
                                      and hi > lo),
                        ))
    return pd.DataFrame(rows)


def write_sensitivity_md(sens: pd.DataFrame, path: str) -> None:
    def f(v):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else f"{v:.3f}"

    lines = [
        "# Detection-damage band: threshold sensitivity\n\n",
        "Band edges from the paired simulation "
        "(results/tables/detection_damage_paired.csv, DGP 2, pi_L = 0.20), "
        "located by linear interpolation between bracketing grid points on "
        "rho in {0, .02, ..., .20, .25, .30, .35, .40, .50, .60}.\n\n"
        "* `rho_lo` = first rho at which SI coverage falls below the coverage "
        "threshold (SI interval becomes unreliable).\n"
        "* `rho_hi` = first rho at which two-sided diagnostic power exceeds "
        "the power threshold (the violation becomes detectable).\n"
        "* The band is the interval [rho_lo, rho_hi] where SI is already "
        "damaged but the diagnostic still usually misses it. `n/a` means the "
        "threshold is never crossed on the grid; a negative width means the "
        "diagnostic fires before SI coverage degrades (no gap).\n"
        "* A dagger on rho_lo marks a coverage threshold that SI already "
        "fails at rho = 0, so the band opens at the left edge of the grid. "
        "At the 0.95 threshold this is a Monte Carlo artifact: SI coverage "
        "at rho = 0 is within one MC SE of nominal, so a 0.95 cut is not "
        "usefully distinguishable from correct coverage.\n\n"
        "Both the SI coverage curve and SE(D_hat) use the SI variance named "
        "in the `SI variance` column, so each block is internally "
        "consistent.\n\n",
        "| n | SI variance | Coverage thr. | Power thr. | alpha | rho_lo | "
        "rho_hi | Band width | Nonempty |\n"
        "|---:|---|---:|---:|---:|---:|---:|---:|:--:|\n",
    ]
    for _, r in sens.iterrows():
        lines.append(
            f"| {int(r.n):,} | {r.si_variance} | {r.coverage_threshold:.2f} "
            f"| {r.power_threshold:.2f} | {r.alpha:.2f} "
            f"| {f(r.rho_lo)}{' &dagger;' if r.si_already_below_at_rho0 else ''} "
            f"| {f(r.rho_hi)} | {f(r.band_width)} "
            f"| {'yes' if r.nonempty else 'no'} |\n"
        )
    with open(path, "w") as fh:
        fh.writelines(lines)


def main() -> None:
    df = load()
    refined = load_refined()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, band_edges = make_figure(df, refined)
    png_path = os.path.join(FIGURES_DIR, "figure_detection_damage.png")
    pdf_path = os.path.join(FIGURES_DIR, "figure_detection_damage.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    for d in PAPER_FIGURE_DIRS:
        if os.path.isdir(d):
            shutil.copyfile(png_path,
                            os.path.join(d, "figure_detection_damage.png"))
            shutil.copyfile(pdf_path,
                            os.path.join(d, "figure_detection_damage.pdf"))

    sens = sensitivity_table(df)
    sens_csv = os.path.join(TABLES_DIR, "detection_damage_sensitivity.csv")
    sens_md = os.path.join(TABLES_DIR, "detection_damage_sensitivity.md")
    sens.to_csv(sens_csv, index=False)
    write_sensitivity_md(sens, sens_md)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {sens_md}")
    for n, (lo, hi) in band_edges.items():
        lo_s = "n/a" if lo is None else f"{lo:.3f}"
        hi_s = "n/a" if hi is None else f"{hi:.3f}"
        print(f"n={n:,}  default-rule band (SI cov < {COVERAGE_THRESHOLD}, "
              f"power > {POWER_THRESHOLD} at alpha={DEFAULT_ALPHA}): "
              f"rho in [{lo_s}, {hi_s}]")


if __name__ == "__main__":
    main()
