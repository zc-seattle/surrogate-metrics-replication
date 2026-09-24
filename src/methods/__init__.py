"""
Estimation methods for surrogate metrics comparison.

Implements 10 methods under a common interface:
    0. labeled_only       — Difference-in-means on Y (labeled units only)
    1. naive_surrogate    — Difference-in-means on S (all units)
    2. surrogate_index    — Athey et al. (2019)
    3. ppi_plus           — PPI++ (Angelopoulos et al. 2023)
    4. greg               — GREG / linear recalibration (the linear special
                            case of recalibrated PPI, Ji, Lei, Zrnic 2025)
    5. composite_proxy    — Tripuraneni et al. (2024)
    6. aipw               — Augmented IPW (doubly robust)
    7. ppi_plus_bootstrap — PPI++ with bootstrap variance (Section 3.5 fix)
    8. ppi_plus_corrected — PPI++ with analytic variance correction
    9. ppi_plus_exact     — PPI++ with the exact-variance tuning rule
                            (lambda_rule="exact", corrected_variance=True)

Each method function takes:
    T:            np.ndarray of treatment indicators (0/1)
    S:            np.ndarray of surrogate outcomes
    Y:            np.ndarray of primary outcomes (may contain NaN for unlabeled)
    Y_hat:        np.ndarray of predicted primary outcomes from f_hat
    labeled_mask: np.ndarray boolean mask (True = labeled)
    **kwargs:     method-specific parameters

Each returns a dict with:
    tau_hat:  float — point estimate of ATE
    var_hat:  float — estimated variance
    ci_lower: float — lower 95% CI bound
    ci_upper: float — upper 95% CI bound
    se_hat:   float — standard error

Paper label -> method ID map
------------------------------
The paper numbers methods 0-6; the method table additionally exposes three PPI++
variants (7, 8, 9).  7 and 8 are variance variants the paper discusses in
Section 3.5; 9 is the correctly tuned PPI++ comparator
(exact-variance tuning rule plus corrected variance).

    Paper label                         Method ID   Function
    ---------------------------------   -----------   -------------------------
    Method 0  Labeled-only              0             labeled_only
    Method 1  Naive surrogate           1             naive_surrogate
    Method 2  Surrogate index (SI)      2             surrogate_index
    Method 3  PPI++                     3             ppi_plus
    Method 4  GREG (linear recalib.)    4             greg
    Method 5  Composite proxy           5             composite_proxy
    Method 6  AIPW                      6             aipw
    (Sec 3.5) PPI++ bootstrap variance  7             ppi_plus_bootstrap
    (Contr C) PPI++ corrected variance  8             ppi_plus_corrected
    Method 9  PPI++ (exact rule)        9             ppi_plus_exact

Use `estimate(method_id, ...)` to dispatch by method ID.

NOTE: the paper's "hybrid" estimator is not one of the numbered methods and has no
method ID. It is a separate function, `hybrid_estimator(...)`, because it
combines the Method 2 and Method 3 outputs (plus their covariance) rather
than consuming the common (T, S, Y, Y_hat, labeled_mask) interface. The
related diagnostic `surrogacy_test(...)` and the covariance helper
`estimate_cov_si_ppi(...)` are likewise standalone functions.
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

# z critical value for 95% CI
Z_ALPHA = stats.norm.ppf(0.975)  # 1.96

#: Default prediction protocol recorded in every result's "config" block.
#: Mirrors src.simulations.simulation.DEFAULT_PROTOCOL; duplicated here as a
#: literal to avoid a circular import (simulation imports from this module).
DEFAULT_PROTOCOL = "allunits_crossfit"

#: Sentinel meaning "the caller did not name an si_variance", which lets
#: `surrogate_index` fall back to the plug-in variance (with a warning) when
#: no design dictionary is supplied, instead of raising.
_SI_VAR_UNSET = "__unset__"

_CONFIG_KEYS = (
    "protocol", "method_id", "lambda_rule", "clip", "per_arm",
    "variance", "si_variance",
)


def _config(
    method_id: Optional[int] = None,
    protocol: str = DEFAULT_PROTOCOL,
    lambda_rule: Optional[str] = None,
    clip: Optional[bool] = None,
    per_arm: Optional[bool] = None,
    variance: Optional[str] = None,
    si_variance: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the configuration block carried by every estimation result.

    The keys are fixed (`_CONFIG_KEYS`); entries that do not apply to the
    method are None.  The result rows are keyed on this block, so it
    must record the values actually used, not the values requested.
    """
    return {
        "protocol": protocol,
        "method_id": None if method_id is None else int(method_id),
        "lambda_rule": lambda_rule,
        "clip": clip,
        "per_arm": per_arm,
        "variance": variance,
        "si_variance": si_variance,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    tau_hat: float,
    var_hat: float,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a standard result dict from point estimate and variance."""
    se_hat = np.sqrt(max(var_hat, 0.0))
    out: Dict[str, Any] = {
        "tau_hat": float(tau_hat),
        "var_hat": float(var_hat),
        "se_hat": float(se_hat),
        "ci_lower": float(tau_hat - Z_ALPHA * se_hat),
        "ci_upper": float(tau_hat + Z_ALPHA * se_hat),
    }
    out["config"] = _config() if config is None else config
    return out


def _neyman_variance(V: np.ndarray, T: np.ndarray) -> float:
    """Neyman (difference-in-means) variance estimator with Bessel correction.

    V_hat = s^2_{V,1}/n_1 + s^2_{V,0}/n_0

    where s^2_{V,t} = 1/(n_t - 1) * sum_i (V_i - Vbar_t)^2 for T_i = t.
    """
    t1 = T == 1
    t0 = T == 0
    n1 = t1.sum()
    n0 = t0.sum()
    if n1 < 2 or n0 < 2:
        return np.inf
    var1 = np.var(V[t1], ddof=1)
    var0 = np.var(V[t0], ddof=1)
    return var1 / n1 + var0 / n0


def _diff_in_means(V: np.ndarray, T: np.ndarray) -> float:
    """Difference-in-means: mean(V | T=1) - mean(V | T=0)."""
    t1 = T == 1
    t0 = T == 0
    if t1.sum() == 0 or t0.sum() == 0:
        return np.nan
    return float(V[t1].mean() - V[t0].mean())


def _sample_cov(A: np.ndarray, B: np.ndarray) -> float:
    """Sample covariance with Bessel correction (ddof=1)."""
    n = len(A)
    if n < 2:
        return 0.0
    return float(np.cov(A, B, ddof=1)[0, 1])


#: Relative eigenvalue floor (rcond) for the rank-truncated inverse of a Gram
#: matrix: eigenvalues below _GRAM_RCOND * lambda_max are dropped.
_GRAM_RCOND = 1e-12


def _psd_solve(Q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rank-truncated solve Q^{-1} v for a symmetric PSD Gram matrix Q.

    The OLS design (1, S, S^2, X) is EXACTLY rank-deficient whenever a
    surrogate is binary, because then S^2 == S: the Criteo-calibrated funnel
    with a binary visit indicator is the case in this paper.  A plain solve or
    a ridge-regularised inverse then returns an enormous, arbitrary component
    along the null direction, which inflates the first-stage variance term by
    an order of magnitude (measured: Var(tau_SI) too large by a factor 4.7 on
    the multi-surrogate cell).  The prediction g' beta and the estimand
    d_0' beta are both well defined on the column space, so the influence
    function must use the pseudo-inverse restricted to it.

    Eigenvalues below `_GRAM_RCOND` (1e-12) times the largest are dropped.
    Measured on the Criteo-calibrated multi-surrogate design: the discarded
    eigenvalue is at most 2.4e-16 of the largest in absolute value (rounding
    noise around the exact zero) and the smallest genuine one is at least
    1.3e-8 of it, so any floor in (1e-16, 1e-8) keeps the same eigenvectors;
    rcond in {1e-10, 1e-12, 1e-14} gives identical variances.

    This truncation is used only on the variance side.  The index itself is
    fit by `src.simulations.simulation._ols_fit`, which solves
    (X'X + 1e-10 I)^{-1} X'y (a fixed absolute ridge, no column dropping).
    """
    w, V = np.linalg.eigh(Q)
    w_max = float(w.max()) if w.size else 0.0
    if w_max <= 0.0:
        return np.zeros_like(v)
    keep = w > _GRAM_RCOND * w_max
    if not np.any(keep):
        return np.zeros_like(v)
    Vk = V[:, keep]
    return Vk @ ((Vk.T @ v) / w[keep])


# ---------------------------------------------------------------------------
# Joint sandwich covariance of (tau_SI, tau_PPI) for the LEARNED linear index
# ---------------------------------------------------------------------------

def joint_influence_cov(
    T: np.ndarray,
    Y: np.ndarray,
    labeled_mask: np.ndarray,
    design: Dict[str, Any],
    lambda_hat: float,
    p_hat: Optional[float] = None,
) -> Dict[str, Any]:
    r"""Stacked estimating-equation (sandwich) covariance of (tau_SI, tau_PPI).

    Both estimators use the SAME linear index g(S, X)' beta, with beta fit by
    OLS on the labeled units under the all-units cross-fitting protocol.  The
    fixed-predictor variance formulas ignore the sampling variation of
    beta_hat; this function restores it.

    Notation
    --------
    g_i        design row (1, S_i, S_i^2, X_i), q columns, from `design["g"]`
    b          = design["beta_full"], the full-labeled-sample OLS fit
    L_i        1 if unit i is labeled, 0 otherwise
    pi_L       = n_L / n
    p_hat      = P(T = 1), estimated by mean(T) unless supplied
    a_i        = T_i / p_hat - (1 - T_i) / (1 - p_hat)   (arm contrast weight)
    e_i        = Y_i - g_i' b for labeled units, 0 otherwise (index residual)
    Q          = (1 / n_L) sum_{i labeled} g_i g_i'      (labeled Gram matrix)
    d0         = mean(g | T = 1) - mean(g | T = 0), over ALL units

    Influence functions
    -------------------
        phi_SI_i  = a_i (g_i' b - mean_{arm(i)}(g' b))
                    + (L_i / pi_L) d0' Q^{-1} g_i e_i

        phi_PPI_i = (L_i / pi_L) a_i [ (Y_i - lambda g_i' b)
                                       - labeled arm mean of (Y - lambda g' b) ]
                    + lambda a_i (g_i' b - mean_{arm(i)}(g' b))

    The first term of phi_SI is the fixed-predictor ("plug-in") piece
    beta' (d_hat - d_0); the second is the first-stage piece
    d_0' (beta_hat - beta), which the plug-in variance omits and which also
    generates most of the covariance with tau_PPI.  lambda is held fixed at
    its estimated value (the estimated-tuning term is second order).

    Returns
    -------
    dict with
        omega       (2, 2) per-unit influence covariance,
                    omega = (1/n) sum_i (phi_i - phibar)(phi_i - phibar)'
        cov_tau     (2, 2) = omega / n, the covariance of (tau_SI, tau_PPI)
        var_si      cov_tau[0, 0]
        var_ppi     cov_tau[1, 1]
        cov_si_ppi  cov_tau[0, 1]
        var_D       Var(D_hat) for D_hat = tau_PPI - tau_SI,
                    = var_si + var_ppi - 2 cov_si_ppi
        p_hat, pi_L, n, n_L

    Raises
    ------
    ValueError
        if `design` carries no design matrix (e.g. a GBT index), in which
        case the joint covariance must be obtained by the paired bootstrap.
    """
    if design is None or design.get("g", None) is None:
        raise ValueError(
            "joint_influence_cov requires a linear design matrix "
            "(design['g']); it is None for a GBT index. Use the paired "
            "bootstrap for the joint covariance in that case."
        )

    G = np.asarray(design["g"], dtype=np.float64)
    b = np.asarray(design["beta_full"], dtype=np.float64)
    T = np.asarray(T)
    Y = np.asarray(Y, dtype=np.float64)
    L = np.asarray(labeled_mask, dtype=bool)

    n, q = G.shape
    n_L = int(L.sum())
    if n_L < q + 2:
        raise ValueError(
            f"joint_influence_cov: too few labeled units ({n_L}) for a "
            f"{q}-column design."
        )
    pi_L = n_L / n

    if p_hat is None:
        p_hat = float(np.mean(T == 1))
    if not (0.0 < p_hat < 1.0):
        raise ValueError(f"joint_influence_cov: degenerate p_hat = {p_hat}.")

    t1 = (T == 1)
    t0 = ~t1
    a = np.where(t1, 1.0 / p_hat, -1.0 / (1.0 - p_hat))

    # --- index predictions and their within-arm centering ------------------
    f = G @ b                                   # g_i' b
    f_centered = np.empty(n, dtype=np.float64)
    f_centered[t1] = f[t1] - f[t1].mean()
    f_centered[t0] = f[t0] - f[t0].mean()

    # --- first-stage score: (L_i / pi_L) d0' Qinv g_i e_i -------------------
    d0 = G[t1].mean(axis=0) - G[t0].mean(axis=0)
    Q = (G[L].T @ G[L]) / n_L
    Qinv_d0 = _psd_solve(Q, d0)

    e = np.zeros(n, dtype=np.float64)
    e[L] = Y[L] - f[L]
    h = np.zeros(n, dtype=np.float64)           # d0' Qinv g_i e_i
    h[L] = (G[L] @ Qinv_d0) * e[L]

    phi_si = a * f_centered + (L / pi_L) * h

    # --- PPI++ rectifier, centered within arm on the LABELED units ---------
    r = Y - lambda_hat * f                      # Y_i - lambda g_i' b
    r_centered = np.zeros(n, dtype=np.float64)
    m1 = L & t1
    m0 = L & t0
    if m1.sum() < 2 or m0.sum() < 2:
        raise ValueError(
            "joint_influence_cov: fewer than two labeled units in an arm."
        )
    r_centered[m1] = r[m1] - r[m1].mean()
    r_centered[m0] = r[m0] - r[m0].mean()

    phi_ppi = (L / pi_L) * a * r_centered + lambda_hat * a * f_centered

    # --- omega and the estimator covariance --------------------------------
    Phi = np.column_stack([phi_si, phi_ppi])
    Phi = Phi - Phi.mean(axis=0, keepdims=True)     # centered
    omega = (Phi.T @ Phi) / n
    cov_tau = omega / n

    var_si = float(cov_tau[0, 0])
    var_ppi = float(cov_tau[1, 1])
    cov_si_ppi = float(cov_tau[0, 1])
    var_D = float(var_si + var_ppi - 2.0 * cov_si_ppi)

    return {
        "omega": omega,
        "cov_tau": cov_tau,
        "var_si": var_si,
        "var_ppi": var_ppi,
        "cov_si_ppi": cov_si_ppi,
        "var_D": var_D,
        "lambda_hat": float(lambda_hat),
        "p_hat": float(p_hat),
        "pi_L": float(pi_L),
        "n": int(n),
        "n_L": int(n_L),
    }


def hybrid_sigma(
    T: np.ndarray,
    Y: np.ndarray,
    labeled_mask: np.ndarray,
    design: Dict[str, Any],
    lambda_hat: float,
    p_hat: Optional[float] = None,
) -> np.ndarray:
    """Sigma_hat = Cov((tau_SI, tau_PPI)) for the hybrid estimator.

    Thin wrapper on `joint_influence_cov`: returns the 2 x 2 matrix

        [[Var(tau_SI),        Cov(tau_SI, tau_PPI)],
         [Cov(tau_SI, tau_PPI), Var(tau_PPI)      ]]

    which is what `hybrid_estimator` needs (its var_si, var_ppi and
    cov_si_ppi arguments are the three distinct entries).
    """
    return joint_influence_cov(
        T, Y, labeled_mask, design, lambda_hat, p_hat=p_hat,
    )["cov_tau"]


# ---------------------------------------------------------------------------
# Method 0: Labeled-Only Difference-in-Means
# ---------------------------------------------------------------------------

def labeled_only(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    **kwargs: Any,
) -> Dict[str, float]:
    """Method 0: Difference-in-means on Y using only labeled units.

    This is the baseline estimator. It ignores the surrogate entirely
    and uses only the subset of units where Y is observed.

    kwargs["config"] lets a caller (e.g. a PPI-family edge-case fallback)
    stamp the result with the configuration that was requested.
    """
    T_L = T[labeled_mask]
    Y_L = Y[labeled_mask]

    tau_hat = _diff_in_means(Y_L, T_L)
    var_hat = _neyman_variance(Y_L, T_L)

    cfg = kwargs.get("config", None) or _config(
        method_id=0, protocol=kwargs.get("protocol", DEFAULT_PROTOCOL),
    )
    return _make_result(tau_hat, var_hat, cfg)


# ---------------------------------------------------------------------------
# Method 1: Naive Surrogate
# ---------------------------------------------------------------------------

def naive_surrogate(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    **kwargs: Any,
) -> Dict[str, float]:
    """Method 1: Difference-in-means on S using all units.

    Estimates tau_S, not tau. Also computes a rescaled version
    tau_rescaled = beta_YS_hat * tau_S_hat where beta_YS_hat is the OLS
    coefficient from regressing Y on S in the labeled data.

    Returns the rescaled estimate as tau_hat (targeting tau), plus extra keys:
        tau_S_hat:      raw surrogate ATE
        tau_S_var:      variance of tau_S_hat
        beta_YS_hat:    OLS coefficient from labeled data
        tau_raw_ci_lower / tau_raw_ci_upper: CI for tau_S
    """
    # Raw surrogate ATE (all units)
    tau_S_hat = _diff_in_means(S, T)
    var_S = _neyman_variance(S, T)

    # Rescaled version using labeled data
    T_L = T[labeled_mask]
    Y_L = Y[labeled_mask]
    S_L = S[labeled_mask]

    # Simple regression: beta_YS = Cov(Y, S) / Var(S).
    # NOTE: This is the *simple* (bivariate) OLS coefficient, NOT the partial
    # coefficient from Y ~ S + X.  When X confounds S and Y (as in DGP 1),
    # the simple coefficient is upward-biased relative to the partial
    # coefficient due to omitted variable bias (OVB).  This is intentional:
    # the "naive" method represents the simplistic practitioner approach that
    # ignores covariates.  Under DGP 1 defaults, the simple beta ≈ 0.54 vs
    # the partial beta = 0.50, producing ~8% upward bias in tau_hat.
    n_L = labeled_mask.sum()
    if n_L >= 2 and np.var(S_L, ddof=1) > 0:
        beta_YS_hat = _sample_cov(Y_L, S_L) / np.var(S_L, ddof=1)
    else:
        beta_YS_hat = 1.0  # fallback: no rescaling

    tau_hat_rescaled = beta_YS_hat * tau_S_hat

    # Variance of rescaled estimate (delta method, treating beta as fixed):
    # Var(beta * tau_S) = beta^2 * Var(tau_S)
    var_hat_rescaled = beta_YS_hat ** 2 * var_S

    result = _make_result(
        tau_hat_rescaled, var_hat_rescaled,
        _config(method_id=1,
                protocol=kwargs.get("protocol", DEFAULT_PROTOCOL)),
    )

    # Store extra info
    result["tau_S_hat"] = float(tau_S_hat)
    result["tau_S_var"] = float(var_S)
    result["tau_S_se"] = float(np.sqrt(max(var_S, 0.0)))
    result["beta_YS_hat"] = float(beta_YS_hat)
    se_S = np.sqrt(max(var_S, 0.0))
    result["tau_raw_ci_lower"] = float(tau_S_hat - Z_ALPHA * se_S)
    result["tau_raw_ci_upper"] = float(tau_S_hat + Z_ALPHA * se_S)

    return result


# ---------------------------------------------------------------------------
# Method 2: Surrogate Index (Athey et al. 2019)
# ---------------------------------------------------------------------------

def surrogate_index(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    **kwargs: Any,
) -> Dict[str, float]:
    """Method 2: Surrogate Index.

    Uses Y_hat (predicted primary outcomes) from the all-units cross-fitted
    prediction model f_hat(S, X) (or an externally-calibrated one).

    tau_hat = mean(Y_hat[T=1]) - mean(Y_hat[T=0])  over ALL units.

    Variance (kwargs["si_variance"]):

      "sandwich" (DEFAULT)
        The joint estimating-equation variance for the LEARNED index,
        `joint_influence_cov(...)["var_si"]`.  Requires the design dictionary
        returned by `train_prediction_model(..., return_design=True)`, passed
        as kwargs["design"], and the PPI++ tuning constant as
        kwargs["lambda_hat"] (only the (1,1) entry is returned here, so
        lambda only affects the reported value through nothing -- it is
        accepted so the same call can produce the full 2 x 2).

      "plugin"
        The Neyman difference-in-means variance of the imputed outcomes, which
        treats the predictions as fixed. This is the SI interval.

      "delta"
        plug-in + d' V_beta d, the delta-method first-stage term of
        `si_first_stage_term`, treating the design-mean contrast d as fixed.
        Requires kwargs["design"] (or the covariates kwargs["X"]).

    kwargs:
        use_unlabeled_only: bool (default False). If True, compute ATE on
            unlabeled units only (sample-splitting variant B).
        first_stage_variance: bool (default False). Deprecated alias for
            si_variance="delta".

    Extra result keys when si_variance is "delta" or "sandwich":
        var_plugin       : the plug-in (predictions-treated-as-fixed) variance
        var_first_stage  : d' V_beta d ("delta" only)
    """
    use_unlabeled_only = kwargs.get("use_unlabeled_only", False)
    protocol = kwargs.get("protocol", DEFAULT_PROTOCOL)
    design = kwargs.get("design", None)

    si_variance = kwargs.get("si_variance", _SI_VAR_UNSET)
    if kwargs.get("first_stage_variance", False) and si_variance is _SI_VAR_UNSET:
        si_variance = "delta"           # deprecated alias
    requested_default = si_variance is _SI_VAR_UNSET
    if requested_default:
        si_variance = "sandwich"
    if si_variance not in ("sandwich", "plugin", "delta"):
        raise ValueError(
            f"Unknown si_variance {si_variance!r}. Choose 'sandwich', "
            f"'plugin' or 'delta'."
        )

    if si_variance in ("sandwich", "delta") and design is None:
        if si_variance == "delta" and kwargs.get("X", None) is not None:
            pass
        elif requested_default:
            warnings.warn(
                "surrogate_index: si_variance defaults to 'sandwich', which "
                "needs the design dict from "
                "train_prediction_model(..., return_design=True); none was "
                "supplied, so the plug-in variance is used and "
                "config['si_variance'] is recorded as 'plugin'.",
                RuntimeWarning, stacklevel=2,
            )
            si_variance = "plugin"
        else:
            raise ValueError(
                f"surrogate_index: si_variance={si_variance!r} requires "
                f"design=... (from train_prediction_model(..., "
                f"return_design=True))."
            )

    if use_unlabeled_only:
        mask = ~labeled_mask
        T_sub = T[mask]
        Yh_sub = Y_hat[mask]
    else:
        T_sub = T
        Yh_sub = Y_hat

    tau_hat = _diff_in_means(Yh_sub, T_sub)
    var_plugin = _neyman_variance(Yh_sub, T_sub)

    cfg = _config(method_id=2, protocol=protocol, si_variance=si_variance)

    if si_variance == "plugin":
        return _make_result(tau_hat, var_plugin, cfg)

    if si_variance == "sandwich":
        joint = joint_influence_cov(
            T, Y, labeled_mask, design,
            lambda_hat=float(kwargs.get("lambda_hat", 0.0)),
            p_hat=kwargs.get("p_hat", None),
        )
        result = _make_result(tau_hat, joint["var_si"], cfg)
        result["var_plugin"] = float(var_plugin)
        result["var_first_stage"] = float(joint["var_si"] - var_plugin)
        return result

    # si_variance == "delta"
    var_fs = si_first_stage_term(
        T, S, Y, labeled_mask,
        X=kwargs.get("X", None),
        use_unlabeled_only=use_unlabeled_only,
        design=design,
    )
    result = _make_result(tau_hat, var_plugin + var_fs, cfg)
    result["var_plugin"] = float(var_plugin)
    result["var_first_stage"] = float(var_fs)
    return result


def si_first_stage_term(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    labeled_mask: np.ndarray,
    X: Optional[np.ndarray] = None,
    use_unlabeled_only: bool = False,
    design: Optional[Dict[str, Any]] = None,
) -> float:
    """Delta-method first-stage variance term for the OLS surrogate index.

    The surrogate index is a plug-in functional of the fitted index
    coefficients.  With design vector g(S, X) = (1, S, S^2, X) (exactly the
    design `src.simulations.simulation._build_design_matrix` builds), the
    index estimator can be written

        tau_SI = (gbar_1 - gbar_0)' beta_hat = d' beta_hat,

    where gbar_t is the mean design vector in arm t over ALL units (labeled
    and unlabeled) and beta_hat is the OLS fit of Y on g(S, X) over the
    labeled units.  Treating d as fixed (it is a sample mean over all n
    units, so its own sampling variation is what the plug-in Neyman variance
    already captures) and applying the delta method to beta_hat gives

        Var_first_stage = d' V_beta d,

    with V_beta the HC1 heteroskedasticity-robust sandwich covariance of the
    full-labeled-set OLS fit.  Total SI variance is the plug-in
    imputed-outcome difference-in-means variance plus this term.

    Multi-surrogate S is handled automatically: `_build_design_matrix`
    widens the design to (1, S_1..S_J, S_1^2..S_J^2, X).

    Notes
    -----
    * beta_hat here is fit ONCE on all labeled units.  The point estimate
      still uses the cross-fitted predictions; only the variance uses this
      full-sample fit, which is the usual delta-method convention (the
      cross-fitted and full-sample fits are asymptotically equivalent).
    * Returns 0.0 when there is too little labeled data to fit the design.
    """
    if design is not None and design.get("g", None) is not None:
        G = np.asarray(design["g"], dtype=np.float64)
    else:
        from src.simulations.simulation import _build_design_matrix
        G = _build_design_matrix(S, X)
    n, p = G.shape

    sel = (~labeled_mask) if use_unlabeled_only else np.ones(n, dtype=bool)
    m1 = sel & (T == 1)
    m0 = sel & (T == 0)
    if m1.sum() < 1 or m0.sum() < 1:
        return 0.0

    d = G[m1].mean(axis=0) - G[m0].mean(axis=0)

    G_L = G[labeled_mask]
    Y_L = Y[labeled_mask]
    n_L = G_L.shape[0]
    if n_L <= p + 1:
        return 0.0

    # Rank-truncated inverse: the design is exactly singular when a surrogate
    # is binary (S^2 == S).  See `_psd_solve`.
    GtG = G_L.T @ G_L
    w, V = np.linalg.eigh(GtG)
    w_max = float(w.max()) if w.size else 0.0
    if w_max <= 0.0:
        return 0.0
    keep = w > _GRAM_RCOND * w_max
    if not np.any(keep):
        return 0.0
    Vk = V[:, keep]
    bread = Vk @ np.diag(1.0 / w[keep]) @ Vk.T

    beta = bread @ (G_L.T @ Y_L)
    resid = Y_L - G_L @ beta

    # HC1 meat: G' diag(e^2) G, scaled by n_L / (n_L - p)
    meat = (G_L * (resid ** 2)[:, None]).T @ G_L
    V_beta = bread @ meat @ bread * (n_L / (n_L - p))

    return float(max(d @ V_beta @ d, 0.0))


# ---------------------------------------------------------------------------
# Method 3: PPI++ (Angelopoulos et al. 2023)
# ---------------------------------------------------------------------------

def ppi_plus(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    **kwargs: Any,
) -> Dict[str, float]:
    """Method 3: Prediction-Powered Inference++ (PPI++).

    tau_hat = tau_Y_labeled + lambda (tau_Yhat_all - tau_Yhat_labeled)

    DEFAULT CONFIGURATION: lambda_rule="exact",
    clip=False, per_arm=False, variance="exact".  A common, unclipped
    coefficient tuned by the exact-variance rule is standard PPI++ mean
    estimation for a scalar coefficient; the clipped and per-arm versions are
    reported as ablations.

    kwargs:
        lam : float, or a (lam_0, lam_1) pair when per_arm=True. If given,
            skips the tuning step.
        lambda_rule : "exact" (default) or "plugin"; see `_ppi_family`.
        clip : bool (default False). Clip the coefficient(s) to [0, 1].
        per_arm : bool (default False). Tune a separate lambda_t per arm.
        variance : "exact" (default) or "plugin"; see `_ppi_family`.
        corrected_variance : deprecated alias, True -> variance="exact".
    """
    kwargs.setdefault("lambda_rule", "exact")
    kwargs.setdefault("clip", False)
    kwargs.setdefault("per_arm", False)
    if "corrected_variance" not in kwargs:
        kwargs.setdefault("variance", "exact")
    return _ppi_family(T, S, Y, Y_hat, labeled_mask, method_id=3, **kwargs)


# ---------------------------------------------------------------------------
# Method 4: GREG / linear recalibration (linear case of recalibrated PPI, Ji et al. 2025)
# ---------------------------------------------------------------------------

def greg(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    **kwargs: Any,
) -> Dict[str, float]:
    """Method 4: GREG, the unconstrained linear recalibration.

    Same structure as PPI++ but beta is UNCONSTRAINED (no clipping to [0,1]).
    This is the generalized regression (GREG) estimator of survey sampling,
    which is the linear special case of recalibrated PPI (Ji, Lei and Zrnic
    2025). The general recalibration is not implemented here, so the paper
    labels this comparator GREG.

    tau_hat = tau_Y_labeled + beta * (tau_Yhat_all - tau_Yhat_labeled)

    DEFAULT CONFIGURATION: lambda_rule="exact",
    clip=False, per_arm=False, variance="exact" -- identical to method 3.
    GREG and PPI++ with a common unclipped coefficient tuned by the
    exact-variance rule ARE the same estimator; the paper states the
    coincidence once and reports one of them.  The method table keeps both ids so
    that older loaders and result files continue to resolve, and so that the
    ablation can vary the configuration of either.
    """
    kwargs.setdefault("lambda_rule", "exact")
    kwargs.setdefault("per_arm", False)
    if "corrected_variance" not in kwargs:
        kwargs.setdefault("variance", "exact")
    kwargs["clip"] = False
    return _ppi_family(T, S, Y, Y_hat, labeled_mask, method_id=4, **kwargs)


def ppi_plus_corrected_variance(
    Y_L: np.ndarray,
    Yhat_L: np.ndarray,
    Yhat_all: np.ndarray,
    T_L: np.ndarray,
    T_all: np.ndarray,
    lambda_hat,
) -> float:
    """Compute the overlap variance correction for PPI++.

    The plug-in variance formula assumes the all-data prediction term and
    the labeled-data residual term are independent. At high labeled fractions
    (pi_L), the labeled set constitutes most of the all-data set, creating a
    covariance the formula ignores. This correction restores nominal coverage.

    Per treatment arm t, the correction is:
        C_t = 2 * lambda_t * (gamma_t - lambda_t * sigma^2_{Yhat,t}) / n_t

    where gamma_t = Cov(Y, Yhat | T=t) and sigma^2_{Yhat,t} = Var(Yhat | T=t).
    Summing C_1 + C_0 and adding it to the plug-in variance yields the exact
    variance, which is why `lambda_rule="exact"` is precisely the minimiser of
    the corrected variance.

    Parameters
    ----------
    Y_L : primary outcomes for labeled units
    Yhat_L : predicted outcomes for labeled units
    Yhat_all : predicted outcomes for all units
    T_L : treatment indicators for labeled units
    T_all : treatment indicators for all units
    lambda_hat : the estimated tuning parameter; a scalar for a common
        coefficient, or a mapping / 2-sequence {0: lam_0, 1: lam_1} for the
        per-arm version.

    Returns
    -------
    float : the correction (to be added to plug-in variance)
    """
    if np.isscalar(lambda_hat):
        lam_by_arm = {0: float(lambda_hat), 1: float(lambda_hat)}
        # No correction needed when a common lambda is zero or negative
        # (the estimator then collapses to labeled-only).
        if float(lambda_hat) <= 0.0:
            return 0.0
    elif isinstance(lambda_hat, dict):
        lam_by_arm = {0: float(lambda_hat[0]), 1: float(lambda_hat[1])}
    else:
        lam_by_arm = {0: float(lambda_hat[0]), 1: float(lambda_hat[1])}

    correction = 0.0

    for t_val in [0, 1]:
        lam_t = lam_by_arm[t_val]
        # Labeled arm
        mask_L_t = T_L == t_val
        nL_t = int(mask_L_t.sum())
        if nL_t < 2:
            return 0.0

        Y_t = Y_L[mask_L_t]
        Yh_L_t = Yhat_L[mask_L_t]

        # All-data arm
        mask_all_t = T_all == t_val
        n_t = int(mask_all_t.sum())
        if n_t < 2:
            return 0.0

        Yh_all_t = Yhat_all[mask_all_t]

        # gamma_t = Cov(Y, Yhat | T=t) on labeled data
        gamma_t = _sample_cov(Y_t, Yh_L_t)

        # sigma^2_{Yhat,t} on all units
        sig2_Yh_t = float(np.var(Yh_all_t, ddof=1))

        # Overlap correction: C_t = 2*lam_t*(gamma_t - lam_t*sig2_Yh_t) / n_t
        correction += 2.0 * lam_t * (gamma_t - lam_t * sig2_Yh_t) / n_t

    return float(correction)


def _ppi_family(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    method_id: int = 3,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Shared implementation for PPI++ (method 3) and GREG (method 4).

    Estimator (common coefficient, per_arm=False):
        tau = tau_Y^L + lam (tau_Yhat^all - tau_Yhat^L)

    Estimator (per_arm=True), with lam_t applied inside arm t:
        tau = [ Ybar^L_1 + lam_1 (Yhatbar_1 - Yhatbar^L_1) ]
            - [ Ybar^L_0 + lam_0 (Yhatbar_0 - Yhatbar^L_0) ]

    Tuning parameter (kwargs["lambda_rule"]), with
    gamma_t = Cov(Y, Yhat | T = t) on labeled data and
    sigma^2_{Yhat,t} = Var(Yhat | T = t) on all units:

      "exact"
        Minimises the EXACT per-arm variance
          sigma^2_Y/n_L + (1/n_L - 1/n)(lam^2 sigma^2_Yhat - 2 lam gamma),
        whose minimiser is gamma / sigma^2_Yhat, independent of pi_L.  Pooled
        across arms with the weights w_t = (1/n_{L,t} - 1/n_t) that the exact
        objective implies:

          lam_exact = sum_t [ w_t gamma_t ] / sum_t [ w_t sigma^2_{Yhat,t} ],

        which reduces to gamma / sigma^2_Yhat in the balanced,
        arm-homogeneous case.  With per_arm=True the w_t cancels inside each
        arm and

          lam_t = gamma_t / sigma^2_{Yhat,t}.

      "plugin" (Eq. (5) of the paper)
        Minimises the PLUG-IN variance objective, which treats the all-data
        prediction term and the labeled rectifier term as independent:

          lam_plugin = sum_t [ gamma_t / n_{L,t} ]
                       / sum_t [ sigma^2_{Yhat,t} (1/n_t + 1/n_{L,t}) ]

        In the balanced, arm-homogeneous case this equals
          gamma / (sigma^2_Yhat (1 + n_L/n)) = lam_exact / (1 + pi_L),
        i.e. it shrinks the exact-variance minimiser by a factor 1/(1+pi_L).
        With per_arm=True,
          lam_t = (gamma_t / n_{L,t}) / (sigma^2_{Yhat,t} (1/n_t + 1/n_{L,t})).

    ``clip=True`` clips whichever coefficients are used (the common lam, or
    each lam_t) to [0, 1].  This is our restriction, not part of published
    PPI++ mean estimation.

    Variance (kwargs["variance"]):

      "plugin"
          V = sum_t [ lam_t^2 sigma^2_{Yhat,t} / n_t
                     + Var(Y - lam_t Yhat | T = t, labeled) / n_{L,t} ]

      "exact" (DEFAULT for method 3)
          V + sum_t C_t,  C_t = 2 lam_t (gamma_t - lam_t sigma^2_{Yhat,t})/n_t,
        the overlap correction generalised to per-arm coefficients.  It is
        algebraically the exact variance of the estimator, so lambda_rule
        "exact" is precisely its minimiser.

    ``corrected_variance=True/False`` is accepted as a deprecated alias for
    variance="exact"/"plugin".
    """
    lam_override = kwargs.get("lam", None)
    lambda_rule = kwargs.get("lambda_rule", "exact")
    clip = bool(kwargs.get("clip", False))
    per_arm = bool(kwargs.get("per_arm", False))
    protocol = kwargs.get("protocol", DEFAULT_PROTOCOL)

    if "corrected_variance" in kwargs:
        variance = "exact" if kwargs["corrected_variance"] else "plugin"
        if "variance" in kwargs and kwargs["variance"] != variance:
            raise ValueError(
                "Pass either variance=... or the deprecated alias "
                "corrected_variance=..., not both with conflicting values."
            )
    else:
        variance = kwargs.get("variance", "exact")

    if lambda_rule not in ("plugin", "exact"):
        raise ValueError(
            f"Unknown lambda_rule {lambda_rule!r}. Choose 'plugin' or 'exact'."
        )
    if variance not in ("plugin", "exact"):
        raise ValueError(
            f"Unknown variance {variance!r}. Choose 'plugin' or 'exact'."
        )

    cfg = _config(
        method_id=method_id, protocol=protocol, lambda_rule=lambda_rule,
        clip=clip, per_arm=per_arm, variance=variance,
    )

    # Partition arrays
    T_L = T[labeled_mask]
    Y_L = Y[labeled_mask]
    Yh_L = Y_hat[labeled_mask]
    Yh_all = Y_hat

    # Counts per arm
    t1_all = T == 1
    t0_all = T == 0
    n1 = int(t1_all.sum())
    n0 = int(t0_all.sum())

    t1_L = T_L == 1
    t0_L = T_L == 0
    nL1 = int(t1_L.sum())
    nL0 = int(t0_L.sum())

    # Edge case: insufficient labeled data in one arm
    if nL1 < 2 or nL0 < 2:
        return labeled_only(T, S, Y, Y_hat, labeled_mask, config=cfg)

    if n1 < 2 or n0 < 2:
        return labeled_only(T, S, Y, Y_hat, labeled_mask, config=cfg)

    # Edge case: when all (or nearly all) units are labeled, the PPI variance
    # formula underestimates variance because it assumes independence between
    # the prediction term (all n) and rectifier term (n_L). When L ~ all,
    # these share units and the ignored negative covariance causes
    # under-coverage.  Fall back to labeled-only when fewer than 5% of units
    # are unlabeled.
    n_unlabeled = int((~labeled_mask).sum())
    if n_unlabeled <= max(10, int(0.05 * len(T))):
        return labeled_only(T, S, Y, Y_hat, labeled_mask, config=cfg)

    # Component estimates
    tau_Y_L = _diff_in_means(Y_L, T_L)
    tau_Yh_all = _diff_in_means(Yh_all, T)
    tau_Yh_L = _diff_in_means(Yh_L, T_L)

    # gamma_t = Cov(Y, Yhat | T=t) on labeled data
    gamma_1 = _sample_cov(Y_L[t1_L], Yh_L[t1_L])
    gamma_0 = _sample_cov(Y_L[t0_L], Yh_L[t0_L])

    # sigma^2_{Yhat,t} on ALL units
    sig2_Yh_1 = float(np.var(Yh_all[t1_all], ddof=1))
    sig2_Yh_0 = float(np.var(Yh_all[t0_all], ddof=1))

    # --- tuning ------------------------------------------------------------
    if lam_override is not None:
        if per_arm:
            lam_arr = np.asarray(lam_override, dtype=float).ravel()
            if lam_arr.size == 1:
                lam_1 = lam_0 = float(lam_arr[0])
            else:
                lam_0, lam_1 = float(lam_arr[0]), float(lam_arr[1])
        else:
            lam_1 = lam_0 = float(np.asarray(lam_override).ravel()[0])
    elif per_arm:
        if lambda_rule == "exact":
            # w_t cancels inside an arm: lam_t = gamma_t / sigma^2_{Yhat,t}.
            lam_1 = gamma_1 / sig2_Yh_1 if sig2_Yh_1 > 0 else 0.0
            lam_0 = gamma_0 / sig2_Yh_0 if sig2_Yh_0 > 0 else 0.0
        else:
            den_1 = sig2_Yh_1 * (1.0 / n1 + 1.0 / nL1)
            den_0 = sig2_Yh_0 * (1.0 / n0 + 1.0 / nL0)
            lam_1 = (gamma_1 / nL1) / den_1 if den_1 > 0 else 0.0
            lam_0 = (gamma_0 / nL0) / den_0 if den_0 > 0 else 0.0
        if clip:
            lam_1 = float(np.clip(lam_1, 0.0, 1.0))
            lam_0 = float(np.clip(lam_0, 0.0, 1.0))
    else:
        if lambda_rule == "exact":
            # Weights implied by the exact variance: w_t = 1/n_{L,t} - 1/n_t.
            w1 = 1.0 / nL1 - 1.0 / n1
            w0 = 1.0 / nL0 - 1.0 / n0
            numerator = w1 * gamma_1 + w0 * gamma_0
            denominator = w1 * sig2_Yh_1 + w0 * sig2_Yh_0
        else:
            numerator = gamma_1 / nL1 + gamma_0 / nL0
            denominator = (sig2_Yh_1 * (1.0 / n1 + 1.0 / nL1)
                           + sig2_Yh_0 * (1.0 / n0 + 1.0 / nL0))

        lam = numerator / denominator if denominator > 0 else 0.0
        if clip:
            lam = float(np.clip(lam, 0.0, 1.0))
        lam_1 = lam_0 = float(lam)

    # --- point estimate ----------------------------------------------------
    if per_arm:
        tau_hat = (
            (Y_L[t1_L].mean()
             + lam_1 * (Yh_all[t1_all].mean() - Yh_L[t1_L].mean()))
            - (Y_L[t0_L].mean()
               + lam_0 * (Yh_all[t0_all].mean() - Yh_L[t0_L].mean()))
        )
    else:
        tau_hat = tau_Y_L + lam_1 * (tau_Yh_all - tau_Yh_L)

    # --- variance ----------------------------------------------------------
    var_hat = 0.0
    for t_val, n_t, mask_L_t, nL_t, sig2_Yh_t, lam_t in [
        (1, n1, t1_L, nL1, sig2_Yh_1, lam_1),
        (0, n0, t0_L, nL0, sig2_Yh_0, lam_0),
    ]:
        # Residuals Y - lam_t * Yhat on labeled units in arm t
        residuals_t = Y_L[mask_L_t] - lam_t * Yh_L[mask_L_t]
        var_resid_t = float(np.var(residuals_t, ddof=1))

        var_hat += lam_t ** 2 * sig2_Yh_t / n_t + var_resid_t / nL_t

    delta_V = 0.0
    if variance == "exact":
        delta_V = ppi_plus_corrected_variance(
            Y_L, Yh_L, Yh_all, T_L, T,
            lam_1 if (not per_arm) else {0: lam_0, 1: lam_1},
        )
        var_hat += delta_V

    result = _make_result(tau_hat, var_hat, cfg)
    # Historical key names: "lambda_hat" for PPI++, "beta_hat" for GREG.
    key = "beta_hat" if method_id == 4 else "lambda_hat"
    result[key] = float(lam_1) if not per_arm else float(0.5 * (lam_1 + lam_0))
    result["lambda_1"] = float(lam_1)
    result["lambda_0"] = float(lam_0)
    if variance == "exact":
        result["delta_V"] = float(delta_V)

    return result


# ---------------------------------------------------------------------------
# Method 5: Composite Proxy (Tripuraneni et al. 2024)
# ---------------------------------------------------------------------------

def composite_proxy(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    **kwargs: Any,
) -> Dict[str, float]:
    """Method 5: Composite Proxy.

    Uses historical experiments to estimate an optimal weight w such that
    tau_hat = w * tau_S (single surrogate case).

    kwargs:
        historical_experiments: list of dicts, each with keys:
            - tau_S_hist: float or np.ndarray — surrogate ATE(s) from hist. exp.
            - tau_hist: float — primary ATE from historical experiment
        If not provided or empty, falls back to surrogate_index.

    Single-surrogate case (S is 1-d):
        w = sum_k(tau_S^(k) * tau^(k)) / sum_k(tau_S^(k))^2
        tau_hat = w * tau_S_current
        Var = w^2 * Var(tau_S)

    Multi-surrogate case (S is J-d):
        w = (sum_k tau_S^(k) tau_S^(k)')^{-1} sum_k tau_S^(k) tau^(k)
        tau_hat = w' @ tau_S
        Var = w' @ Sigma_{tau_S} @ w
    """
    hist_exps: Optional[List[Dict]] = kwargs.get("historical_experiments", None)

    if hist_exps is None or len(hist_exps) == 0:
        # Fall back to surrogate index
        return surrogate_index(T, S, Y, Y_hat, labeled_mask, **kwargs)

    # Determine if multi-surrogate (S has shape (n, J)) or single (S shape (n,))
    if S.ndim == 1:
        return _composite_proxy_single(T, S, Y, Y_hat, labeled_mask, hist_exps)
    else:
        return _composite_proxy_multi(T, S, Y, Y_hat, labeled_mask, hist_exps)


def _composite_proxy_single(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    hist_exps: List[Dict],
) -> Dict[str, float]:
    """Single-surrogate composite proxy.

    w = sum_k(tau_S^(k) * tau^(k)) / sum_k(tau_S^(k))^2
    tau_hat = w * tau_S
    Var = w^2 * Var(tau_S)
    """
    # Compute current surrogate ATE
    tau_S = _diff_in_means(S, T)
    var_tau_S = _neyman_variance(S, T)

    # Estimate weight from historical experiments
    numerator = 0.0
    denominator = 0.0
    for exp in hist_exps:
        tau_S_k = float(exp["tau_S_hist"])
        tau_k = float(exp["tau_hist"])
        numerator += tau_S_k * tau_k
        denominator += tau_S_k ** 2

    if denominator == 0:
        # Degenerate case: fall back to surrogate index
        return surrogate_index(T, S, Y, Y_hat, labeled_mask)

    w = numerator / denominator

    tau_hat = w * tau_S
    var_hat = w ** 2 * var_tau_S

    result = _make_result(tau_hat, var_hat, _config(method_id=5))
    result["w_hat"] = float(w)
    return result


def _composite_proxy_multi(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    hist_exps: List[Dict],
) -> Dict[str, float]:
    """Multi-surrogate composite proxy.

    S has shape (n, J).
    w = (sum_k tau_S^(k) tau_S^(k)')^{-1} sum_k tau_S^(k) tau^(k)
    tau_hat = w' @ tau_S
    Var = w' @ Sigma_{tau_S} @ w
    """
    n, J = S.shape
    t1 = T == 1
    t0 = T == 0
    n1 = int(t1.sum())
    n0 = int(t0.sum())

    if n1 < 2 or n0 < 2:
        return labeled_only(T, S[:, 0] if J > 0 else S, Y, Y_hat, labeled_mask)

    # Current surrogate ATEs: tau_S_j = mean(S_j | T=1) - mean(S_j | T=0)
    tau_S_vec = S[t1].mean(axis=0) - S[t0].mean(axis=0)  # shape (J,)

    # Variance of tau_S: Sigma_{tau_S} = Sigma_{S,1}/n_1 + Sigma_{S,0}/n_0
    Sigma_S1 = np.cov(S[t1], rowvar=False, ddof=1)  # (J, J)
    Sigma_S0 = np.cov(S[t0], rowvar=False, ddof=1)
    Sigma_tau_S = Sigma_S1 / n1 + Sigma_S0 / n0  # (J, J)

    # Estimate weights from historical experiments
    Gamma = np.zeros((J, J))
    mu = np.zeros(J)
    for exp in hist_exps:
        tau_S_k = np.asarray(exp["tau_S_hist"], dtype=float).ravel()
        tau_k = float(exp["tau_hist"])
        if len(tau_S_k) != J:
            raise ValueError(
                f"Historical experiment has {len(tau_S_k)} surrogates, "
                f"but current data has {J}."
            )
        Gamma += np.outer(tau_S_k, tau_S_k)
        mu += tau_S_k * tau_k

    # w = Gamma^{-1} mu
    try:
        w = np.linalg.solve(Gamma, mu)
    except np.linalg.LinAlgError:
        return surrogate_index(T, S[:, 0], Y, Y_hat, labeled_mask)

    tau_hat = float(w @ tau_S_vec)
    var_hat = float(w @ Sigma_tau_S @ w)

    result = _make_result(tau_hat, var_hat, _config(method_id=5))
    result["w_hat"] = w.tolist()
    return result


# ---------------------------------------------------------------------------
# Method 6: AIPW (Augmented Inverse Probability Weighted)
# ---------------------------------------------------------------------------

def aipw(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    **kwargs: Any,
) -> Dict[str, float]:
    """Method 6: Augmented Inverse Probability Weighted (AIPW) estimator.

    Doubly-robust estimator for ATE under missing outcomes.  Combines an
    outcome model (Y_hat) with an observation-probability model (propensity
    for being labeled).

    phi_i = (M_i / pi_i) * (Y_i - Y_hat_i) + Y_hat_i   (DR imputed outcome)
    tau_hat = mean(phi | T=1) - mean(phi | T=0)

    Under MCAR, pi_i = n_L / n (constant).
    Under MAR, pi_i is estimated via logistic regression of M on S.

    Variance:  Var(phi | T=1)/n_1 + Var(phi | T=0)/n_0

    kwargs:
        propensity_model: str, "constant" (default, for MCAR) or "logistic"
            (for MAR).  When "logistic", estimates pi(S) via logistic
            regression of M on (1, S, S^2).
    """
    propensity_model = kwargs.get("propensity_model", "constant")

    n = len(T)
    M = labeled_mask.astype(np.float64)

    # Estimate observation propensity
    if propensity_model == "logistic":
        pi = _estimate_propensity_logistic(S, M)
    else:
        # Constant propensity (MCAR)
        n_L = int(labeled_mask.sum())
        pi = np.full(n, n_L / n)

    # Clip propensity to avoid extreme weights
    pi = np.clip(pi, 0.01, 0.99)

    # AIPW score per unit: augmented outcome imputation
    # For missing Y: phi_i = (M_i / pi_i) * (Y_i - Y_hat_i) + Y_hat_i
    # This gives the doubly-robust imputed outcome for each unit.
    correction = (M / pi) * (Y - Y_hat)

    # For unlabeled units, Y may be arbitrary; the M=0 zeroes out the
    # correction term, so we need to make sure (Y - Y_hat) doesn't produce
    # NaN.  Y should be valid for all units in our DGP setup, but guard
    # against NaN just in case.
    correction = np.where(np.isfinite(correction), correction, 0.0)

    phi = correction + Y_hat  # DR-imputed outcome for each unit

    # ATE = mean(phi | T=1) - mean(phi | T=0)
    t1 = T == 1
    t0 = T == 0
    n1 = int(t1.sum())
    n0 = int(t0.sum())

    cfg = _config(method_id=6,
                  protocol=kwargs.get("protocol", DEFAULT_PROTOCOL))

    if n1 == 0 or n0 == 0:
        return _make_result(np.nan, np.inf, cfg)

    tau_hat = float(phi[t1].mean() - phi[t0].mean())

    # Influence-function variance:
    # IF_i = (2T_i - 1) * phi_i - tau_hat  (simplified for balanced design)
    # More precisely, decompose into arm-specific terms:
    # Var = Var(phi | T=1)/n_1 + Var(phi | T=0)/n_0
    var_hat = float(np.var(phi[t1], ddof=1) / n1 + np.var(phi[t0], ddof=1) / n0)

    result = _make_result(tau_hat, var_hat, cfg)
    result["propensity_model"] = propensity_model
    return result


def _estimate_propensity_logistic(
    S: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """Estimate P(M=1 | S) via logistic regression on (1, S, S^2).

    For multi-surrogate (S has shape (n, J)), uses (1, S_1, ..., S_J,
    S_1^2, ..., S_J^2).
    """
    from sklearn.linear_model import LogisticRegression

    if S.ndim == 1:
        design = np.column_stack([S, S ** 2])
    else:
        design = np.column_stack([S, S ** 2])

    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(design, M.astype(int))
    pi = lr.predict_proba(design)[:, 1]
    return pi


# ---------------------------------------------------------------------------
# Method 7: PPI++ with Bootstrap Variance
# ---------------------------------------------------------------------------

def ppi_plus_bootstrap(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    **kwargs: Any,
) -> Dict[str, float]:
    """Method 7: PPI++ with bootstrap variance correction.

    Addresses the PPI++ variance anomaly at high labeled fractions
    (Section 3.5) by replacing the analytic variance with a bootstrap
    estimate.

    Procedure:
        1. Point estimate: use the original PPI++ point estimate.
        2. For B bootstrap replicates:
           a. Stratified resample of labeled set (preserving T structure).
           b. Resample unlabeled set with replacement.
           c. Re-estimate lambda_hat on the bootstrap sample.
           d. Compute tau_hat_boot using bootstrap lambda and data.
        3. Variance: bootstrap variance of tau_hat_boot across replicates.
        4. CI: percentile interval or normal-based using bootstrap variance.

    kwargs:
        B: int, number of bootstrap replicates (default 200).
        ci_method: str, "percentile" (default) or "normal".
    """
    B = kwargs.get("B", 200)
    ci_method = kwargs.get("ci_method", "percentile")
    rng = np.random.default_rng(kwargs.get("boot_seed", None))

    # --- Original PPI++ point estimate ---
    original_result = ppi_plus(
        T, S, Y, Y_hat, labeled_mask,
        lambda_rule="plugin", clip=True, per_arm=False, variance="plugin",
    )
    tau_hat_orig = original_result["tau_hat"]

    # --- Setup indices ---
    labeled_idx = np.where(labeled_mask)[0]
    unlabeled_idx = np.where(~labeled_mask)[0]
    n = len(T)
    n_L = len(labeled_idx)
    n_U = len(unlabeled_idx)

    # Stratified indices within labeled set
    labeled_T = T[labeled_idx]
    labeled_t1_idx = labeled_idx[labeled_T == 1]
    labeled_t0_idx = labeled_idx[labeled_T == 0]
    nL1 = len(labeled_t1_idx)
    nL0 = len(labeled_t0_idx)

    # Edge case: insufficient data for bootstrap
    if nL1 < 2 or nL0 < 2:
        return original_result

    # --- Bootstrap ---
    tau_boots = np.empty(B)
    for b in range(B):
        # Stratified resample of labeled set
        boot_L_t1 = rng.choice(labeled_t1_idx, size=nL1, replace=True)
        boot_L_t0 = rng.choice(labeled_t0_idx, size=nL0, replace=True)
        boot_L = np.concatenate([boot_L_t1, boot_L_t0])

        # Resample unlabeled set
        if n_U > 0:
            boot_U = rng.choice(unlabeled_idx, size=n_U, replace=True)
            boot_all = np.concatenate([boot_L, boot_U])
        else:
            boot_all = boot_L

        # Extract bootstrap data
        T_b = T[boot_all]
        Y_hat_b = Y_hat[boot_all]

        # labeled_mask for bootstrap: first n_L entries are labeled
        mask_b = np.zeros(len(boot_all), dtype=bool)
        mask_b[:n_L] = True

        T_L_b = T_b[mask_b]
        Y_L_b = Y[boot_L]
        Yh_L_b = Y_hat_b[mask_b]

        # Re-estimate lambda on bootstrap sample
        t1_L_b = T_L_b == 1
        t0_L_b = T_L_b == 0
        nL1_b = int(t1_L_b.sum())
        nL0_b = int(t0_L_b.sum())

        t1_all_b = T_b == 1
        t0_all_b = T_b == 0
        n1_b = int(t1_all_b.sum())
        n0_b = int(t0_all_b.sum())

        if nL1_b < 2 or nL0_b < 2 or n1_b < 2 or n0_b < 2:
            tau_boots[b] = tau_hat_orig
            continue

        gamma_1_b = _sample_cov(Y_L_b[t1_L_b], Yh_L_b[t1_L_b])
        gamma_0_b = _sample_cov(Y_L_b[t0_L_b], Yh_L_b[t0_L_b])
        sig2_1_b = float(np.var(Y_hat_b[t1_all_b], ddof=1))
        sig2_0_b = float(np.var(Y_hat_b[t0_all_b], ddof=1))

        num_b = gamma_1_b / nL1_b + gamma_0_b / nL0_b
        den_b = sig2_1_b * (1.0 / n1_b + 1.0 / nL1_b) + sig2_0_b * (1.0 / n0_b + 1.0 / nL0_b)

        if den_b > 0:
            lam_b = float(np.clip(num_b / den_b, 0.0, 1.0))
        else:
            lam_b = 0.0

        # Compute bootstrap PPI++ estimate
        tau_Y_L_b = _diff_in_means(Y_L_b, T_L_b)
        tau_Yh_all_b = _diff_in_means(Y_hat_b, T_b)
        tau_Yh_L_b = _diff_in_means(Yh_L_b, T_L_b)

        tau_boots[b] = tau_Y_L_b + lam_b * (tau_Yh_all_b - tau_Yh_L_b)

    # --- Aggregate bootstrap results ---
    var_hat = float(np.var(tau_boots, ddof=1))
    se_hat = np.sqrt(max(var_hat, 0.0))

    if ci_method == "percentile":
        ci_lower = float(np.percentile(tau_boots, 2.5))
        ci_upper = float(np.percentile(tau_boots, 97.5))
    else:
        ci_lower = float(tau_hat_orig - Z_ALPHA * se_hat)
        ci_upper = float(tau_hat_orig + Z_ALPHA * se_hat)

    result = {
        "tau_hat": float(tau_hat_orig),
        "var_hat": float(var_hat),
        "se_hat": float(se_hat),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "config": _config(
            method_id=7,
            protocol=kwargs.get("protocol", DEFAULT_PROTOCOL),
            lambda_rule="plugin", clip=True, per_arm=False,
            variance="bootstrap",
        ),
    }
    result["lambda_hat"] = original_result.get("lambda_hat", np.nan)
    result["ci_method"] = ci_method
    result["B"] = B

    return result


# ---------------------------------------------------------------------------
# Surrogacy Diagnostic Test
# ---------------------------------------------------------------------------

def surrogacy_test(
    tau_si: float,
    tau_ppi: float,
    var_si: Optional[float] = None,
    var_ppi: Optional[float] = None,
    cov_si_ppi: Optional[float] = None,
    alternative: str = "two-sided",
    joint: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Formal diagnostic test for surrogacy violations.

    Tests H₀: surrogacy holds AND model correctly specified
    vs H₁: surrogacy violated or model misspecified.

    Under H₀, both τ̂_SI and τ̂_PPI++ target τ, so D̂ = τ̂_PPI++ - τ̂_SI
    has mean zero. Under H₁ with direct effect δ > 0, E[D̂] = δ > 0.

    Parameters
    ----------
    tau_si : float
        Surrogate index point estimate.
    tau_ppi : float
        PPI++ point estimate.
    var_si : float
        Estimated variance of τ̂_SI.
    var_ppi : float
        Estimated variance of τ̂_PPI++.
    cov_si_ppi : float
        Estimated covariance between τ̂_SI and τ̂_PPI++.
    alternative : str
        "two-sided" (default, H₁: δ ≠ 0), "greater" (one-sided, H₁: δ > 0),
        or "less". The paper reports the two-sided test throughout, so it is
        the default here; pass alternative="greater" explicitly for the
        one-sided size/power comparisons in Section 2.6.
    joint : dict or None
        The output of `joint_influence_cov`.  When supplied it overrides
        var_si, var_ppi and cov_si_ppi with the learned-index sandwich
        entries, and Var(D̂) is read from joint["var_D"] directly (which is
        the same number, but computed without the subtraction rounding).
        This is the default path; see `surrogacy_test_sandwich`.

    Returns
    -------
    dict with keys:
        D_hat : float — discrepancy statistic τ̂_PPI++ - τ̂_SI
        SE_D : float — standard error of D̂
        T_n : float — test statistic D̂ / SE(D̂)
        p_value : float — p-value under H₀
        var_D : float — variance of D̂
    """
    D_hat = tau_ppi - tau_si

    if joint is not None:
        var_si = joint["var_si"]
        var_ppi = joint["var_ppi"]
        cov_si_ppi = joint["cov_si_ppi"]
    if var_si is None or var_ppi is None or cov_si_ppi is None:
        raise ValueError(
            "surrogacy_test needs var_si, var_ppi and cov_si_ppi, or a "
            "joint=... dict from joint_influence_cov."
        )

    # Var(D̂) = Var(τ̂_PPI++) + Var(τ̂_SI) - 2 Cov(τ̂_SI, τ̂_PPI++)
    var_D_raw = var_ppi + var_si - 2.0 * cov_si_ppi
    var_D = max(var_D_raw, 0.0)  # guard against numerical negatives

    if var_D <= 0.0:
        # A non-positive Var(D̂) means the estimated covariance swamped the
        # two marginal variances (a finite-sample artifact, most common at
        # high π_L where τ̂_SI and τ̂_PPI++ are nearly collinear). The test
        # statistic is forced to 0, i.e. the discrepancy is treated as
        # undetectable, which makes the test conservative rather than
        # spuriously significant.
        warnings.warn(
            f"surrogacy_test: Var(D_hat) = {var_D_raw:.6g} <= 0 "
            f"(var_si={var_si:.6g}, var_ppi={var_ppi:.6g}, "
            f"cov_si_ppi={cov_si_ppi:.6g}); forcing T_n = 0 "
            f"(p-value will be non-significant).",
            RuntimeWarning,
            stacklevel=2,
        )

    SE_D = np.sqrt(var_D) if var_D > 0 else np.inf
    T_n = D_hat / SE_D if SE_D > 0 and np.isfinite(SE_D) else 0.0

    if alternative == "greater":
        p_value = 1.0 - stats.norm.cdf(T_n)
    elif alternative == "less":
        p_value = stats.norm.cdf(T_n)
    else:  # two-sided
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(T_n)))

    return {
        "D_hat": float(D_hat),
        "SE_D": float(SE_D),
        "T_n": float(T_n),
        "p_value": float(p_value),
        "var_D": float(var_D),
    }


def surrogacy_test_sandwich(
    T: np.ndarray,
    Y: np.ndarray,
    labeled_mask: np.ndarray,
    design: Dict[str, Any],
    lambda_hat: float,
    tau_si: float,
    tau_ppi: float,
    p_hat: Optional[float] = None,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """SI--PPI++ estimator-disagreement diagnostic with the joint sandwich.

    Builds the 2 x 2 covariance of (tau_SI, tau_PPI) for the LEARNED linear
    index with `joint_influence_cov` and feeds it to `surrogacy_test`, so
    SE(D̂) accounts for the first-stage estimation of beta.  The returned dict
    carries the usual keys plus "joint" (the full `joint_influence_cov`
    output).
    """
    joint = joint_influence_cov(
        T, Y, labeled_mask, design, lambda_hat, p_hat=p_hat,
    )
    out = surrogacy_test(tau_si, tau_ppi, alternative=alternative, joint=joint)
    out["joint"] = joint
    return out


def estimate_cov_si_ppi(
    T: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    lambda_hat: float,
    design: Optional[Dict[str, Any]] = None,
    p_hat: Optional[float] = None,
) -> float:
    """Estimate Cov(τ̂_SI, τ̂_PPI++).

    With ``design`` supplied (the dict from
    `train_prediction_model(..., return_design=True)`) this returns the
    learned-index sandwich entry `joint_influence_cov(...)["cov_si_ppi"]`,
    which is the default.

    Without a design it falls back to the FIXED-f expression
    (Section B.3), which treats the predictions as a known function:

        Cov(τ̂_SI, τ̂_PPI++) = Σ_t [λ* σ²_{Ŷ,t}/n_t + γ_t/n_t]
                            = Σ_t [(λ* σ²_{Ŷ,t} + γ_t) / n_t]

    where γ_t = Cov(Y - λ*Ŷ, Ŷ | T=t) on labeled data.  That path is kept so
    the old numbers can be reproduced, and because it is the only option for
    a non-linear (GBT) index.

    Parameters
    ----------
    T : treatment indicators (all units)
    Y : primary outcomes (all units, but only labeled used for γ)
    Y_hat : predicted outcomes (all units)
    labeled_mask : boolean mask
    lambda_hat : estimated PPI++ tuning parameter
    design : design dict from train_prediction_model(..., return_design=True)
    p_hat : P(T = 1); estimated by mean(T) when None

    Returns
    -------
    float : estimated covariance
    """
    if design is not None and design.get("g", None) is not None:
        return float(joint_influence_cov(
            T, Y, labeled_mask, design, lambda_hat, p_hat=p_hat,
        )["cov_si_ppi"])

    T_L = T[labeled_mask]
    Y_L = Y[labeled_mask]
    Yh_L = Y_hat[labeled_mask]

    cov_total = 0.0
    for t_val in [0, 1]:
        t_all = T == t_val
        n_t = int(t_all.sum())

        t_L = T_L == t_val
        nL_t = int(t_L.sum())

        if n_t < 2 or nL_t < 2:
            continue

        # σ²_{Ŷ,t} on all units in arm t
        sig2_Yh_t = float(np.var(Y_hat[t_all], ddof=1))

        # γ_t = Cov(Y - λ*Ŷ, Ŷ | T=t) on labeled data
        resid_L_t = Y_L[t_L] - lambda_hat * Yh_L[t_L]
        gamma_t = _sample_cov(resid_L_t, Yh_L[t_L])

        cov_total += (lambda_hat * sig2_Yh_t + gamma_t) / n_t

    return cov_total


# ---------------------------------------------------------------------------
# Adaptive Hybrid Estimator
# ---------------------------------------------------------------------------

def hybrid_estimator(
    tau_si: float,
    tau_ppi: float,
    var_si: float,
    var_ppi: float,
    cov_si_ppi: float,
    c: float = 1.5,
    alpha: float = 0.05,
    M: int = 10_000,
    rng_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Adaptive hybrid estimator.

    Interpolates between τ̂_SI and τ̂_PPI++ using a Cauchy-kernel weight
    based on the surrogacy diagnostic test statistic.

    w = c / (c + T²_n), where T_n = D̂ / SE(D̂).
    τ̂_hybrid = w · τ̂_SI + (1 - w) · τ̂_PPI++.

    Under H₀ (surrogacy holds): T_n ~ N(0,1), w → 1, hybrid → SI.
    Under H₁ (large violation): T²_n >> c, w → 0, hybrid → PPI++.

    CI is constructed via simulation-calibrated approach (Proposition 7):
    draw from estimated bivariate normal, compute hybrid on each draw,
    take percentiles.

    Parameters
    ----------
    tau_si : float
        Surrogate index point estimate.
    tau_ppi : float
        PPI++ point estimate.
    var_si : float
        Estimated variance of τ̂_SI.
    var_ppi : float
        Estimated variance of τ̂_PPI++.
    cov_si_ppi : float
        Estimated covariance between τ̂_SI and τ̂_PPI++.
    c : float
        Cauchy-kernel tuning parameter (default 1.5, minimax optimal).
    alpha : float
        Significance level for CI (default 0.05 for 95% CI).
    M : int
        Number of simulation draws for CI (default 10,000).
    rng_seed : int or None
        Random seed for reproducibility of simulation draws.

    Returns
    -------
    dict with keys:
        tau_hybrid : float — point estimate
        w : float — Cauchy-kernel weight (1 = full SI, 0 = full PPI++)
        ci_lower : float — lower CI bound (simulation-calibrated)
        ci_upper : float — upper CI bound (simulation-calibrated)
        T_n : float — test statistic used for weighting
        SE_D : float — standard error of discrepancy
    """
    # Compute test statistic. Only T_n and SE_D are used below, so the
    # choice of `alternative` does not affect the hybrid estimate; it is
    # passed explicitly so the weighting is pinned to one convention.
    test_result = surrogacy_test(
        tau_si, tau_ppi, var_si, var_ppi, cov_si_ppi,
        alternative="two-sided",
    )
    T_n = test_result["T_n"]
    SE_D = test_result["SE_D"]

    # Cauchy-kernel weight
    w = c / (c + T_n ** 2)

    # Point estimate
    tau_hybrid = w * tau_si + (1.0 - w) * tau_ppi

    # Simulation-calibrated CI (Proposition 7)
    Sigma = np.array([
        [var_si, cov_si_ppi],
        [cov_si_ppi, var_ppi],
    ])

    # Ensure Sigma is positive semi-definite
    eigvals = np.linalg.eigvalsh(Sigma)
    if np.any(eigvals < 0):
        # Project to nearest PSD matrix
        eigvals_clipped = np.maximum(eigvals, 0.0)
        Q = np.linalg.eigh(Sigma)[1]
        Sigma = Q @ np.diag(eigvals_clipped) @ Q.T

    rng = np.random.default_rng(rng_seed)
    mu = np.array([tau_si, tau_ppi])

    try:
        draws = rng.multivariate_normal(mu, Sigma, size=M)
    except np.linalg.LinAlgError:
        # Fallback: independent normals
        draws = np.column_stack([
            rng.normal(tau_si, np.sqrt(max(var_si, 0)), size=M),
            rng.normal(tau_ppi, np.sqrt(max(var_ppi, 0)), size=M),
        ])

    tau_si_draws = draws[:, 0]
    tau_ppi_draws = draws[:, 1]

    # Compute D̂ and T_n on each draw (using fixed SE from original data)
    D_draws = tau_ppi_draws - tau_si_draws
    if SE_D > 0 and np.isfinite(SE_D):
        T_n_draws = D_draws / SE_D
    else:
        T_n_draws = np.zeros(M)

    w_draws = c / (c + T_n_draws ** 2)
    tau_hybrid_draws = w_draws * tau_si_draws + (1.0 - w_draws) * tau_ppi_draws

    ci_lower = float(np.percentile(tau_hybrid_draws, 100 * alpha / 2))
    ci_upper = float(np.percentile(tau_hybrid_draws, 100 * (1.0 - alpha / 2)))

    return {
        "tau_hybrid": float(tau_hybrid),
        "w": float(w),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "T_n": float(T_n),
        "SE_D": float(SE_D),
    }


# ---------------------------------------------------------------------------
# CUPED-Adjusted Labeled-Only Baseline
# ---------------------------------------------------------------------------

def labeled_only_cuped(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    X: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """CUPED-adjusted labeled-only estimator (supplementary baseline).

    Reduces variance by regressing out pre-treatment covariates X from Y
    on labeled units, then running difference-in-means on the adjusted
    outcome.

    Steps:
        1. Extract labeled units.
        2. Compute theta via OLS: theta = (X_L' X_L)^{-1} X_L' Y_L
           (multivariate regression if X has multiple columns).
        3. Compute Y_adj = Y_L - X_L @ theta.
        4. tau_hat = mean(Y_adj | T=1) - mean(Y_adj | T=0).
        5. var_hat = Var(Y_adj | T=1)/n1 + Var(Y_adj | T=0)/n0  (Neyman).

    If X is None, falls back to unadjusted labeled-only.
    """
    T_L = T[labeled_mask]
    Y_L = Y[labeled_mask]

    if X is None:
        return labeled_only(T, S, Y, Y_hat, labeled_mask, **kwargs)

    X_L = X[labeled_mask]
    if X_L.ndim == 1:
        X_L = X_L.reshape(-1, 1)

    n_L, p = X_L.shape
    if n_L < p + 2:
        # Not enough data for regression; fall back to unadjusted
        return labeled_only(T, S, Y, Y_hat, labeled_mask, **kwargs)

    # OLS: theta = (X'X)^{-1} X'Y
    XtX = X_L.T @ X_L
    XtX += 1e-10 * np.eye(p)  # ridge for numerical stability
    XtY = X_L.T @ Y_L
    theta = np.linalg.solve(XtX, XtY)

    # Adjusted outcome
    Y_adj = Y_L - X_L @ theta

    # Difference-in-means on adjusted outcome
    tau_hat = _diff_in_means(Y_adj, T_L)
    var_hat = _neyman_variance(Y_adj, T_L)

    return _make_result(tau_hat, var_hat, _config(method_id=0))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def cross_ppi(
    T: np.ndarray,
    S: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    labeled_mask: np.ndarray,
    seed: int = 0,
    **kwargs: Any,
) -> Dict[str, float]:
    """Cross-PPI estimator.

    Splits the labeled data into two halves. In each fold:
      - Half A trains the prediction model f_A.
      - Half B computes the PPI++ rectifier using f_A's predictions.
    Then swaps roles. The final estimate averages the two folds.

    This avoids overfitting bias from using the same labeled data for both
    training f and computing the correction term.

    Parameters
    ----------
    T : treatment indicators (all units)
    S : surrogates (all units)
    X : covariates (all units)
    Y : primary outcomes (all units)
    labeled_mask : boolean mask (True = labeled)
    seed : random seed for fold splitting
    """
    from src.simulations.simulation import train_prediction_model as _train_pred

    rng = np.random.default_rng(seed + 9999)

    n = len(T)
    labeled_idx = np.where(labeled_mask)[0]
    n_L = len(labeled_idx)

    # Shuffle and split labeled indices into two halves
    perm = rng.permutation(n_L)
    half = n_L // 2
    fold_A_idx = labeled_idx[perm[:half]]
    fold_B_idx = labeled_idx[perm[half:]]

    tau_hats = []
    var_hats = []

    for train_idx, rect_idx in [(fold_A_idx, fold_B_idx),
                                (fold_B_idx, fold_A_idx)]:
        # Build mask for training fold
        train_mask = np.zeros(n, dtype=bool)
        train_mask[train_idx] = True

        # Train prediction model on train fold
        cf_rng = np.random.default_rng(seed + int(hashlib.sha256(str(tuple(train_idx[:5].tolist())).encode()).hexdigest()[:8], 16) % 100000)
        Y_hat = _train_pred(
            S, X, Y, train_mask,
            n_folds=5, rng=cf_rng, prediction_model="ols", seed=seed,
        )

        # Compute PPI++ using rectifier fold for the labeled correction
        rect_mask = np.zeros(n, dtype=bool)
        rect_mask[rect_idx] = True

        # PPI++ components
        T_rect = T[rect_mask]
        Y_rect = Y[rect_mask]
        Yh_rect = Y_hat[rect_mask]

        t1_rect = T_rect == 1
        t0_rect = T_rect == 0
        nL1 = int(t1_rect.sum())
        nL0 = int(t0_rect.sum())

        t1_all = T == 1
        t0_all = T == 0
        n1 = int(t1_all.sum())
        n0 = int(t0_all.sum())

        if nL1 < 2 or nL0 < 2 or n1 < 2 or n0 < 2:
            # Fall back to labeled-only
            return labeled_only(T, S, Y, Y_hat, labeled_mask)

        # Optimal lambda from rectifier fold
        gamma_1 = _sample_cov(Y_rect[t1_rect], Yh_rect[t1_rect])
        gamma_0 = _sample_cov(Y_rect[t0_rect], Yh_rect[t0_rect])
        sig2_1 = float(np.var(Y_hat[t1_all], ddof=1))
        sig2_0 = float(np.var(Y_hat[t0_all], ddof=1))

        num = gamma_1 / nL1 + gamma_0 / nL0
        den = sig2_1 * (1.0 / n1 + 1.0 / nL1) + sig2_0 * (1.0 / n0 + 1.0 / nL0)
        lam = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0

        # Point estimate
        tau_Y_rect = _diff_in_means(Y_rect, T_rect)
        tau_Yh_all = _diff_in_means(Y_hat, T)
        tau_Yh_rect = _diff_in_means(Yh_rect, T_rect)

        tau_hat = tau_Y_rect + lam * (tau_Yh_all - tau_Yh_rect)

        # Variance
        var_hat = 0.0
        for t_val, n_t, mask_rect_t, nL_t in [
            (1, n1, t1_rect, nL1),
            (0, n0, t0_rect, nL0),
        ]:
            t_all_mask = T == t_val
            sig2_t = float(np.var(Y_hat[t_all_mask], ddof=1))
            residuals_t = Y_rect[mask_rect_t] - lam * Yh_rect[mask_rect_t]
            var_resid_t = float(np.var(residuals_t, ddof=1))
            var_hat += lam ** 2 * sig2_t / n_t + var_resid_t / nL_t

        tau_hats.append(tau_hat)
        var_hats.append(var_hat)

    # Average the two folds
    tau_final = np.mean(tau_hats)
    # Variance of average: Var((X1+X2)/2) ≈ (V1+V2)/4 for independent folds
    var_final = np.mean(var_hats) / 2.0  # conservative: treat as avg of 2 indep

    return _make_result(tau_final, var_final, _config(method_id=None))


def ppi_plus_corrected(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    **kwargs: Any,
) -> Dict[str, float]:
    """Method 8: PPI++ with the overlap variance correction.

    """
    kwargs.setdefault("lambda_rule", "plugin")
    kwargs.setdefault("clip", True)
    kwargs.setdefault("per_arm", False)
    kwargs.pop("corrected_variance", None)
    kwargs["variance"] = "exact"
    return _ppi_family(T, S, Y, Y_hat, labeled_mask, method_id=8, **kwargs)


def ppi_plus_exact(
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    **kwargs: Any,
) -> Dict[str, float]:
    """Method 9: ALIAS of method 3 (PPI++) with its default configuration.

    Method 3 is now the exact-rule, unclipped, exact-variance estimator, so id
    9 is a plain alias. It stays registered so that loaders and older result
    files referring to method 9 keep working; METHOD_DISPLAY_NAMES[9] reads
    "PPI++ (alias)" to make the duplication visible in any table.
    """
    return ppi_plus(T, S, Y, Y_hat, labeled_mask, **kwargs)


METHODS = {
    0: labeled_only,
    1: naive_surrogate,
    2: surrogate_index,
    3: ppi_plus,
    4: greg,
    5: composite_proxy,
    6: aipw,
    7: ppi_plus_bootstrap,
    8: ppi_plus_corrected,
    9: ppi_plus_exact,
}

METHOD_NAMES = {
    0: "labeled_only",
    1: "naive_surrogate",
    2: "surrogate_index",
    3: "ppi_plus",
    # Method 4 is the unconstrained linear recalibration of the predictions:
    # the GREG estimator of survey sampling, which is the linear special case
    # of recalibrated PPI (Ji, Lei, Zrnic 2025). The paper labels it GREG
    # because the general recalibration is not implemented here.
    4: "greg",
    5: "composite_proxy",
    6: "aipw",
    7: "ppi_boot",
    8: "ppi_corrected",
    9: "ppi_plus_exact",
}

METHOD_NAME_ALIASES = {
}

# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------

METHOD_DISPLAY_NAMES = {
    0: "Labeled-Only",
    1: "Naive Surrogate",
    2: "Surrogate Index",
    3: "PPI++",
    4: "GREG",
    5: "Composite Proxy",
    6: "AIPW",
    7: "PPI++ Boot",
    9: "PPI++ (alias)",
}

#: Configuration defaults per method id, used by `display_name` to decide
#: which ablation suffixes to print.  Keys that are None never print.
_DISPLAY_DEFAULTS = {
    2: {"si_variance": "sandwich"},
    3: {"lambda_rule": "exact", "clip": False, "per_arm": False,
        "variance": "exact"},
    # GREG's defaults now coincide with PPI++'s: same estimator, kept as a
    # separate id only so older loaders and result files keep resolving.
    4: {"lambda_rule": "exact", "clip": False, "per_arm": False,
        "variance": "exact"},
}

_PPI_SUFFIXES = (
    ("lambda_rule", "plugin", "plug-in lambda"),
    ("variance", "plugin", "plug-in variance"),
    ("clip", True, "clipped"),
    ("per_arm", True, "per-arm lambda"),
)

_SI_SUFFIXES = {
    "plugin": "plug-in variance",
    "delta": "delta-method first stage",
}


def display_name(method_id: int, config: Optional[Dict[str, Any]] = None) -> str:
    """Table label for a (method id, configuration) pair.

    The base label is `METHOD_DISPLAY_NAMES[method_id]`, abbreviated to
    "PPI++" for id 3 and "SI" for id 2; every departure from that method's
    default configuration is appended in parentheses, comma-separated in the
    fixed order (tuning rule, variance, clipping, per-arm).  Examples:

        display_name(3, {...defaults...})              -> "PPI++"
        display_name(3, {"lambda_rule": "plugin"})     -> "PPI++ (plug-in lambda)"
        display_name(3, {"variance": "plugin"})        -> "PPI++ (plug-in variance)"
        display_name(3, {"clip": True})                -> "PPI++ (clipped)"
        display_name(3, {"per_arm": True})             -> "PPI++ (per-arm lambda)"
        display_name(2, {"si_variance": "sandwich"})   -> "SI"
        display_name(2, {"si_variance": "plugin"})     -> "SI (plug-in variance)"
        display_name(2, {"si_variance": "delta"})      -> "SI (delta-method first stage)"

    Unspecified keys take the method's default, so a partial config is fine.
    """
    method_id = int(method_id)
    config = dict(config or {})

    if method_id == 2:
        base = "SI"
        si_var = config.get("si_variance", None)
        if si_var is None:
            si_var = _DISPLAY_DEFAULTS[2]["si_variance"]
        suffix = _SI_SUFFIXES.get(si_var, None)
        return f"{base} ({suffix})" if suffix else base

    if method_id in (3, 4, 8, 9):
        base = "PPI++" if method_id != 4 else "GREG"
        if method_id == 9:
            base = "PPI++"
        defaults = _DISPLAY_DEFAULTS.get(
            method_id, _DISPLAY_DEFAULTS[4 if method_id == 4 else 3]
        )
        parts = []
        for key, flag, label in _PPI_SUFFIXES:
            value = config.get(key, None)
            if value is None:
                value = defaults.get(key, None)
            if value == flag and defaults.get(key, None) != flag:
                parts.append(label)
        return f"{base} ({', '.join(parts)})" if parts else base

    return METHOD_DISPLAY_NAMES.get(method_id, f"Method {method_id}")


def method_id_from_name(name: str) -> int:
    """Resolve a method-name string to its method ID."""
    for m_id, m_name in METHOD_NAMES.items():
        if m_name == name:
            return m_id
    if name in METHOD_NAME_ALIASES:
        return METHOD_NAME_ALIASES[name]
    raise ValueError(f"Unknown method name {name!r}")


def estimate(
    method: int,
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    **kwargs: Any,
) -> Dict[str, float]:
    """Dispatch to a method by its integer ID (0-9).

    Parameters
    ----------
    method : int
        Method ID (0 through 9).
    T, S, Y, Y_hat, labeled_mask : np.ndarray
        Standard inputs (see module docstring).
    **kwargs
        Method-specific parameters.

    Returns
    -------
    dict with tau_hat, var_hat, se_hat, ci_lower, ci_upper (and extras).
    """
    if method not in METHODS:
        raise ValueError(
            f"Unknown method {method}. Choose from {list(METHODS.keys())}."
        )
    result = METHODS[method](T, S, Y, Y_hat, labeled_mask, **kwargs)

    # Guarantee the configuration block on every result, and make sure the
    # method id it reports is the id that was actually dispatched (method 9
    # delegates to method 3, so it would otherwise report 3).
    cfg = result.get("config", None)
    if cfg is None:
        cfg = _config(
            method_id=method,
            protocol=kwargs.get("protocol", DEFAULT_PROTOCOL),
        )
    else:
        cfg = dict(cfg)
    cfg["method_id"] = int(method)
    for key in _CONFIG_KEYS:
        cfg.setdefault(key, None)
    result["config"] = cfg
    return result
