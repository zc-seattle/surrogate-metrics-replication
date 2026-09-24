"""
Data-Generating Processes (DGPs) for surrogate metrics simulation.

Each DGP function takes parameters and returns a dictionary with keys:
    T          : (n,) int array, treatment assignment {0, 1}
    X          : (n, p) float array, pre-treatment covariates
    S          : (n,) float array, surrogate outcome (always observed)
    Y          : (n,) float array, primary outcome (observed for all, but
                 only labeled units should be used in estimation)
    Y_hat      : None (placeholder; filled in by simulation runner after
                 prediction model is trained)
    labeled_mask : (n,) bool array, True for labeled units
    true_tau   : float, the true ATE on Y

DGP 4 additionally returns:
    calibration_data : dict with keys X_cal, S_cal, Y_cal for the
                       calibration-period data

DGP 6 returns a list of per-experiment dicts, plus portfolio-level info.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.special import expit


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _assign_treatment(n: int, rng: np.random.Generator) -> np.ndarray:
    """Bernoulli(0.5) treatment assignment."""
    return rng.binomial(1, 0.5, size=n).astype(np.int32)


def _assign_labels_mcar(
    n: int, pi_L: float, rng: np.random.Generator
) -> np.ndarray:
    """MCAR labeling: randomly select n_L = floor(pi_L * n) units."""
    n_L = int(np.floor(pi_L * n))
    n_L = min(n_L, n)
    indices = rng.permutation(n)
    mask = np.zeros(n, dtype=bool)
    mask[indices[:n_L]] = True
    return mask


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# DGP 1: Valid Surrogate (Baseline)
# ---------------------------------------------------------------------------

def generate_dgp1(
    n: int = 10_000,
    pi_L: float = 0.20,
    alpha_S: float = 5.0,
    beta_SX: float = 1.0,
    gamma_S: float = 0.3,
    sigma_S: float = 2.0,
    alpha_Y: float = 0.0,
    beta_YS: float = 0.5,
    beta_YX: float = 0.2,
    sigma_Y: float = 1.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    DGP 1: Valid Surrogate.

    Treatment affects Y entirely through S.  The Prentice criteria hold.
    True ATE: tau = beta_YS * gamma_S.
    """
    rng = _make_rng(seed)

    X = rng.standard_normal((n, 1))
    T = _assign_treatment(n, rng)

    eps_S = rng.normal(0, sigma_S, size=n)
    S = alpha_S + beta_SX * X[:, 0] + gamma_S * T + eps_S

    eps_Y = rng.normal(0, sigma_Y, size=n)
    Y = alpha_Y + beta_YS * S + beta_YX * X[:, 0] + eps_Y

    labeled_mask = _assign_labels_mcar(n, pi_L, rng)
    true_tau = beta_YS * gamma_S

    return dict(
        T=T, X=X, S=S, Y=Y, Y_hat=None,
        labeled_mask=labeled_mask, true_tau=true_tau,
    )


# ---------------------------------------------------------------------------
# DGP 2: Partial Mediation
# ---------------------------------------------------------------------------

def generate_dgp2(
    n: int = 10_000,
    pi_L: float = 0.20,
    rho: float = 0.3,
    alpha_S: float = 5.0,
    beta_SX: float = 1.0,
    gamma_S: float = 0.3,
    sigma_S: float = 2.0,
    alpha_Y: float = 0.0,
    beta_YS: float = 0.5,
    beta_YX: float = 0.2,
    sigma_Y: float = 1.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    DGP 2: Partial Mediation.

    Treatment has a direct effect on Y (delta) that bypasses S.
    delta = rho / (1 - rho) * beta_YS * gamma_S.
    True ATE: tau = beta_YS * gamma_S + delta.
    """
    rng = _make_rng(seed)

    mediated = beta_YS * gamma_S  # 0.15 with defaults
    if rho >= 1.0:
        raise ValueError("rho must be < 1")
    delta = (rho / (1.0 - rho)) * mediated

    X = rng.standard_normal((n, 1))
    T = _assign_treatment(n, rng)

    eps_S = rng.normal(0, sigma_S, size=n)
    S = alpha_S + beta_SX * X[:, 0] + gamma_S * T + eps_S

    eps_Y = rng.normal(0, sigma_Y, size=n)
    Y = alpha_Y + beta_YS * S + beta_YX * X[:, 0] + delta * T + eps_Y

    labeled_mask = _assign_labels_mcar(n, pi_L, rng)
    true_tau = mediated + delta

    return dict(
        T=T, X=X, S=S, Y=Y, Y_hat=None,
        labeled_mask=labeled_mask, true_tau=true_tau,
    )


# ---------------------------------------------------------------------------
# DGP 3: Heterogeneous Surrogate Quality
# ---------------------------------------------------------------------------

def generate_dgp3(
    n: int = 10_000,
    pi_L: float = 0.20,
    pi_group: float = 0.3,
    beta_YS_high: float = 0.8,
    beta_YS_low: float = 0.2,
    gamma_S_high: float = 0.5,
    gamma_S_low: float = 0.2,
    alpha_S: float = 5.0,
    beta_SX: float = 1.0,
    sigma_S: float = 2.0,
    alpha_Y: float = 0.0,
    beta_YX: float = 0.2,
    sigma_Y: float = 1.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    DGP 3: Heterogeneous Surrogate Quality.

    Two latent groups with different surrogate-outcome slopes and treatment
    effects on S.  A pooled surrogate index is biased (Simpson's paradox).

    True ATE: tau = pi_group * gamma_S_high * beta_YS_high
                  + (1 - pi_group) * gamma_S_low * beta_YS_low.
    """
    rng = _make_rng(seed)

    G = rng.binomial(1, pi_group, size=n)  # 1 = power user (group 1)
    X = rng.standard_normal((n, 1))
    T = _assign_treatment(n, rng)

    gamma_S_i = np.where(G == 1, gamma_S_high, gamma_S_low)
    beta_YS_i = np.where(G == 1, beta_YS_high, beta_YS_low)

    eps_S = rng.normal(0, sigma_S, size=n)
    S = alpha_S + beta_SX * X[:, 0] + gamma_S_i * T + eps_S

    eps_Y = rng.normal(0, sigma_Y, size=n)
    Y = alpha_Y + beta_YS_i * S + beta_YX * X[:, 0] + eps_Y

    labeled_mask = _assign_labels_mcar(n, pi_L, rng)
    # Use realized group fractions for finite-sample true ATE
    n_group1 = G.sum()
    true_tau = (
        n_group1 * gamma_S_high * beta_YS_high
        + (n - n_group1) * gamma_S_low * beta_YS_low
    ) / n

    return dict(
        T=T, X=X, S=S, Y=Y, Y_hat=None,
        labeled_mask=labeled_mask, true_tau=true_tau,
    )


# ---------------------------------------------------------------------------
# DGP 4: Temporal Degradation (Distribution Shift)
# ---------------------------------------------------------------------------

def generate_dgp4(
    n: int = 10_000,
    pi_L: float = 0.20,
    n_cal: int = 50_000,
    alpha_S: float = 5.0,
    beta_SX: float = 1.0,
    gamma_S: float = 0.3,
    sigma_S: float = 2.0,
    alpha_Y: float = 0.0,
    beta_YS_0: float = 0.5,
    delta_beta: float = 0.2,
    beta_YX: float = 0.2,
    sigma_Y: float = 1.0,
    mu_shift: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    DGP 4: Temporal Degradation.

    Calibration-period data has slope beta_YS_0.  Experiment-period data has
    slope beta_YS_0 + delta_beta.  Methods using calibration-period f_hat are
    biased; PPI-family methods correct via the labeled set.

    True ATE (experiment period): tau = (beta_YS_0 + delta_beta) * gamma_S.

    Returns an extra key 'calibration_data' with {X_cal, S_cal, Y_cal}.
    """
    rng = _make_rng(seed)

    beta_YS_1 = beta_YS_0 + delta_beta

    # --- Calibration period (no treatment) ---
    X_cal = rng.standard_normal((n_cal, 1))
    eps_S_cal = rng.normal(0, sigma_S, size=n_cal)
    S_cal = alpha_S + beta_SX * X_cal[:, 0] + eps_S_cal
    eps_Y_cal = rng.normal(0, sigma_Y, size=n_cal)
    Y_cal = alpha_Y + beta_YS_0 * S_cal + beta_YX * X_cal[:, 0] + eps_Y_cal

    # --- Experiment period (with drift) ---
    X = rng.normal(mu_shift, 1.0, size=(n, 1))
    T = _assign_treatment(n, rng)
    eps_S = rng.normal(0, sigma_S, size=n)
    S = alpha_S + beta_SX * X[:, 0] + gamma_S * T + eps_S
    eps_Y = rng.normal(0, sigma_Y, size=n)
    Y = alpha_Y + beta_YS_1 * S + beta_YX * X[:, 0] + eps_Y

    labeled_mask = _assign_labels_mcar(n, pi_L, rng)
    true_tau = beta_YS_1 * gamma_S

    return dict(
        T=T, X=X, S=S, Y=Y, Y_hat=None,
        labeled_mask=labeled_mask, true_tau=true_tau,
        calibration_data=dict(X_cal=X_cal, S_cal=S_cal, Y_cal=Y_cal),
    )


# ---------------------------------------------------------------------------
# DGP 5: Sparse / Delayed Outcome
# ---------------------------------------------------------------------------

def _calibrate_eta0(
    q: float, eta_S: float, mu_S: float, var_S: float,
) -> float:
    """
    Calibrate eta_0 so that marginal observation rate is approximately q
    under logistic MAR:  P(M=1 | S) = logit^{-1}(eta_0 + eta_S * S).

    Uses numerical integration over the marginal distribution of S
    (assumed Normal with given mean and variance) via Gauss-Hermite
    quadrature, then solves for eta_0 by bisection.

    Falls back to the probit approximation if scipy is unavailable.
    """
    from scipy.optimize import brentq

    # Gauss-Hermite quadrature for E[expit(eta_0 + eta_S * S)]
    # where S ~ N(mu_S, var_S)
    sd_S = np.sqrt(var_S)
    nodes, weights = np.polynomial.hermite.hermgauss(30)
    # Transform nodes: S = mu_S + sqrt(2) * sd_S * node
    s_vals = mu_S + np.sqrt(2.0) * sd_S * nodes
    # Hermite weights are for exp(-x^2), normalize to get density weights
    w = weights / np.sqrt(np.pi)

    def marginal_rate(eta_0_val):
        return float(np.sum(w * expit(eta_0_val + eta_S * s_vals))) - q

    # Bracket the root: for very negative eta_0 rate -> 0, for very positive -> 1
    eta_0 = brentq(marginal_rate, -20.0, 20.0)
    return eta_0


def generate_dgp5(
    n: int = 10_000,
    q: float = 0.15,
    missingness: str = "MCAR",
    alpha_S: float = 5.0,
    beta_SX: float = 1.0,
    gamma_S: float = 0.3,
    sigma_S: float = 2.0,
    alpha_Y: float = 0.0,
    beta_YS: float = 0.5,
    beta_YX: float = 0.2,
    sigma_Y: float = 1.0,
    eta_S: float = 0.3,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    DGP 5: Sparse / Delayed Outcome.

    Same structural model as DGP 1 (valid surrogate).  Labeling is
    determined by the observation mechanism (MCAR or MAR), not imposed
    exogenously.  pi_L ~ q.

    True ATE: tau = beta_YS * gamma_S.
    """
    rng = _make_rng(seed)

    X = rng.standard_normal((n, 1))
    T = _assign_treatment(n, rng)

    eps_S = rng.normal(0, sigma_S, size=n)
    S = alpha_S + beta_SX * X[:, 0] + gamma_S * T + eps_S

    eps_Y = rng.normal(0, sigma_Y, size=n)
    Y = alpha_Y + beta_YS * S + beta_YX * X[:, 0] + eps_Y

    # Observation mechanism
    if missingness.upper() == "MCAR":
        labeled_mask = rng.random(n) < q
    elif missingness.upper() == "MAR":
        mu_S = alpha_S + 0.5 * gamma_S  # approximate marginal mean of S
        var_S = sigma_S**2 + beta_SX**2  # approximate marginal var of S
        eta_0 = _calibrate_eta0(q, eta_S, mu_S, var_S)
        prob = expit(eta_0 + eta_S * S)
        labeled_mask = rng.random(n) < prob
    else:
        raise ValueError(f"Unknown missingness type: {missingness}")

    # Ensure at least a few labeled units per arm for estimation
    # (in extremely sparse settings, this is a safety guard)
    n_labeled = labeled_mask.sum()
    if n_labeled < 10:
        # Force at least 10 random labels
        extra = rng.choice(np.where(~labeled_mask)[0], size=10 - n_labeled, replace=False)
        labeled_mask[extra] = True

    true_tau = beta_YS * gamma_S

    return dict(
        T=T, X=X, S=S, Y=Y, Y_hat=None,
        labeled_mask=labeled_mask, true_tau=true_tau,
    )


# ---------------------------------------------------------------------------
# DGP 6: Multiple Weak Experiments (Portfolio Setting)
# ---------------------------------------------------------------------------

def generate_dgp6(
    K: int = 100,
    pi_L: float = 0.20,
    n_min: int = 500,
    n_max: int = 5_000,
    pi_0: float = 0.5,
    sigma_tau: float = 0.10,
    alpha_S: float = 5.0,
    beta_SX: float = 1.0,
    sigma_S: float = 2.0,
    alpha_Y: float = 0.0,
    beta_YS: float = 0.5,
    beta_YX: float = 0.2,
    sigma_Y: float = 1.0,
    alpha_decision: float = 0.05,
    rho_mix: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    DGP 6: Multiple Weak Experiments (Portfolio).

    Generates K independent experiments, each with its own sample size
    and true treatment effect.

    When rho_mix > 0, a fraction (30%) of experiments have partial
    mediation with direct-effect share = rho_mix, while the remaining
    70% have valid surrogacy (rho=0).  This tests whether methods that
    are biased for individual experiments with violated surrogacy still
    accumulate worse regret at the portfolio level.

    Returns a dict with:
        experiments : list of K dicts, each with the standard keys
        true_taus   : (K,) array of true ATEs
        oracle_gain : sum of max(tau_k, 0)
        alpha_decision : significance level for launch decisions
    """
    rng = _make_rng(seed)

    # Draw per-experiment parameters
    n_ks = rng.integers(n_min, n_max + 1, size=K)
    is_null = rng.random(K) < pi_0
    tau_mediated_ks = np.where(
        is_null,
        0.0,
        rng.normal(0, sigma_tau, size=K),
    )

    # Determine which experiments have partial mediation
    if rho_mix > 0:
        has_direct_effect = rng.random(K) < 0.3
    else:
        has_direct_effect = np.zeros(K, dtype=bool)

    # Compute true ATEs including any direct effects
    tau_ks = np.empty(K)
    for k in range(K):
        mediated_k = tau_mediated_ks[k]
        if has_direct_effect[k] and rho_mix < 1.0:
            delta_k = (rho_mix / (1.0 - rho_mix)) * mediated_k
        else:
            delta_k = 0.0
        tau_ks[k] = mediated_k + delta_k

    experiments: List[Dict[str, Any]] = []
    for k in range(K):
        n_k = int(n_ks[k])
        mediated_k = float(tau_mediated_ks[k])

        # gamma_S_k such that beta_YS * gamma_S_k = mediated_k
        gamma_S_k = mediated_k / beta_YS if beta_YS != 0.0 else 0.0

        # Direct effect for this experiment
        if has_direct_effect[k] and rho_mix < 1.0:
            delta_k = (rho_mix / (1.0 - rho_mix)) * mediated_k
        else:
            delta_k = 0.0

        # Derive a deterministic per-experiment seed
        exp_seed = seed * 100_000 + k + 1

        exp_rng = _make_rng(exp_seed)

        X_k = exp_rng.standard_normal((n_k, 1))
        T_k = _assign_treatment(n_k, exp_rng)

        eps_S_k = exp_rng.normal(0, sigma_S, size=n_k)
        S_k = alpha_S + beta_SX * X_k[:, 0] + gamma_S_k * T_k + eps_S_k

        eps_Y_k = exp_rng.normal(0, sigma_Y, size=n_k)
        Y_k = (alpha_Y + beta_YS * S_k + beta_YX * X_k[:, 0]
               + delta_k * T_k + eps_Y_k)

        labeled_mask_k = _assign_labels_mcar(n_k, pi_L, exp_rng)

        experiments.append(dict(
            T=T_k, X=X_k, S=S_k, Y=Y_k, Y_hat=None,
            labeled_mask=labeled_mask_k, true_tau=float(tau_ks[k]),
        ))

    oracle_gain = float(np.sum(np.maximum(tau_ks, 0.0)))

    return dict(
        experiments=experiments,
        true_taus=tau_ks,
        oracle_gain=oracle_gain,
        alpha_decision=alpha_decision,
    )


# ---------------------------------------------------------------------------
# DGP 7: Multi-Surrogate
# ---------------------------------------------------------------------------

def generate_dgp7(
    n: int = 10_000,
    pi_L: float = 0.20,
    gamma_1: float = 0.3,
    gamma_2: float = 0.2,
    gamma_3: float = 0.1,
    beta_1: float = 0.3,
    beta_2: float = 0.4,
    beta_3: float = 0.2,
    beta_YX: float = 0.2,
    sigma_1: float = 2.0,
    sigma_2: float = np.sqrt(2.0),
    sigma_3: float = 1.0,
    sigma_Y: float = 1.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    DGP 7: Multi-Surrogate.

    Three surrogates capture different aspects of the treatment effect:
        S1 (engagement/sessions):  S1 = 5 + X + gamma_1 * T + eps_1
        S2 (content/completions):  S2 = 3 + 0.5*X + gamma_2 * T + eps_2
        S3 (social/shares):        S3 = 1 + 0.3*X + gamma_3 * T + eps_3
        Y = beta_1*S1 + beta_2*S2 + beta_3*S3 + beta_YX*X + eps_Y

    True ATE: tau = beta_1*gamma_1 + beta_2*gamma_2 + beta_3*gamma_3.

    Returns S as (n, 3) array with columns [S1, S2, S3].
    """
    rng = _make_rng(seed)

    X = rng.standard_normal((n, 1))
    T = _assign_treatment(n, rng)

    eps_1 = rng.normal(0, sigma_1, size=n)
    eps_2 = rng.normal(0, sigma_2, size=n)
    eps_3 = rng.normal(0, sigma_3, size=n)
    eps_Y = rng.normal(0, sigma_Y, size=n)

    S1 = 5.0 + X[:, 0] + gamma_1 * T + eps_1
    S2 = 3.0 + 0.5 * X[:, 0] + gamma_2 * T + eps_2
    S3 = 1.0 + 0.3 * X[:, 0] + gamma_3 * T + eps_3

    Y = (beta_1 * S1 + beta_2 * S2 + beta_3 * S3
         + beta_YX * X[:, 0] + eps_Y)

    S = np.column_stack([S1, S2, S3])  # (n, 3)

    labeled_mask = _assign_labels_mcar(n, pi_L, rng)
    true_tau = beta_1 * gamma_1 + beta_2 * gamma_2 + beta_3 * gamma_3

    return dict(
        T=T, X=X, S=S, Y=Y, Y_hat=None,
        labeled_mask=labeled_mask, true_tau=true_tau,
    )


# ---------------------------------------------------------------------------
# DGP 8: Nonlinear Surrogate-Outcome
# ---------------------------------------------------------------------------

def generate_dgp8(
    n: int = 10_000,
    pi_L: float = 0.20,
    alpha_S: float = 5.0,
    beta_SX: float = 1.0,
    gamma_S: float = 0.3,
    sigma_S: float = 2.0,
    beta_YS_linear: float = 0.5,
    beta_YS_quad: float = -0.03,
    beta_YX: float = 0.2,
    sigma_Y: float = 1.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    DGP 8: Nonlinear Surrogate-Outcome Relationship.

    S follows the same linear model as DGP 1:
        S_i = alpha_S + beta_SX * X_i + gamma_S * T_i + eps_S_i

    Y has a quadratic (concave) dependence on S:
        Y_i = beta_YS_linear * S_i + beta_YS_quad * (S_i - alpha_S)^2
              + beta_YX * X_i + eps_Y_i

    The quadratic term makes the S -> Y relationship nonlinear.
    Tests whether prediction models with S^2 features handle this well.

    True ATE (analytic):
        E[S|T=t] = alpha_S + gamma_S*t  (since E[X]=0)
        ATE = beta_YS_linear * gamma_S
              + beta_YS_quad * [(E[S|T=1] - alpha_S)^2 - (E[S|T=0] - alpha_S)^2]
            = beta_YS_linear * gamma_S + beta_YS_quad * gamma_S^2

    With defaults: 0.5*0.3 + (-0.03)*0.09 = 0.15 - 0.0027 = 0.1473
    """
    rng = _make_rng(seed)

    X = rng.standard_normal((n, 1))
    T = _assign_treatment(n, rng)

    eps_S = rng.normal(0, sigma_S, size=n)
    S = alpha_S + beta_SX * X[:, 0] + gamma_S * T + eps_S

    eps_Y = rng.normal(0, sigma_Y, size=n)
    Y = (beta_YS_linear * S
         + beta_YS_quad * (S - alpha_S) ** 2
         + beta_YX * X[:, 0]
         + eps_Y)

    labeled_mask = _assign_labels_mcar(n, pi_L, rng)

    # Analytic true ATE:
    # E[Y|T=1] - E[Y|T=0] = beta_YS_linear * gamma_S
    #   + beta_YS_quad * [E[(S-alpha_S)^2 | T=1] - E[(S-alpha_S)^2 | T=0]]
    # E[(S-alpha_S)^2 | T=t] = Var(S|T=t) + (E[S|T=t] - alpha_S)^2
    #                        = (beta_SX^2 + sigma_S^2) + (gamma_S*t)^2
    # (since E[X]=0 and Var(X)=1, Var(S|T=t) = beta_SX^2 + sigma_S^2)
    # Difference: (gamma_S*1)^2 - (gamma_S*0)^2 = gamma_S^2
    # (variances cancel because they are equal across arms)
    true_tau = beta_YS_linear * gamma_S + beta_YS_quad * gamma_S ** 2

    return dict(
        T=T, X=X, S=S, Y=Y, Y_hat=None,
        labeled_mask=labeled_mask, true_tau=true_tau,
    )


# ---------------------------------------------------------------------------
# DGP 9: Antagonistic Surrogate
# ---------------------------------------------------------------------------

def generate_dgp9(
    n: int = 10_000,
    pi_L: float = 0.20,
    alpha_S: float = 5.0,
    beta_SX: float = 1.0,
    gamma_S: float = 0.3,
    sigma_S: float = 2.0,
    beta_YS: float = -0.3,
    beta_YX: float = 0.8,
    delta: float = 0.5,
    sigma_Y: float = 1.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    DGP 9: Antagonistic Surrogate.

    S positively responds to treatment, but Y is negatively related to S.
    This makes the optimal PPI++ beta* negative, differentiating GREG
    from PPI++.

    S_i = alpha_S + beta_SX * X_i + gamma_S * T_i + eps_S
    Y_i = 10 + beta_YS * S_i + beta_YX * X_i + delta * T_i + eps_Y

    True ATE: tau = beta_YS * gamma_S + delta
    With defaults: -0.3 * 0.3 + 0.5 = 0.41

    The surrogate effect is beta_YS * gamma_S = -0.09 (negative via S),
    direct effect is delta = 0.5.
    SI targets beta_YS * gamma_S = -0.09 (badly wrong, wrong sign!)
    PPI++ lambda* should be ~0 (surrogate is misleading)
    GREG beta* should be negative (invert the prediction)
    """
    rng = _make_rng(seed)

    X = rng.standard_normal((n, 1))
    T = _assign_treatment(n, rng)

    eps_S = rng.normal(0, sigma_S, size=n)
    S = alpha_S + beta_SX * X[:, 0] + gamma_S * T + eps_S

    eps_Y = rng.normal(0, sigma_Y, size=n)
    Y = 10.0 + beta_YS * S + beta_YX * X[:, 0] + delta * T + eps_Y

    labeled_mask = _assign_labels_mcar(n, pi_L, rng)
    true_tau = beta_YS * gamma_S + delta

    return dict(
        T=T, X=X, S=S, Y=Y, Y_hat=None,
        labeled_mask=labeled_mask, true_tau=true_tau,
    )


# ---------------------------------------------------------------------------
# DGP 10: Nonlinear + Partial Mediation
# ---------------------------------------------------------------------------

def generate_dgp10(
    n: int = 10_000,
    pi_L: float = 0.20,
    rho: float = 0.2,
    alpha_S: float = 5.0,
    beta_SX: float = 1.0,
    gamma_S: float = 0.3,
    sigma_S: float = 2.0,
    beta_YS_linear: float = 0.5,
    beta_YS_quad: float = -0.03,
    beta_YX: float = 0.2,
    sigma_Y: float = 1.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    DGP 10: Nonlinear + Partial Mediation.

    Combines DGP 2 (partial mediation) + DGP 8 (nonlinearity).

    S_i = alpha_S + beta_SX * X_i + gamma_S * T_i + eps_S
    Y_i = beta_YS_linear * S_i + beta_YS_quad * (S_i - alpha_S)^2
           + beta_YX * X_i + delta * T_i + eps_Y

    delta = rho / (1 - rho) * mediated_effect
    where mediated_effect = beta_YS_linear * gamma_S + beta_YS_quad * gamma_S^2

    Sweep rho = delta / (mediated + delta) over {0, 0.2, 0.4}.
    """
    rng = _make_rng(seed)

    # Mediated ATE (nonlinear, same as DGP 8):
    mediated = beta_YS_linear * gamma_S + beta_YS_quad * gamma_S ** 2

    if rho >= 1.0:
        raise ValueError("rho must be < 1")
    delta = (rho / (1.0 - rho)) * mediated if rho > 0 else 0.0

    X = rng.standard_normal((n, 1))
    T = _assign_treatment(n, rng)

    eps_S = rng.normal(0, sigma_S, size=n)
    S = alpha_S + beta_SX * X[:, 0] + gamma_S * T + eps_S

    eps_Y = rng.normal(0, sigma_Y, size=n)
    Y = (beta_YS_linear * S
         + beta_YS_quad * (S - alpha_S) ** 2
         + beta_YX * X[:, 0]
         + delta * T
         + eps_Y)

    labeled_mask = _assign_labels_mcar(n, pi_L, rng)
    true_tau = mediated + delta

    return dict(
        T=T, X=X, S=S, Y=Y, Y_hat=None,
        labeled_mask=labeled_mask, true_tau=true_tau,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

DGP_GENERATORS = {
    1: generate_dgp1,
    2: generate_dgp2,
    3: generate_dgp3,
    4: generate_dgp4,
    5: generate_dgp5,
    6: generate_dgp6,
    7: generate_dgp7,
    8: generate_dgp8,
    9: generate_dgp9,
    10: generate_dgp10,
}
