"""
Simulation runner for surrogate metrics Monte Carlo study.

Main entry point: run_simulation(dgp_config, method_list, R, ...) which
returns a pandas DataFrame of results across replications.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.dgps.dgps import DGP_GENERATORS
from src.methods import (
    estimate,
    METHOD_NAMES as _M_NAMES,
    # Display-name machinery now lives with the methods so that
    # `display_name(method_id, config)` can see the configuration defaults.
    # Re-exported here because scripts and tests import it from this module.
    METHOD_DISPLAY_NAMES,
    display_name,
)
from src.utils.config import resolve_specs


def api_estimate(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Call the estimator through `src.utils.estimator_api`.

    Imported inside the call rather than at module scope: estimator_api
    imports this module (for `train_prediction_model`), so a top-level import
    here would be circular.
    """
    from src.utils.estimator_api import estimate as _estimate
    return _estimate(*args, **kwargs)

# Lazy import for GBT to avoid hard dependency when not used
_GBT_CLASS = None

def _get_gbt_class():
    global _GBT_CLASS
    if _GBT_CLASS is None:
        from sklearn.ensemble import GradientBoostingRegressor
        _GBT_CLASS = GradientBoostingRegressor
    return _GBT_CLASS


# ---------------------------------------------------------------------------
# Prediction model: OLS with 5-fold cross-fitting
# ---------------------------------------------------------------------------

def _ols_fit(
    X_design: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Fit the linear index: returns beta = (X'X + 1e-10 I)^{-1} X'y.

    This is the normal equation with a fixed absolute ridge of 1e-10 on the
    diagonal, solved by ``np.linalg.solve``.  It does NOT drop collinear
    columns.  When the design is exactly rank-deficient (a binary surrogate
    makes S^2 == S, as on the Criteo-calibrated multi-surrogate design), the
    ridge makes X'X + 1e-10 I invertible and the solve returns, to within
    O(1e-10 / smallest genuine eigenvalue of X'X), the minimum-norm
    least-squares solution; the fitted values X beta, which are all the
    estimators use, are the least-squares fit on the column space.

    The variance side does not use this ridge: `src.methods.joint_influence_cov`
    and `src.methods.si_first_stage_term` invert the Gram matrix by
    eigen-truncation (`src.methods._psd_solve`), dropping eigenvalues below
    1e-12 times the largest (`src.methods._GRAM_RCOND`).
    """
    XtX = X_design.T @ X_design
    XtX += 1e-10 * np.eye(XtX.shape[0])
    Xty = X_design.T @ y
    beta = np.linalg.solve(XtX, Xty)
    return beta


def _build_design_matrix(S: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Build OLS design: [1, S, S^2, X_cols].

    For 1-d S: [1, S, S^2, X_cols].
    For multi-surrogate S (n, J): [1, S_1, ..., S_J, S_1^2, ..., S_J^2, X_cols].

    Following Section 5.3: f(s) = alpha + beta_1 * s + beta_2 * s^2,
    plus covariate columns X.
    """
    n = S.shape[0]
    intercept = np.ones((n, 1))

    if S.ndim == 1:
        S_cols = S.reshape(-1, 1)
        S2_cols = (S ** 2).reshape(-1, 1)
    else:
        S_cols = S
        S2_cols = S ** 2

    parts = [intercept, S_cols, S2_cols]
    if X is not None and X.shape[1] > 0:
        parts.append(X)
    return np.hstack(parts)


PROTOCOL_ALLUNITS = "allunits_crossfit"
#: All-units cross-fitting with the fold assignment stratified by
#: (arm, labeled status): a permutation within each of the four strata, so
#: every fold holds the same share (to within one unit) of the labeled and of
#: the unlabeled units of each arm.  Opt-in; the reported results use the
#: unstratified PROTOCOL_ALLUNITS.
PROTOCOL_ALLUNITS_STRAT = "allunits_crossfit_stratified"
PROTOCOL_MIXED_FIT = "mixed_fit"
DEFAULT_PROTOCOL = PROTOCOL_ALLUNITS
#: Protocols that assign a fold to every unit (labeled or not).
ALLUNITS_PROTOCOLS = (PROTOCOL_ALLUNITS, PROTOCOL_ALLUNITS_STRAT)


def allunits_fold_ids(
    n: int,
    n_folds: int,
    rng: np.random.Generator,
    protocol: str = PROTOCOL_ALLUNITS,
    T: Optional[np.ndarray] = None,
    labeled_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fold assignment of all n units for the all-units protocols.

    ``allunits_crossfit``: one permutation of {0, ..., n-1} cut into K
    contiguous blocks.

    ``allunits_crossfit_stratified``: the same construction applied
    separately within each (arm, labeled) stratum, in the fixed order
    (T=0, unlabeled), (T=0, labeled), (T=1, unlabeled), (T=1, labeled).  Each
    fold then holds floor or ceil of n_s / K units of every stratum s, so the
    labeled fold shares n_{L,tk} / n_{L,t} and the all-units fold shares
    n_{tk} / n_t within each arm agree to within one unit per stratum (and
    exactly when every stratum size is a multiple of K).
    """
    fold_ids = np.zeros(n, dtype=int)
    if protocol == PROTOCOL_ALLUNITS_STRAT:
        if T is None or labeled_mask is None:
            raise ValueError(
                "protocol 'allunits_crossfit_stratified' needs the treatment "
                "vector T and the labeled mask to stratify the folds."
            )
        T = np.asarray(T)
        L = np.asarray(labeled_mask, dtype=bool)
        for t in (0, 1):
            for lab in (False, True):
                idx = np.where((T == t) & (L == lab))[0]
                m = len(idx)
                perm = idx[rng.permutation(m)]
                for k in range(n_folds):
                    fold_ids[perm[k * m // n_folds:(k + 1) * m // n_folds]] = k
        return fold_ids
    perm = rng.permutation(n)
    for k in range(n_folds):
        fold_ids[perm[k * n // n_folds:(k + 1) * n // n_folds]] = k
    return fold_ids


def unit_clustered_fold_ids(
    orig_ids: np.ndarray,
    n_folds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fold assignment of bootstrap rows by ORIGINAL unit.

    A bootstrap resample holds duplicated rows.  Permuting rows (as
    `allunits_fold_ids` does) can put copies of one original unit on both
    sides of a fold, so the held-out copy is predicted by an index trained on
    another copy of itself.  Here the distinct original ids are permuted and
    cut into K contiguous blocks (the `allunits_fold_ids` construction applied
    to units rather than rows) and every row inherits its unit's fold.  Used
    only by the bootstrap checks (``--clustered-folds``); the estimators on
    the observed sample are unaffected.
    """
    uniq, inv = np.unique(np.asarray(orig_ids), return_inverse=True)
    m = len(uniq)
    perm = rng.permutation(m)
    ufold = np.zeros(m, dtype=int)
    for k in range(n_folds):
        ufold[perm[k * m // n_folds:(k + 1) * m // n_folds]] = k
    return ufold[inv]


def _degenerate_fold_error(k: int, n_train: int, q: int) -> ValueError:
    return ValueError(
        f"degenerate cross-fitting fold: fold {k} leaves {n_train} labeled "
        f"training units outside it, fewer than the {q + 1} needed to fit a "
        f"{q}-column index.  The all-units protocol has no silent fallback "
        f"(a full-sample fit would put a unit's own label in its prediction); "
        f"use fewer folds or more labeled units."
    )


def train_prediction_model(
    S: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    labeled_mask: np.ndarray,
    n_folds: int = 5,
    rng: Optional[np.random.Generator] = None,
    protocol: str = DEFAULT_PROTOCOL,
    model: str = "ols",
    return_design: bool = False,
    prediction_model: Optional[str] = None,
    seed: int = 0,
    T: Optional[np.ndarray] = None,
):
    """Fit the prediction index f_hat and impute Y for every unit.

    Three protocols are available.

    ``protocol="allunits_crossfit"``
        EVERY unit -- labeled and unlabeled -- is assigned to one of K folds
        by an rng-driven permutation of {0, ..., n-1}.  For fold k the index
        is fit by OLS on the LABELED units OUTSIDE fold k, and that fit
        predicts every unit inside fold k.  Writing g_i = g(S_i, X_i) for the
        design row and beta_(-k) for the fold-k-excluded fit,

            Yhat_i = g_i' beta_(-k(i))      for every i = 1, ..., n,

        so no unit's prediction uses its own label and tau_SI = sum_k (n_k/n)
        d_k' beta_(-k) is a single M-estimator whose fold remainder is exactly
        mean-zero under MCAR conditional on the training folds.

    ``protocol="allunits_crossfit_stratified"`` (opt-in)
        As above, but the fold assignment is a permutation WITHIN each (arm,
        labeled) stratum (`allunits_fold_ids`), so the labeled and the
        all-units fold shares within each arm agree to within one unit. Needs
        ``T``.

    ``protocol="mixed_fit"``
        The mixed fit: labeled units get held-out-fold predictions (folds
        assigned among the n_L labeled units only), unlabeled units get the
        full-labeled-sample fit.

    Parameters
    ----------
    S : (n,) or (n, J) surrogate values for all units
    X : (n, p) covariates for all units
    Y : (n,) primary outcomes (only labeled entries are used)
    labeled_mask : (n,) bool, True for labeled units
    n_folds : number of cross-fitting folds K
    rng : random number generator for the fold assignment
    protocol : "allunits_crossfit" (default), "allunits_crossfit_stratified"
        or "mixed_fit"
    model : "ols" (default) or "gbt" (gradient-boosted trees)
    return_design : if True, also return the design dictionary described below
    prediction_model : deprecated alias for `model` (kept for old callers)
    seed : seed for the GBT random_state
    T : (n,) treatment indicator; required by the stratified protocol and
        ignored by the others

    Raises
    ------
    ValueError
        Under the all-units protocols, when a fold leaves fewer than q + 1
        labeled training units for a q-column index, or when there are fewer
        than 2K labeled units.  There is no fallback to the full-sample fit.

    Returns
    -------
    Y_hat : (n,) predicted outcomes for all units
    design : dict, only when return_design=True, with keys
        g         : (n, q) design matrix, columns (1, S, S^2, X) for OLS
                    (None for GBT -- the joint sandwich is unavailable and
                    `joint_influence_cov` raises, so use the bootstrap)
        fold_ids : (n,) int fold assignment
        coefs     : (K, q) per-fold coefficient matrix beta_(-k)
        beta_full : (q,) full-labeled-sample OLS fit
        protocol  : the protocol string actually used
        model     : the model string actually used
        n_folds   : K
        labeled_mask : the mask used
    """
    if prediction_model is not None:
        model = prediction_model
    if protocol not in ALLUNITS_PROTOCOLS + (PROTOCOL_MIXED_FIT,):
        raise ValueError(
            f"Unknown protocol {protocol!r}. Choose one of "
            f"{ALLUNITS_PROTOCOLS + (PROTOCOL_MIXED_FIT,)}."
        )
    if protocol == PROTOCOL_ALLUNITS_STRAT and T is None:
        raise ValueError(
            "protocol 'allunits_crossfit_stratified' needs T (the treatment "
            "vector) to stratify the folds by (arm, labeled status)."
        )

    if model == "gbt":
        Y_hat = _train_prediction_model_gbt(
            S, X, Y, labeled_mask, n_folds=n_folds, rng=rng, seed=seed,
            protocol=protocol, T=T,
        )
        if not return_design:
            return Y_hat
        return Y_hat, dict(
            g=None, fold_ids=None, coefs=None, beta_full=None,
            protocol=protocol, model="gbt", n_folds=n_folds,
            labeled_mask=np.asarray(labeled_mask, dtype=bool),
        )

    if rng is None:
        rng = np.random.default_rng(42)

    n = S.shape[0]
    Y_hat = np.empty(n, dtype=np.float64)

    labeled_idx = np.where(labeled_mask)[0]
    unlabeled_idx = np.where(~labeled_mask)[0]
    n_L = len(labeled_idx)

    G_all = _build_design_matrix(S, X) if return_design else None

    # If very few labeled units, the mixed-fit protocol skips cross-fitting and
    # uses the full fit; the all-units protocols refuse instead.
    if n_L < n_folds * 2 and protocol in ALLUNITS_PROTOCOLS:
        raise ValueError(
            f"too few labeled units for {n_folds}-fold all-units "
            f"cross-fitting: n_L = {n_L} < 2K = {2 * n_folds}."
        )
    if n_L < n_folds * 2:
        design_L = _build_design_matrix(S[labeled_idx], X[labeled_idx])
        beta = _ols_fit(design_L, Y[labeled_idx])
        design_all = _build_design_matrix(S, X)
        Y_hat[:] = design_all @ beta
        if not return_design:
            return Y_hat
        return Y_hat, dict(
            g=G_all,
            fold_ids=np.zeros(n, dtype=int),
            coefs=np.tile(beta, (n_folds, 1)),
            beta_full=beta,
            protocol=protocol, model="ols", n_folds=n_folds,
            labeled_mask=np.asarray(labeled_mask, dtype=bool),
        )

    # --- Full-labeled-sample fit (used by both protocols) -------------------
    # NOTE: under the mixed-fit protocol this fit is computed AFTER the fold
    # loop; the rng is not touched by either, so the numerical result is
    # identical either way.
    design_L_full = _build_design_matrix(S[labeled_idx], X[labeled_idx])
    beta_full = _ols_fit(design_L_full, Y[labeled_idx])

    if protocol in ALLUNITS_PROTOCOLS:
        # Fold assignment over ALL n units (stratified by (arm, labeled) under
        # the stratified protocol).
        fold_ids = allunits_fold_ids(n, n_folds, rng, protocol, T=T,
                                     labeled_mask=labeled_mask)

        q = design_L_full.shape[1]
        coefs = np.empty((n_folds, q), dtype=np.float64)
        for k in range(n_folds):
            test_idx = np.where(fold_ids == k)[0]
            train_idx = labeled_idx[fold_ids[labeled_idx] != k]
            if len(train_idx) < q + 1:
                raise _degenerate_fold_error(k, len(train_idx), q)
            beta_k = _ols_fit(
                _build_design_matrix(S[train_idx], X[train_idx]),
                Y[train_idx],
            )
            coefs[k] = beta_k
            if len(test_idx) > 0:
                Y_hat[test_idx] = (
                    _build_design_matrix(S[test_idx], X[test_idx]) @ beta_k
                )
    else:
        # --- mixed-fit protocol: folds over labeled units only -----------------
        fold_ids_L = np.zeros(n_L, dtype=int)
        perm = rng.permutation(n_L)
        for k in range(n_folds):
            start = k * n_L // n_folds
            end = (k + 1) * n_L // n_folds
            fold_ids_L[perm[start:end]] = k

        coefs = np.empty((n_folds, design_L_full.shape[1]), dtype=np.float64)
        coefs[:] = beta_full
        for k in range(n_folds):
            train_in_fold = fold_ids_L != k
            test_in_fold = fold_ids_L == k

            train_idx = labeled_idx[train_in_fold]
            test_idx = labeled_idx[test_in_fold]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            design_train = _build_design_matrix(S[train_idx], X[train_idx])
            beta_k = _ols_fit(design_train, Y[train_idx])
            coefs[k] = beta_k

            design_test = _build_design_matrix(S[test_idx], X[test_idx])
            Y_hat[test_idx] = design_test @ beta_k

        if len(unlabeled_idx) > 0:
            design_U = _build_design_matrix(S[unlabeled_idx], X[unlabeled_idx])
            Y_hat[unlabeled_idx] = design_U @ beta_full

        fold_ids = np.full(n, -1, dtype=int)
        fold_ids[labeled_idx] = fold_ids_L

    if not return_design:
        return Y_hat
    return Y_hat, dict(
        g=G_all,
        fold_ids=fold_ids,
        coefs=coefs,
        beta_full=beta_full,
        protocol=protocol, model="ols", n_folds=n_folds,
        labeled_mask=np.asarray(labeled_mask, dtype=bool),
    )


def _train_prediction_model_gbt(
    S: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    labeled_mask: np.ndarray,
    n_folds: int = 5,
    rng: Optional[np.random.Generator] = None,
    seed: int = 0,
    protocol: str = DEFAULT_PROTOCOL,
    T: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Train GBT prediction model with cross-fitting.

    Uses the same feature set as OLS: (S, S^2, X) for 1-d S,
    or (S_1,...,S_J, S_1^2,...,S_J^2, X) for multi-d S.

    Under ``protocol="allunits_crossfit"`` every unit is assigned to a fold
    and predicted from the model fit on the labeled units outside its fold;
    under ``protocol="mixed_fit"`` labeled units are cross-fitted among
    themselves and unlabeled units use the full-labeled-sample model.
    """
    GBT = _get_gbt_class()

    if rng is None:
        rng = np.random.default_rng(42)

    n = S.shape[0]
    Y_hat = np.empty(n, dtype=np.float64)

    labeled_idx = np.where(labeled_mask)[0]
    unlabeled_idx = np.where(~labeled_mask)[0]
    n_L = len(labeled_idx)

    def _build_features(S_sub, X_sub):
        """Build feature matrix: [S, S^2, X] (no intercept for GBT)."""
        if S_sub.ndim == 1:
            S_cols = S_sub.reshape(-1, 1)
            S2_cols = (S_sub ** 2).reshape(-1, 1)
        else:
            S_cols = S_sub
            S2_cols = S_sub ** 2
        parts = [S_cols, S2_cols]
        if X_sub is not None and X_sub.shape[1] > 0:
            parts.append(X_sub)
        return np.hstack(parts)

    def _fit_gbt(features, targets):
        model = GBT(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=seed,
        )
        model.fit(features, targets)
        return model

    # If very few labeled units, the mixed-fit protocol skips cross-fitting; the
    # all-units protocols refuse.
    if n_L < n_folds * 2 and protocol in ALLUNITS_PROTOCOLS:
        raise ValueError(
            f"too few labeled units for {n_folds}-fold all-units "
            f"cross-fitting: n_L = {n_L} < 2K = {2 * n_folds}."
        )
    if n_L < n_folds * 2:
        feat_L = _build_features(S[labeled_idx], X[labeled_idx])
        model = _fit_gbt(feat_L, Y[labeled_idx])
        feat_all = _build_features(S, X)
        Y_hat[:] = model.predict(feat_all)
        return Y_hat

    if protocol in ALLUNITS_PROTOCOLS:
        # Assign folds to ALL units; predict every unit in fold k from the
        # model fit on the labeled units outside fold k.
        fold_ids = allunits_fold_ids(n, n_folds, rng, protocol, T=T,
                                     labeled_mask=labeled_mask)

        for k in range(n_folds):
            test_idx = np.where(fold_ids == k)[0]
            train_idx = labeled_idx[fold_ids[labeled_idx] != k]
            if len(test_idx) == 0:
                continue
            if len(train_idx) < 2:
                raise _degenerate_fold_error(k, len(train_idx), 1)
            model_k = _fit_gbt(
                _build_features(S[train_idx], X[train_idx]), Y[train_idx],
            )
            Y_hat[test_idx] = model_k.predict(
                _build_features(S[test_idx], X[test_idx])
            )
        return Y_hat

    # --- mixed-fit protocol ---------------------------------------------------
    # Assign folds to labeled units
    fold_ids = np.zeros(n_L, dtype=int)
    perm = rng.permutation(n_L)
    for k in range(n_folds):
        start = k * n_L // n_folds
        end = (k + 1) * n_L // n_folds
        fold_ids[perm[start:end]] = k

    # Cross-fitted predictions for labeled units
    for k in range(n_folds):
        train_in_fold = fold_ids != k
        test_in_fold = fold_ids == k
        train_idx = labeled_idx[train_in_fold]
        test_idx = labeled_idx[test_in_fold]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        feat_train = _build_features(S[train_idx], X[train_idx])
        model_k = _fit_gbt(feat_train, Y[train_idx])
        feat_test = _build_features(S[test_idx], X[test_idx])
        Y_hat[test_idx] = model_k.predict(feat_test)

    # Full-labeled-set model for unlabeled units
    feat_L_full = _build_features(S[labeled_idx], X[labeled_idx])
    model_full = _fit_gbt(feat_L_full, Y[labeled_idx])

    if len(unlabeled_idx) > 0:
        feat_U = _build_features(S[unlabeled_idx], X[unlabeled_idx])
        Y_hat[unlabeled_idx] = model_full.predict(feat_U)

    return Y_hat


def train_prediction_model_calibration(
    S_cal: np.ndarray,
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    S: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """
    Train prediction model on external calibration data and predict for
    experiment units.  Used by DGP 4 for the surrogate index method.

    No cross-fitting needed since calibration data is separate.
    """
    design_cal = _build_design_matrix(S_cal, X_cal)
    beta = _ols_fit(design_cal, Y_cal)
    design_exp = _build_design_matrix(S, X)
    return design_exp @ beta


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------

def derive_seed(
    master_seed: int, dgp_id: int, config_id: int, pi_L_id: int, rep: int
) -> int:
    """
    Deterministic seed per (dgp, config, pi_L, replication) cell.
    Uses SHA-256 hash truncated to 32-bit integer range.
    """
    key = f"{master_seed}-{dgp_id}-{config_id}-{pi_L_id}-{rep}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16)


# ---------------------------------------------------------------------------
# Composite proxy weight estimation from historical experiments
# ---------------------------------------------------------------------------

def estimate_composite_weight_from_historical(
    K_hist: int = 30,
    beta_YS: float = 0.5,
    gamma_S: float = 0.3,
    sigma_tau: float = 0.10,
    seed: int = 999,
    **dgp_kwargs: Any,
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Generate K_hist historical experiments and estimate the composite proxy
    weight.  Returns (w, hist_experiments_list).

    The hist_experiments_list can be passed to the composite_proxy method via
    kwargs['historical_experiments'].
    """
    rng = np.random.default_rng(seed)

    hist_experiments: List[Dict[str, float]] = []

    for k in range(K_hist):
        tau_k = rng.normal(0, sigma_tau)
        gamma_S_k = tau_k / beta_YS if beta_YS != 0 else 0.0

        n_k = rng.integers(2000, 10001)
        exp_rng = np.random.default_rng(seed * 1000 + k + 1)

        X_k = exp_rng.standard_normal((n_k, 1))
        T_k = exp_rng.binomial(1, 0.5, size=n_k)

        sigma_S = dgp_kwargs.get("sigma_S", 2.0)
        eps_S = exp_rng.normal(0, sigma_S, size=n_k)
        S_k = (dgp_kwargs.get("alpha_S", 5.0)
               + dgp_kwargs.get("beta_SX", 1.0) * X_k[:, 0]
               + gamma_S_k * T_k + eps_S)

        sigma_Y = dgp_kwargs.get("sigma_Y", 1.0)
        eps_Y = exp_rng.normal(0, sigma_Y, size=n_k)
        Y_k = (dgp_kwargs.get("alpha_Y", 0.0)
               + beta_YS * S_k
               + dgp_kwargs.get("beta_YX", 0.2) * X_k[:, 0]
               + eps_Y)

        t1 = T_k == 1
        t0 = T_k == 0
        tau_S_hat_k = float(S_k[t1].mean() - S_k[t0].mean())
        tau_Y_hat_k = float(Y_k[t1].mean() - Y_k[t0].mean())

        hist_experiments.append(dict(
            tau_S_hist=tau_S_hat_k,
            tau_hist=tau_Y_hat_k,
        ))

    # Also compute the scalar weight for backwards compatibility
    tau_S_arr = np.array([e["tau_S_hist"] for e in hist_experiments])
    tau_Y_arr = np.array([e["tau_hist"] for e in hist_experiments])
    denom = np.sum(tau_S_arr ** 2)
    w = float(np.sum(tau_S_arr * tau_Y_arr) / denom) if denom > 1e-15 else 0.0

    return w, hist_experiments


def estimate_composite_weight_from_historical_multi(
    K_hist: int = 30,
    J: int = 3,
    gammas: Optional[Sequence[float]] = None,
    betas: Optional[Sequence[float]] = None,
    sigma_tau: float = 0.10,
    seed: int = 999,
    **dgp_kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Generate K_hist historical experiments for the multi-surrogate composite
    proxy.  Returns a list of dicts with keys tau_S_hist (length-J array)
    and tau_hist (scalar).

    Parameters
    ----------
    K_hist : number of historical experiments
    J : number of surrogates
    gammas : default treatment effects on each surrogate
    betas : outcome coefficients for each surrogate
    sigma_tau : std of per-experiment ATE variation
    seed : RNG seed
    """
    if gammas is None:
        gammas = [0.3, 0.2, 0.1]
    if betas is None:
        betas = [0.3, 0.4, 0.2]

    rng = np.random.default_rng(seed)
    hist_experiments: List[Dict[str, Any]] = []

    for k in range(K_hist):
        # Each historical experiment has a random scaling of the treatment
        scale = rng.normal(1.0, sigma_tau / 0.19)  # scale around 1
        gammas_k = [g * scale for g in gammas]

        n_k = rng.integers(2000, 10001)
        exp_rng = np.random.default_rng(seed * 1000 + k + 1)

        X_k = exp_rng.standard_normal((n_k, 1))
        T_k = exp_rng.binomial(1, 0.5, size=n_k)

        # Generate J surrogates
        intercepts = [5.0, 3.0, 1.0]
        x_coefs = [1.0, 0.5, 0.3]
        sigmas = [
            dgp_kwargs.get("sigma_1", 2.0),
            dgp_kwargs.get("sigma_2", np.sqrt(2.0)),
            dgp_kwargs.get("sigma_3", 1.0),
        ]

        S_k = np.zeros((n_k, J))
        for j in range(J):
            eps_j = exp_rng.normal(0, sigmas[j], size=n_k)
            S_k[:, j] = intercepts[j] + x_coefs[j] * X_k[:, 0] + gammas_k[j] * T_k + eps_j

        sigma_Y = dgp_kwargs.get("sigma_Y", 1.0)
        eps_Y = exp_rng.normal(0, sigma_Y, size=n_k)
        Y_k = sum(betas[j] * S_k[:, j] for j in range(J)) + dgp_kwargs.get("beta_YX", 0.2) * X_k[:, 0] + eps_Y

        t1 = T_k == 1
        t0 = T_k == 0
        tau_S_hat_k = S_k[t1].mean(axis=0) - S_k[t0].mean(axis=0)  # (J,)
        tau_Y_hat_k = float(Y_k[t1].mean() - Y_k[t0].mean())

        hist_experiments.append(dict(
            tau_S_hist=tau_S_hat_k.tolist(),
            tau_hist=tau_Y_hat_k,
        ))

    return hist_experiments


# ---------------------------------------------------------------------------
# Single-replication runner
# ---------------------------------------------------------------------------

def run_single_replication(
    dgp_id: int,
    dgp_kwargs: Dict[str, Any],
    method_ids: Optional[Sequence[int]] = None,
    seed: int = 0,
    historical_experiments: Optional[List[Dict[str, float]]] = None,
    prediction_model: Optional[str] = None,
    model: str = "ols",
    protocol: str = DEFAULT_PROTOCOL,
    method_specs: Optional[Sequence[Any]] = None,
    include_ablation: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run one replication: generate data, fit the index, apply every method
    configuration to the SAME draw and the SAME predictions.

    Running the primary and ablation configurations together is what makes the
    ablation tables paired with the headline tables: the only thing that
    differs between a primary row and its ablation row is the estimator
    configuration, never the data.

    Parameters
    ----------
    dgp_id : which DGP (1-10)
    dgp_kwargs : parameters passed to the DGP generator (must include seed)
    method_ids : method ids, resolved to their default configurations
    seed : seed for the cross-fitting fold assignment
    historical_experiments : for method 5, list of historical experiment dicts
    prediction_model : deprecated alias for `model`
    model : "ols" (default) or "gbt"
    protocol : "allunits_crossfit" (default) or "mixed_fit"
    method_specs : explicit MethodSpec list (overrides method_ids)
    include_ablation : also run the ablation configurations

    Returns
    -------
    List of dicts, one per method configuration, with keys
        method_id, method_key, method_label, method_name, protocol,
        lambda_rule, clip, per_arm, variance, si_variance,
        tau_hat, V_hat, ci_lower, ci_upper, true_tau
    """
    if prediction_model is not None:
        model = prediction_model
    specs = resolve_specs(method_ids, method_specs, include_ablation)

    # Generate data
    dgp_func = DGP_GENERATORS[dgp_id]
    data = dgp_func(**dgp_kwargs)

    if dgp_id == 6:
        return _run_portfolio_replication(
            data, specs, seed, historical_experiments,
            model=model, protocol=protocol,
        )

    # Fit the prediction index once; every configuration shares it.
    cf_rng = np.random.default_rng(seed + 7777)
    Y_hat, design = train_prediction_model(
        data["S"], data["X"], data["Y"],
        data["labeled_mask"], n_folds=5, rng=cf_rng,
        protocol=protocol, model=model, return_design=True, seed=seed,
        T=data["T"],
    )

    # For DGP 4, also compute calibration-based predictions for the SI method:
    # the externally calibrated index is the point of that DGP.
    Y_hat_cal = None
    if dgp_id == 4 and "calibration_data" in data:
        cal = data["calibration_data"]
        Y_hat_cal = train_prediction_model_calibration(
            cal["S_cal"], cal["X_cal"], cal["Y_cal"],
            data["S"], data["X"],
        )

    true_tau = data["true_tau"]
    T = data["T"]
    S = data["S"]
    Y = data["Y"]
    labeled_mask = data["labeled_mask"]

    results = []
    for spec in specs:
        m_id = spec.method_id
        cfg = dict(spec.cfg())
        mkwargs: Dict[str, Any] = {}
        # Choose which Y_hat to use
        if dgp_id == 4 and m_id == 2 and Y_hat_cal is not None:
            yh = Y_hat_cal
            # The calibration index is fit on n_cal = 50,000 external units,
            # so the learned-index first-stage term is O(1/n_cal) and the
            # sandwich reduces to the plug-in variance; ask for it explicitly
            # rather than letting the estimator warn and fall back.
            cfg["si_variance"] = "plugin"
        else:
            yh = Y_hat
            if m_id in (2, 5):
                # Method 5 falls back to the surrogate index when it has no
                # historical experiments, so it needs the design as well.
                if design.get("g") is None:
                    # GBT predictions have no linear design, so the joint
                    # sandwich is unavailable.  Report the plug-in variance and
                    # record it (si_variance="plugin"): a heuristic interval
                    # that treats the trees as fixed.  The tree-refitting
                    # paired bootstrap that measures its shortfall is
                    # scripts/run_gbt_variance_check.gbt_paired_bootstrap
                    # (too costly to run per replication of the GBT grid).
                    cfg["si_variance"] = "plugin"
                else:
                    mkwargs["design"] = design

        if m_id == 5 and historical_experiments is not None:
            mkwargs["historical_experiments"] = historical_experiments
        if m_id == 7:
            # The bootstrap draws must be a function of the replication seed
            # (itself from derive_seed); an unseeded generator made the
            # bootstrap-variance rows differ from run to run.
            mkwargs["boot_seed"] = int(seed) + 1111
        if m_id == 6:
            # Pass propensity model: use logistic for MAR DGPs
            missingness = dgp_kwargs.get("missingness", None)
            if missingness is not None and missingness.upper() == "MAR":
                mkwargs["propensity_model"] = "logistic"
            else:
                mkwargs["propensity_model"] = "constant"

        res = api_estimate(
            m_id, T, S, Y, yh, labeled_mask, protocol=protocol,
            **cfg, **mkwargs,
        )
        cfg = res.get("config", {})

        # Normalize key names: methods module uses var_hat; we store as V_hat
        results.append(dict(
            method_id=m_id,
            method_key=spec.key,
            method_label=spec.label,
            method_name=spec.label,
            protocol=cfg.get("protocol", protocol),
            lambda_rule=cfg.get("lambda_rule"),
            clip=cfg.get("clip"),
            per_arm=cfg.get("per_arm"),
            variance=cfg.get("variance"),
            si_variance=cfg.get("si_variance"),
            tau_hat=res["tau_hat"],
            V_hat=res["var_hat"],
            ci_lower=res["ci_lower"],
            ci_upper=res["ci_upper"],
            true_tau=true_tau,
        ))

    return results


def _run_portfolio_replication(
    portfolio_data: Dict[str, Any],
    specs: Sequence[Any],
    seed: int,
    historical_experiments: Optional[List[Dict[str, float]]] = None,
    model: str = "ols",
    protocol: str = DEFAULT_PROTOCOL,
) -> List[Dict[str, Any]]:
    """
    Run one replication for DGP 6 (portfolio of K experiments).

    Every experiment in the portfolio is fit and estimated under the same
    protocol and the same method configurations as the single-experiment DGPs.
    """
    experiments = portfolio_data["experiments"]
    true_taus = portfolio_data["true_taus"]
    alpha_decision = portfolio_data["alpha_decision"]
    K = len(experiments)
    z_crit = stats.norm.ppf(1.0 - alpha_decision / 2.0)

    method_decisions: Dict[str, np.ndarray] = {
        s.key: np.zeros(K, dtype=bool) for s in specs
    }

    for k, exp_data in enumerate(experiments):
        cf_rng = np.random.default_rng(seed * 10000 + k + 8888)
        Y_hat_k, design_k = train_prediction_model(
            exp_data["S"], exp_data["X"], exp_data["Y"],
            exp_data["labeled_mask"], n_folds=5, rng=cf_rng,
            protocol=protocol, model=model, return_design=True,
            seed=seed * 10000 + k, T=exp_data["T"],
        )

        T_k = exp_data["T"]
        S_k = exp_data["S"]
        Y_k = exp_data["Y"]
        lm_k = exp_data["labeled_mask"]

        for spec in specs:
            mkwargs: Dict[str, Any] = {}
            if spec.method_id in (2, 5):
                mkwargs["design"] = design_k
            if spec.method_id == 5 and historical_experiments is not None:
                mkwargs["historical_experiments"] = historical_experiments

            res = api_estimate(
                spec.method_id, T_k, S_k, Y_k, Y_hat_k, lm_k,
                protocol=protocol, **spec.cfg(), **mkwargs,
            )

            tau_hat = res["tau_hat"]
            var_hat = res["var_hat"]

            if (not np.isnan(tau_hat) and not np.isnan(var_hat)
                    and var_hat > 0):
                z_stat = tau_hat / np.sqrt(var_hat)
                reject = abs(z_stat) > z_crit
                method_decisions[spec.key][k] = reject and tau_hat > 0

    oracle_gain = portfolio_data["oracle_gain"]
    results = []
    for spec in specs:
        decisions = method_decisions[spec.key]

        regret = 0.0
        for k in range(K):
            tau_k = true_taus[k]
            d_k = decisions[k]
            if tau_k > 0 and not d_k:
                regret += tau_k
            elif tau_k < 0 and d_k:
                regret += -tau_k

        results.append(dict(
            method_id=spec.method_id,
            method_key=spec.key,
            method_label=spec.label,
            method_name=spec.label,
            protocol=protocol,
            lambda_rule=spec.lambda_rule,
            clip=spec.clip,
            per_arm=spec.per_arm,
            variance=spec.variance,
            si_variance=spec.si_variance,
            tau_hat=np.nan,  # not meaningful for portfolio
            V_hat=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            true_tau=np.nanmean(true_taus),
            cumulative_regret=regret,
            oracle_gain=oracle_gain,
            decisions=decisions.copy(),
            true_taus=true_taus.copy(),
        ))

    return results


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    dgp_id: int,
    dgp_params: Dict[str, Any],
    method_ids: Optional[Sequence[int]] = None,
    R: int = 2000,
    master_seed: int = 42,
    config_id: int = 0,
    pi_L_id: int = 0,
    historical_experiments: Optional[List[Dict[str, float]]] = None,
    verbose: bool = False,
    prediction_model: Optional[str] = None,
    model: str = "ols",
    protocol: str = DEFAULT_PROTOCOL,
    method_specs: Optional[Sequence[Any]] = None,
    include_ablation: bool = False,
) -> pd.DataFrame:
    """
    Run the full Monte Carlo simulation for one (DGP, config, pi_L) cell.

    Parameters
    ----------
    dgp_id : DGP number (1-10)
    dgp_params : DGP-specific parameters (will be merged with seed per rep)
    method_ids : method ids, resolved to their default configurations
    R : number of Monte Carlo replications
    master_seed : master seed for reproducibility
    config_id : integer config identifier (for seed hashing)
    pi_L_id : integer pi_L config identifier (for seed hashing)
    historical_experiments : for method 5
    verbose : whether to print progress
    prediction_model : deprecated alias for `model`
    model : "ols" (default) or "gbt"
    protocol : "allunits_crossfit" (default) or "mixed_fit"
    method_specs : explicit MethodSpec list (overrides method_ids)
    include_ablation : also run the ablation configurations, paired

    Returns
    -------
    DataFrame with columns:
        replication, method_id, method_key, method_label, method_name,
        protocol, lambda_rule, clip, per_arm, variance, si_variance,
        tau_hat, V_hat, ci_lower, ci_upper, true_tau
        [cumulative_regret, oracle_gain for DGP 6]

    Feed it to `rows_from_replications` to get the
    ResultRow records that every table and figure reads.
    """
    if prediction_model is not None:
        model = prediction_model
    specs = resolve_specs(method_ids, method_specs, include_ablation)

    all_results: List[Dict[str, Any]] = []

    for r in range(R):
        if verbose and (r + 1) % 100 == 0:
            print(f"  Replication {r + 1}/{R}")

        rep_seed = derive_seed(master_seed, dgp_id, config_id, pi_L_id, r)

        dgp_kwargs = dict(dgp_params)
        dgp_kwargs["seed"] = rep_seed

        rep_results = run_single_replication(
            dgp_id=dgp_id,
            dgp_kwargs=dgp_kwargs,
            seed=rep_seed,
            historical_experiments=historical_experiments,
            model=model,
            protocol=protocol,
            method_specs=specs,
        )

        for res in rep_results:
            res["replication"] = r
            all_results.append(res)

    df = pd.DataFrame(all_results)
    return df
