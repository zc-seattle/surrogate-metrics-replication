"""
Single entry point for the estimator interface.

Every runner and script in this repository calls the estimator through this
module rather than importing ``src.methods`` or
``src.simulations.simulation.train_prediction_model`` directly.  Two reasons:

* it pins the call contract in one place, so a change on the
  estimator side is absorbed here instead of in thirty scripts;
* it stamps the paper label and the configuration block that the results

The contract
------------
::

    train_prediction_model(S, X, Y, labeled_mask, n_folds=5, rng=None,
                           protocol="allunits_crossfit", model="ols",
                           return_design=False)
        -> Y_hat, or (Y_hat, design) when return_design=True

    estimate(method_id, T, S, Y, Y_hat, labeled_mask, **cfg)
        cfg: lambda_rule in {"exact","plugin"} (default "exact"),
             clip (False), per_arm (False),
             variance in {"exact","plugin"} (default "exact")     [id 3]
             si_variance in {"sandwich","plugin","delta"}
                 (default "sandwich") plus design=design          [id 2]
        result["config"] = {protocol, method_id, lambda_rule, clip,
                            per_arm, variance, si_variance}

    display_name(method_id, config)        -> paper label
    joint_influence_cov(T, Y, labeled_mask, design, lambda_hat)
        -> {var_si, var_ppi, cov_si_ppi, var_D, ...}
    hybrid_sigma(T, Y, labeled_mask, design, lambda_hat) -> 2x2 covariance
    surrogacy_test(...)                    (accepts the sandwich covariance)

Method id 9 is an alias of id 3 in its default configuration.

"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import numpy as np

from src import methods as _methods
from src.simulations import simulation as _sim
from src.utils.config import (
    ALL_METHOD_SPECS,
    DEFAULT_PROTOCOL,
    PROTOCOLS,
)

__all__ = [
    "train_prediction_model",
    "estimate",
    "display_name",
    "joint_influence_cov",
    "hybrid_sigma",
    "surrogacy_test",
    "surrogacy_test_sandwich",
    "estimate_cov_si_ppi",
    "paired_bootstrap",
    "hybrid_estimator",
    "estimator_api_status",
    "DEFAULT_PROTOCOL",
    "PROTOCOLS",
]


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

def _accepts(func: Any, *names: str) -> bool:
    """True if `func` names every one of `names` as a parameter."""
    if func is None:
        return False
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return False
    return all(n in params for n in names)


_TPM = getattr(_sim, "train_prediction_model", None)
_HAS_PROTOCOL_TPM = _accepts(_TPM, "protocol", "return_design")
_HAS_JOINT_COV = hasattr(_methods, "joint_influence_cov")
_HAS_HYBRID_SIGMA = hasattr(_methods, "hybrid_sigma")
_HAS_DISPLAY_NAME = hasattr(_methods, "display_name")
# The configuration-axis contract is present as soon as src.methods exposes
# the config block it stamps on every result.  `estimate` is called
# positionally, so the name of its first parameter does not matter.
_HAS_CONFIG_AXES = hasattr(_methods, "_CONFIG_KEYS")


def estimator_api_status() -> Dict[str, Any]:
    """What the estimator layer currently implements, for smoke reporting."""
    parts = {
        "train_prediction_model(protocol=, return_design=)": _HAS_PROTOCOL_TPM,
        "methods configuration axes": _HAS_CONFIG_AXES,
        "methods.joint_influence_cov": _HAS_JOINT_COV,
        "methods.hybrid_sigma": _HAS_HYBRID_SIGMA,
        "methods.display_name": _HAS_DISPLAY_NAME,
    }
    required = dict(parts)
    required.pop("methods.display_name")
    parts["all_required_landed"] = all(required.values())
    return parts


def _require(flag: bool, what: str) -> None:
    if not flag:
        raise NotImplementedError(
            f"estimator_api: {what} has not landed yet in src/methods or "
            f"src/simulations/simulation.py. This is the WP1 estimator work "
            f"package; see the interface contract in this module's docstring."
        )


# ---------------------------------------------------------------------------
# Prediction model
# ---------------------------------------------------------------------------

def train_prediction_model(
    S: np.ndarray,
    X: Optional[np.ndarray],
    Y: np.ndarray,
    labeled_mask: np.ndarray,
    n_folds: int = 5,
    rng: Optional[np.random.Generator] = None,
    protocol: str = DEFAULT_PROTOCOL,
    model: str = "ols",
    return_design: bool = False,
    prediction_model: Optional[str] = None,
    seed: int = 0,
) -> Any:
    """Fit the prediction index and impute the outcome for every unit.

    ``protocol="allunits_crossfit"`` assigns every unit (labeled or not) to one
    of ``n_folds`` folds, fits the index on the labeled units outside the fold,
    and predicts the whole fold from that fit.

    With ``return_design=True`` the second return value is the design
    dictionary the SI sandwich variance, the diagnostic and the hybrid
    consume.  Treat it as opaque: hand it to ``estimate(2, ..., design=...)``,
    ``joint_influence_cov`` and ``hybrid_sigma``.
    """
    if prediction_model is not None:
        model = prediction_model
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol {protocol!r} not in {PROTOCOLS}")
    _require(_HAS_PROTOCOL_TPM, "train_prediction_model(protocol=, ...)")
    return _TPM(
        S, X, Y, labeled_mask, n_folds=n_folds, rng=rng, protocol=protocol,
        model=model, return_design=return_design, seed=seed,
    )


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

_DEFAULT_CFG = {
    "lambda_rule": "exact",
    "clip": False,
    "per_arm": False,
    "variance": "exact",
    "si_variance": "sandwich",
}

#: Which configuration axes apply to which method ids.
_PPI_IDS = (3, 7, 8, 9)
_SI_IDS = (2,)


def _normalize_cfg(method_id: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill the configuration axes that apply to `method_id`."""
    out: Dict[str, Any] = {}
    if method_id in _PPI_IDS:
        out["lambda_rule"] = cfg.get("lambda_rule", _DEFAULT_CFG["lambda_rule"])
        out["clip"] = bool(cfg.get("clip", _DEFAULT_CFG["clip"]))
        out["per_arm"] = bool(cfg.get("per_arm", _DEFAULT_CFG["per_arm"]))
        out["variance"] = cfg.get("variance", _DEFAULT_CFG["variance"])
    if method_id in _SI_IDS:
        out["si_variance"] = cfg.get("si_variance", _DEFAULT_CFG["si_variance"])
    return out


def estimate(
    method_id: int,
    T: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    protocol: str = DEFAULT_PROTOCOL,
    **cfg: Any,
) -> Dict[str, Any]:
    """Dispatch to a method under an explicit configuration.

    Returns the estimator's result dict plus ``result["config"]``, the block
    the result rows record.  Method-specific extras (historical
    experiments for id 5, the propensity model for id 6, ``design`` for id 2)
    pass straight through.
    """
    _require(_HAS_CONFIG_AXES, "the src.methods configuration axes")

    axes = _normalize_cfg(method_id, cfg)
    kwargs = {
        k: v for k, v in cfg.items()
        if k not in ("lambda_rule", "clip", "per_arm", "variance",
                     "si_variance")
    }
    kwargs.update(axes)
    kwargs.setdefault("protocol", protocol)

    call_id = 3 if method_id == 9 else method_id
    result = dict(_methods.estimate(
        call_id, T, S, Y, Y_hat, labeled_mask, **kwargs
    ))

    # Record what was actually used: the estimator may downgrade an axis (for
    # instance si_variance="sandwich" without a design dict falls back to
    # "plugin"), and the method table must show the downgrade.
    used = dict(result.get("config") or {})
    result["config"] = {
        "protocol": used.get("protocol", protocol),
        "method_id": method_id,
        "lambda_rule": used.get("lambda_rule", axes.get("lambda_rule")),
        "clip": used.get("clip", axes.get("clip")),
        "per_arm": used.get("per_arm", axes.get("per_arm")),
        "variance": used.get("variance", axes.get("variance")),
        "si_variance": used.get("si_variance", axes.get("si_variance")),
    }
    return result


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

_FALLBACK_LABELS = {
    0: "Labeled-Only",
    1: "Naive Surrogate",
    2: "SI",
    3: "PPI++",
    4: "GREG",
    5: "Composite Proxy",
    6: "AIPW",
    7: "PPI++ (bootstrap variance)",
    8: "PPI++ (plug-in lambda)",
    9: "PPI++",
}


def display_name(
    method_id: int, config: Optional[Dict[str, Any]] = None
) -> str:
    """The paper label for a (method id, configuration) pair.

    Falls back to the bare method name for a configuration that matches no
    spec.
    """
    cfg = dict(config or {})
    axes = _normalize_cfg(method_id, cfg)
    lookup_id = 3 if method_id == 9 else method_id
    for spec in ALL_METHOD_SPECS:
        if spec.method_id != lookup_id:
            continue
        if all(axes.get(k) == v for k, v in spec.cfg().items()):
            return spec.label
    if _HAS_DISPLAY_NAME:
        return _methods.display_name(method_id, cfg)
    return _FALLBACK_LABELS.get(method_id, f"Method {method_id}")


# ---------------------------------------------------------------------------
# Joint covariance, diagnostic, hybrid
# ---------------------------------------------------------------------------

def joint_influence_cov(
    T: np.ndarray,
    Y: np.ndarray,
    labeled_mask: np.ndarray,
    design: Dict[str, Any],
    lambda_hat: float,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Joint sandwich covariance of (tau_SI, tau_PPI) for the learned index.

    Returns at least ``var_si``, ``var_ppi``, ``cov_si_ppi`` and ``var_D``,
    where ``var_D = var_si + var_ppi - 2 cov_si_ppi`` is the variance of the
    SI--PPI++ estimator-disagreement statistic.
    """
    _require(_HAS_JOINT_COV, "joint_influence_cov")
    return _methods.joint_influence_cov(
        T, Y, labeled_mask, design, lambda_hat, **kwargs
    )


def hybrid_sigma(
    T: np.ndarray,
    Y: np.ndarray,
    labeled_mask: np.ndarray,
    design: Dict[str, Any],
    lambda_hat: float,
    **kwargs: Any,
) -> np.ndarray:
    """The 2x2 covariance of (tau_SI, tau_PPI) the hybrid is built from."""
    _require(_HAS_HYBRID_SIGMA, "hybrid_sigma")
    return np.asarray(
        _methods.hybrid_sigma(T, Y, labeled_mask, design, lambda_hat, **kwargs)
    )


def surrogacy_test(*args: Any, **kwargs: Any) -> Dict[str, float]:
    """SI--PPI++ estimator-disagreement diagnostic.

    Feed it ``var_si``, ``var_ppi`` and ``cov_si_ppi`` from
    :func:`joint_influence_cov`, or pass ``joint=<joint_influence_cov output>``.
    """
    return _methods.surrogacy_test(*args, **kwargs)


def surrogacy_test_sandwich(
    T: np.ndarray,
    Y: np.ndarray,
    labeled_mask: np.ndarray,
    design: Dict[str, Any],
    lambda_hat: float,
    tau_si: float,
    tau_ppi: float,
    **kwargs: Any,
) -> Dict[str, float]:
    """The diagnostic with SE(D) from the joint sandwich.

    This is the one-call form: it builds the 2 x 2 covariance for the learned
    index and returns the usual test keys plus ``joint``.
    """
    return _methods.surrogacy_test_sandwich(
        T, Y, labeled_mask, design, lambda_hat, tau_si, tau_ppi, **kwargs
    )


def estimate_cov_si_ppi(
    T: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
    labeled_mask: np.ndarray,
    lambda_hat: float,
    design: Optional[Dict[str, Any]] = None,
) -> float:
    """Cov(tau_SI, tau_PPI).

    With ``design`` it is the joint sandwich covariance for the learned index;
    without it, the fixed-predictor covariance, which is what the
    ablation columns report.
    """
    if design is None:
        return _methods.estimate_cov_si_ppi(
            T, Y, Y_hat, labeled_mask, lambda_hat
        )
    return _methods.estimate_cov_si_ppi(
        T, Y, Y_hat, labeled_mask, lambda_hat, design=design
    )


def paired_bootstrap(*args: Any, **kwargs: Any) -> Dict[str, float]:
    """Joint paired bootstrap of (tau_SI, tau_PPI), stratified by arm.

    ``paired_bootstrap(G, Y, T, labeled_mask, B, seed)`` resamples units
    within arm (labels attached), REFITS THE OLS INDEX on the prebuilt design
    ``G`` under the all-units protocol and recomputes both estimators.  It is
    the validation reference for the joint sandwich (run_joint_cov_check.py,
    run_nonrandom_label_variance_check.py).  It does not refit trees: the GBT
    rows of the comparison carry ``si_variance="plugin"``, and the
    tree-refitting analogue is ``gbt_paired_bootstrap`` in
    scripts/run_gbt_variance_check.py.
    """
    return _load_paired_bootstrap()(*args, **kwargs)


def _load_paired_bootstrap():
    """Import `paired_bootstrap` from scripts/run_joint_cov_check.py.

    `scripts/` is a directory of entry points, not a package, so the function
    is loaded by path.
    """
    global _PAIRED_BOOTSTRAP
    if _PAIRED_BOOTSTRAP is None:
        import importlib.util
        import os

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            "scripts", "run_joint_cov_check.py",
        )
        spec = importlib.util.spec_from_file_location(
            "_run_joint_cov_check", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PAIRED_BOOTSTRAP = module.paired_bootstrap
    return _PAIRED_BOOTSTRAP


_PAIRED_BOOTSTRAP = None


def hybrid_estimator(*args: Any, **kwargs: Any) -> Dict[str, float]:
    """Adaptive hybrid; feed it the :func:`hybrid_sigma` entries."""
    return _methods.hybrid_estimator(*args, **kwargs)
