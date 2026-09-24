"""
Configuration management for surrogate metrics simulation.

Defines default parameter values for each DGP, sweep ranges, the experiment
matrix, the prediction protocols, the method-configuration table (primary
versus ablation), and the ``ResultRow`` schema that every results writer emits.

"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Default parameters per DGP
# ---------------------------------------------------------------------------

DGP1_DEFAULTS: Dict[str, Any] = dict(
    n=10_000,
    alpha_S=5.0,
    beta_SX=1.0,
    gamma_S=0.3,
    sigma_S=2.0,
    alpha_Y=0.0,
    beta_YS=0.5,
    beta_YX=0.2,
    sigma_Y=1.0,
)

DGP2_DEFAULTS: Dict[str, Any] = dict(
    **DGP1_DEFAULTS,
    rho=0.3,
)

DGP3_DEFAULTS: Dict[str, Any] = dict(
    n=10_000,
    pi_group=0.3,
    beta_YS_high=0.8,
    beta_YS_low=0.2,
    gamma_S_high=0.5,
    gamma_S_low=0.2,
    alpha_S=5.0,
    beta_SX=1.0,
    sigma_S=2.0,
    alpha_Y=0.0,
    beta_YX=0.2,
    sigma_Y=1.0,
)

DGP4_DEFAULTS: Dict[str, Any] = dict(
    n=10_000,
    n_cal=50_000,
    alpha_S=5.0,
    beta_SX=1.0,
    gamma_S=0.3,
    sigma_S=2.0,
    alpha_Y=0.0,
    beta_YS_0=0.5,
    delta_beta=0.2,
    beta_YX=0.2,
    sigma_Y=1.0,
    mu_shift=0.0,
)

DGP5_DEFAULTS: Dict[str, Any] = dict(
    n=10_000,
    q=0.15,
    missingness="MCAR",
    alpha_S=5.0,
    beta_SX=1.0,
    gamma_S=0.3,
    sigma_S=2.0,
    alpha_Y=0.0,
    beta_YS=0.5,
    beta_YX=0.2,
    sigma_Y=1.0,
    eta_S=0.3,
)

DGP6_DEFAULTS: Dict[str, Any] = dict(
    K=100,
    n_min=500,
    n_max=5_000,
    pi_0=0.5,
    sigma_tau=0.10,
    alpha_S=5.0,
    beta_SX=1.0,
    sigma_S=2.0,
    alpha_Y=0.0,
    beta_YS=0.5,
    beta_YX=0.2,
    sigma_Y=1.0,
    alpha_decision=0.05,
    rho_mix=0.0,
)

DGP7_DEFAULTS: Dict[str, Any] = dict(
    n=10_000,
    gamma_1=0.3,
    gamma_2=0.2,
    gamma_3=0.1,
    beta_1=0.3,
    beta_2=0.4,
    beta_3=0.2,
    beta_YX=0.2,
    sigma_1=2.0,
    sigma_2=1.4142135623730951,  # sqrt(2)
    sigma_3=1.0,
    sigma_Y=1.0,
)

DGP8_DEFAULTS: Dict[str, Any] = dict(
    n=10_000,
    alpha_S=5.0,
    beta_SX=1.0,
    gamma_S=0.3,
    sigma_S=2.0,
    beta_YS_linear=0.5,
    beta_YS_quad=-0.03,
    beta_YX=0.2,
    sigma_Y=1.0,
)

DGP9_DEFAULTS: Dict[str, Any] = dict(
    n=10_000,
    alpha_S=5.0,
    beta_SX=1.0,
    gamma_S=0.3,
    sigma_S=2.0,
    beta_YS=-0.3,
    beta_YX=0.8,
    delta=0.5,
    sigma_Y=1.0,
)

DGP10_DEFAULTS: Dict[str, Any] = dict(
    n=10_000,
    alpha_S=5.0,
    beta_SX=1.0,
    gamma_S=0.3,
    sigma_S=2.0,
    beta_YS_linear=0.5,
    beta_YS_quad=-0.03,
    beta_YX=0.2,
    sigma_Y=1.0,
    rho=0.2,
)

DGP_DEFAULTS = {
    1: DGP1_DEFAULTS,
    2: DGP2_DEFAULTS,
    3: DGP3_DEFAULTS,
    4: DGP4_DEFAULTS,
    5: DGP5_DEFAULTS,
    6: DGP6_DEFAULTS,
    7: DGP7_DEFAULTS,
    8: DGP8_DEFAULTS,
    9: DGP9_DEFAULTS,
    10: DGP10_DEFAULTS,
}


# ---------------------------------------------------------------------------
# Sweep ranges
# ---------------------------------------------------------------------------

PI_L_SWEEP_STANDARD = [0.05, 0.10, 0.20, 0.50, 0.80, 1.00]

DGP_PI_L_SWEEPS: Dict[int, List[float]] = {
    1: PI_L_SWEEP_STANDARD,
    2: PI_L_SWEEP_STANDARD,
    3: PI_L_SWEEP_STANDARD,
    4: PI_L_SWEEP_STANDARD,
    5: [0.05, 0.10, 0.15, 0.20, 0.30],  # determined by q, not pi_L
    6: [0.10, 0.20, 0.50, 1.00],
    7: PI_L_SWEEP_STANDARD,
    8: PI_L_SWEEP_STANDARD,
    9: PI_L_SWEEP_STANDARD,
    10: PI_L_SWEEP_STANDARD,
}

DGP_PARAM_SWEEPS: Dict[int, Dict[str, List[Any]]] = {
    1: {},  # no parameter sweep for baseline
    2: {"rho": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]},
    3: {"beta_YS_low": [0.0, 0.1, 0.2, 0.3, 0.4]},
    4: {"delta_beta": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]},
    5: {"q": [0.05, 0.10, 0.15, 0.20, 0.30]},
    6: {"K": [50, 100, 200]},
    7: {},  # no parameter sweep for multi-surrogate baseline
    8: {},  # no parameter sweep for nonlinear baseline
    9: {},  # no parameter sweep for antagonistic baseline
    10: {"rho": [0.0, 0.2, 0.4]},
}

# Sample size sweep (shared)
N_SWEEP = [2_000, 5_000, 10_000, 50_000]

# All methods
ALL_METHODS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Default number of replications
DEFAULT_R = 2_000

# Default master seed
DEFAULT_MASTER_SEED = 42


# ---------------------------------------------------------------------------
# Configuration data class
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Configuration for a single simulation cell."""
    dgp_id: int
    dgp_params: Dict[str, Any]
    pi_L: float
    method_ids: List[int]
    R: int = DEFAULT_R
    master_seed: int = DEFAULT_MASTER_SEED
    config_id: int = 0
    pi_L_id: int = 0
    label: str = ""

    def to_dgp_kwargs(self) -> Dict[str, Any]:
        """
        Build the kwargs dict to pass to the DGP generator.

        For DGPs 1-4, pi_L is passed directly.
        For DGP 5, pi_L is determined by q (not passed as pi_L).
        For DGP 6, pi_L is passed directly.
        """
        kwargs = dict(self.dgp_params)
        if self.dgp_id == 5:
            # pi_L determined by q, which should already be in dgp_params
            pass
        else:
            kwargs["pi_L"] = self.pi_L
        return kwargs


# ---------------------------------------------------------------------------
# Experiment matrix generation
# ---------------------------------------------------------------------------

def generate_experiment_matrix(
    dgp_ids: Optional[Sequence[int]] = None,
    method_ids: Optional[Sequence[int]] = None,
    R: int = DEFAULT_R,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> List[SimulationConfig]:
    """
    Generate the full experiment matrix as a list of SimulationConfig objects.

    This creates one config per (DGP, parameter-config, pi_L) cell.
    Methods are included in each config (all configs run all methods).

    Parameters
    ----------
    dgp_ids : which DGPs to include (default: all 1-6)
    method_ids : which methods to include (default: all 0-5)
    R : replications per cell
    master_seed : master seed

    Returns
    -------
    List of SimulationConfig objects.
    """
    if dgp_ids is None:
        dgp_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if method_ids is None:
        method_ids = ALL_METHODS

    configs: List[SimulationConfig] = []
    global_config_id = 0

    for dgp_id in dgp_ids:
        defaults = DGP_DEFAULTS[dgp_id]
        pi_L_values = DGP_PI_L_SWEEPS[dgp_id]
        param_sweep = DGP_PARAM_SWEEPS[dgp_id]

        # Generate parameter configurations
        if not param_sweep:
            # No sweep -- single default config
            param_configs = [{}]
        else:
            # Cartesian product of all sweep params
            keys = sorted(param_sweep.keys())
            value_lists = [param_sweep[k] for k in keys]
            param_configs = []
            for vals in itertools.product(*value_lists):
                param_configs.append(dict(zip(keys, vals)))

        for p_idx, param_override in enumerate(param_configs):
            dgp_params = dict(defaults)
            dgp_params.update(param_override)

            # For DGP 5, pi_L is determined by q
            if dgp_id == 5:
                # The pi_L sweep IS the q sweep
                q_values = param_sweep.get("q", [0.15])
                # Only iterate over pi_L values if this param_override
                # sets q; avoid double-looping.
                q_val = param_override.get("q", dgp_params.get("q", 0.15))
                pi_L_iter = [q_val]  # pi_L ~ q for DGP 5
            else:
                pi_L_iter = pi_L_values

            for pi_L_idx, pi_L in enumerate(pi_L_iter):
                # Build label
                override_str = ", ".join(
                    f"{k}={v}" for k, v in sorted(param_override.items())
                )
                label = f"DGP{dgp_id}"
                if override_str:
                    label += f" ({override_str})"
                label += f", pi_L={pi_L}"

                cfg = SimulationConfig(
                    dgp_id=dgp_id,
                    dgp_params=dgp_params,
                    pi_L=pi_L,
                    method_ids=list(method_ids),
                    R=R,
                    master_seed=master_seed,
                    config_id=global_config_id,
                    pi_L_id=pi_L_idx,
                    label=label,
                )
                configs.append(cfg)
                global_config_id += 1

    return configs


def get_default_config(
    dgp_id: int,
    pi_L: float = 0.20,
    R: int = DEFAULT_R,
    master_seed: int = DEFAULT_MASTER_SEED,
    **overrides: Any,
) -> SimulationConfig:
    """
    Get a single simulation config with defaults for a given DGP.

    Parameters
    ----------
    dgp_id : DGP number (1-6)
    pi_L : labeled fraction
    R : number of replications
    master_seed : master seed
    **overrides : any DGP-specific parameter overrides

    Returns
    -------
    SimulationConfig
    """
    defaults = dict(DGP_DEFAULTS[dgp_id])
    defaults.update(overrides)

    label = f"DGP{dgp_id}, pi_L={pi_L}"
    if overrides:
        override_str = ", ".join(f"{k}={v}" for k, v in sorted(overrides.items()))
        label += f" ({override_str})"

    return SimulationConfig(
        dgp_id=dgp_id,
        dgp_params=defaults,
        pi_L=pi_L,
        method_ids=ALL_METHODS,
        R=R,
        master_seed=master_seed,
        config_id=0,
        pi_L_id=0,
        label=label,
    )


def count_total_cells(
    dgp_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Count the total number of simulation cells (matching Section 6.4).

    Returns a dict with per-DGP and total cell counts.
    """
    configs = generate_experiment_matrix(dgp_ids=dgp_ids)

    counts: Dict[int, int] = {}
    for cfg in configs:
        dgp = cfg.dgp_id
        counts[dgp] = counts.get(dgp, 0) + 1

    total = sum(counts.values())
    # Each cell runs all 6 methods, so multiply by method count
    # Actually, each config already includes all methods, so the cell count
    # from the framework counts (DGP x pi_L x param_config x method)
    n_methods = len(ALL_METHODS)

    return dict(
        per_dgp_configs=counts,
        total_configs=total,
        methods_per_config=n_methods,
        total_cells=total * n_methods,
        replications_per_cell=DEFAULT_R,
        total_runs=total * n_methods * DEFAULT_R,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

#: Version 2 = {"registry_version": 2, "rows": [...]} where every element of
#: "rows" is a ResultRow.
REGISTRY_VERSION = 2

#: Prediction protocols. ``allunits_crossfit``: every unit is assigned to one
#: of K folds; the index is fit on the *labeled* units outside the fold and
#: used to predict *every* unit in the fold. ``allunits_crossfit_stratified``
#: (opt-in): as ``allunits_crossfit`` with the folds assigned by permutation
#: within each (arm, labeled) stratum. The reported results use the
#: unstratified default.
PROTOCOLS = ("allunits_crossfit", "mixed_fit", "allunits_crossfit_stratified")
DEFAULT_PROTOCOL = "allunits_crossfit"

#: Targets a row can be estimating.
TARGETS = ("ATE", "masking_target")

#: Allowed values of the per-method configuration axes.
LAMBDA_RULES = ("exact", "plugin", None)
VARIANCE_RULES = ("exact", "plugin", "bootstrap", None)
#: "bootstrap" is a paired-bootstrap SI variance.  No reported result row
#: uses it: GBT rows, where the joint sandwich is unavailable (the design
#: matrix is None), record "plugin"; the tree-refitting bootstrap is run only
#: in the two-cell check scripts/run_gbt_variance_check.py.
SI_VARIANCE_RULES = ("sandwich", "plugin", "delta", "bootstrap", None)


@dataclass(frozen=True)
class MethodSpec:
    """One method *configuration*: a method id plus its estimator options.

    ``label`` is the paper label; it is the key that tables and figures select
    on.  ``cfg()`` is the kwargs dict handed to
    ``src.utils.estimator_api.estimate``.
    """

    method_id: int
    label: str
    lambda_rule: Optional[str] = None
    clip: Optional[bool] = None
    per_arm: Optional[bool] = None
    variance: Optional[str] = None
    si_variance: Optional[str] = None
    tier: str = "primary"          # "primary" or "ablation"
    key: str = ""                  # short machine key, unique across specs

    def cfg(self) -> Dict[str, Any]:
        """Estimator kwargs for this configuration (only the set axes)."""
        out: Dict[str, Any] = {}
        if self.lambda_rule is not None:
            out["lambda_rule"] = self.lambda_rule
        if self.clip is not None:
            out["clip"] = self.clip
        if self.per_arm is not None:
            out["per_arm"] = self.per_arm
        if self.variance is not None:
            out["variance"] = self.variance
        if self.si_variance is not None:
            out["si_variance"] = self.si_variance
        return out

    def row_fields(self) -> Dict[str, Any]:
        """The schema fields this spec contributes to a ResultRow."""
        return dict(
            method_id=self.method_id,
            method_label=self.label,
            lambda_rule=self.lambda_rule,
            clip=self.clip,
            per_arm=self.per_arm,
            variance=self.variance,
            si_variance=self.si_variance,
        )


# --- The paper's primary configurations (headline tables and figures) ------
#
# "PPI++" unqualified = exact-variance tuning rule, a common unclipped
# coefficient across arms, and the exact (overlap-corrected) variance.  This
# is standard PPI++ for a scalar coefficient; method id 9 is an alias of
# id 3 in that default configuration.

# GREG (method 4) is NOT primary: with a common, unclipped coefficient tuned
# by the exact rule, PPI++ and GREG are the identical point estimator, so a
# headline table that printed both would print the same estimator twice.  GREG
# stays in the ablation, where the comparison is the point of the row.
PRIMARY_METHOD_SPECS: Tuple[MethodSpec, ...] = (
    MethodSpec(0, "Labeled-Only", tier="primary", key="lo"),
    MethodSpec(1, "Naive Surrogate", tier="primary", key="ns"),
    MethodSpec(2, "SI", si_variance="sandwich", tier="primary", key="si"),
    MethodSpec(
        3, "PPI++", lambda_rule="exact", clip=False, per_arm=False,
        variance="exact", tier="primary", key="ppi",
    ),
    MethodSpec(5, "Composite Proxy", tier="primary", key="cp"),
    MethodSpec(6, "AIPW", tier="primary", key="aipw"),
)

# --- Ablation configurations (the old rules) -------------------------------
#
# Run in the same replication as the primary configurations, on the same
# draws, so the ablation tables are paired with the headline tables.

ABLATION_METHOD_SPECS: Tuple[MethodSpec, ...] = (
    # GREG: the same point estimator as the primary PPI++ under a common,
    # unclipped, exact-rule coefficient; kept so the equivalence can be read
    # off the table rather than asserted.
    MethodSpec(4, "GREG", tier="ablation", key="greg"),
    # "PPI++ (clipped)" is the plug-in-rule PPI++ point estimator: DGP 9 is where
    # the clip binds, so that DGP's table prints it beside the primary.
    MethodSpec(
        3, "PPI++ (plug-in lambda)", lambda_rule="plugin", clip=False,
        per_arm=False, variance="exact", tier="ablation", key="ppi_plugin_lam",
    ),
    MethodSpec(
        3, "PPI++ (plug-in variance)", lambda_rule="exact", clip=False,
        per_arm=False, variance="plugin", tier="ablation",
        key="ppi_plugin_var",
    ),
    MethodSpec(
        3, "PPI++ (clipped)", lambda_rule="exact", clip=True, per_arm=False,
        variance="exact", tier="ablation", key="ppi_clipped",
    ),
    MethodSpec(
        3, "PPI++ (per-arm lambda)", lambda_rule="exact", clip=False,
        per_arm=True, variance="exact", tier="ablation", key="ppi_per_arm",
    ),
    # The plug-in-rule default estimator AND its plug-in-rule interval: the
    # plug-in tuning rule with a clipped common coefficient, paired with the
    # plug-in (uncorrected) variance. The point estimator is the one method id
    # 8 carries ("PPI++ (plug-in lambda, clipped)"); only the variance differs,
    # so the two rows isolate the overlap correction at the plug-in rule.
    MethodSpec(
        3, "PPI++ (plug-in lambda, plug-in variance)", lambda_rule="plugin",
        clip=True, per_arm=False, variance="plugin", tier="ablation",
        key="ppi_pluginrule",
    ),
    MethodSpec(
        2, "SI (plug-in variance)", si_variance="plugin", tier="ablation",
        key="si_plugin_var",
    ),
    MethodSpec(
        2, "SI (delta-method first stage)", si_variance="delta",
        tier="ablation", key="si_delta",
    ),
)

ALL_METHOD_SPECS: Tuple[MethodSpec, ...] = (
    PRIMARY_METHOD_SPECS + ABLATION_METHOD_SPECS
)

#: Also aliased: the spelling `src.methods.display_name` generates for the
#: plug-in-rule ablation configuration. That function names every departure
#: from the default in a fixed order, so it writes out the clip; the method
#: label does not. Both spellings mean method-spec key "ppi_pluginrule".
LABEL_ALIASES: Dict[str, str] = {
    "PPI++ (plug-in lambda, plug-in variance, clipped)":
        "PPI++ (plug-in lambda, plug-in variance)",
    "Surrogate Index": "SI",
    "Surr. Index": "SI",
    "Naive Surr.": "Naive Surrogate",
    "Composite": "Composite Proxy",
}


def canonical_label(label: str) -> str:
    """Normalize a method display string to a paper label."""
    return LABEL_ALIASES.get(str(label).strip(), str(label).strip())


def spec_by_key(key: str) -> MethodSpec:
    """Look up a MethodSpec by its short machine key."""
    for spec in ALL_METHOD_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"No MethodSpec with key {key!r}")


def spec_by_label(label: str) -> MethodSpec:
    """Look up a MethodSpec by its paper label.

    Falls back to the pinned specs of the extra method ids (7, 8, 9), so a
    table that prints "PPI++ (bootstrap variance)" or
    "PPI++ (plug-in lambda, clipped)" still resolves to a spec.
    """
    want = canonical_label(label)
    for spec in ALL_METHOD_SPECS:
        if spec.label == want:
            return spec
    for spec in _EXTRA_ID_SPECS.values():
        if spec.label == want:
            return spec
    raise KeyError(f"No MethodSpec labelled {label!r}")


# ---------------------------------------------------------------------------
# ResultRow
# ---------------------------------------------------------------------------

#: Identification fields.  Every row carries all of them.
RESULT_ROW_ID_FIELDS: Tuple[str, ...] = (
    "dgp",           # int, or a string tag for non-DGP analyses
                     #   ("hillstrom", "lalonde", "criteo", "semisynthetic", ...)
    "config_name",   # str, the cell label, e.g. "DGP2_rho0.4_piL0.20"
    "params",        # dict, the DGP / analysis parameters of the cell
    "pi_L",          # float labeled fraction (nan when not applicable)
    "n",             # int sample size per replication (nan when it varies)
    "R",             # int replications behind the metrics
    "protocol",      # one of PROTOCOLS
    "method_id",     # int method id
    "method_label",  # str paper label (the selection key)
    "lambda_rule",   # "exact" | "plugin" | None
    "clip",          # bool | None
    "per_arm",       # bool | None
    "variance",      # "exact" | "plugin" | None
    "si_variance",   # "sandwich" | "plugin" | "delta" | None
    "alpha",         # float test level behind `rejection` / `cdr` / the CI
    "target",        # "ATE" | "masking_target"
    "seed",          # int master seed the cell's replication seeds derive from
)

#: Metric fields.  Writers emit the ones their analysis produces; the rest are
#: filled with nan so every row has the same keys.
RESULT_ROW_METRIC_FIELDS: Tuple[str, ...] = (
    "true_tau",
    "n_valid",
    "bias",
    "rel_bias",
    "rmse",
    "coverage",
    "ci_width",
    "rejection",
    "cdr",
    "relative_efficiency",
    "mc_se_bias",
    "mc_se_rmse",
    "mc_se_coverage",
    "mc_se_ci_width",
    "mc_se_rejection",
    "mc_se_cdr",
)

#: Optional portfolio (DGP 6) metrics, emitted only by the portfolio runners.
RESULT_ROW_PORTFOLIO_FIELDS: Tuple[str, ...] = (
    "mean_cumul_regret",
    "rel_regret",
    "oracle_gain",
    "mc_se_regret",
)

RESULT_ROW_FIELDS: Tuple[str, ...] = (
    RESULT_ROW_ID_FIELDS + RESULT_ROW_METRIC_FIELDS
)


def make_result_row(
    *,
    dgp: Any,
    config_name: str,
    params: Optional[Dict[str, Any]] = None,
    pi_L: float = float("nan"),
    n: Any = float("nan"),
    R: int = 0,
    protocol: str = DEFAULT_PROTOCOL,
    spec: Optional[MethodSpec] = None,
    method_id: Optional[int] = None,
    method_label: Optional[str] = None,
    alpha: float = 0.05,
    target: str = "ATE",
    seed: int = DEFAULT_MASTER_SEED,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one ResultRow dict in the canonical schema.

    ``metrics`` supplies whichever of RESULT_ROW_METRIC_FIELDS the analysis
    computed; missing ones become nan.  ``extra`` adds analysis-specific
    columns (portfolio regret, diagnostic power, masking-target gaps, ...);
    they are kept verbatim alongside the schema fields.
    """
    nan = float("nan")
    row: Dict[str, Any] = {
        "dgp": dgp,
        "config_name": config_name,
        "params": dict(params or {}),
        "pi_L": float(pi_L) if pi_L is not None else nan,
        "n": n,
        "R": int(R),
        "protocol": protocol,
        "alpha": float(alpha),
        "target": target,
        "seed": int(seed),
    }

    if spec is not None:
        row.update(spec.row_fields())
        if method_label is not None:
            row["method_label"] = method_label
    else:
        if method_id is None or method_label is None:
            raise ValueError(
                "make_result_row needs either spec= or "
                "(method_id=, method_label=)"
            )
        row.update(
            method_id=int(method_id),
            method_label=method_label,
            lambda_rule=None,
            clip=None,
            per_arm=None,
            variance=None,
            si_variance=None,
        )

    metrics = dict(metrics or {})
    for field_name in RESULT_ROW_METRIC_FIELDS:
        row[field_name] = metrics.pop(field_name, nan)
    # Anything left in `metrics` was not part of the schema: keep it, but it
    # will not be validated.
    row.update(metrics)
    row.update(extra or {})

    if row["protocol"] not in PROTOCOLS:
        raise ValueError(
            f"protocol {row['protocol']!r} not in {PROTOCOLS}"
        )
    if row["target"] not in TARGETS:
        raise ValueError(f"target {row['target']!r} not in {TARGETS}")
    return row


def validate_result_row(row: Dict[str, Any]) -> None:
    """Raise ValueError if `row` is missing a schema field or has a bad value."""
    missing = [f for f in RESULT_ROW_FIELDS if f not in row]
    if missing:
        raise ValueError(f"ResultRow missing fields: {missing}")
    if row["protocol"] not in PROTOCOLS:
        raise ValueError(f"bad protocol {row['protocol']!r}")
    if row["target"] not in TARGETS:
        raise ValueError(f"bad target {row['target']!r}")
    if row["lambda_rule"] not in LAMBDA_RULES:
        raise ValueError(f"bad lambda_rule {row['lambda_rule']!r}")
    if row["variance"] not in VARIANCE_RULES:
        raise ValueError(f"bad variance {row['variance']!r}")
    if row["si_variance"] not in SI_VARIANCE_RULES:
        raise ValueError(f"bad si_variance {row['si_variance']!r}")


# ---------------------------------------------------------------------------
# Spec resolution for the runners
# ---------------------------------------------------------------------------

#: Method ids that are not paper-primary configurations but that older
#: callers still ask for by id.
_EXTRA_ID_SPECS: Dict[int, MethodSpec] = {
    # Method 7 is pinned to the plug-in rule with a clipped coefficient and
    # a bootstrap variance; method 8 to the plug-in rule, clipped, with the
    # exact variance.  Neither is a primary configuration.
    7: MethodSpec(
        7, "PPI++ (bootstrap variance)", lambda_rule="plugin", clip=True,
        per_arm=False, variance="bootstrap", tier="ablation", key="ppi_boot",
    ),
    8: MethodSpec(
        8, "PPI++ (plug-in lambda, clipped)", lambda_rule="plugin", clip=True,
        per_arm=False, variance="exact", tier="ablation", key="ppi_m8",
    ),
    # Id 9 is an alias of id 3 in its default configuration: the same
    # estimator, so it resolves to the primary PPI++ spec.
    9: spec_by_key("ppi"),
}


def default_spec_for_id(method_id: int) -> MethodSpec:
    """The configuration a bare method id means under the defaults.

    Primary configurations win; ids that are only ablations (GREG, the PPI++
    variance variants) resolve to their ablation spec.
    """
    for spec in PRIMARY_METHOD_SPECS:
        if spec.method_id == method_id:
            return spec
    if method_id in _EXTRA_ID_SPECS:
        return _EXTRA_ID_SPECS[method_id]
    for spec in ABLATION_METHOD_SPECS:
        if spec.method_id == method_id:
            return spec
    raise KeyError(f"No default MethodSpec for method id {method_id}")


def resolve_specs(
    method_ids: Optional[Sequence[int]] = None,
    specs: Optional[Sequence[MethodSpec]] = None,
    include_ablation: bool = False,
) -> List[MethodSpec]:
    """Resolve a runner's method selection to a list of MethodSpec.

    ``specs`` wins if given.  Otherwise ``method_ids`` are resolved to their
    default (primary) configurations; with ``method_ids=None`` the full
    primary list is used.  ``include_ablation=True`` appends the ablation
    configurations, which is how the default runs keep the ablation paired
    with the headline results: same draws, same predictions, same replication.
    """
    if specs is not None:
        out = list(specs)
    elif method_ids is None:
        out = list(PRIMARY_METHOD_SPECS)
    else:
        seen: set = set()
        out = []
        for m_id in method_ids:
            spec = default_spec_for_id(int(m_id))
            if spec.key in seen:
                continue
            seen.add(spec.key)
            out.append(spec)
    if include_ablation:
        have = {s.key for s in out}
        out += [s for s in ABLATION_METHOD_SPECS if s.key not in have]
    return out
