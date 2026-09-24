"""
The schema itself lives in ``src/utils/config.py`` (``make_result_row``,
``RESULT_ROW_FIELDS``, ``MethodSpec``, ``PRIMARY_METHOD_SPECS``,
``ABLATION_METHOD_SPECS``).  This module is the reading side: every table
writer and figure script selects rows through ``select(...)`` rather
than by pattern-matching method display strings.

On-disk format (results schema version 2)::

    {
      "registry_version": 2,
      "generated_by": "run_dgp.py",
      "protocol": "allunits_crossfit",
      "rows": [ {ResultRow}, ... ]
    }

Version-1 files are still readable; ``load_rows`` upgrades them
in memory and stamps ``protocol="mixed_fit"`` on every upgraded row so that a
mixed read can never silently blend conventions.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.utils.config import (
    ABLATION_METHOD_SPECS,
    spec_by_label,
    DEFAULT_PROTOCOL,
    MethodSpec,
    PRIMARY_METHOD_SPECS,
    REGISTRY_VERSION,
    RESULT_ROW_FIELDS,
    canonical_label,
    make_result_row,
    validate_result_row,
)

__all__ = [
    "primary_methods",
    "ablation_methods",
    "primary_labels",
    "ablation_labels",
    "write_rows",
    "load_rows",
    "select",
    "select_one",
    "rows_to_frame",
    "registry_version_of",
    "rows_from_replications",
    "metrics_from_replications",
    "metrics_from_errors",
    "replications_long",
    "rows_from_summary",
    "write_table_rows",
]


# ---------------------------------------------------------------------------
# Method-configuration lists
# ---------------------------------------------------------------------------

def primary_methods() -> List[MethodSpec]:
    """The paper's primary configurations, in table order.

    LO, NS, SI (sandwich), PPI++ (exact tuning rule, common unclipped
    coefficient, exact variance), GREG, CP, AIPW.
    """
    return list(PRIMARY_METHOD_SPECS)


def ablation_methods() -> List[MethodSpec]:
    """The ablation configurations (the old rules), in table order.

    GREG, PPI++ (plug-in lambda), PPI++ (plug-in variance), PPI++ (clipped),
    PPI++ (per-arm lambda), PPI++ (plug-in lambda, plug-in variance) (the
    plug-in-rule estimator and its plug-in-rule interval), SI (plug-in variance), SI
    (delta-method first stage).
    """
    return list(ABLATION_METHOD_SPECS)


def primary_labels() -> List[str]:
    return [s.label for s in PRIMARY_METHOD_SPECS]


def ablation_labels() -> List[str]:
    return [s.label for s in ABLATION_METHOD_SPECS]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """Make numpy scalars / arrays and nan JSON-safe."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "shape", ()) == ():
        obj = obj.item()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    return str(obj)


#: Row keys that mark a per-replication record.
_REPLICATION_KEYS = ("rep", "replication")


def write_rows(
    path: str,
    rows: Sequence[Dict[str, Any]],
    generated_by: str = "",
    protocol: str = DEFAULT_PROTOCOL,
    validate: bool = True,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Write result rows to `path` as a version-2 eval JSON.

    Returns the path written.  With ``validate=True`` every row is checked
    against the schema first, so a writer that forgets a field fails loudly at
    write time rather than silently at table time.

    A row that carries a replication index (``rep`` / ``replication``) is
    rejected whatever ``validate`` says: per-replication output belongs in a
    raw file (``results/tables/*_raw.csv`` or ``results/raw/``), summarized
    with :func:`rows_from_replications` or :func:`metrics_from_errors`.
    """
    for row in rows:
        hit = [k for k in _REPLICATION_KEYS if k in row]
        if hit:
            raise ValueError(
                f"{os.path.basename(path)}: row carries replication field(s) "
                f"{hit}; result files hold summary rows only.  Write the "
                f"replications to a raw file and summarize them with "
                f"rows_from_replications / metrics_from_errors."
            )
    if validate:
        for row in rows:
            validate_result_row(row)
    payload: Dict[str, Any] = {
        "registry_version": REGISTRY_VERSION,
        "generated_by": generated_by,
        "protocol": protocol,
        "rows": [_jsonable(r) for r in rows],
    }
    if extra_meta:
        payload.update(_jsonable(extra_meta))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def registry_version_of(payload: Any) -> int:
    """Return the results schema version of a loaded JSON payload (1 or 2)."""
    if isinstance(payload, dict) and "rows" in payload:
        return int(payload.get("registry_version", REGISTRY_VERSION))
    return 1


_V1_LABELS = {
    0: "Labeled-Only",
    1: "Naive Surrogate",
    2: "SI",
    3: "PPI++ (version-1 plug-in rule)",
    4: "GREG",
    5: "Composite Proxy",
    6: "AIPW",
    7: "PPI++ (bootstrap variance)",
    8: "PPI++ (version-1 corrected variance)",
    9: "PPI++ (exact lambda)",
}


def _upgrade_v1(payload: Any, source: str) -> List[Dict[str, Any]]:
    """Upgrade a version-1 eval JSON to result rows.

    The v1 files are lists of ``{dgp_id, label, elapsed, eval: [...]}`` (or
    ``{..., methods: {...}}`` for the portfolio DGP).
    """
    rows: List[Dict[str, Any]] = []
    entries = payload if isinstance(payload, list) else [payload]

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        dgp = entry.get("dgp_id", entry.get("dgp", "unknown"))
        config_name = entry.get("label", "")
        pi_L = float("nan")
        # Labels look like "DGP2_rho0.4_piL0.20"; recover pi_L when present.
        if "piL" in str(config_name):
            try:
                pi_L = float(str(config_name).split("piL")[1].split("_")[0])
            except (ValueError, IndexError):
                pi_L = float("nan")

        eval_rows = entry.get("eval")
        if eval_rows is None and "methods" in entry:
            eval_rows = []
            for m_id, metrics in (entry.get("methods") or {}).items():
                d = dict(metrics)
                d["method_id"] = int(m_id)
                eval_rows.append(d)
        for er in eval_rows or []:
            m_id = int(er.get("method_id", -1))
            label = _V1_LABELS.get(m_id) or canonical_label(
                er.get("method_name", er.get("method_label", ""))
            ) or f"Method {m_id}"
            metrics = {
                k: er.get(k, float("nan"))
                for k in (
                    "true_tau", "n_valid", "bias", "rel_bias", "rmse",
                    "coverage", "cdr", "mc_se_bias", "mc_se_coverage",
                    "mc_se_cdr",
                )
            }
            metrics["rejection"] = er.get("rejection_rate", float("nan"))
            metrics["relative_efficiency"] = er.get(
                "relative_efficiency", float("nan")
            )
            extra = {
                k: v for k, v in er.items()
                if k in ("mean_regret", "mean_cumul_regret", "rel_regret",
                         "relative_regret", "oracle_gain", "std_regret",
                         "mc_se_oracle_gain", "mc_se_rel_regret",
                         "rel_regret_mean_of_ratios",
                         "mc_se_rel_regret_mean_of_ratios")
            }
            rows.append(make_result_row(
                dgp=dgp,
                config_name=config_name,
                params={},
                pi_L=pi_L,
                R=int(er.get("n_valid", 0) or 0),
                protocol="mixed_fit",
                method_id=int(er.get("method_id", -1)),
                method_label=label or f"Method {er.get('method_id')}",
                metrics=metrics,
                extra={"source_file": source, **extra},
            ))
    return rows


def load_rows(
    paths: Sequence[str] | str,
    upgrade_v1: bool = True,
    missing_ok: bool = False,
) -> List[Dict[str, Any]]:
    """Load result rows from one or several eval JSON files.

    Version-2 files are read directly.
    """
    if isinstance(paths, str):
        paths = [paths]

    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not os.path.exists(path):
            if missing_ok:
                continue
            raise FileNotFoundError(path)
        with open(path) as fh:
            payload = json.load(fh)
        version = registry_version_of(payload)
        if version >= 2:
            file_rows = payload.get("rows", [])
            for row in file_rows:
                row.setdefault("source_file", os.path.basename(path))
                for field_name in RESULT_ROW_FIELDS:
                    row.setdefault(field_name, float("nan"))
            rows.extend(file_rows)
        else:
            if not upgrade_v1:
                raise ValueError(
                    f"{path} is a schema-version-1 artifact; regenerate it "
                    f"or pass upgrade_v1=True"
                )
            rows.extend(_upgrade_v1(payload, os.path.basename(path)))
    return rows


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _matches(row: Dict[str, Any], key: str, want: Any) -> bool:
    if want is None:
        return True
    have = row.get(key, None)
    if key == "method_label":
        have = canonical_label(have) if have is not None else have
        if isinstance(want, (list, tuple, set)):
            return have in {canonical_label(w) for w in want}
        return have == canonical_label(want)
    if isinstance(want, (list, tuple, set)):
        return have in want
    if isinstance(want, float) and isinstance(have, (int, float)):
        return abs(float(have) - want) < 1e-12
    return have == want


def select(
    rows: Iterable[Dict[str, Any]],
    **criteria: Any,
) -> List[Dict[str, Any]]:
    """Select result rows by any schema field.

    Every keyword is a schema field name; the value is either a scalar or a
    collection of accepted values.

    Examples
    --------
    >>> select(rows, method_label="PPI++", protocol="allunits_crossfit")
    >>> select(rows, dgp=2, pi_L=0.20, method_label=primary_labels())
    """
    out = []
    for row in rows:
        if all(_matches(row, k, v) for k, v in criteria.items()):
            out.append(row)
    return out


def select_one(
    rows: Iterable[Dict[str, Any]],
    default: Any = None,
    **criteria: Any,
) -> Optional[Dict[str, Any]]:
    """Select exactly one row; returns `default` if none, raises if several."""
    hits = select(rows, **criteria)
    if not hits:
        return default
    if len(hits) > 1:
        raise ValueError(
            f"select_one matched {len(hits)} rows for {criteria}"
        )
    return hits[0]


def rows_to_frame(rows: Iterable[Dict[str, Any]]):
    """Flatten result rows to a pandas DataFrame.

    ``params`` is expanded into ``param_<key>`` columns so that a cell's DGP
    parameters (rho, delta_beta, q, K, ...) are selectable from the CSV.
    """
    import pandas as pd

    flat = []
    for row in rows:
        d = {k: v for k, v in row.items() if k != "params"}
        for pk, pv in (row.get("params") or {}).items():
            d[f"param_{pk}"] = pv
        flat.append(d)
    frame = pd.DataFrame(flat)
    ordered = [c for c in RESULT_ROW_FIELDS if c in frame.columns]
    rest = [c for c in frame.columns if c not in ordered]
    return frame[ordered + sorted(rest)]


# ---------------------------------------------------------------------------
# Monte Carlo metrics -> result rows
# ---------------------------------------------------------------------------

def metrics_from_replications(
    tau_hat,
    true_tau,
    var_hat=None,
    ci_lower=None,
    ci_upper=None,
    alpha: float = 0.05,
    rmse_baseline: float = float("nan"),
) -> Dict[str, float]:
    """Monte Carlo metrics for one method configuration in one cell.

    ``true_tau`` may be a scalar or a per-replication array (DGP 3 and the
    portfolio DGP draw a new truth each replication); the comparisons are made
    replication by replication either way.

    Returns the RESULT_ROW_METRIC_FIELDS subset that these inputs determine.
    """
    import numpy as np
    from scipy import stats as _stats

    tau_hat = np.asarray(tau_hat, dtype=float)
    truth = np.broadcast_to(
        np.asarray(true_tau, dtype=float), tau_hat.shape
    ).astype(float)
    nan = float("nan")

    ok = ~np.isnan(tau_hat)
    n_valid = int(ok.sum())
    out: Dict[str, float] = {
        "true_tau": float(np.nanmean(truth)) if truth.size else nan,
        "n_valid": n_valid,
    }
    if n_valid < 2:
        return out

    err = tau_hat[ok] - truth[ok]
    bias = float(err.mean())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mean_truth = float(np.mean(truth[ok]))
    out.update(
        bias=bias,
        rel_bias=(bias / mean_truth * 100.0) if abs(mean_truth) > 1e-12 else nan,
        rmse=rmse,
        mc_se_bias=float(err.std(ddof=1) / np.sqrt(n_valid)),
        # MC SE of the RMSE via the delta method on the mean squared error.
        mc_se_rmse=float(
            (err ** 2).std(ddof=1) / np.sqrt(n_valid) / (2.0 * rmse)
        ) if rmse > 1e-15 else nan,
    )
    if not np.isnan(rmse_baseline) and rmse > 1e-15:
        out["relative_efficiency"] = float(rmse_baseline / rmse)

    if ci_lower is not None and ci_upper is not None:
        lo = np.asarray(ci_lower, dtype=float)[ok]
        hi = np.asarray(ci_upper, dtype=float)[ok]
        good = ~(np.isnan(lo) | np.isnan(hi))
        if good.sum() >= 2:
            covers = (lo[good] <= truth[ok][good]) & (truth[ok][good] <= hi[good])
            cov = float(covers.mean())
            width = hi[good] - lo[good]
            out.update(
                coverage=cov,
                mc_se_coverage=float(np.sqrt(cov * (1 - cov) / good.sum())),
                ci_width=float(width.mean()),
                mc_se_ci_width=float(width.std(ddof=1) / np.sqrt(good.sum())),
            )

    if var_hat is not None:
        V = np.asarray(var_hat, dtype=float)[ok]
        good = ~np.isnan(V)
        if good.sum() >= 2:
            se = np.sqrt(np.maximum(V[good], 0.0))
            z = np.where(se > 1e-15, np.abs(tau_hat[ok][good] / se), 0.0)
            rejects = z > _stats.norm.ppf(1.0 - alpha / 2.0)
            rej = float(rejects.mean())
            null_true = np.abs(truth[ok][good]) < 1e-12
            right_sign = np.sign(tau_hat[ok][good]) == np.sign(truth[ok][good])
            correct = np.where(null_true, ~rejects, rejects & right_sign)
            cdr = float(correct.astype(float).mean())
            out.update(
                rejection=rej,
                mc_se_rejection=float(np.sqrt(rej * (1 - rej) / good.sum())),
                cdr=cdr,
                mc_se_cdr=float(np.sqrt(cdr * (1 - cdr) / good.sum())),
            )
    return out


def metrics_from_errors(
    err,
    *,
    true_tau: Any = float("nan"),
    covers=None,
    ci_width=None,
    reject=None,
    rmse_baseline: float = float("nan"),
) -> Dict[str, float]:
    """Monte Carlo metrics from per-replication errors and indicator columns.

    For the analyses whose raw files store the estimation error
    ``tau_hat - target`` and a coverage indicator rather than the interval
    endpoints (the real-data label-masking runs, the semi-synthetic testbed,
    the adaptive-c check).  Same definitions as
    :func:`metrics_from_replications`: bias = mean error, rmse = root mean
    squared error, rel_bias = 100 * bias / mean target, MC SEs from the
    replication spread (delta method for the RMSE), binomial SEs for the
    coverage and rejection rates, RE = ``rmse_baseline / rmse``.

    ``err`` may be None for a row that has only an indicator (a diagnostic's
    rejection rate); ``reject`` is the per-replication rejection indicator.
    """
    import numpy as np

    nan = float("nan")
    out: Dict[str, float] = {}

    def _rate(x, name, se_name):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if len(x) >= 1:
            p = float(x.mean())
            out[name] = p
            out[se_name] = float(np.sqrt(p * (1.0 - p) / len(x)))
        return len(x)

    truth = np.asarray(true_tau, dtype=float).ravel()
    truth = truth[~np.isnan(truth)]
    mean_truth = float(truth.mean()) if truth.size else nan
    out["true_tau"] = mean_truth

    n_ind = 0
    if err is not None:
        e = np.asarray(err, dtype=float)
        e = e[~np.isnan(e)]
        n_valid = int(len(e))
        out["n_valid"] = n_valid
        if n_valid >= 1:
            # Point metrics need one replication; their MC SEs need two.
            bias = float(e.mean())
            rmse = float(np.sqrt(np.mean(e ** 2)))
            out.update(
                bias=bias,
                rel_bias=(bias / mean_truth * 100.0)
                if np.isfinite(mean_truth) and abs(mean_truth) > 1e-12
                else nan,
                rmse=rmse,
            )
            if n_valid >= 2:
                out["mc_se_bias"] = float(e.std(ddof=1) / np.sqrt(n_valid))
                out["mc_se_rmse"] = float(
                    (e ** 2).std(ddof=1) / np.sqrt(n_valid) / (2.0 * rmse)
                ) if rmse > 1e-15 else nan
            if np.isfinite(rmse_baseline) and rmse > 1e-15:
                out["relative_efficiency"] = float(rmse_baseline / rmse)
    if covers is not None:
        n_ind = _rate(covers, "coverage", "mc_se_coverage")
    if ci_width is not None:
        w = np.asarray(ci_width, dtype=float)
        w = w[~np.isnan(w)]
        if len(w) >= 1:
            out["ci_width"] = float(w.mean())
        if len(w) >= 2:
            out["mc_se_ci_width"] = float(w.std(ddof=1) / np.sqrt(len(w)))
    if reject is not None:
        n_ind = _rate(reject, "rejection", "mc_se_rejection")
    if err is None:
        out["n_valid"] = n_ind
    return out


def replications_long(
    df,
    estimators: Sequence[Any],
    *,
    rep_col: str = "rep",
    true_col: str = "true_tau",
    protocol: Optional[str] = None,
):
    """Reshape a wide per-replication frame into the long layout of
    :func:`rows_from_replications`.

    The hybrid-family scripts store one row per replication with one column
    group per estimator (``tau_si``, ``ci_lower_si``, ``ci_upper_si``, ...).
    ``estimators`` lists ``(method, tau_col, ci_lower_col, ci_upper_col)``
    tuples, where ``method`` is a method-spec key (``"lo"``, ``"si"``,
    ``"ppi"``, ...; the MethodSpec supplies id, label and configuration axes)
    or a ``(key, label)`` pair for a method outside the method table (the hybrid,
    the binary switch), which is filed under ``method_id = -1``.  Interval
    columns may be None.
    """
    import numpy as np
    import pandas as pd
    from src.utils.config import spec_by_key

    parts = []
    for method, tau_col, lo_col, hi_col in estimators:
        if isinstance(method, str):
            spec = spec_by_key(method)
            key, mid, label = spec.key, spec.method_id, spec.label
            axes = spec.row_fields()
        else:
            key, label = method
            mid = -1
            axes = {}
        part = pd.DataFrame({
            "replication": df[rep_col].to_numpy(),
            "method_key": key,
            "method_id": mid,
            "method_label": label,
            "tau_hat": df[tau_col].to_numpy(dtype=float),
            "ci_lower": (df[lo_col].to_numpy(dtype=float) if lo_col
                         else np.full(len(df), np.nan)),
            "ci_upper": (df[hi_col].to_numpy(dtype=float) if hi_col
                         else np.full(len(df), np.nan)),
            "true_tau": df[true_col].to_numpy(dtype=float),
        })
        for axis in ("lambda_rule", "clip", "per_arm", "variance",
                     "si_variance"):
            part[axis] = axes.get(axis)
        if protocol is not None:
            part["protocol"] = protocol
        elif "protocol" in df.columns:
            part["protocol"] = df["protocol"].to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def rows_from_replications(
    df,
    *,
    dgp: Any,
    config_name: str,
    params: Optional[Dict[str, Any]] = None,
    pi_L: float = float("nan"),
    n: Any = float("nan"),
    R: int = 0,
    protocol: str = DEFAULT_PROTOCOL,
    alpha: float = 0.05,
    target: str = "ATE",
    seed: int = 42,
    baseline_label: str = "Labeled-Only",
) -> List[Dict[str, Any]]:
    """Turn a per-replication DataFrame into one result row per method config.

    ``df`` is what ``src.simulations.simulation.run_simulation`` returns: one
    row per (replication, method configuration) with columns ``method_key``,
    ``method_id``, ``method_label``, the four configuration axes, ``tau_hat``,
    ``V_hat``, ``ci_lower``, ``ci_upper``, ``true_tau`` and, for the portfolio
    DGP, ``cumulative_regret`` and ``oracle_gain``.

    Relative efficiency is RMSE(baseline) / RMSE(method) with the baseline
    named by ``baseline_label`` (the paper's Labeled-Only).
    """
    import numpy as np

    key_col = "method_key" if "method_key" in df.columns else "method_id"
    groups = {k: g.sort_values("replication") for k, g in df.groupby(key_col)}

    # Baseline RMSE for the relative-efficiency column.
    rmse_baseline = float("nan")
    for g in groups.values():
        label = canonical_label(g["method_label"].iloc[0]) \
            if "method_label" in g.columns else ""
        if label == canonical_label(baseline_label):
            m = metrics_from_replications(
                g["tau_hat"].values, g["true_tau"].values, alpha=alpha
            )
            rmse_baseline = m.get("rmse", float("nan"))
            break

    rows: List[Dict[str, Any]] = []
    for key, g in groups.items():
        metrics = metrics_from_replications(
            g["tau_hat"].values,
            g["true_tau"].values,
            var_hat=g["V_hat"].values if "V_hat" in g.columns else None,
            ci_lower=g["ci_lower"].values if "ci_lower" in g.columns else None,
            ci_upper=g["ci_upper"].values if "ci_upper" in g.columns else None,
            alpha=alpha,
            rmse_baseline=rmse_baseline,
        )
        extra: Dict[str, Any] = {"method_key": key}
        if "cumulative_regret" in g.columns:
            # The oracle gain V* = sum_k max(tau_k, 0) is redrawn with the
            # portfolio in every replication, so it is averaged over
            # replications and relative regret is the RATIO OF MEANS (the
            # same definition as scripts/run_portfolio_sensitivity.py).
            from src.evaluation import summarize_portfolio_regret

            reg = g["cumulative_regret"].values.astype(float)
            og_vals = (
                g["oracle_gain"].values.astype(float)
                if "oracle_gain" in g.columns
                else np.full(len(reg), float("nan"))
            )
            extra.update(summarize_portfolio_regret(reg, og_vals))

        first = g.iloc[0]
        rows.append(make_result_row(
            dgp=dgp,
            config_name=config_name,
            params=params or {},
            pi_L=pi_L,
            n=n,
            R=int(R or g["replication"].nunique()),
            protocol=first.get("protocol", protocol),
            method_id=int(first["method_id"]),
            method_label=str(first.get("method_label", "")),
            alpha=alpha,
            target=target,
            seed=seed,
            metrics=metrics,
            extra=extra,
        ))
        # Configuration axes come from the replication rows, which record what
        # the estimator actually used.
        for axis in ("lambda_rule", "clip", "per_arm", "variance",
                     "si_variance"):
            if axis in g.columns:
                val = first[axis]
                rows[-1][axis] = None if (
                    val is None or (isinstance(val, float) and np.isnan(val))
                ) else val
    return rows


def rows_from_summary(
    df,
    *,
    dgp: Any,
    target: str = "masking_target",
    protocol: str = DEFAULT_PROTOCOL,
    alpha: float = 0.05,
    seed: int = 42,
    n: Any = float("nan"),
    R: Any = None,
    config_name_col: Optional[str] = None,
    config_name: Optional[str] = None,
    params_cols: Sequence[str] = (),
    label_col: str = "method_name",
    extra_cols: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Convert an already-summarized per-method table into result rows.

    For the analyses that compute their own Monte Carlo summary (the real-data
    label-masking runs, the semi-synthetic testbed, the drift and calendar
    checks) rather than going through ``rows_from_replications``.  Column names
    are matched to the schema where they agree and mapped where they do not
    (``rejection_rate`` -> ``rejection``).

    ``target`` defaults to ``"masking_target"``: in the real-data analyses the
    estimand is the full-sample difference in means, not an ATE.
    """
    import numpy as np

    # "variance" is a CONFIGURATION axis in the schema (exact / plugin /
    # bootstrap), so a summary column of the same name -- the Monte Carlo
    # variance of tau_hat -- is carried as `mc_variance` instead.
    alias = {
        "rejection_rate": "rejection",
        "mean_regret": "mean_cumul_regret",
        "variance": "mc_variance",
    }
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        metrics: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for col in df.columns:
            key = alias.get(col, col)
            val = r[col]
            if key in RESULT_ROW_FIELDS:
                metrics[key] = val
            elif col in extra_cols or key in (
                "ess_multiplier", "surrogacy_test_rejection_rate",
                "mc_se_surrogacy_test", "mean_cumul_regret", "rel_regret",
                "mc_variance", "oracle_gain", "mc_se_regret",
                "mc_se_oracle_gain", "mc_se_rel_regret",
                "rel_regret_mean_of_ratios",
                "mc_se_rel_regret_mean_of_ratios",
            ):
                extra[key] = val
        label = canonical_label(str(r[label_col])) if label_col in df else ""
        name = (str(r[config_name_col]) if config_name_col
                else (config_name or str(dgp)))
        params = {c: r[c] for c in params_cols if c in df.columns}
        pi_L = float(r["pi_L"]) if "pi_L" in df.columns else float("nan")
        n_row = r["n"] if "n" in df.columns else n
        R_row = R if R is not None else (
            int(r["n_valid"]) if "n_valid" in df.columns
            and not np.isnan(r["n_valid"]) else 0
        )
        rows.append(make_result_row(
            dgp=dgp, config_name=name, params=params, pi_L=pi_L, n=n_row,
            R=R_row, protocol=protocol,
            method_id=int(r["method_id"]) if "method_id" in df.columns else -1,
            method_label=label or "unknown",
            alpha=alpha, target=target, seed=seed,
            metrics=metrics, extra=extra,
        ))
    return rows


#: Summary-column names that mean a schema metric under a different spelling.
_METRIC_ALIASES = {
    "rejection_rate": "rejection", "reject_rate": "rejection",
    "reject_rate_2s": "rejection", "power": "rejection",
    "mean_regret": "mean_cumul_regret", "rel_regret_pct": "rel_regret",
    "variance": "mc_variance", "mean_var": "mc_variance",
    "mean_width": "ci_width", "mc_se_width": "mc_se_ci_width",
    "mc_se_power": "mc_se_rejection",
    "bias": "bias", "rmse": "rmse", "coverage": "coverage",
    "re_vs_lo": "relative_efficiency", "re": "relative_efficiency",
    "cdr": "cdr",
}


def write_table_rows(
    path: str,
    df,
    *,
    dgp: Any,
    analysis: str,
    protocol: str = DEFAULT_PROTOCOL,
    label_col: Optional[str] = None,
    label_map: Optional[Dict[str, str]] = None,
    config_cols: Sequence[str] = (),
    pi_L_col: str = "pi_L",
    n_col: str = "n",
    R_col: Optional[str] = None,
    R: Optional[int] = None,
    alpha: float = 0.05,
    target: str = "ATE",
    seed: int = 42,
    generated_by: str = "",
    validate: bool = True,
) -> str:
    """Emit result rows for any already-summarized results table.

    The general-purpose writer behind "every script that produces a paper
    table emits result rows".  It handles the three shapes the repository
    actually has:

    * one row per (cell, method) -- pass ``label_col`` (and ``label_map`` when
      the table's spelling is not a paper label);
    * one row per cell with no method column (a diagnostic power curve, a
      variance-convention comparison, a timing table) -- leave ``label_col``
      unset and every row is filed under ``method_label=analysis`` with
      ``method_id=-1``;
    * replication-level tables -- summarize them with
      :func:`rows_from_replications` instead.

    Columns whose names are schema metrics (or a known alias of one) become
    metrics; everything else is carried verbatim in the row, so nothing the
    script computed is lost.
    """
    import numpy as np

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        metrics: Dict[str, Any] = {}
        extra: Dict[str, Any] = {"analysis": analysis}
        for col in df.columns:
            key = _METRIC_ALIASES.get(str(col).lower(), str(col))
            val = r[col]
            if key in RESULT_ROW_FIELDS and key not in (
                "dgp", "config_name", "params", "pi_L", "n", "R", "protocol",
                "method_id", "method_label", "lambda_rule", "clip", "per_arm",
                "variance", "si_variance", "alpha", "target", "seed",
            ):
                metrics[key] = val
            elif col not in (label_col, pi_L_col, n_col, R_col):
                extra[key] = val

        if label_col and label_col in df.columns:
            raw = str(r[label_col])
            label = (label_map or {}).get(raw, canonical_label(raw))
        else:
            label = analysis
        try:
            method_id = spec_by_label(label).method_id
        except KeyError:
            method_id = -1

        cfg_name = "; ".join(
            f"{c}={r[c]}" for c in config_cols if c in df.columns
        ) or analysis
        params = {c: r[c] for c in config_cols if c in df.columns}

        def _num(col, default=float("nan")):
            if col and col in df.columns:
                try:
                    return float(r[col])
                except (TypeError, ValueError):
                    return default
            return default

        R_row = R if R is not None else int(_num(R_col, 0) or 0)
        rows.append(make_result_row(
            dgp=dgp, config_name=cfg_name, params=params,
            pi_L=_num(pi_L_col), n=_num(n_col), R=R_row, protocol=protocol,
            method_id=method_id, method_label=label,
            alpha=alpha, target=target, seed=seed,
            metrics=metrics, extra=extra,
        ))
        try:
            spec = spec_by_label(label)
            rows[-1].update(spec.row_fields())
            rows[-1]["method_label"] = label
        except KeyError:
            pass

    return write_rows(path, rows, generated_by=generated_by,
                      protocol=protocol, validate=validate)
