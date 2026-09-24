"""
Flatten results/tables/dgp{N}_eval.json to the sibling dgp{N}_eval.csv.

Why this exists
---------------
The eval JSON is authoritative: it holds the result rows. The CSV is a
convenience flattening of exactly those rows, for spreadsheets and for the
replication package. This script is a pure re-serialization: it runs no
simulation code and recomputes no estimate, so the CSV can always be rebuilt
without the hours the simulations cost.

Columns
-------
The full ResultRow schema -- dgp, config_name, pi_L, n, R, protocol,
method_id, method_label, lambda_rule, clip, per_arm, variance, si_variance,
alpha, target, seed, then the metrics -- plus one ``param_<key>`` column per
entry of the row's ``params`` dict (rho, delta_beta, q, K, missingness, ...).

Usage
-----
    python scripts/export_eval_csv.py            # rewrite the CSVs
    python scripts/export_eval_csv.py --check    # report drift only
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.registry import load_rows, registry_version_of, rows_to_frame

TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")

#: The core-grid DGPs whose eval JSON this script flattens.
DGP_IDS = [1, 2, 3, 4, 5, 6]

#: Sort keys, in order, that give a deterministic row order.
SORT_PREFERENCE = [
    "dgp", "param_rho", "param_beta_YS_low", "param_delta_beta", "param_q",
    "param_K", "param_missingness", "pi_L", "method_id", "method_label",
]


def build_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten result rows and sort them deterministically."""
    df = rows_to_frame(rows)
    sort_keys = [k for k in SORT_PREFERENCE if k in df.columns]
    if sort_keys:
        df = df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    return df


def _drift_note(csv_path: str, new_df: pd.DataFrame) -> tuple[str, bool]:
    if not os.path.exists(csv_path):
        return "no existing CSV", False
    old_df = pd.read_csv(csv_path)
    if len(old_df) != len(new_df):
        return f"row count {len(old_df)} -> {len(new_df)}", True
    shared = [c for c in ("rmse", "coverage", "bias")
              if c in old_df.columns and c in new_df.columns]
    deltas = {}
    for c in shared:
        try:
            deltas[c] = float(
                (old_df[c].to_numpy() - new_df[c].to_numpy()).__abs__().max()
            )
        except (TypeError, ValueError):
            return f"column {c} is not comparable", True
    worst = max(deltas.values()) if deltas else 0.0
    if worst > 1e-9:
        return ("max |delta| " + ", ".join(f"{c}={v:.6g}"
                                           for c, v in deltas.items()), True)
    return "already consistent", False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report drift without rewriting the CSVs")
    ap.add_argument("--dgps", type=int, nargs="+", default=DGP_IDS)
    ap.add_argument("--suffix", default="",
                    help="eval file suffix, e.g. '_mixed_fit'")
    args = ap.parse_args()

    any_drift = False
    for dgp_id in args.dgps:
        json_path = os.path.join(
            TABLES_DIR, f"dgp{dgp_id}_eval{args.suffix}.json"
        )
        csv_path = os.path.join(
            TABLES_DIR, f"dgp{dgp_id}_eval{args.suffix}.csv"
        )

        if not os.path.exists(json_path):
            print(f"DGP{dgp_id}: no JSON at {json_path}, skipping")
            continue

        import json as _json
        with open(json_path) as fh:
            version = registry_version_of(_json.load(fh))
        rows = load_rows(json_path)
        new_df = build_frame(rows)

        note, drift = _drift_note(csv_path, new_df)
        any_drift = any_drift or drift
        stamp = "" if version >= 2 else " [upgraded from results schema v1]"

        if args.check:
            print(f"DGP{dgp_id}: {note}{stamp}")
        else:
            new_df.to_csv(csv_path, index=False)
            print(f"DGP{dgp_id}: wrote {csv_path} "
                  f"({len(new_df)} rows; {note}){stamp}")

    if args.check and any_drift:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
