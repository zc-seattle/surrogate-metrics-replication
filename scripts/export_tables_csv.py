#!/usr/bin/env python3
"""
Write CSV versions of the result tables that the analysis scripts save only
as markdown (``<stem>.md``) or JSON (``<stem>_rows.json``, ``*.json``).

Run it after the analysis scripts (``reproduce.sh`` does this last). It reads
``results/tables/`` and writes, next to the inputs:

* ``<stem>.csv`` for summary tables saved only as ``<stem>_rows.json``
  (adaptive_c, bootstrap_sensitivity, criteo_main, cuped_comparison,
  enrollment_drift, multisurrogate, semisynthetic);
* ``<stem>_summary.csv``, the summary rows of analyses whose ``<stem>.csv``
  holds one row per replication (aipw_full_eval,
  corrected_variance_verification, dgp9, dgp10, gbt_comparison,
  hybrid_ci_validation, hybrid_eval, surrogacy_test_misspec,
  switch_vs_hybrid);
* ``criteo_scale.csv`` and ``limit_minimax_by_c.csv`` from their markdown
  tables, with the printed values;
* ``criteo_descriptive.csv``, ``limit_experiment.csv`` and
  ``semisynth_params.csv`` as key-value flattenings of the JSON files.

Inputs that are missing (for example the Criteo tables when the Criteo data
were not used) are skipped with a message.

Usage:
    python scripts/export_tables_csv.py [--tables results/tables] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.registry import load_rows, rows_to_frame  # noqa: E402

ROWS_ONLY = ("adaptive_c", "bootstrap_sensitivity", "criteo_main",
             "cuped_comparison", "enrollment_drift", "multisurrogate",
             "semisynthetic")

PER_REPLICATION = (
    ("aipw_full_eval_rows.json", "aipw_full_eval_summary.csv"),
    ("corrected_variance_verification_rows.json",
     "corrected_variance_verification_summary.csv"),
    ("dgp9_rows.json", "dgp9_summary.csv"),
    ("dgp10_rows.json", "dgp10_summary.csv"),
    ("gbt_comparison_rows.json", "gbt_comparison_summary.csv"),
    ("hybrid_ci_validation_rows.json", "hybrid_ci_validation_summary.csv"),
    ("hybrid_eval_rows.json", "hybrid_eval_summary.csv"),
    ("surrogacy_test_misspec_rows.json", "surrogacy_test_misspec_summary.csv"),
    ("switch_vs_hybrid_rows.json", "switch_vs_hybrid_summary.csv"),
)

JSON_FILES = ("criteo_descriptive", "limit_experiment", "semisynth_params")


def md_table(path):
    """Header and body rows of the first pipe table in a markdown file."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    continue
                rows.append(cells)
            elif rows:
                break
    return rows[0], rows[1:]


def flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from flatten(v, key + ".")
        elif isinstance(v, list):
            for i, x in enumerate(v):
                yield f"{key}[{i}]", x
        else:
            yield key, v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tables",
                    default=os.path.join(PROJECT_ROOT, "results", "tables"))
    ap.add_argument("--out", default=None,
                    help="output directory (default: the --tables directory)")
    args = ap.parse_args()
    src, out = args.tables, args.out or args.tables
    os.makedirs(out, exist_ok=True)

    def have(name):
        ok = os.path.exists(os.path.join(src, name))
        if not ok:
            print(f"  skip: {name} not found")
        return ok

    def from_rows(rows_json, out_csv):
        if not have(rows_json):
            return
        df = rows_to_frame(load_rows(os.path.join(src, rows_json)))
        df = df.drop(columns=[c for c in ("source_file",) if c in df.columns])
        df.to_csv(os.path.join(out, out_csv), index=False)
        print(f"  {rows_json} -> {out_csv} ({len(df)} rows)")

    for stem in ROWS_ONLY:
        from_rows(f"{stem}_rows.json", f"{stem}.csv")
    for rows_json, out_csv in PER_REPLICATION:
        from_rows(rows_json, out_csv)

    if have("criteo_scale.md"):
        _, body = md_table(os.path.join(src, "criteo_scale.md"))
        pd.DataFrame(
            [[r[0].replace(",", ""), r[1]] + r[2:6] for r in body],
            columns=["n", "R", "si_bias", "si_coverage", "ppi_coverage",
                     "diagnostic_rejection"],
        ).to_csv(os.path.join(out, "criteo_scale.csv"), index=False)
        print(f"  criteo_scale.md -> criteo_scale.csv ({len(body)} rows)")

    if have("limit_minimax_by_c.md"):
        _, body = md_table(os.path.join(src, "limit_minimax_by_c.md"))
        pd.DataFrame(
            [[r[0].split()[0], "c*" in r[0].replace("\\", "")] + r[1:3]
             for r in body],
            columns=["c", "is_limit_minimax_c", "worst_case_excess_risk",
                     "ratio_to_c_star"],
        ).to_csv(os.path.join(out, "limit_minimax_by_c.csv"), index=False)
        print(f"  limit_minimax_by_c.md -> limit_minimax_by_c.csv "
              f"({len(body)} rows)")

    for stem in JSON_FILES:
        if not have(f"{stem}.json"):
            continue
        with open(os.path.join(src, f"{stem}.json")) as fh:
            d = json.load(fh)
        df = pd.DataFrame(list(flatten(d)), columns=["key", "value"])
        df.to_csv(os.path.join(out, f"{stem}.csv"), index=False)
        print(f"  {stem}.json -> {stem}.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
