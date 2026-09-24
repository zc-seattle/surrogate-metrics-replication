#!/usr/bin/env bash
# Reproduce every result table of the paper, in dependency order.
#
#   bash reproduce.sh
#
# The full run takes many hours on 8 cores. Hillstrom and LaLonde are
# downloaded on first use (network access needed). The Criteo steps at the end
# run only when CRITEO_CSV points to criteo-uplift-v2.1.csv (see README.md).
# Every output goes to results/tables/.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0
PY="${PYTHON:-python}"
run() { echo "+ $PY $*"; "$PY" "$@"; }

# Core simulation grid (DGPs 1-6) and its CSV export
for d in 1 2 3 4 5 6; do
  run run_dgp.py --dgp "$d" --R 2000
done
run scripts/export_eval_csv.py

# Further designs (DGPs 7-11) and the AIPW cells
run run_auxiliary_simulations.py
run run_dgp9.py
run run_dgp10.py
run run_aipw_full.py
run scripts/run_enrollment_drift.py
run scripts/run_calendar_review.py
run scripts/run_portfolio_sensitivity.py

# PPI++ variance
run run_corrected_variance.py
run run_bootstrap_sensitivity.py

# SI--PPI++ estimator-disagreement diagnostic and the detection-damage gap
run scripts/power_curve_extended.py
run scripts/run_frontier_coverage.py
run scripts/run_detection_damage_paired.py
run scripts/run_detection_damage_paired.py --refine
run scripts/detection_damage_edge_ci.py
run scripts/make_detection_damage_figure.py
run scripts/surrogacy_test_misspec.py

# Hybrid estimator
run run_hybrid_eval.py --R 2000
run scripts/hybrid_sensitivity_c.py
run run_switch_vs_hybrid.py
run scripts/hybrid_ci_validation.py
run scripts/minimax_objective.py
run scripts/run_adaptive_c.py
run scripts/run_limit_experiment.py

# Robustness and variance checks
run run_gbt_comparison.py --R 200
run scripts/run_gbt_variance_check.py
run scripts/run_gbt_variance_check.py --clustered-folds --cells 1 --R 200 --B 50
run scripts/run_nonrandom_label_variance_check.py
run run_cuped_comparison.py
run scripts/cross_ppi_comparison.py
run scripts/cp_best_case.py
run scripts/computational_timing.py

# Public data: Hillstrom and LaLonde
run scripts/hillstrom_expanded.py
run scripts/hillstrom_hybrid.py
run scripts/hillstrom_hybrid.py --si-variance
run scripts/lalonde_negative_control.py

# Criteo and the Criteo-calibrated designs
if [ -n "${CRITEO_CSV:-}" ]; then
  run scripts/run_criteo.py --data "$CRITEO_CSV"
  run scripts/run_semisynthetic.py --data "$CRITEO_CSV"
  run scripts/run_multisurrogate.py
  run scripts/multisurrogate_population_offset.py
  run scripts/real_data_noninvariance.py --criteo "$CRITEO_CSV"
  run scripts/run_joint_cov_check.py
  run scripts/run_joint_cov_check.py --cells 1,2,3,4 --clustered-folds
  run scripts/run_joint_cov_check.py --cells 2,1,3,5,4 -R 2000 --no-boot --out-suffix _R2000
else
  echo "CRITEO_CSV is not set: skipping the Criteo, semi-synthetic, multi-surrogate and joint-covariance steps."
  run scripts/real_data_noninvariance.py
fi

# CSV versions of the markdown and JSON outputs
run scripts/export_tables_csv.py
