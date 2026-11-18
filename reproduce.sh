#!/usr/bin/env bash
# Re-runs the analyses of the paper from the derived data.
#
#   bash reproduce.sh [stage ...]
#
# Stages (default: main vendored supporting figures):
#   main         RQ1-RQ5 and the specification curve on the primary data (no vendored code)
#   vendored     the same analyses with vendored code kept (results_v6/with_vendored/)
#   supporting   variance estimators, bootstraps, observation windows, intent checks,
#                tracker ground truth, modifier variants (results_v6/round1..round4/)
#   permutations label-permutation null distributions (several hours)
#   figures      the figures of the paper
#
# Requires the AIDev parquet files in data/ (python src/download_data.py) and R with
# the survival package. Outputs overwrite the files shipped in results_v6/.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python} -u"
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(main vendored supporting figures)
run() { echo "=== $*"; "$@"; }

bash scripts/unpack_data.sh

main() {
  export DATA_TAG=v6confnv RESULTS_DIR=results_v6
  run $PY src/build_intent_labels_v6.py
  for s in analyze_rq1_master analyze_rq1_def2_stratified analyze_rq2 analyze_rq1_bot_excluded check_ph_assumption \
           sensitivity_observation_window sensitivity_shadow_ai check_sampling_caps check_ph_piecewise \
           early_churn_holdout rq1_largest_prs; do
    run $PY src/$s.py
  done
  run $PY src/analyze_rq3.py --repo-meta data/repo_meta_v6.csv
  run $PY src/compute_intent_kappa.py
  run $PY src/intent_kappa_v6.py --topup-v6 --topup-v6-sheet results_v6/intent-top-up-v6-labeled-sheet-v001-v256.xlsx \
      --corpus data/death_commit_messages_v6confnv.csv
  run $PY src/analyze_intent_aggregation.py
  run $PY src/analyze_intent_aggregation.py --full-corpus
  L=results_v6/labels618
  run $PY src/analyze_intent_aggregation.py --gold $L/gold.csv --sample $L/sample.csv --intent-kw $L/keyword.csv \
      --sample-llm-dir $L --output results_v6/intent_aggregation_results_618.txt
  run $PY src/analyze_intent_aggregation.py --full-corpus --gold $L/gold.csv --sample $L/sample.csv --intent-kw $L/keyword.csv \
      --sample-llm-dir $L --output results_v6/intent_aggregation_full_corpus_618.txt
  run $PY src/intent_weighted_tier_test_v6.py
  run $PY src/intent_topup_sample_size_v6.py
  for s in analyze_rq4 analyze_rq4_intent_sample rq4_self_modification rq3_keyword_llm_disagreement check_early_churn \
           compare_tracking_methods specification_curve emulate_prior_studies \
           describe_event_mix; do
    run $PY src/$s.py
  done
  VERIFIED_MERGE=1 run $PY src/analyze_rq1_master.py
  VERIFIED_MERGE=1 run $PY src/analyze_rq4.py
}

vendored() {
  export DATA_TAG=v6conf RESULTS_DIR=results_v6/with_vendored
  for s in analyze_rq1_master analyze_rq1_def2_stratified analyze_rq2 analyze_rq1_bot_excluded check_ph_assumption \
           sensitivity_observation_window sensitivity_shadow_ai check_sampling_caps check_ph_piecewise \
           early_churn_holdout rq1_largest_prs; do
    run $PY src/$s.py
  done
  run $PY src/analyze_rq3.py --repo-meta data/repo_meta_v6.csv
  VERIFIED_MERGE=1 run $PY src/analyze_rq1_master.py
}

supporting() {
  export DATA_TAG=v6confnv RESULTS_DIR=results_v6
  # variance estimators and the specification curve with clustered errors
  run $PY src/export_spec_data_r1.py
  run Rscript src/cox_variance_r1.R
  for w in c8 c8h c10h; do run Rscript src/cox_variance_r1.R 4 $w; done
  run $PY src/summarize_c1_c2_r1.py c1
  run $PY src/summarize_c5_r1.py
  # Definition 2 followed past leaf-only edits
  run $PY src/measure_ast_continue_r1.py
  run $PY src/c3_refit_r1.py
  run Rscript src/c3_cluster_r1.R
  run $PY src/def2_continued_r2.py
  # within-repository bootstraps, covariates, per-agent estimates
  run $PY src/c4_bootstrap_r1.py
  run $PY src/file_covariates_r1.py
  run $PY src/c8_covariate_within_r1.py
  run $PY src/c9f_per_agent_within_r1.py
  run $PY src/per_agent_boot_r2.py
  run Rscript src/piecewise_r1.R
  run $PY src/hunk_size_clustered_r3.py
  run $PY src/km_two_panel_r3.py
  run $PY src/km_two_panel_r4.py
  # intent checks
  run $PY src/intent_tier_confusion_r1.py
  run $PY src/c9b_labelled_vs_corpus_r1.py
  run $PY src/r4_intent_kappa.py
  run $PY src/r4_llm_cost.py
  # tracker ground truth
  run $PY src/tracker_gt_score_r1.py
  run $PY src/tracker_gt_ci_r2.py
  run $PY src/tracker_gt_ci_r3.py
  # who modifies agent code (RQ5 variants)
  run $PY src/rq5_r2.py e6 e7
  run $PY src/rq5_r2.py e8 e9
  run $PY src/balance_r2.py
}

permutations() {
  export DATA_TAG=v6confnv RESULTS_DIR=results_v6
  run $PY src/spec_null_r1.py
  run $PY src/summarize_c1_c2_r1.py c2
  run $PY src/spec_null_r2.py run --variant full --engine efron --tiers human_supervised,fully_autonomous
  for v in nopdfme coh270_150 coh8m_150 c8 c8h c10h; do
    run $PY src/spec_null_r2.py run --variant $v --engine efron
    run $PY src/spec_null_r2.py run --variant $v --engine breslow
  done
  run $PY src/spec_null_r2.py summarize
}

figures() {
  export DATA_TAG=v6confnv RESULTS_DIR=results_v6
  run $PY src/plot_pipeline.py
  run $PY src/plot_rq1_ladder.py
  run $PY src/plot_round1_figs_r1.py
  run $PY src/plot_figs_r2.py
  run $PY src/paper_figs_r3.py
  run $PY src/forest_r6.py
  run $PY src/figs_r6.py
}

for st in "${STAGES[@]}"; do "$st"; done
