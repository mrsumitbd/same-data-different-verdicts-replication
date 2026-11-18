# Replication Package: Revisiting the Survival of Agent-Attributed Code

This repository contains the code, data, human labels, and annotation guides for the paper

> Musfiqur Rahman, Mahi Begoug, and Emad Shihab. *Revisiting the Survival of Agent-Attributed Code: How Measurement Choices Decide the Verdict.*

The study re-analyzes the 854,139 lines that 2,802 merged pull requests added to 215 repositories of the AIDev dataset. It varies nine measurement choices in five research questions and builds a specification curve of 88 survival analyses.

## Contents

| Path | What it holds |
|---|---|
| `src/` | Mining, analysis, and plotting scripts (Python and R) |
| `config/` | Agent taxonomy, agent logins, observation window, path settings |
| `earlier_study/` | The miner of our earlier study and its output tables. Our births come from these tables |
| `data/` | Derived datasets (gzipped CSV) and small metadata files |
| `results/` | Inputs from the first labeling round: the 385-commit intent sample with human and LLM labels, the LLM labels of the earlier corpus, and the AST validation sample |
| `results_v6/` | Every output the paper reports, as produced by the scripts. Re-running the scripts overwrites these files |
| `annotation_guides/` | The guides used for each manual labeling task |
| `reproduce.sh` | Runs the analyses in order |

### Datasets

Each line dataset has one row per added source line and records when, and by which commit, the line was first modified. `v6confnv` is the primary dataset, without vendored code. `v6conf` keeps vendored code and is used for the sensitivity analysis in `results_v6/with_vendored/`.

| File (in `data/`) | Content |
|---|---|
| `survival_events_line_v6confnv_analysis.csv` | Primary line data used by RQ1 to RQ5 (identity tracking, observation window applied) |
| `survival_events_line_v6confnv.csv`, `..._ast.csv` | The same lines before the window, without and with the AST classification of each modification |
| `survival_events_line_pos_v6confnv*.csv` | Line-number tracking of the same lines (the third tracker in RQ1) |
| `survival_events_hunk_v6confnv.csv`, `survival_events_hunk_pos_v6confnv.csv` | Hunk-level data |
| `death_commit_messages_v6confnv.csv` | The 5,303 modification commits with messages and keyword labels |
| `*_v6conf*` | The same files with vendored code kept |
| `death_commit_messages.csv`, `survival_events_hunk_v2.csv` | Inputs of the first intent sample (385 commits) |
| `intent_api_cache.json` | Commit messages fetched from the GitHub API |
| `results_v6/diff_cache_full_corpus.csv` | The diffs the LLM classifiers received |

## Setup

Python 3.10 and R 4.5 with the `survival` package (3.8.3). We ran the analyses with the versions in `requirements.txt`.

```
pip install -r requirements.txt
python src/download_data.py          # AIDev parquet files from Hugging Face into data/
bash scripts/unpack_data.sh          # decompress the derived datasets
```

`download_data.py` fetches several large AIDev files. The analyses need `all_pull_request.parquet`, `human_pull_request.parquet`, `pull_request.parquet` and `all_repository.parquet`.

## Reproducing the results

```
bash reproduce.sh                    # main, vendored, supporting, figures
bash reproduce.sh main               # one stage
bash reproduce.sh permutations       # label-permutation nulls (several hours)
```

Scripts select the dataset through two environment variables, which `reproduce.sh` sets: `DATA_TAG=v6confnv RESULTS_DIR=results_v6` for the primary analyses, and `DATA_TAG=v6conf RESULTS_DIR=results_v6/with_vendored` for the sensitivity analysis. Seeds are fixed (42 unless a script says otherwise). The bootstraps use 500 resamples, so the `main` stage alone takes about seven hours on an Apple Silicon Mac. The other stages are shorter, except `permutations`.

Some script names follow an earlier numbering of the research questions. For example, `analyze_rq4.py` produces the modifier results of RQ5 and the intent shares of RQ4.

## Where each result comes from

| Paper | Script | Output (in `results_v6/`) |
|---|---|---|
| Fig. 1, pipeline | `plot_pipeline.py` | `visuals/pipeline.pdf` |
| Human-baseline validation, shadow AI sensitivity | `validate_attribution.py`, `sensitivity_shadow_ai.py` | `data/attribution_validation.csv`, `sensitivity_shadow_ai.csv`, `second_rater/human_baseline/hb_estimates.txt` |
| AST rule validation | `score_ast_validation.py` | `results/ast_validation_scores.txt`, `round1/c9a_leaf_precision.txt` |
| Intent labels and sample sizes | `build_intent_labels_v6.py`, `intent_topup_sample_size_v6.py` | `labels618/`, `intent_topup_sample_size.txt` |
| LLM cost | `r4_llm_cost.py` | `round4/task3_llm_tokens_cost.txt` |
| Second-rater agreement | `score_second_rater.py` | `second_rater_kappas.txt` |
| RQ1, measurement ladder (Fig. 3) | `analyze_rq1_master.py`, `cox_variance_r1.R`, `plot_figs_r2.py` | `rq1_master_results.txt`, `round1/c1_variance_verdicts.txt`, `round2/visuals/rq1_ladder_clustered.pdf` |
| RQ1, Kaplan-Meier curves (Fig. 4) | `km_two_panel_r4.py` | `round4/rq1_km_def1_def2_line.pdf`, `round4/km_two_panel.txt` |
| RQ1, three trackers (Table 5) | `compare_tracking_methods.py`, `tracker_gt_score_r1.py`, `tracker_gt_ci_r3.py` | `tracking_method_comparison.txt`, `round1/tracker_gt/`, `round3/tracker_gt_ci_r3.txt` |
| RQ1, hunk size (Table 6) | `hunk_size_clustered_r3.py` | `round3/hunk_size_clustered.txt` |
| RQ1, observation window (Step 5) | `spec_null_r2.py`, `balance_r2.py` | `round2/e1_window_curve.txt` |
| RQ1, largest PRs, vendored code | `rq1_largest_prs.py` | `rq1_largest_prs.txt`, `with_vendored/rq1_master_results.txt` |
| RQ2, definitions and event mix | `analyze_rq2.py`, `analyze_rq1_def2_stratified.py`, `describe_event_mix.py`, `def2_continued_r2.py` | `rq2_results.txt`, `rq1_def2_stratified_results.txt`, `event_mix.txt`, `round2/e5_def2_continued.txt` |
| RQ2, definition figure (Fig. 5) | `figs_r6.py` | `round6/figA_rq2_definitions.pdf`, `round6/figs_r6_values.txt` |
| RQ3, repository coverage | `analyze_rq3.py` | `rq3_results.txt` |
| RQ3, across vs within, verdict counts | `analyze_rq1_master.py`, `c4_bootstrap_r1.py`, `summarize_c1_c2_r1.py` | `rq1_master_results.txt`, `round1/c4_bootstrap.csv`, `round1/c1_verdict_counts.csv` |
| RQ3, permutation nulls (Fig. 6) | `spec_null_r2.py`, `figs_r6.py` | `round2/work/null_*_efron.csv.gz`, `round6/figB_permutation_null.pdf` |
| RQ3, per agent and forest plot (Fig. 7) | `per_agent_boot_r2.py`, `c8_covariate_within_r1.py`, `forest_r6.py` | `round2/e3_per_agent_within.txt`, `round1/c8_covariate_within.txt`, `round6/rq3_forest.pdf` |
| RQ4, agreement with human labels | `intent_kappa_v6.py`, `intent_message_only_r1.py`, `r4_intent_kappa.py` | `intent_kappa_v6.txt`, `round1/c7_message_only.txt`, `round4/task1_task2_intent.txt` |
| RQ4, intent shares and corrected shares | `analyze_intent_aggregation.py` | `intent_aggregation_full_corpus_618.txt` |
| RQ4, tier checks and keyword disagreement | `intent_tier_confusion_r1.py`, `intent_weighted_tier_test_v6.py`, `c9b_labelled_vs_corpus_r1.py`, `rq3_keyword_llm_disagreement.py` | `round1/c6_intent_tier_confusion.txt`, `intent_human_label_weighted_test.txt`, `round1/c9b_labelled_vs_corpus.txt`, `rq3_keyword_llm_disagreement.txt` |
| RQ4, intent figure | `paper_figs_r3.py` | `round3/rq4_intent_stacked.pdf` |
| RQ5, who modifies agent code | `analyze_rq4.py`, `rq4_self_modification.py` | `rq4_results.txt`, `rq4_self_modification.txt`, `round2/*_selfmod_sample_labelled.xlsx` |
| RQ5, units, windows, identification rules | `rq5_r2.py` | `round2/e6_modifier_unit_window.txt`, `e7_identification_rule.txt`, `e8_mixed_commits.txt`, `e9_self_modification.txt` |
| RQ5, agent-share figure | `paper_figs_r3.py` | `round3/rq5_agent_share.pdf` |
| Specification curve | `specification_curve.py`, `plot_figs_r2.py` | `spec_curve.csv`, `round2/visuals/spec_curve_clustered.pdf` |
| Verdicts by observation window, permutation tests | `summarize_c5_r1.py`, `spec_null_r1.py`, `spec_null_r2.py` | `round1/c5_window_curve.txt`, `round2/e1_e4_permutation_summary.csv` |
| Published designs re-run on our data | `emulate_prior_studies.py` | `emulation_prior_studies.txt` |
| Threats: proportional hazards, early churn, sampling caps | `check_ph_assumption.py`, `check_ph_piecewise.py`, `early_churn_holdout.py`, `check_sampling_caps.py` | `ph_assumption_test.txt`, `ph_piecewise.txt`, `early_churn_holdout.txt`, `sampling_caps_check.txt` |

`paper_figs_r3.py` and `forest_r6.py` draw the values reported in the paper's tables, which are copied in `results_v6/paper_tables.tex`. They do not recompute them. `figs_r6.py` reads the result files and checks every plotted value against them.

## Rebuilding the data from scratch

The derived datasets can be rebuilt from the AIDev data and clones of the 215 repositories (about 94 GB). The steps, in the order we ran them:

1. `earlier_study/scripts/mine_survival.py` and `build_tables.py`: mine the merged PRs of the earlier study and write `earlier_study/tables/`.
2. `src/build_conf_births_v6.py`: births of the 854,139 lines from those tables.
3. `V5_COHORT=conf215births python src/measure_retention_v5.py --phase track`, then `--phase combine`: identity tracking.
4. `src/measure_ast_v5.py` and `src/derive_v5_datasets.py`: AST classification of each modification and the analysis files.
5. `src/measure_position_v6.py`: line-number tracking.
6. `src/build_death_commit_messages.py`: modification commits and keyword labels.
7. `src/classify_intent_full_corpus.py`: LLM intent labels through Together AI (needs an API key; about US$47 for the three models).
8. `src/fetch_modifier_authors.py` and `src/fetch_commit_pulls_r2.py`: commit authors and PRs from the GitHub REST API (needs a token).

`src/explain_tracker_disagreement_v6.py` also needs the clones (it runs `git blame --reverse`), so `reproduce.sh` does not run it. Its output is `results_v6/tracker_disagreement_breakdown.txt`.

Each script documents its arguments at the top of the file. The LLM and GitHub API outputs are included, so these steps are needed only to rebuild the data.

## Human labels

- **Commit intent:** 618 modification commits labeled by the first author (`results_v6/labels618/gold.csv`, built from `results/intent_annotation_*.xlsx` and `results_v6/intent-top-up-v6-labeled-sheet-v001-v256.xlsx`).
- **AST rule:** `results/ast_validation_*`.
- **Tracker ground truth:** 299 lines, with evidence pages, in `results_v6/round1/tracker_gt/`. The key maps each line to the claims of the three trackers.
- **Self-modification:** 20 Copilot and 20 Devin commits, in `results_v6/round2/`.
- **Human-baseline review:** `data/attribution_validation.csv`.

A second rater labeled up to 100 items of each task. The agreement values are in `results_v6/second_rater_kappas.txt`. The second rater's labels are not distributed, so `score_second_rater.py`, `hb_validation_estimates.py` and Task 1 of `r4_intent_kappa.py` cannot be re-run from this package.

## License

The code is released under the MIT License (see `LICENSE`). The data derived from AIDev follows the AIDev license.
