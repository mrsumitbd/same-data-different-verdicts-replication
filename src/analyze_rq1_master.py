"""
analyze_rq1_master.py

Produces all numbers needed for RQ1 in one clean pass.
Outputs a structured results file that the paper is written from directly.

RQ1: How does agent aggregation affect conclusions about code retention?

TWO-TIER TAXONOMY (updated from three-tier):
  human_supervised  = pair_programmer + batch_generator
                      (Copilot, Cursor, Claude Code, Codex -- variably
                      human-supervised; AIDev metadata does not record
                      which collaboration mode was used per PR, so
                      these four tools are grouped conservatively)
  fully_autonomous  = autonomous (Devin only -- no per-step human
                      oversight, operates as independent PR author)
  human             = human baseline

The script runs four steps corresponding to the methodological ladder:
  Step 1: Pooled (all agents) vs human, syntactic definition, unadjusted Cox
  Step 2: Two-tier-disaggregated vs human, syntactic definition, unadjusted Cox
  Step 3: Two-tier-disaggregated vs human, AST structural definition, unadjusted Cox
  Step 4: Two-tier-disaggregated vs human, syntactic definition, repo-stratified
          Cox with cluster-robust SE + cluster bootstrap (confound-corrected)

Also produces:
  - Median retention times per group (for results table)
  - Hunk-level stratified Cox by block size (block-level mechanism)

Usage:
    python src/analyze_rq1_master.py \
        --line-data data/survival_events_line_v5_analysis.csv \
        --hunk-data data/survival_events_hunk_v5.csv \
        --output results/rq1_master_results.txt \
        --n-bootstrap 500

Robustness check (verified merges only; outputs to results/robustness_verified_merge/):
    VERIFIED_MERGE=1 python src/analyze_rq1_master.py
"""

import argparse
import re
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.observation_window import apply_observation_window, output_path
from config.paths import P

OUTLIER_PR_KEYS = {('microsoft/typespec', 7115)}
GENERATED_PATH_PATTERNS = [
    r'/dist/', r'/third_party/', r'/vendor/', r'/generated/',
    r'_pb2\.py$', r'\.min\.js$', r'immutable_entry',
    r'_generated\.', r'zz_generated\.',
]

# Two-tier taxonomy
HUMAN_SUPERVISED_TIERS = {'pair_programmer', 'batch_generator'}
FULLY_AUTONOMOUS_TIERS  = {'autonomous'}
TIERS = ['human_supervised', 'fully_autonomous']
SEED = 42


def is_generated(unit_id: str) -> bool:
    filepath = str(unit_id).split('::')[0]
    return any(re.search(p, filepath) for p in GENERATED_PATH_PATTERNS)


def add_two_tier_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'two_tier' column mapping agent_tier to the two-tier taxonomy."""
    def map_tier(row):
        if not row['is_agent']:
            return 'human'
        if row['agent_tier'] in HUMAN_SUPERVISED_TIERS:
            return 'human_supervised'
        if row['agent_tier'] in FULLY_AUTONOMOUS_TIERS:
            return 'fully_autonomous'
        return 'unknown'
    df = df.copy()
    df['two_tier'] = df.apply(map_tier, axis=1)
    return df


def preprocess_hunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['_key'] = list(zip(df['slug'], df['pr_number']))
    df = df[~df['_key'].isin(OUTLIER_PR_KEYS)]
    df = df[~df['unit_id'].apply(is_generated)]
    df = df.drop(columns=['_key'])
    return df


def pooled_cox(agent: pd.DataFrame, human: pd.DataFrame, label: str, out: list) -> None:
    """Step 1/2/3 style: unadjusted Cox, no stratification."""
    cox_df = pd.concat([
        agent[['duration_days', 'event']].assign(is_tier=1),
        human[['duration_days', 'event']].assign(is_tier=0)
    ], ignore_index=True).dropna()

    lr = logrank_test(
        agent['duration_days'], human['duration_days'],
        event_observed_A=agent['event'], event_observed_B=human['event']
    )

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='duration_days', event_col='event')
    s = cph.summary
    hr = float(np.exp(s.loc['is_tier', 'coef']))
    ci_low = float(np.exp(s.loc['is_tier', 'coef lower 95%']))
    ci_high = float(np.exp(s.loc['is_tier', 'coef upper 95%']))
    p_cox = float(s.loc['is_tier', 'p'])

    kmf = KaplanMeierFitter()
    kmf.fit(agent['duration_days'], agent['event'])
    median_agent = kmf.median_survival_time_

    kmf.fit(human['duration_days'], human['event'])
    median_human = kmf.median_survival_time_

    out.append({
        'label': label,
        'n_agent': len(agent),
        'n_human': len(human),
        'mod_rate_agent': round(agent['event'].mean() * 100, 1),
        'mod_rate_human': round(human['event'].mean() * 100, 1),
        'median_agent': round(float(median_agent), 1),
        'median_human': round(float(median_human), 1),
        'hr': round(hr, 4),
        'ci_low': round(ci_low, 4),
        'ci_high': round(ci_high, 4),
        'p_cox': f'{p_cox:.2e}',
        'logrank_p': f'{lr.p_value:.2e}',
    })


def fit_bootstrap_draw(strat_input: pd.DataFrame, sampled: np.ndarray) -> float:
    """Fit one cluster-bootstrap resample; NaN if the fit fails."""
    pieces = []
    for i, repo in enumerate(sampled):
        piece = strat_input[strat_input['slug'] == repo].copy()
        piece['_boot_stratum'] = f'{repo}__draw{i}'
        pieces.append(piece)
    resampled = pd.concat(pieces, ignore_index=True)
    try:
        cph_b = CoxPHFitter()
        cph_b.fit(resampled[['duration_days', 'event', 'is_tier', '_boot_stratum']],
                  duration_col='duration_days', event_col='event',
                  strata=['_boot_stratum'])
        return float(np.exp(cph_b.summary.loc['is_tier', 'coef']))
    except Exception:
        return np.nan


def parallel_bootstrap_hrs(strat_input: pd.DataFrame, rng: np.random.Generator,
                           n_bootstrap: int) -> list:
    """Bootstrap HRs in draw order (NaN = failed fit).

    All resamples are drawn up front from rng in the same order as a serial
    loop, so results match it; fits run in parallel with one BLAS thread per
    worker.
    """
    repos = strat_input['slug'].unique()
    draws = [rng.choice(repos, size=len(repos), replace=True)
             for _ in range(n_bootstrap)]
    with parallel_config(backend='loky', inner_max_num_threads=1):
        return Parallel(n_jobs=-1)(
            delayed(fit_bootstrap_draw)(strat_input, sampled) for sampled in draws
        )


def stratified_cox_bootstrap(
    agent: pd.DataFrame,
    human: pd.DataFrame,
    label: str,
    out: list,
    n_bootstrap: int = 500,
) -> None:
    """Step 4: repo-stratified Cox + cluster-robust SE + cluster bootstrap."""
    combined = pd.concat([
        agent.assign(is_tier=1),
        human.assign(is_tier=0)
    ], ignore_index=True)

    # Analytic stratified Cox with cluster-robust SE
    combined['_cluster'] = combined['slug'].copy()
    strat_input = combined[['duration_days', 'event', 'is_tier', 'slug', '_cluster']].dropna()

    n_contributing = int(
        (strat_input.groupby('slug')['is_tier'].nunique() >= 2).sum()
    )
    n_total_strata = int(strat_input['slug'].nunique())

    hr_analytic = np.nan
    ci_low_analytic = np.nan
    ci_high_analytic = np.nan
    p_analytic = np.nan
    try:
        cph = CoxPHFitter()
        cph.fit(strat_input, duration_col='duration_days', event_col='event',
                strata=['slug'], cluster_col='_cluster')
        s = cph.summary
        hr_analytic = float(np.exp(s.loc['is_tier', 'coef']))
        ci_low_analytic = float(np.exp(s.loc['is_tier', 'coef lower 95%']))
        ci_high_analytic = float(np.exp(s.loc['is_tier', 'coef upper 95%']))
        p_analytic = float(s.loc['is_tier', 'p'])
    except Exception as e:
        print(f'  Stratified Cox failed for {label}: {e}')

    # Cluster bootstrap
    rng = np.random.default_rng(SEED)
    hrs_all = parallel_bootstrap_hrs(strat_input, rng, n_bootstrap)
    hrs_boot = [hr_b for hr_b in hrs_all if np.isfinite(hr_b)]
    failures = len(hrs_all) - len(hrs_boot)

    boot_median = np.median(hrs_boot) if hrs_boot else np.nan
    boot_ci_low = np.percentile(hrs_boot, 2.5) if hrs_boot else np.nan
    boot_ci_high = np.percentile(hrs_boot, 97.5) if hrs_boot else np.nan

    out.append({
        'label': label,
        'n_contributing_strata': n_contributing,
        'n_total_strata': n_total_strata,
        'hr_analytic': round(hr_analytic, 4),
        'ci_low_analytic': round(ci_low_analytic, 4),
        'ci_high_analytic': round(ci_high_analytic, 4),
        'p_analytic': f'{p_analytic:.2e}',
        'boot_median_hr': round(boot_median, 4),
        'boot_ci_low': round(boot_ci_low, 4),
        'boot_ci_high': round(boot_ci_high, 4),
        'boot_n_valid': len(hrs_boot),
        'boot_failures': failures,
    })




def run(line_data: str, hunk_data: str, output: str, n_bootstrap: int) -> None:
    print('Loading data...')
    line = apply_observation_window(pd.read_csv(line_data), label='line')
    hunk = apply_observation_window(pd.read_csv(hunk_data), label='hunk')
    hunk = preprocess_hunk(hunk)

    # Apply two-tier remapping
    line = add_two_tier_column(line)
    hunk = add_two_tier_column(hunk)

    print(f'Two-tier distribution (line):')
    print(line['two_tier'].value_counts().to_string())

    human_line = line[line['two_tier'] == 'human'].copy()
    human_hunk = hunk[hunk['two_tier'] == 'human'].copy()

    results = {}

    # ── STEP 1: Pooled (all agents) vs human, syntactic, unadjusted ──
    print('\n=== STEP 1: Pooled vs human (syntactic, unadjusted) ===')
    step1_out = []
    all_agent_line = line[line['is_agent'] == True].copy()
    pooled_cox(all_agent_line, human_line, 'pooled_all_agents', step1_out)
    results['step1'] = pd.DataFrame(step1_out)
    print(results['step1'].to_string(index=False))

    # ── STEP 2: Two-tier-disaggregated vs human, syntactic, unadjusted ──
    print('\n=== STEP 2: Two-tier-disaggregated vs human (syntactic, unadjusted) ===')
    step2_out = []
    for tier in TIERS:
        agent = line[line['two_tier'] == tier].copy()
        pooled_cox(agent, human_line, tier, step2_out)
    results['step2'] = pd.DataFrame(step2_out)
    print(results['step2'][['label','n_agent','mod_rate_agent','mod_rate_human',
                             'median_agent','median_human','hr','ci_low','ci_high',
                             'p_cox','logrank_p']].to_string(index=False))

    # ── STEP 3: AST structural definition ──────────────────────────────
    print('\n=== STEP 3: Two-tier-disaggregated vs human (AST structural, unadjusted) ===')
    print('NOTE: event_def2 = 1 only where ast_death==1.0 (structural change).')
    step3_out = []

    line_def2 = line.copy()
    line_def2['event_def2'] = (line_def2['ast_death'] == 1.0).astype(int)

    human_def2 = line_def2[line_def2['two_tier'] == 'human'].copy()
    human_def2_ev = human_def2[['duration_days', 'event_def2', 'slug']].rename(
        columns={'event_def2': 'event'})

    for tier in TIERS:
        agent_def2 = line_def2[line_def2['two_tier'] == tier].copy()
        agent_def2_ev = agent_def2[['duration_days', 'event_def2', 'slug']].rename(
            columns={'event_def2': 'event'})
        pooled_cox(agent_def2_ev, human_def2_ev, f'{tier}_ast', step3_out)
    results['step3'] = pd.DataFrame(step3_out)
    print(results['step3'][['label','n_agent','mod_rate_agent','mod_rate_human',
                             'hr','ci_low','ci_high','p_cox','logrank_p']].to_string(index=False))

    # ── STEP 4: Repo-stratified + cluster bootstrap ────────────────────
    print(f'\n=== STEP 4: Repo-stratified + cluster bootstrap (n={n_bootstrap}) ===')
    print('This will take several minutes...')
    step4_out = []
    for tier in TIERS:
        print(f'  Running {tier}...')
        agent = line[line['two_tier'] == tier].copy()
        stratified_cox_bootstrap(agent, human_line, tier, step4_out, n_bootstrap)
        print(f'  Done: {step4_out[-1]}')
    results['step4'] = pd.DataFrame(step4_out)
    print(results['step4'].to_string(index=False))

    # ── HUNK-LEVEL by block size ───────────────────────────────────────
    print('\n=== Hunk-level stratified by block size ===')
    hunk_out = []
    for tier in TIERS:
        tier_hunk = hunk[hunk['two_tier'] == tier]
        size_groups = {
            'single_line': tier_hunk['hunk_size'] == 1,
            'small_block': (tier_hunk['hunk_size'] >= 2) & (tier_hunk['hunk_size'] <= 10),
            'large_block': tier_hunk['hunk_size'] >= 11,
        }
        size_groups_human = {
            'single_line': human_hunk['hunk_size'] == 1,
            'small_block': (human_hunk['hunk_size'] >= 2) & (human_hunk['hunk_size'] <= 10),
            'large_block': human_hunk['hunk_size'] >= 11,
        }
        for size_label in ['single_line', 'small_block', 'large_block']:
            agent_sub = tier_hunk[size_groups[size_label]]
            human_sub = human_hunk[size_groups_human[size_label]]

            if len(agent_sub) < 10 or len(human_sub) < 10:
                hunk_out.append({
                    'tier': tier, 'size_group': size_label,
                    'n_agent': len(agent_sub), 'n_human': len(human_sub),
                    'hr': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                    'p': 'insufficient data',
                })
                continue

            cox_df = pd.concat([
                agent_sub[['duration_days', 'event']].assign(is_tier=1),
                human_sub[['duration_days', 'event']].assign(is_tier=0)
            ], ignore_index=True).dropna()

            try:
                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col='duration_days', event_col='event')
                s = cph.summary
                hr = float(np.exp(s.loc['is_tier', 'coef']))
                ci_low = float(np.exp(s.loc['is_tier', 'coef lower 95%']))
                ci_high = float(np.exp(s.loc['is_tier', 'coef upper 95%']))
                p = float(s.loc['is_tier', 'p'])
                hunk_out.append({
                    'tier': tier, 'size_group': size_label,
                    'n_agent': len(agent_sub), 'n_human': len(human_sub),
                    'hr': round(hr, 4), 'ci_low': round(ci_low, 4),
                    'ci_high': round(ci_high, 4), 'p': f'{p:.2e}',
                })
            except Exception as e:
                hunk_out.append({
                    'tier': tier, 'size_group': size_label,
                    'n_agent': len(agent_sub), 'n_human': len(human_sub),
                    'hr': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                    'p': f'failed: {e}',
                })

    results['hunk_size'] = pd.DataFrame(hunk_out)
    print(results['hunk_size'].to_string(index=False))

    # ── Save all results ───────────────────────────────────────────────
    with open(output, 'w') as f:
        for section, df in results.items():
            f.write(f'\n{"="*70}\n')
            f.write(f'SECTION: {section}\n')
            f.write(f'{"="*70}\n')
            f.write(df.to_string(index=False))
            f.write('\n')
    print(f'\nAll results written to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--line-data', default=P('data/survival_events_line_v5_analysis.csv'))
    parser.add_argument('--hunk-data', default=P('data/survival_events_hunk_v5.csv'))
    parser.add_argument('--output', default=P('results/rq1_master_results.txt'))
    parser.add_argument('--n-bootstrap', type=int, default=500)
    args = parser.parse_args()
    args.output = output_path(args.output)
    run(args.line_data, args.hunk_data, args.output, args.n_bootstrap)