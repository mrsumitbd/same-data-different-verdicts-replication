"""
analyze_rq1_def2_stratified.py

Computes the missing cell in the specification grid:
Definition 2 (AST structural) × repo-stratified Cox with cluster bootstrap.

A claim that results hold "under all three corrections simultaneously"
needs the joint cell, but the stratified Cox uses
Definition 1. The joint cell -- AST-structural definition AND
repo-stratification -- is never estimated, and the two corrections push
in opposite directions for the fully-autonomous tier (Def 2 moves HR
above 1.0; stratification moves it toward 1.0 from below). The joint
result is not predictable from the marginals.

This script reports the full 2x2 specification grid:
  {Def 1, Def 2} x {unadjusted, stratified + bootstrap}

for both two-tier groups, so the paper can present the complete picture.

Outputs:
  results/rq1_def2_stratified_results.txt

Usage:
    python src/analyze_rq1_def2_stratified.py \
        --line-data data/survival_events_line_v5_analysis.csv \
        --output results/rq1_def2_stratified_results.txt \
        --n-bootstrap 500

Robustness check (verified merges only; outputs to results/robustness_verified_merge/):
    VERIFIED_MERGE=1 python src/analyze_rq1_def2_stratified.py
"""

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.observation_window import apply_observation_window, output_path
from config.paths import P

HUMAN_SUPERVISED_TIERS = {'pair_programmer', 'batch_generator'}
FULLY_AUTONOMOUS_TIERS  = {'autonomous'}
TIERS = ['human_supervised', 'fully_autonomous']
SEED = 42


def add_two_tier_column(df: pd.DataFrame) -> pd.DataFrame:
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


def unadjusted_cox(agent: pd.DataFrame, human: pd.DataFrame) -> dict:
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
    return {
        'n_agent':      len(agent),
        'n_human':      len(human),
        'mod_rate_agent': round(agent['event'].mean() * 100, 1),
        'mod_rate_human': round(human['event'].mean() * 100, 1),
        'hr':      round(float(np.exp(s.loc['is_tier', 'coef'])), 4),
        'ci_low':  round(float(np.exp(s.loc['is_tier', 'coef lower 95%'])), 4),
        'ci_high': round(float(np.exp(s.loc['is_tier', 'coef upper 95%'])), 4),
        'p_cox':   f"{float(s.loc['is_tier', 'p']):.2e}",
        'logrank_p': f'{lr.p_value:.2e}',
    }


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


def stratified_bootstrap_cox(agent: pd.DataFrame, human: pd.DataFrame,
                              n_bootstrap: int = 500) -> dict:
    combined = pd.concat([
        agent.assign(is_tier=1),
        human.assign(is_tier=0)
    ], ignore_index=True)
    strat_input = combined[['duration_days', 'event',
                             'is_tier', 'slug']].dropna().copy()
    strat_input['_cluster'] = strat_input['slug']

    n_contributing = int(
        (strat_input.groupby('slug')['is_tier'].nunique() >= 2).sum()
    )
    n_total = int(strat_input['slug'].nunique())

    hr_a = p_a = np.nan
    try:
        cph = CoxPHFitter()
        cph.fit(strat_input, duration_col='duration_days', event_col='event',
                strata=['slug'], cluster_col='_cluster')
        s = cph.summary
        hr_a = float(np.exp(s.loc['is_tier', 'coef']))
        p_a  = float(s.loc['is_tier', 'p'])
    except Exception as e:
        print(f'  Stratified Cox failed: {e}')

    rng = np.random.default_rng(SEED)
    hrs_boot = [hr_b for hr_b in parallel_bootstrap_hrs(strat_input, rng, n_bootstrap)
                if np.isfinite(hr_b)]

    return {
        'n_contributing_strata': n_contributing,
        'n_total_strata': n_total,
        'hr_analytic': round(hr_a, 4),
        'p_analytic': f'{p_a:.2e}',
        'boot_median_hr': round(np.median(hrs_boot), 4) if hrs_boot else np.nan,
        'boot_ci_low':  round(np.percentile(hrs_boot, 2.5), 4) if hrs_boot else np.nan,
        'boot_ci_high': round(np.percentile(hrs_boot, 97.5), 4) if hrs_boot else np.nan,
        'boot_n_valid': len(hrs_boot),
    }


def run(line_data: str, output: str, n_bootstrap: int) -> None:
    print('Loading data...')
    line = apply_observation_window(pd.read_csv(line_data), label='line')
    line = add_two_tier_column(line)

    # Def 1 event column (syntactic — already present as 'event')
    # Def 2 event column (AST structural)
    line['event_def2'] = (line['ast_death'] == 1.0).astype(int)

    human_def1 = line[line['two_tier'] == 'human'][
        ['duration_days', 'event', 'slug']].copy()
    human_def2 = line[line['two_tier'] == 'human'][
        ['duration_days', 'event_def2', 'slug']].rename(
        columns={'event_def2': 'event'}).copy()

    out_lines = []
    out_lines.append('Full 2x2 specification grid: {Def 1, Def 2} x {Unadjusted, Stratified}')
    out_lines.append('='*70)

    for tier in TIERS:
        print(f'\n=== {tier} ===')
        out_lines.append(f'\n{"="*70}')
        out_lines.append(f'TIER: {tier}')
        out_lines.append('='*70)

        agent_def1 = line[line['two_tier'] == tier][
            ['duration_days', 'event', 'slug']].copy()
        agent_def2 = line[line['two_tier'] == tier][
            ['duration_days', 'event_def2', 'slug']].rename(
            columns={'event_def2': 'event'}).copy()

        # Cell 1: Def 1 x Unadjusted (already known, reported for completeness)
        print('  Computing Def1 x Unadjusted...')
        r = unadjusted_cox(agent_def1, human_def1)
        out_lines.append(
            f'\n[Def 1 x Unadjusted] (n_agent={r["n_agent"]:,}, n_human={r["n_human"]:,})')
        out_lines.append(
            f'  mod_rate: agent={r["mod_rate_agent"]}%, human={r["mod_rate_human"]}%')
        out_lines.append(
            f'  HR={r["hr"]} [{r["ci_low"]}, {r["ci_high"]}], p={r["p_cox"]}, logrank_p={r["logrank_p"]}')

        # Cell 2: Def 2 x Unadjusted (already known, reported for completeness)
        print('  Computing Def2 x Unadjusted...')
        r = unadjusted_cox(agent_def2, human_def2)
        out_lines.append(
            f'\n[Def 2 x Unadjusted] (n_agent={r["n_agent"]:,}, n_human={r["n_human"]:,})')
        out_lines.append(
            f'  mod_rate: agent={r["mod_rate_agent"]}%, human={r["mod_rate_human"]}%')
        out_lines.append(
            f'  HR={r["hr"]} [{r["ci_low"]}, {r["ci_high"]}], p={r["p_cox"]}, logrank_p={r["logrank_p"]}')

        # Cell 3: Def 1 x Stratified (already known, reported for completeness)
        print(f'  Computing Def1 x Stratified + bootstrap (n={n_bootstrap})...')
        r = stratified_bootstrap_cox(agent_def1, human_def1, n_bootstrap)
        out_lines.append(
            f'\n[Def 1 x Stratified + Bootstrap] '
            f'(contributing_strata={r["n_contributing_strata"]}/{r["n_total_strata"]})')
        out_lines.append(
            f'  HR_analytic={r["hr_analytic"]}, p={r["p_analytic"]}')
        out_lines.append(
            f'  Bootstrap median HR={r["boot_median_hr"]}, '
            f'95% CI=[{r["boot_ci_low"]}, {r["boot_ci_high"]}], '
            f'n_valid={r["boot_n_valid"]}')

        # Cell 4: Def 2 x Stratified (THE MISSING CELL)
        print(f'  Computing Def2 x Stratified + bootstrap (n={n_bootstrap}) -- THE MISSING CELL...')
        r = stratified_bootstrap_cox(agent_def2, human_def2, n_bootstrap)
        out_lines.append(
            f'\n[Def 2 x Stratified + Bootstrap] *** THE JOINT CORRECTION ***'
            f'\n  (contributing_strata={r["n_contributing_strata"]}/{r["n_total_strata"]})')
        out_lines.append(
            f'  HR_analytic={r["hr_analytic"]}, p={r["p_analytic"]}')
        out_lines.append(
            f'  Bootstrap median HR={r["boot_median_hr"]}, '
            f'95% CI=[{r["boot_ci_low"]}, {r["boot_ci_high"]}], '
            f'n_valid={r["boot_n_valid"]}')

        sig = 'NS' if r['boot_ci_low'] <= 1.0 <= r['boot_ci_high'] else 'SIGNIFICANT'
        relation = 'includes' if sig == 'NS' else 'excludes'
        out_lines.append(f'  Bootstrap CI interpretation: {sig} (CI {relation} 1.0: '
                         f'[{r["boot_ci_low"]}, {r["boot_ci_high"]}])')

    result_text = '\n'.join(out_lines)
    print('\n' + result_text)
    with open(output, 'w') as f:
        f.write(result_text)
    print(f'\nResults written to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--line-data', default=P('data/survival_events_line_v5_analysis.csv'))
    parser.add_argument('--output',    default=P('results/rq1_def2_stratified_results.txt'))
    parser.add_argument('--n-bootstrap', type=int, default=500)
    args = parser.parse_args()
    args.output = output_path(args.output)
    run(args.line_data, args.output, args.n_bootstrap)