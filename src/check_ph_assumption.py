"""
check_ph_assumption.py

Checks the proportional-hazards (PH) assumption for the RQ1 Cox models
using the Schoenfeld-residual test (lifelines proportional_hazard_test).

Restored from the deleted analyze_rq1-check_ph_assumption.py, which tested
an earlier covariate model on prediction_features_v1.csv. This version
tests the models the paper reports, with the same data and two-tier
mapping as analyze_rq1_master.py:
  Step 1: pooled agents vs human, Def 1 (syntactic), unadjusted
  Step 2: each two-tier group vs human, Def 1, unadjusted
  Step 3: each two-tier group vs human, Def 2 (AST structural), unadjusted
  Step 4: each two-tier group vs human, Def 1, repo-stratified

Each test is run with two time transforms (rank and km). H0: the
is_tier effect is constant over time; p < 0.05 indicates a violation.

TWO-TIER TAXONOMY:
  human_supervised = pair_programmer + batch_generator
  fully_autonomous = autonomous (Devin only)
  human            = human baseline

Outputs:
  results/ph_assumption_test.txt

Usage:
    python src/check_ph_assumption.py \
        --line-data data/survival_events_line_v5_analysis.csv \
        --output results/ph_assumption_test.txt

Robustness check (verified merges only; outputs to results/robustness_verified_merge/):
    VERIFIED_MERGE=1 python src/check_ph_assumption.py
"""

import argparse

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.observation_window import apply_observation_window, output_path
from config.paths import P

HUMAN_SUPERVISED_TIERS = {'pair_programmer', 'batch_generator'}
FULLY_AUTONOMOUS_TIERS = {'autonomous'}
TIERS = ['human_supervised', 'fully_autonomous']
TIME_TRANSFORMS = ['rank', 'km']


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


def ph_test(agent: pd.DataFrame, human: pd.DataFrame, label: str,
            out: list, stratified: bool = False) -> None:
    """Fit the Cox model used in analyze_rq1_master.py and test PH for is_tier."""
    cols = ['duration_days', 'event', 'slug'] if stratified else ['duration_days', 'event']
    cox_df = pd.concat([
        agent[cols].assign(is_tier=1),
        human[cols].assign(is_tier=0)
    ], ignore_index=True).dropna()

    cph = CoxPHFitter()
    if stratified:
        cph.fit(cox_df, duration_col='duration_days', event_col='event', strata=['slug'])
    else:
        cph.fit(cox_df, duration_col='duration_days', event_col='event')
    row = {
        'model': label,
        'n': len(cox_df),
        'hr': round(float(np.exp(cph.summary.loc['is_tier', 'coef'])), 4),
    }
    for transform in TIME_TRANSFORMS:
        res = proportional_hazard_test(cph, cox_df, time_transform=transform)
        row[f'stat_{transform}'] = round(float(res.summary.loc['is_tier', 'test_statistic']), 2)
        row[f'p_{transform}'] = f"{float(res.summary.loc['is_tier', 'p']):.2e}"
    out.append(row)
    print(f'  {row}')


def run(line_data: str, output: str) -> None:
    print('Loading data...')
    line = add_two_tier_column(apply_observation_window(pd.read_csv(line_data, usecols=[
        'slug', 'pr_number', 'is_agent', 'agent_tier', 'event', 'duration_days',
        'ast_death', 'birth_date', 'death_date']), label='line'))
    human = line[line['two_tier'] == 'human']

    rows = []
    print('\nStep 1: pooled vs human (Def 1)')
    ph_test(line[line['is_agent'] == True], human, 'step1_pooled_def1', rows)

    print('\nStep 2: two-tier vs human (Def 1)')
    for tier in TIERS:
        ph_test(line[line['two_tier'] == tier], human, f'step2_{tier}_def1', rows)

    print('\nStep 3: two-tier vs human (Def 2)')
    line_def2 = line.assign(event=(line['ast_death'] == 1.0).astype(int))
    human_def2 = line_def2[line_def2['two_tier'] == 'human']
    for tier in TIERS:
        ph_test(line_def2[line_def2['two_tier'] == tier], human_def2,
                f'step3_{tier}_def2', rows)

    print('\nStep 4: two-tier vs human (Def 1, repo-stratified)')
    for tier in TIERS:
        ph_test(line[line['two_tier'] == tier], human,
                f'step4_{tier}_def1_stratified', rows, stratified=True)

    results = pd.DataFrame(rows)
    with open(output, 'w') as f:
        f.write('Schoenfeld-residual PH test for is_tier '
                '(H0: constant effect over time; p < 0.05 = violation)\n\n')
        f.write(results.to_string(index=False))
        f.write('\n')
    print(f'\n{results.to_string(index=False)}')
    print(f'\nResults written to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--line-data', default=P('data/survival_events_line_v5_analysis.csv'))
    parser.add_argument('--output',    default=P('results/ph_assumption_test.txt'))
    args = parser.parse_args()
    args.output = output_path(args.output)
    run(args.line_data, args.output)
