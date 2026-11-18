"""
analyze_rq2.py

Produces all numbers needed for RQ2 in one clean pass.

RQ2: How does modification operationalization affect conclusions?

Two operationalization dimensions:

  1. MODIFICATION DEFINITION: syntactic (Def 1) vs AST-structural (Def 2)
     Numbers come from analyze_rq1_master.py (Steps 2 and 3), which already
     produces the Def 1 and Def 2 HRs by two-tier grouping. This script
     does NOT rerun those Cox models -- it reads from rq1_master_results.txt
     and reformats for the RQ2 table.

     Section D adds the Def 2 composition (leaf-only vs structural share of
     modified lines, null ast_death share) and a Def 2 sensitivity Cox that
     drops modified lines whose AST status is unknown.

  2. INTENT CLASSIFICATION: keyword vs each LLM vs LLM aggregations,
     reported side by side (no primary classifier). That analysis is in
     analyze_intent_aggregation.py (--full-corpus for the corpus, default
     mode for the 385 human-labelled commits); get_majority_tier() below is
     the tier mapping it uses.

TWO-TIER TAXONOMY:
  human_supervised = pair_programmer + batch_generator
  fully_autonomous = autonomous (Devin only)
  human            = human baseline

Outputs:
  results/rq2_results.txt

Usage:
    python src/analyze_rq2.py \
        --line-data data/survival_events_line_v5_analysis.csv \
        --rq1-results results/rq1_master_results.txt \
        --output results/rq2_results.txt

Robustness check (verified merges only; outputs to results/robustness_verified_merge/):
    VERIFIED_MERGE=1 python src/analyze_rq2.py
"""

import argparse
import io
import re

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
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
HUMAN_SUPERVISED_TIERS = {'pair_programmer', 'batch_generator'}
FULLY_AUTONOMOUS_TIERS  = {'autonomous'}


def is_generated(unit_id: str) -> bool:
    filepath = str(unit_id).split('::')[0]
    return any(re.search(p, filepath) for p in GENERATED_PATH_PATTERNS)


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


def get_majority_tier(hunks: pd.DataFrame) -> pd.DataFrame:
    """Map (slug, death_sha) -> two_tier via majority vote.

    Takes line-level data: every modification commit appears there, whereas
    the hunk-level file covers fewer than half of them.
    """
    hunks = hunks.copy()
    hunks['_key'] = list(zip(hunks['slug'], hunks['pr_number']))
    hunks = hunks[~hunks['_key'].isin(OUTLIER_PR_KEYS)]
    hunks = hunks[~hunks['unit_id'].apply(is_generated)]
    hunks = add_two_tier_column(hunks)

    def majority(s): return s.value_counts().index[0]
    tier_map = (
        hunks.groupby(['slug', 'death_sha'])['two_tier']
        .agg(majority)
        .reset_index()
        .rename(columns={'death_sha': 'sha'})
    )
    return tier_map


def read_rq1_section(path: str, section: str) -> pd.DataFrame:
    """Read one SECTION table from rq1_master_results.txt."""
    text = open(path).read()
    block = text.split(f'SECTION: {section}\n')[1].split('='*70)[1]
    return pd.read_csv(io.StringIO(block.strip()), sep=r'\s+').set_index('label')


def unadjusted_cox(agent: pd.DataFrame, human: pd.DataFrame) -> dict:
    """Same unadjusted Cox as pooled_cox in analyze_rq1_master.py."""
    cox_df = pd.concat([
        agent[['duration_days', 'event']].assign(is_tier=1),
        human[['duration_days', 'event']].assign(is_tier=0)
    ], ignore_index=True).dropna()
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='duration_days', event_col='event')
    s = cph.summary
    return {
        'hr': float(np.exp(s.loc['is_tier', 'coef'])),
        'ci_low': float(np.exp(s.loc['is_tier', 'coef lower 95%'])),
        'ci_high': float(np.exp(s.loc['is_tier', 'coef upper 95%'])),
        'p': float(s.loc['is_tier', 'p']),
    }


def run(line_data: str,
        rq1_results: str, output: str) -> None:

    print('Loading data...')
    line = pd.read_csv(line_data, usecols=[
        'slug', 'pr_number', 'is_agent', 'agent_tier', 'unit_id', 'death_sha',
        'event', 'duration_days', 'ast_death', 'birth_date', 'death_date'])
    line = apply_observation_window(line, label='line')


    out_lines = []

    # ── SECTION A: Modification definition comparison ─────────────────
    # Read Def1 and Def2 HRs from RQ1 master results file
    out_lines.append('\n' + '='*70)
    out_lines.append('SECTION A: Modification definition (Def1 syntactic vs Def2 structural)')
    out_lines.append('NOTE: HRs computed by analyze_rq1_master.py (Steps 2 and 3)')
    out_lines.append('='*70)
    step2 = read_rq1_section(rq1_results, 'step2')
    step3 = read_rq1_section(rq1_results, 'step3')
    for title, table, suffix in [('Def 1 (syntactic, unadjusted):', step2, ''),
                                 ('Def 2 (AST structural, unadjusted):', step3, '_ast')]:
        out_lines.append(f'\n{title}')
        for tier in ['human_supervised', 'fully_autonomous']:
            r = table.loc[tier + suffix]
            note = ' (REVERSED)' if r['hr'] > 1 else ''
            out_lines.append(
                f"  {tier}: HR={r['hr']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}], "
                f"p={r['p_cox']:.2e}{note}"
            )
    for title, table, suffix in [('Def 1 modification rates:', step2, ''),
                                 ('Def 2 modification rates:', step3, '_ast')]:
        hs = table.loc['human_supervised' + suffix]
        fa = table.loc['fully_autonomous' + suffix]
        out_lines.append(f'\n{title}')
        out_lines.append(
            f"  human_supervised: {hs['mod_rate_agent']}%  |  "
            f"fully_autonomous: {fa['mod_rate_agent']}%  |  human: {hs['mod_rate_human']}%"
        )

    # ── SECTION D: Def 2 composition and sensitivity ──────────────────
    out_lines.append('\n' + '='*70)
    out_lines.append('SECTION D: Def 2 composition and null-AST sensitivity')
    out_lines.append('(line-level, two-tier mapping as in analyze_rq1_master.py)')
    out_lines.append('='*70)

    line = add_two_tier_column(line)
    modified = line[line['event'] == 1]

    out_lines.append('\nLeaf-only share of modified lines with known AST status '
                     '(ast_death==0 among event==1, ast_death non-null):')
    for tier in ['human_supervised', 'fully_autonomous', 'human']:
        sub = modified[(modified['two_tier'] == tier) & modified['ast_death'].notna()]
        n_leaf = (sub['ast_death'] == 0).sum()
        out_lines.append(f'  {tier}: {n_leaf}/{len(sub)} ({n_leaf/len(sub)*100:.1f}%)')

    out_lines.append('\nModified lines with null ast_death (event==1):')
    for tier in ['human_supervised', 'fully_autonomous', 'human']:
        sub = modified[modified['two_tier'] == tier]
        n_null = sub['ast_death'].isna().sum()
        out_lines.append(f'  {tier}: {n_null}/{len(sub)} ({n_null/len(sub)*100:.1f}%)')

    # Def 2 sensitivity: drop modified lines whose AST status is unknown
    known = line[~((line['event'] == 1) & line['ast_death'].isna())].copy()
    known['event'] = (known['ast_death'] == 1.0).astype(int)
    human_known = known[known['two_tier'] == 'human']
    out_lines.append('\nDef 2 sensitivity (unadjusted Cox, event==1 & null ast_death dropped):')
    for tier in ['human_supervised', 'fully_autonomous']:
        agent_known = known[known['two_tier'] == tier]
        r = unadjusted_cox(agent_known, human_known)
        out_lines.append(
            f"  {tier} (n_agent={len(agent_known)}, n_human={len(human_known)}): "
            f"HR={r['hr']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}], p={r['p']:.2e}"
        )

    # Print and save
    result_text = '\n'.join(out_lines)
    print(result_text)
    with open(output, 'w') as f:
        f.write(result_text)
    print(f'\nResults written to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--line-data',    default=P('data/survival_events_line_v5_analysis.csv'))
    parser.add_argument('--rq1-results', default=P('results/rq1_master_results.txt'))
    parser.add_argument('--output',      default=P('results/rq2_results.txt'))
    args = parser.parse_args()
    args.output = output_path(args.output)
    args.rq1_results = output_path(args.rq1_results)
    run(args.line_data,
        args.rq1_results, args.output)