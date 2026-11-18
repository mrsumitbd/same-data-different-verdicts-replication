"""
analyze_rq4.py

Produces all numbers needed for RQ4 in one clean pass.

RQ4: What does corrected measurement reveal?

Two parts:

  1. MODIFIER IDENTITY ANALYSIS
     Who performs modifications to agent-attributed code -- humans or agents?
     - modifier_type distribution by birth tier (two-tier)
     - Agent-to-agent modification: same tool or different?
     Intent by modifier type is reported per classification approach in
     analyze_intent_aggregation.py --full-corpus (Section B there).

  2. INTEGRATED CORRECTED PICTURE
     Summarizes what the data show after all three corrections:
     (a) No significant retention differences (read from rq1_master_results.txt Step 4)
     (b) Human maintenance is primary

TWO-TIER TAXONOMY:
  human_supervised = pair_programmer + batch_generator
  fully_autonomous = autonomous (Devin only)
  human            = human baseline

Outputs:
  results/rq4_results.txt

Usage:
    python src/analyze_rq4.py \
        --line-data data/survival_events_line_v5_analysis.csv \
        --commits data/death_commit_messages_v5.csv \
        --modifier-authors results/modifier_authors.csv \
        --rq1-results results/rq1_master_results.txt \
        --output results/rq4_results.txt

Robustness check (verified merges only; outputs to results/robustness_verified_merge/):
    VERIFIED_MERGE=1 python src/analyze_rq4.py
"""

import argparse
import io
import re

import pandas as pd
import numpy as np
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


def get_birth_tier_map(hunks: pd.DataFrame) -> pd.DataFrame:
    """Map (slug, death_sha) -> birth two_tier via majority vote.

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
        .rename(columns={'death_sha': 'sha', 'two_tier': 'birth_tier'})
    )
    return tier_map


def read_rq1_section(path: str, section: str) -> pd.DataFrame:
    """Read one SECTION table from rq1_master_results.txt."""
    text = open(path).read()
    block = text.split(f'SECTION: {section}\n')[1].split('='*70)[1]
    return pd.read_csv(io.StringIO(block.strip()), sep=r'\s+').set_index('label')


def run(line_data: str, commits: str,
        modifier_authors: str, rq1_results: str, output: str) -> None:

    print('Loading data...')
    hunks = pd.read_csv(line_data, usecols=[
        'slug', 'pr_number', 'is_agent', 'agent_tier', 'unit_id', 'death_sha',
        'event', 'duration_days', 'birth_date', 'death_date'])
    hunks = apply_observation_window(hunks, label='line')
    hunks = hunks.dropna(subset=['death_sha'])
    mod_commits = pd.read_csv(commits)[['slug', 'sha']].drop_duplicates()
    authors = pd.read_csv(modifier_authors)

    birth_tier_map = get_birth_tier_map(hunks)
    print(f'Birth tier map: {len(birth_tier_map):,} unique (slug, sha) pairs')

    # Modification commits with birth tier and modifier identity
    merged = mod_commits.merge(authors, on=['slug', 'sha'], how='left')
    merged = merged.merge(birth_tier_map, on=['slug', 'sha'], how='left')
    merged = merged.dropna(subset=['birth_tier', 'modifier_type'])

    print(f'Merged dataset: {len(merged):,} modification commits with tier + modifier info')

    out_lines = []

    # ── SECTION A: Modifier type by birth tier ────────────────────────
    out_lines.append('\n' + '='*70)
    out_lines.append('SECTION A: Modifier type distribution by birth tier')
    out_lines.append('='*70)

    total_overall = len(merged)
    for tier in ['human_supervised', 'fully_autonomous', 'human']:
        sub = merged[merged['birth_tier'] == tier]
        if len(sub) == 0:
            continue
        vc = sub['modifier_type'].value_counts()
        out_lines.append(f'\n  {tier} (n={len(sub)}):')
        for mt in ['human', 'agent', 'unknown']:
            n = vc.get(mt, 0)
            out_lines.append(f'    {mt}: {n} ({n/len(sub)*100:.1f}%)')

    out_lines.append(f'\nOverall modifier_type distribution (all birth tiers):')
    vc_all = merged['modifier_type'].value_counts()
    for mt in ['human', 'agent', 'unknown']:
        n = vc_all.get(mt, 0)
        out_lines.append(f'  {mt}: {n} ({n/total_overall*100:.1f}%)')

    agent_born = merged[merged['birth_tier'].isin(['human_supervised', 'fully_autonomous'])]
    n_human_mod = (agent_born['modifier_type'] == 'human').sum()
    out_lines.append(
        f'\nAgent-attributed code (both tiers) modified by humans: '
        f'{n_human_mod}/{len(agent_born)} ({n_human_mod/len(agent_born)*100:.1f}%)'
    )

    # ── SECTION B: Agent-to-agent: same tool or different? ────────────
    out_lines.append('\n' + '='*70)
    out_lines.append('SECTION B: Agent modifier identity by birth tier')
    out_lines.append('(Who modifies agent-attributed code when an agent does it?)')
    out_lines.append('='*70)

    agent_mods = merged[merged['modifier_type'] == 'agent']
    out_lines.append(f'\nTotal agent-modified commits: {len(agent_mods)}')

    for tier in ['human_supervised', 'fully_autonomous', 'human']:
        sub = agent_mods[agent_mods['birth_tier'] == tier]
        if len(sub) == 0:
            continue
        out_lines.append(f'\n  {tier} modified by agent (n={len(sub)}):')
        top = sub['author_login'].value_counts().head(8)
        for login, count in top.items():
            pct = count / len(sub) * 100
            out_lines.append(f'    {login}: {count} ({pct:.1f}%)')

    # ── SECTION C: Integrated corrected picture ───────────────────────
    out_lines.append('\n' + '='*70)
    out_lines.append('SECTION C: Integrated corrected picture summary')
    out_lines.append('='*70)

    out_lines.append('\nRetention (repo-stratified, bootstrap CI; from rq1_master_results.txt):')
    step4 = read_rq1_section(rq1_results, 'step4')
    for tier in ['human_supervised', 'fully_autonomous']:
        r = step4.loc[tier]
        out_lines.append(
            f'  {tier}: HR={r["hr_analytic"]:.3f} '
            f'[{r["boot_ci_low"]:.3f}, {r["boot_ci_high"]:.3f}], p={r["p_analytic"]:.3f}'
        )

    # Measurement contrast: across repositories (RQ1 step 2: pooled, unstratified Cox)
    # vs within repositories (step 4: repo-stratified Cox, cluster bootstrap CI).
    out_lines.append('\nMeasurement contrast, across vs within repositories (from rq1_master_results.txt):')
    out_lines.append(f'  {"tier":17s} {"across: HR [95% CI]":>26s} {"within: HR [boot 95% CI]":>28s} '
                     f'{"ratio":>6s}  direction / significance')
    step2 = read_rq1_section(rq1_results, 'step2')
    for tier in ['human_supervised', 'fully_autonomous']:
        x, w = step2.loc[tier], step4.loc[tier]
        sig_x = not (x['ci_low'] <= 1 <= x['ci_high'])
        sig_w = not (w['boot_ci_low'] <= 1 <= w['boot_ci_high'])
        same_dir = (x['hr'] - 1) * (w['hr_analytic'] - 1) > 0
        out_lines.append(
            f'  {tier:17s} {x["hr"]:8.3f} [{x["ci_low"]:.3f}, {x["ci_high"]:.3f}]   '
            f'{w["hr_analytic"]:8.3f} [{w["boot_ci_low"]:.3f}, {w["boot_ci_high"]:.3f}]   '
            f'{w["hr_analytic"] / x["hr"]:6.3f}  '
            f'{"same" if same_dir else "opposite"} direction; '
            f'across {"significant" if sig_x else "n.s."}, within {"significant" if sig_w else "n.s."}')

    out_lines.append('\nIntent: reported per classification approach in '
                     'results/intent_aggregation_full_corpus.txt')

    out_lines.append('\nModifier identity overall:')
    for mt in ['human', 'agent', 'unknown']:
        n = vc_all.get(mt, 0)
        out_lines.append(f'  {mt}: {n} ({n/total_overall*100:.1f}%)')

    result_text = '\n'.join(out_lines)
    print(result_text)
    with open(output, 'w') as f:
        f.write(result_text)
    print(f'\nResults written to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--line-data',        default=P('data/survival_events_line_v5_analysis.csv'))
    parser.add_argument('--commits',          default=P('data/death_commit_messages_v5.csv'))
    parser.add_argument('--modifier-authors', default=P('results/modifier_authors.csv'))
    parser.add_argument('--rq1-results',      default=P('results/rq1_master_results.txt'))
    parser.add_argument('--output',           default=P('results/rq4_results.txt'))
    args = parser.parse_args()
    args.output = output_path(args.output)
    args.rq1_results = output_path(args.rq1_results)
    run(args.line_data, args.commits, args.modifier_authors,
        args.rq1_results, args.output)