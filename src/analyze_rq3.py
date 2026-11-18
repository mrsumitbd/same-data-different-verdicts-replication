"""
analyze_rq3.py

Produces all numbers needed for RQ3 in one clean pass.

RQ3: How does repo-level confound correction affect conclusions?

Two parts:

  1. REPO DISTRIBUTION DIAGNOSTIC
     - Repo coverage per tier (how many cohort repos contain each tier)
     - Pairwise Jaccard similarity between tier repo sets
     - Mann-Whitney tests: do repos containing each tier differ from
       repos that don't on contributor count, stars, language?

  2. STRATIFIED COX WITH BOOTSTRAP
     Numbers come from analyze_rq1_master.py (Step 4), which already
     produces the stratified HRs and bootstrap CIs by two-tier grouping.
     This script reads those results and adds the repo diagnostic context.

TWO-TIER TAXONOMY:
  human_supervised = pair_programmer + batch_generator
  fully_autonomous = autonomous (Devin only)
  human            = human baseline

Outputs:
  results/rq3_results.txt

Usage:
    python src/analyze_rq3.py \
        --line-data data/survival_events_line_v5_analysis.csv \
        --repo-meta data/aidev_filtered_v1.csv \
        --rq1-results results/rq1_master_results.txt \
        --output results/rq3_results.txt

Robustness check (verified merges only; outputs to results/robustness_verified_merge/):
    VERIFIED_MERGE=1 python src/analyze_rq3.py
"""

import argparse
import io

import numpy as np
import pandas as pd
from scipy import stats
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.observation_window import apply_observation_window, output_path
from config.paths import P

HUMAN_SUPERVISED_TIERS = {'pair_programmer', 'batch_generator'}
FULLY_AUTONOMOUS_TIERS  = {'autonomous'}


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


def jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def read_rq1_section(path: str, section: str) -> pd.DataFrame:
    """Read one SECTION table from rq1_master_results.txt."""
    text = open(path).read()
    block = text.split(f'SECTION: {section}\n')[1].split('='*70)[1]
    return pd.read_csv(io.StringIO(block.strip()), sep=r'\s+').set_index('label')


def run(line_data: str, repo_meta: str, rq1_results: str, output: str) -> None:
    print('Loading data...')
    line = apply_observation_window(pd.read_csv(line_data), label='line')
    repos = pd.read_csv(repo_meta)

    line = add_two_tier_column(line)

    # Universe: cohort repositories that contribute lines after the
    # observation window, not every repository in the metadata file.
    all_repos = set(line['slug'].unique())
    repos = repos[repos['slug'].isin(all_repos)]
    n_total = len(all_repos)

    out_lines = []

    # ── SECTION A: Repo distribution diagnostic ───────────────────────
    out_lines.append('\n' + '='*70)
    out_lines.append('SECTION A: Repository distribution diagnostic')
    out_lines.append('='*70)

    tier_repo_sets = {}
    for tier in ['human_supervised', 'fully_autonomous', 'human']:
        repo_set = set(line[line['two_tier'] == tier]['slug'].unique())
        tier_repo_sets[tier] = repo_set
        pct = len(repo_set) / n_total * 100
        out_lines.append(
            f'  {tier}: {len(repo_set)} / {n_total} repos ({pct:.1f}%)'
        )

    out_lines.append('\nPairwise Jaccard similarity:')
    tiers = ['human_supervised', 'fully_autonomous']
    for i, t1 in enumerate(tiers):
        for t2 in tiers[i+1:]:
            jac = jaccard(tier_repo_sets[t1], tier_repo_sets[t2])
            overlap = len(tier_repo_sets[t1] & tier_repo_sets[t2])
            out_lines.append(
                f'  {t1} ∩ {t2}: {overlap} repos, Jaccard={jac:.3f}'
            )

    # Repos containing both tiers
    both = tier_repo_sets['human_supervised'] & tier_repo_sets['fully_autonomous']
    out_lines.append(f'\nRepos containing BOTH agent tiers: {len(both)}')

    # Repos containing exactly one agent tier
    only_hs = tier_repo_sets['human_supervised'] - tier_repo_sets['fully_autonomous']
    only_fa = tier_repo_sets['fully_autonomous'] - tier_repo_sets['human_supervised']
    neither = all_repos - (tier_repo_sets['human_supervised'] | tier_repo_sets['fully_autonomous'])
    out_lines.append(f'Repos with ONLY human_supervised: {len(only_hs)}')
    out_lines.append(f'Repos with ONLY fully_autonomous: {len(only_fa)}')
    out_lines.append(f'Repos with NO agent tiers: {len(neither)}')

    one_tier = len(only_hs) + len(only_fa)
    out_lines.append(
        f'Repos with exactly one agent tier: {one_tier} / {n_total} '
        f'({one_tier/n_total*100:.1f}%)'
    )

    if 'language' in repos.columns:
        lang_all = repos['language'].value_counts(normalize=True).head(5) * 100
        out_lines.append(f'\nCohort language distribution ({n_total} repos, top 5): '
                         + ', '.join(f'{k} {v:.1f}%' for k, v in lang_all.items()))

    # ── SECTION B: Systematic differences between tier repo sets ──────
    out_lines.append('\n' + '='*70)
    out_lines.append('SECTION B: Systematic differences (Mann-Whitney U tests)')
    out_lines.append('='*70)

    numeric_cols = ['stars', 'forks', 'size_kb', 'open_issues', 'contributors', 'loc']
    repos_indexed = repos.set_index('slug')

    for tier in ['human_supervised', 'fully_autonomous']:
        with_tier = repos_indexed.loc[repos_indexed.index.isin(tier_repo_sets[tier])]
        without_tier = repos_indexed.loc[~repos_indexed.index.isin(tier_repo_sets[tier])]
        out_lines.append(
            f'\n  {tier}: n_with={len(with_tier)}, n_without={len(without_tier)}'
        )
        for col in numeric_cols:
            if col not in repos_indexed.columns:
                continue
            w = with_tier[col].dropna()
            wo = without_tier[col].dropna()
            if len(w) < 2 or len(wo) < 2:
                continue
            stat, p = stats.mannwhitneyu(w, wo, alternative='two-sided')
            flag = '  <-- DIFFERS (p<0.05)' if p < 0.05 else ''
            out_lines.append(
                f'    {col:15s}: median_with={w.median():>10,.0f}  '
                f'median_without={wo.median():>10,.0f}  p={p:.4f}{flag}'
            )

        if 'language' in repos_indexed.columns:
            lang_with = with_tier['language'].value_counts(normalize=True).head(3)
            lang_wo   = without_tier['language'].value_counts(normalize=True).head(3)
            out_lines.append(f'    Top languages WITH {tier}: {dict(lang_with.round(3))}')
            out_lines.append(f'    Top languages WITHOUT {tier}: {dict(lang_wo.round(3))}')

    # ── SECTION C: Stratified Cox results (from RQ1 master) ──────────
    out_lines.append('\n' + '='*70)
    out_lines.append('SECTION C: Repo-stratified Cox + bootstrap results')
    out_lines.append('NOTE: Computed by analyze_rq1_master.py (Step 4)')
    out_lines.append('Bootstrap CI reported (500 resamples); analytic CI for')
    out_lines.append('internal validation only, not reported in paper.')
    out_lines.append('='*70)
    out_lines.append(
        '\nUnadjusted (RQ1 Step 2) vs repo-stratified + bootstrap (RQ1 Step 4):\n'
    )
    step2 = read_rq1_section(rq1_results, 'step2')
    step4 = read_rq1_section(rq1_results, 'step4')
    for i, tier in enumerate(['human_supervised', 'fully_autonomous']):
        u, s = step2.loc[tier], step4.loc[tier]
        sig = '(NS)' if s['p_analytic'] >= 0.05 else '(SIG)'
        out_lines.append(('\n' if i else '') + f'  {tier}:')
        out_lines.append(
            f"    Unadjusted:  HR={u['hr']:.4f} [{u['ci_low']:.4f}, {u['ci_high']:.4f}], "
            f"p={u['p_cox']:.2e}"
        )
        out_lines.append(
            f"    Stratified:  HR={s['hr_analytic']:.4f}, bootstrap CI="
            f"[{s['boot_ci_low']:.4f}, {s['boot_ci_high']:.4f}], p={s['p_analytic']:.2e} {sig}"
        )
        out_lines.append(
            f"    Contributing strata: {int(s['n_contributing_strata'])} / "
            f"{int(s['n_total_strata'])} repos"
        )

    # ── SECTION D: Within-repo heterogeneity (most extreme examples) ──
    out_lines.append('\n' + '='*70)
    out_lines.append('SECTION D: Within-repo heterogeneity (Discussion examples)')
    out_lines.append('='*70)

    human_line = line[line['two_tier'] == 'human']
    for tier in ['human_supervised', 'fully_autonomous']:
        tier_line = line[line['two_tier'] == tier]
        per_repo = []
        for repo in tier_line['slug'].unique():
            t = tier_line[tier_line['slug'] == repo]
            h = human_line[human_line['slug'] == repo]
            if len(t) < 20 or len(h) < 20:
                continue
            per_repo.append({
                'slug': repo,
                'n_tier': len(t),
                'n_human': len(h),
                'tier_rate': round(t['event'].mean() * 100, 1),
                'human_rate': round(h['event'].mean() * 100, 1),
                'gap_pp': round((t['event'].mean() - h['event'].mean()) * 100, 1),
            })
        if not per_repo:
            continue
        repo_df = pd.DataFrame(per_repo).sort_values('gap_pp')
        out_lines.append(f'\n  {tier} -- most extreme within-repo gaps (n>=20 each):')
        out_lines.append('  Tier MUCH LESS modified than human in same repo (most negative gap):')
        out_lines.append(
            repo_df.head(3)[['slug','n_tier','n_human','tier_rate','human_rate','gap_pp']]
            .to_string(index=False)
        )
        out_lines.append('  Tier MUCH MORE modified than human in same repo (most positive gap):')
        out_lines.append(
            repo_df.tail(3)[['slug','n_tier','n_human','tier_rate','human_rate','gap_pp']]
            .to_string(index=False)
        )

    result_text = '\n'.join(out_lines)
    print(result_text)
    with open(output, 'w') as f:
        f.write(result_text)
    print(f'\nResults written to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--line-data', default=P('data/survival_events_line_v5_analysis.csv'))
    parser.add_argument('--repo-meta', default=P('data/aidev_filtered_v1.csv'))
    parser.add_argument('--rq1-results', default=P('results/rq1_master_results.txt'))
    parser.add_argument('--output',    default=P('results/rq3_results.txt'))
    args = parser.parse_args()
    args.output = output_path(args.output)
    args.rq1_results = output_path(args.rq1_results)
    run(args.line_data, args.repo_meta, args.rq1_results, args.output)