"""
check_early_churn.py

What drives the first-week spike in the tier HR (results/ph_piecewise.txt)?

For modifications within 7 days of merge, by tier:
  - who made them: the PR's own author (same GitHub login as the birth PR's
    author), another agent, or another human (modifier login rules:
    config/agent_logins.py via results/modifier_authors.csv);
  - whitespace-only share;
  - concentration: share of early modifications in the 5 PRs with the most
    early modifications;
  - intent mix (majority vote of the three LLM labels; keyword for reference).
Then refits the 0-7 day HR (delayed-entry Cox, unadjusted and repository-
stratified) treating (a) self-modifications and (b) whitespace-only changes as
non-events (the line is censored at that time).

Usage:
    python src/check_early_churn.py
Output:
    results/early_churn_check.txt
"""

import os
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.observation_window import apply_observation_window
from config.paths import P
from analyze_rq1_master import add_two_tier_column, is_generated, OUTLIER_PR_KEYS
from measure_retention_line_v2 import normalize_slug

TIERS = ['human_supervised', 'fully_autonomous']
EARLY = 7.0


def hr_0_7(df, tier, stratified):
    x = df[df['two_tier'].isin([tier, 'human'])]
    d = pd.DataFrame({'stop': np.minimum(x['duration_days'], EARLY),
                      'ev': ((x['event'] == 1) & (x['duration_days'] <= EARLY)).astype(int),
                      't': (x['two_tier'] == tier).astype(int)})
    if stratified:
        d['s'] = x['slug'].values
        d = d[d.groupby('s')['t'].transform('nunique') == 2]
        d['c'] = d['s']
        c = CoxPHFitter().fit(d, 'stop', 'ev', strata=['s'], cluster_col='c')
    else:
        c = CoxPHFitter().fit(d, 'stop', 'ev')
    r = c.summary.loc['t']
    return f"HR={r['exp(coef)']:.3f} [{r['exp(coef) lower 95%']:.3f}, {r['exp(coef) upper 95%']:.3f}] p={r['p']:.2g}"


def main():
    df = add_two_tier_column(apply_observation_window(
        pd.read_csv(P('data/survival_events_line_v5_analysis.csv'), low_memory=False), label='line'))
    df = df[~df['unit_id'].apply(is_generated)]
    df = df[~pd.Series(list(zip(df['slug'], df['pr_number'])), index=df.index).isin(OUTLIER_PR_KEYS)]

    # birth PR author
    a = pd.read_parquet(P('data/all_pull_request.parquet'), columns=['number', 'repo_url', 'user'])
    h = pd.read_parquet(P('data/human_pull_request.parquet'), columns=['number', 'repo_url', 'user'])
    prs = pd.concat([a.assign(is_agent=True), h.assign(is_agent=False)])
    prs['slug'] = prs['repo_url'].apply(normalize_slug)
    prs = prs.rename(columns={'number': 'pr_number', 'user': 'pr_author'})[['slug', 'pr_number', 'is_agent', 'pr_author']]
    df = df.merge(prs.drop_duplicates(['slug', 'pr_number', 'is_agent']), on=['slug', 'pr_number', 'is_agent'], how='left')
    mods = pd.read_csv(P('results/modifier_authors.csv')).rename(columns={'sha': 'death_sha'})
    df = df.merge(mods[['slug', 'death_sha', 'author_login', 'modifier_type']], on=['slug', 'death_sha'], how='left')

    early = df[(df['event'] == 1) & (df['duration_days'] <= EARLY)].copy()
    same = early['author_login'].str.lower() == early['pr_author'].astype(str).str.lower()
    early['who'] = np.where(same, 'same author as the PR',
                            np.where(early['modifier_type'] == 'agent', 'another agent', 'another human'))
    early['ws'] = early['whitespace_only'].astype(str).eq('True')

    labels = {m: pd.read_csv(P(f'results/intent_{m}_full_corpus_t0.csv'))[['slug', 'sha', 'label']]
              .drop_duplicates(['slug', 'sha']).rename(columns={'sha': 'death_sha', 'label': m})
              for m in ['deepseek', 'gpt_oss', 'minimax']}
    for m, l in labels.items():
        early = early.merge(l, on=['slug', 'death_sha'], how='left')
    vote = early[['deepseek', 'gpt_oss', 'minimax']].mode(axis=1)
    early['intent'] = np.where(early[['deepseek', 'gpt_oss', 'minimax']].nunique(axis=1) == 3, 'undecided', vote[0])

    out = ['=' * 70, f'Early churn: modifications within {EARLY:g} days of merge', '=' * 70]
    tot = df.groupby('two_tier').size()
    out.append('Early modifications as % of units: ' + ', '.join(f'{t} {v:.1f}%' for t, v in (early.groupby('two_tier').size() / tot * 100).items()))
    out += ['', 'Who made them (% of early modifications):',
            (pd.crosstab(early['two_tier'], early['who'], normalize='index') * 100).round(1).to_string()]
    out.append('\nWhitespace-only (% of early modifications): ' + ', '.join(f'{t} {v:.1f}%' for t, v in (early.groupby('two_tier')['ws'].mean() * 100).items()))
    out.append('Event types (% of early modifications):')
    out.append((pd.crosstab(early['two_tier'], early['event_type'], normalize='index') * 100).round(1).to_string())
    out.append('\nConcentration: share of early modifications in the 5 PRs with the most early modifications:')
    for t, g in early.groupby('two_tier'):
        c = g.groupby(['slug', 'pr_number']).size()
        out.append(f'  {t:17s} {100 * c.nlargest(5).sum() / len(g):.1f}% (PRs with early modifications: {len(c)})')
    out += ['', 'Intent of early modifications (LLM majority vote, % by tier):',
            (pd.crosstab(early['two_tier'], early['intent'], normalize='index') * 100).round(1).to_string()]

    out += ['', '0-7 day HR vs human:']
    variants = {'all modifications': df}
    x = df.copy()
    is_self = pd.Series(False, index=x.index)
    is_self.loc[early.index] = same.values
    x.loc[is_self.reindex(x.index, fill_value=False), 'event'] = 0
    variants['self-modifications treated as non-events'] = x
    y = df.copy()
    wso = (y['event'] == 1) & y['whitespace_only'].astype(str).eq('True')
    y.loc[wso, 'event'] = 0
    variants['whitespace-only changes treated as non-events'] = y
    for name, d in variants.items():
        out.append(f'  {name}:')
        for t in TIERS:
            out.append(f'    {t:17s} unadjusted {hr_0_7(d, t, False)} | repo-stratified {hr_0_7(d, t, True)}')
    text = '\n'.join(out) + '\n'
    open(P('results/early_churn_check.txt'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
