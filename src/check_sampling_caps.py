"""
check_sampling_caps.py

Is the difference between capped and uncapped line samples real?

The v2 miner kept at most MAX_FILES_PER_PR source files per PR (the first in
`git show --name-only` order) and a random MAX_LINES_PER_FILE lines per file.
This script, on the uncapped v5 data:
  (1) re-applies that cap rule (first 20 files per PR in path order, 100 random
      lines per file) with five seeds and fits the unadjusted tier HRs;
  (2) reports the share of lines the caps drop and the PR-size distribution by tier;
  (3) fits uncapped HRs with each PR weighted equally (weight = 1 / lines in
      the PR, robust SE), the estimand the caps move towards.

Usage:
    python src/check_sampling_caps.py
Output:
    results/sampling_caps_check.txt
"""

import os
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.constants import MAX_FILES_PER_PR, MAX_LINES_PER_FILE
from config.observation_window import apply_observation_window
from config.paths import P, DATA_LABEL
from analyze_rq1_master import add_two_tier_column, is_generated, OUTLIER_PR_KEYS

TIERS = ['human_supervised', 'fully_autonomous']


def hr(df, tier, weights=False):
    x = df[df['two_tier'].isin([tier, 'human'])]
    d = pd.DataFrame({'d': x['duration_days'].values, 'e': x['event'].astype(int).values,
                      't': (x['two_tier'] == tier).astype(int).values})
    if weights:
        d['w'] = x['w'].values
        c = CoxPHFitter().fit(d, 'd', 'e', weights_col='w', robust=True)
    else:
        c = CoxPHFitter().fit(d, 'd', 'e')
    r = c.summary.loc['t']
    return r['exp(coef)'], r['exp(coef) lower 95%'], r['exp(coef) upper 95%']


def cap(df, seed):
    df = df.assign(path=df['unit_id'].str.rsplit('::', n=1).str[0])
    files = df[['slug', 'pr_number', 'path']].drop_duplicates().sort_values(['slug', 'pr_number', 'path'])
    files['rank'] = files.groupby(['slug', 'pr_number']).cumcount()
    keep_files = files[files['rank'] < MAX_FILES_PER_PR].drop(columns='rank')
    x = df.merge(keep_files, on=['slug', 'pr_number', 'path'])
    rng = np.random.default_rng(seed)
    x['_r'] = rng.random(len(x))
    x['_k'] = x.groupby(['slug', 'pr_number', 'path'])['_r'].rank(method='first')
    return x[x['_k'] <= MAX_LINES_PER_FILE]


def main():
    df = add_two_tier_column(apply_observation_window(
        pd.read_csv(P('data/survival_events_line_v5_analysis.csv'), low_memory=False), label='line'))
    df = df[~df['unit_id'].apply(is_generated)]
    df = df[~pd.Series(list(zip(df['slug'], df['pr_number'])), index=df.index).isin(OUTLIER_PR_KEYS)]
    out = ['=' * 70, f'Sampling caps check (uncapped {DATA_LABEL} data)', '=' * 70]
    base = {t: hr(df, t) for t in TIERS}
    out.append('Uncapped, line-weighted: ' + '; '.join(f'{t} HR={v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]' for t, v in base.items()))
    out.append(f'\n(1) Old cap rule re-applied ({MAX_FILES_PER_PR} files per PR, {MAX_LINES_PER_FILE} random lines per file):')
    for seed in [42, 1, 2, 3, 4]:
        c = cap(df, seed)
        res = {t: hr(c, t) for t in TIERS}
        out.append(f'  seed {seed:2d}: n={len(c):,}  ' + '; '.join(f'{t} HR={v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]' for t, v in res.items()))
    c = cap(df, 42)
    out.append('\n(2) Share of lines dropped by the caps (seed 42), and lines per PR:')
    for t, g in df.groupby('two_tier'):
        kept = (c['two_tier'] == t).sum()
        sizes = g.groupby(['slug', 'pr_number']).size()
        out.append(f'  {t:17s} lines {len(g):7,d} kept {kept:7,d} dropped {100 * (1 - kept / len(g)):5.1f}% | '
                   f'PRs {len(sizes):4d}, lines/PR median {sizes.median():.0f}, IQR {sizes.quantile(.25):.0f}-{sizes.quantile(.75):.0f}, '
                   f'max {sizes.max():,}; top 5 PRs hold {100 * sizes.nlargest(5).sum() / len(g):.1f}% of lines')
    df['w'] = 1.0 / df.groupby(['slug', 'pr_number'])['unit_id'].transform('size')
    out.append('\n(3) Uncapped, each PR weighted equally (weight = 1 / lines in PR, robust SE):')
    for t in TIERS:
        v = hr(df, t, weights=True)
        out.append(f'  {t:17s} HR={v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]')
    text = '\n'.join(out) + '\n'
    open(P('results/sampling_caps_check.txt'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
