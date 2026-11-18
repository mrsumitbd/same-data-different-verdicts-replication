"""
early_churn_holdout.py

Held-out test of the early-churn effect for fully-autonomous (Devin) code.

The first-week spike (results/early_churn_check.txt) was found after looking
at the data. To test it honestly, repositories are split at random into two
halves. On half A the early window w in {3, 7, 14, 30} days that maximises the
fully-autonomous HR (0-w days, repository-stratified Cox with cluster-robust
SE) is chosen; that single window is then tested on half B. Repeated for five
seeds (42, 1, 2, 3, 4). Only repositories with both FA and human lines can
contribute to a stratified estimate; the split is drawn among those.

Usage:
    python src/early_churn_holdout.py
Output:
    results/early_churn_holdout.txt
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

WINDOWS = [3, 7, 14, 30]
TIER = 'fully_autonomous'


def fit(df, w):
    d = pd.DataFrame({'stop': np.minimum(df['duration_days'], w),
                      'ev': ((df['event'] == 1) & (df['duration_days'] <= w)).astype(int),
                      't': (df['two_tier'] == TIER).astype(int), 's': df['slug'].values})
    d = d[d.groupby('s')['t'].transform('nunique') == 2]
    d['c'] = d['s']
    c = CoxPHFitter().fit(d, 'stop', 'ev', strata=['s'], cluster_col='c')
    r = c.summary.loc['t']
    return r['exp(coef)'], r['exp(coef) lower 95%'], r['exp(coef) upper 95%'], r['p'], d['s'].nunique()


def main():
    df = add_two_tier_column(apply_observation_window(
        pd.read_csv(P('data/survival_events_line_v5_analysis.csv'), low_memory=False), label='line'))
    df = df[~df['unit_id'].apply(is_generated)]
    df = df[~pd.Series(list(zip(df['slug'], df['pr_number'])), index=df.index).isin(OUTLIER_PR_KEYS)]
    df = df[df['two_tier'].isin([TIER, 'human'])]
    both = df.groupby('slug')['two_tier'].nunique()
    repos = np.array(sorted(both[both == 2].index))
    df = df[df['slug'].isin(repos)]
    out = ['=' * 70, 'Held-out early churn, fully-autonomous vs human, repository-stratified', '=' * 70,
           f'repositories with FA and human lines: {len(repos)}; windows tried on half A: {WINDOWS} days', '']
    for w in WINDOWS:
        hr, lo, hi, p, k = fit(df, w)
        out.append(f'full sample, 0-{w} d: HR={hr:.3f} [{lo:.3f}, {hi:.3f}] p={p:.2g} ({k} repos)')
    out.append('')
    passed = 0
    for seed in [42, 1, 2, 3, 4]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(repos)
        A, B = set(perm[:len(perm) // 2]), set(perm[len(perm) // 2:])
        fa = {w: fit(df[df['slug'].isin(A)], w) for w in WINDOWS}
        best = max(WINDOWS, key=lambda w: fa[w][0])
        hr, lo, hi, p, k = fit(df[df['slug'].isin(B)], best)
        ok = lo > 1
        passed += ok
        out.append(f'seed {seed:2d}: half A picks 0-{best} d (A: HR={fa[best][0]:.3f}, p={fa[best][3]:.2g}, {fa[best][4]} repos) '
                   f'-> half B: HR={hr:.3f} [{lo:.3f}, {hi:.3f}] p={p:.2g} ({k} repos) {"REPLICATES" if ok else "does not replicate"}')
    out.append(f'\nreplicated (half-B CI above 1) in {passed} of 5 splits')
    text = '\n'.join(out) + '\n'
    open(P('results/early_churn_holdout.txt'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
